"""`studiomix serve`: a small local web app for phones (Termux) and desktops.

Standard library only. Uploads go to a job folder, one job is processed at a time in a
background thread, and the page polls for progress. By default it listens on 127.0.0.1, so
only the device it runs on can reach it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import webbrowser
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__
from .presets import PRESETS, TUNE_STYLES, get_preset

MAX_UPLOAD = 1024 * 1024 * 1024  # 1 GB per request
AUDIO_EXT = {".wav", ".wave", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a", ".aac", ".opus"}

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_worker_lock = threading.Lock()  # phones are slow: one mix at a time


def default_jobs_dir() -> Path:
    downloads = Path.home() / "storage" / "downloads"  # Termux, after termux-setup-storage
    if downloads.is_dir():
        return downloads / "studiomix"
    return Path.cwd() / "studiomix-jobs"


def _safe_name(name: str, fallback: str) -> str:
    base = Path(name or "").name
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(base).stem).strip(" .") or fallback
    ext = Path(base).suffix.lower()
    return stem[:80] + (ext if ext in AUDIO_EXT else ".wav")


def _parse_multipart(content_type: str, body: bytes):
    msg = BytesParser(policy=policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    fields: dict[str, list[str]] = {}
    files: dict[str, list[tuple[str, bytes]]] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        data = part.get_payload(decode=True) or b""
        if filename is not None:
            if data:
                files.setdefault(name, []).append((filename, data))
        else:
            fields.setdefault(name, []).append(data.decode("utf-8", "replace"))
    return fields, files


def _run_job(job: dict) -> None:
    from .cli import check_harmonies, parse_time_ranges
    from .dsp import pitch
    from .engine import master_mix, run

    def progress(msg: str) -> None:
        with _jobs_lock:
            job["log"].append({"t": round(time.time() - job["started"], 1), "msg": msg})

    with _worker_lock:
        with _jobs_lock:
            job["status"] = "running"
            job["started"] = time.time()
        try:
            o = job["options"]
            prof = None
            if job["mode"] == "learn":
                from . import profiles

                p_ = profiles.learn(job["refs"], job["name"], progress=progress)
                with _jobs_lock:
                    job["status"] = "done"
                    job["result"] = {"profile": p_["name"], "text": profiles.describe(p_), "files": {}}
                return
            if o.get("profile"):
                from . import profiles

                prof = profiles.load(o["profile"])
                job["preset"] = profiles.apply(prof, job["preset"], keep_loudness=o.get("lufs_set", False))
            if job["mode"] == "mix":
                log = master_mix(job["mix"], job["out"], job["preset"], name=job["name"],
                                 reference_path=job.get("reference"), vocal_lift_db=o.get("vocal_lift", 0.0),
                                 deliver_extra=o.get("deliver"), verbose=False, progress=progress, profile=prof)
            else:
                if job["mode"] == "studio":
                    from . import studio

                    progress("laying your takes on the beat's timeline")
                    tr = studio.build_tracks(job["root"], job["session"])
                    job["lead"], job["beat"] = tr["lead"], tr["beat"]
                    job["adlibs"] = [tr["adlib"]] if "adlib" in tr else []
                if o.get("key"):
                    pitch.parse_key(o["key"])
                check_harmonies(o.get("harmonies"))
                log = run(
                    job["lead"], job["beat"], job["out"], job["preset"], name=job["name"],
                    adlib_paths=job["adlibs"], key=o.get("key") or None, key_changes=o.get("key_changes", False),
                    stack_at=parse_time_ranges(o.get("stack_at")), verbose=False, progress=progress,
                    deliver_extra=o.get("deliver"), profile=prof,
                )
            with _jobs_lock:
                job["status"] = "done"
                job["result"] = {
                    "output": log["output"],
                    "key": log.get("key", {}).get("key", "—" if job["mode"] == "mix" else "tuning off"),
                    "findings": log.get("diagnosis", {}).get("findings", []),
                    "checks": log["delivery_check"], "files": log["files"],
                    "seconds": log["processing_seconds"],
                }
        except Exception as e:  # report every failure to the page
            traceback.print_exc()
            with _jobs_lock:
                job["status"] = "error"
                job["error"] = str(e) or e.__class__.__name__


class Handler(BaseHTTPRequestHandler):
    server_version = f"studiomix/{__version__}"
    jobs_dir: Path = Path(".")

    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            return self._send(200, render_page().encode(), "text/html; charset=utf-8")
        m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})", self.path)
        if m:
            with _jobs_lock:
                job = _jobs.get(m.group(1))
                if not job:
                    return self._json(404, {"error": "no such job"})
                view = {k: job.get(k) for k in ("id", "status", "log", "error", "result", "name", "mode")}
                view["out_dir"] = str(job["out"])
            return self._json(200, view)
        m = re.fullmatch(r"/api/studio/([A-Za-z0-9_-]{1,40})", self.path)
        if m:
            from . import studio

            return self._json(200, studio.info(self.jobs_dir, m.group(1)))
        m = re.fullmatch(r"/studio/([A-Za-z0-9_-]{1,40})/(beat|take)(?:/([0-9a-f]{10}))?", self.path)
        if m:
            from . import studio

            path = studio.file_path(self.jobs_dir, m.group(1), m.group(2), m.group(3))
            if path is None or not path.exists():
                return self._json(404, {"error": "no such file"})
            ctype = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".flac": "audio/flac",
                     ".m4a": "audio/mp4", ".aac": "audio/aac"}.get(path.suffix, "application/octet-stream")
            return self._send(200, path.read_bytes(), ctype)
        m = re.fullmatch(r"/jobs/([0-9a-f]{12})/([^/]+)", self.path)
        if m:
            with _jobs_lock:
                job = _jobs.get(m.group(1))
                allowed = set((job or {}).get("result", {}).get("files", {}).values()) if job else set()
            from urllib.parse import unquote

            fname = unquote(m.group(2))
            if not job or fname not in allowed:
                return self._json(404, {"error": "no such file"})
            path = Path(job["out"]) / fname
            ctype = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".json": "application/json",
                     ".txt": "text/plain; charset=utf-8"}.get(path.suffix, "application/octet-stream")
            return self._send(200, path.read_bytes(), ctype,
                              {"Content-Disposition": f'inline; filename="{fname}"'})
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        st = re.fullmatch(r"/api/studio/([A-Za-z0-9_-]{1,40})/(beat|take|delete/([0-9a-f]{10}))", self.path)
        if self.path not in ("/api/jobs", "/api/learn") and not st:
            return self._json(404, {"error": "not found"})
        if st and st.group(3):
            from . import studio

            studio.delete_take(self.jobs_dir, st.group(1), st.group(3))
            return self._json(200, studio.info(self.jobs_dir, st.group(1)))
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD:
            return self._json(413, {"error": "upload too large (max 1 GB)"})
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            return self._json(400, {"error": "expected a form upload"})
        fields, files = _parse_multipart(ctype, self.rfile.read(length))
        f = lambda k, d="": (fields.get(k) or [d])[0].strip()  # noqa: E731
        if st:
            from . import studio

            try:
                if st.group(2) == "beat":
                    if not files.get("beat"):
                        return self._json(400, {"error": "choose a beat"})
                    studio.save_beat(self.jobs_dir, st.group(1), *files["beat"][0])
                else:
                    if not files.get("take"):
                        return self._json(400, {"error": "no audio received"})
                    studio.add_take(self.jobs_dir, st.group(1), files["take"][0][1], f("role", "lead"),
                                    float(f("offset_s", "0") or 0), float(f("latency_ms", "0") or 0))
            except (ValueError, RuntimeError) as e:
                return self._json(400, {"error": str(e)})
            return self._json(200, studio.info(self.jobs_dir, st.group(1)))
        if self.path == "/api/learn":
            if not files.get("refs"):
                return self._json(400, {"error": "choose one or more reference songs"})
            from . import profiles

            try:
                pname = profiles._safe(f("profile_name") or "my-sound")
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            job_id = uuid.uuid4().hex[:12]
            ref_dir = self.jobs_dir / "references" / pname
            ref_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for i, it in enumerate(files["refs"]):
                pth = ref_dir / _safe_name(it[0], f"ref{i + 1}")
                pth.write_bytes(it[1])
                paths.append(pth)
            job = {"id": job_id, "status": "queued", "log": [], "name": pname, "out": ref_dir, "mode": "learn",
                   "refs": paths, "options": {}, "started": time.time(), "preset": None}
            with _jobs_lock:
                _jobs[job_id] = job
            threading.Thread(target=_run_job, args=(job,), daemon=True).start()
            return self._json(200, {"id": job_id})

        mode = f("mode") if f("mode") in ("mix", "studio") else "stems"
        if mode == "studio":
            from . import studio

            try:
                info = studio.info(self.jobs_dir, f("session"))
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            if not info["beat"] or not any(t["role"] == "lead" for t in info["takes"]):
                return self._json(400, {"error": "load a beat and record at least one lead take first"})
        if mode == "mix" and not files.get("mix"):
            return self._json(400, {"error": "choose your mix file"})
        if mode == "stems" and (not files.get("lead") or not files.get("beat")):
            return self._json(400, {"error": "a lead vocal and a beat are required"})
        from .cli import parse_deliver

        try:
            deliver = parse_deliver(",".join(fields.get("deliver", [])))
        except ValueError as e:
            return self._json(400, {"error": str(e)})

        preset_name = f("preset", "pop")
        if preset_name not in PRESETS:
            return self._json(400, {"error": f"unknown preset {preset_name}"})
        overrides: dict = {}
        tune = f("tune")
        if tune in TUNE_STYLES:
            r, h, a = TUNE_STYLES[tune]
            overrides.update(tune_retune_ms=r, tune_humanize=h, tune_amount=a)
        try:
            if f("flex"):
                overrides["tune_flex_cents"] = float(f("flex"))
            if f("lufs"):
                overrides["target_lufs"] = float(f("lufs"))
            if f("vocal_level"):
                overrides["vocal_balance_db"] = float(f("vocal_level"))
        except ValueError:
            return self._json(400, {"error": "numbers only for flex / loudness / vocal level"})
        if f("doubles") == "on":
            overrides["doubles"] = True
        harmonies = ",".join(fields.get("harmony", []))
        if harmonies:
            overrides["harmonies"] = harmonies
        preset = get_preset(preset_name, **overrides)

        job_id = uuid.uuid4().hex[:12]
        first = f("session") if mode == "studio" else files["mix" if mode == "mix" else "lead"][0][0]
        name = re.sub(r"[^A-Za-z0-9._ -]+", "_", f("name") or Path(first).stem)[:60] or "song"
        out = self.jobs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
        src = out / "inputs"
        src.mkdir(parents=True, exist_ok=True)

        def save(item, fallback):
            p = src / _safe_name(item[0], fallback)
            p.write_bytes(item[1])
            return p

        job = {
            "id": job_id, "status": "queued", "log": [], "name": name, "out": out, "preset": preset, "mode": mode,
            "options": {"key": f("key"), "key_changes": f("key_changes") == "on", "harmonies": harmonies,
                        "stack_at": f("stack_at"), "deliver": deliver,
                        "vocal_lift": 2.0 if f("vocal_lift") == "on" else 0.0,
                        "profile": f("profile"), "lufs_set": bool(f("lufs"))},
            "started": time.time(),
        }
        if mode == "studio":
            job["root"], job["session"] = self.jobs_dir, f("session")
        elif mode == "mix":
            job["mix"] = save(files["mix"][0], "mix")
            if files.get("reference"):
                job["reference"] = save(files["reference"][0], "reference")
        else:
            job["lead"] = save(files["lead"][0], "lead")
            job["beat"] = save(files["beat"][0], "beat")
            job["adlibs"] = [save(it, f"adlib{i + 1}") for i, it in enumerate(files.get("adlibs", []))]
        with _jobs_lock:
            _jobs[job_id] = job
        threading.Thread(target=_run_job, args=(job,), daemon=True).start()
        return self._json(200, {"id": job_id})


def render_page() -> str:
    presets = "".join(f'<option value="{p.name}"{" selected" if p.name == "pop" else ""}>'
                      f'{p.name} - {p.description}</option>' for p in PRESETS.values())
    from . import profiles

    profs = "".join(f'<option value="{n}">{n}</option>' for n in profiles.list_profiles())
    return (PAGE.replace("{{PRESETS}}", presets).replace("{{VERSION}}", __version__)
            .replace("{{PROFILES}}", profs))


def serve_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="studiomix serve", description="Run the studiomix app in your browser.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = only this device (default); 0.0.0.0 = anyone on your Wi-Fi")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--jobs-dir", type=Path, default=None,
                    help="where songs are saved (default: Downloads/studiomix on Termux, else ./studiomix-jobs)")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    a = ap.parse_args(argv)
    Handler.jobs_dir = (a.jobs_dir or default_jobs_dir()).expanduser()
    Handler.jobs_dir.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}/"
    print(f"studiomix {__version__} is running at {url}")
    print(f"songs are saved in {Handler.jobs_dir}")
    print("keep this window open while mixing; press Ctrl+C to stop")
    if not a.no_open:
        if shutil.which("termux-open-url"):
            subprocess.Popen(["termux-open-url", url])
        elif os.environ.get("DISPLAY") or os.name == "nt" or shutil.which("open"):
            webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="dark light">
<title>studiomix</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%23ff5a36'/%3E%3Cpath d='M9 20v-8M14 24V8M19 21V11M24 18v-4' stroke='white' stroke-width='3' stroke-linecap='round'/%3E%3C/svg%3E">
<style>
:root {
  --bg: #0e0f12; --card: #17191e; --line: #2a2d35; --text: #eceef2; --muted: #9aa0ab;
  --accent: #ff5a36; --accent-ink: #fff; --ok: #3ccf8e; --warn: #f5b942; --chip: #22252c;
}
@media (prefers-color-scheme: light) {
  :root { --bg: #f5f5f2; --card: #ffffff; --line: #e1e1dc; --text: #16171a; --muted: #62666f;
          --chip: #efefea; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 16px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; -webkit-tap-highlight-color: transparent; }
main { max-width: 640px; margin: 0 auto; padding: 20px 16px 96px; }
header { display: flex; align-items: baseline; justify-content: space-between; margin: 4px 2px 18px; }
h1 { font-size: 26px; letter-spacing: -0.02em; margin: 0; }
h1 span { color: var(--accent); }
.ver { color: var(--muted); font-size: 13px; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 16px; margin-bottom: 14px; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 0 0 12px; font-weight: 600; }
label.file { display: flex; align-items: center; gap: 12px; padding: 12px; border: 1px dashed var(--line);
  border-radius: 12px; margin-bottom: 10px; cursor: pointer; min-height: 56px; }
label.file:last-child { margin-bottom: 0; }
label.file input { display: none; }
.file .icon { width: 36px; height: 36px; border-radius: 10px; background: var(--chip); display: grid; place-items: center; flex: none; font-size: 18px; }
.file .t { font-weight: 600; } .file .s { color: var(--muted); font-size: 13px; word-break: break-all; }
.file.has { border-style: solid; border-color: var(--accent); }
.row { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 0; border-top: 1px solid var(--line); }
.row:first-of-type { border-top: 0; padding-top: 0; }
.row .lbl { font-weight: 500; } .row .hint { color: var(--muted); font-size: 13px; }
select, input[type=text], input[type=number] { font: inherit; color: var(--text); background: var(--chip);
  border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; width: 100%; }
.seg { display: grid; grid-template-columns: repeat(4, 1fr); background: var(--chip); border-radius: 12px; padding: 4px; gap: 4px; }
.seg input { display: none; }
.seg label { text-align: center; padding: 9px 0; border-radius: 9px; font-weight: 600; font-size: 14px; cursor: pointer; color: var(--muted); }
.seg input:checked + label { background: var(--card); color: var(--text); box-shadow: 0 1px 2px rgba(0,0,0,.25); }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chips input { display: none; }
.chips label { padding: 8px 12px; border-radius: 999px; background: var(--chip); border: 1px solid var(--line); font-size: 14px; cursor: pointer; }
.chips input:checked + label { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
.switch { position: relative; width: 50px; height: 30px; flex: none; }
.switch input { opacity: 0; width: 0; height: 0; }
.switch span { position: absolute; inset: 0; background: var(--line); border-radius: 999px; transition: .2s; }
.switch span:before { content: ""; position: absolute; width: 24px; height: 24px; left: 3px; top: 3px; background: #fff; border-radius: 50%; transition: .2s; }
.switch input:checked + span { background: var(--accent); }
.switch input:checked + span:before { transform: translateX(20px); }
.stack { display: grid; gap: 8px; width: 100%; }
details summary { cursor: pointer; color: var(--muted); font-weight: 600; font-size: 14px; }
details[open] summary { margin-bottom: 10px; }
.go { position: fixed; left: 0; right: 0; bottom: 0; padding: 12px 16px calc(12px + env(safe-area-inset-bottom));
  background: linear-gradient(transparent, var(--bg) 35%); }
.go button { display: block; width: 100%; max-width: 608px; margin: 0 auto; border: 0; border-radius: 14px; padding: 16px;
  font: 700 17px system-ui, sans-serif; background: var(--accent); color: var(--accent-ink); cursor: pointer; }
.go button:disabled { opacity: .5; }
#progress, #result { display: none; }
.bar { height: 6px; background: var(--chip); border-radius: 999px; overflow: hidden; margin: 6px 0 14px; }
.bar i { display: block; height: 100%; width: 0; background: var(--accent); transition: width .3s; }
.steps { list-style: none; margin: 0; padding: 0; font-size: 14px; }
.steps li { padding: 6px 0; border-top: 1px solid var(--line); display: flex; gap: 10px; }
.steps li:first-child { border-top: 0; }
.steps .t { color: var(--muted); width: 48px; flex: none; font-variant-numeric: tabular-nums; }
.steps li.now .m:after { content: " …"; }
.big { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px; }
.big div { background: var(--chip); border-radius: 12px; padding: 10px; }
.big b { display: block; font-size: 20px; font-variant-numeric: tabular-nums; }
.big span { color: var(--muted); font-size: 12px; }
audio { width: 100%; margin: 4px 0 12px; }
.checks { list-style: none; padding: 0; margin: 0 0 12px; font-size: 14px; }
.checks li { padding: 4px 0; } .checks .ok { color: var(--ok); } .checks .warn { color: var(--warn); }
.dl a { display: flex; justify-content: space-between; padding: 12px; border: 1px solid var(--line); border-radius: 12px;
  margin-bottom: 8px; color: var(--text); text-decoration: none; }
.dl a span { color: var(--muted); font-size: 13px; }
.err { color: #ff6b6b; font-weight: 600; }
button.small { margin-top: 10px; width: 100%; border: 1px solid var(--accent); background: transparent; color: var(--accent);
  border-radius: 12px; padding: 12px; font: 700 15px system-ui, sans-serif; cursor: pointer; }
.modes { grid-template-columns: 1fr 1fr 1fr; margin-bottom: 14px; }
.modes label { padding: 11px 0; font-size: 14px; }
body[data-mode=mix] .m-vox, body[data-mode=mix] .m-stems, body[data-mode=mix] .m-rec,
body[data-mode=stems] .m-mix, body[data-mode=stems] .m-rec,
body[data-mode=studio] .m-mix, body[data-mode=studio] .m-stems { display: none; }
.rec { display: flex; gap: 10px; align-items: center; margin-top: 12px; }
.recbtn { width: 76px; height: 76px; border-radius: 50%; border: 0; background: #e5322d; color: #fff;
  font: 800 15px system-ui, sans-serif; flex: none; cursor: pointer; box-shadow: 0 0 0 4px rgba(229,50,45,.25); }
.recbtn.on { animation: pulse 1s infinite; border-radius: 22px; }
@keyframes pulse { 50% { box-shadow: 0 0 0 10px rgba(229,50,45,.15); } }
.meter { flex: 1; }
.meter .lvl { height: 10px; background: var(--chip); border-radius: 999px; overflow: hidden; }
.meter .lvl i { display: block; height: 100%; width: 0; background: var(--ok); transition: width 60ms; }
.meter .lvl i.hot { background: #ff4d4d; }
.meter .time { font: 700 26px ui-monospace, monospace; margin-top: 6px; font-variant-numeric: tabular-nums; }
.takes { list-style: none; padding: 0; margin: 10px 0 0; }
.takes li { display: flex; align-items: center; gap: 8px; padding: 10px 0; border-top: 1px solid var(--line); font-size: 14px; }
.takes li .grow { flex: 1; } .takes .warn { color: var(--warn); font-size: 12px; }
.takes button, .mini { border: 1px solid var(--line); background: var(--chip); color: var(--text); border-radius: 10px;
  padding: 8px 12px; font: 600 14px system-ui, sans-serif; cursor: pointer; }
.tag { font-size: 11px; font-weight: 700; padding: 3px 7px; border-radius: 6px; background: var(--chip); text-transform: uppercase; }
.tag.lead { background: var(--accent); color: var(--accent-ink); }
#diag { margin: 0 0 12px; padding-left: 18px; font-size: 14px; }
#diag li { margin: 4px 0; }
.muted { color: var(--muted); font-size: 13px; }
</style>
</head>
<body>
<main>
<header><h1>studio<span>mix</span></h1><div class="ver">v{{VERSION}}</div></header>

<form id="f">
<div class="seg modes">
  <input type="radio" name="mode" id="m0" value="studio"><label for="m0">🎙️ Record</label>
  <input type="radio" name="mode" id="m1" value="stems" checked><label for="m1">🎤 Upload vocals</label>
  <input type="radio" name="mode" id="m2" value="mix"><label for="m2">🎚️ Finished mix</label>
</div>

<section class="m-rec" id="studio">
  <h2>Studio</h2>
  <div class="row"><div class="stack"><div class="lbl">Song</div>
    <input type="text" id="session" placeholder="Song name, e.g. night-drive" autocomplete="off"></div></div>
  <label class="file"><input type="file" id="studiobeat" accept="audio/*,.wav,.mp3,.flac,.m4a">
    <div class="icon">🥁</div><div><div class="t">Beat</div><div class="s" id="beatname">Tap to choose (saved with the song)</div></div></label>
  <div class="row"><div class="stack"><div class="lbl">Microphone</div><select id="micsel"><option value="">Default microphone</option></select>
    <div class="hint">Best: the phone's own mic or wired earbuds. Bluetooth earbuds: fine for listening, poor as a mic.</div></div></div>
  <div class="row"><div><div class="lbl">Earbud delay <b id="latval">0 ms</b></div>
    <div class="hint">Calibrate once per pair of earbuds: hold an earbud against the phone's mic, tap Calibrate.</div></div>
    <button type="button" class="mini" id="calib">Calibrate</button></div>
  <div class="row"><div class="stack"><div class="lbl">Recording</div>
    <div class="seg" style="grid-template-columns:1fr 1fr">
      <input type="radio" name="role" id="r1" value="lead" checked><label for="r1">Lead</label>
      <input type="radio" name="role" id="r2" value="adlib"><label for="r2">Ad-lib</label>
    </div>
    <div style="display:flex;gap:8px;margin-top:8px">
      <input type="text" id="startat" value="0:00" style="width:110px" aria-label="start at">
      <div class="hint" style="align-self:center">start at (the beat starts 3 s before, so you hear the lead-in)</div>
    </div>
    <div class="rec">
      <button type="button" class="recbtn" id="recbtn">REC</button>
      <div class="meter"><div class="lvl"><i id="lvl"></i></div><div class="time" id="rectime">0:00.0</div>
        <div class="hint" id="rechint">Earbuds in, tap REC, perform, tap STOP.</div></div>
    </div>
    <label style="display:flex;gap:8px;align-items:center;margin-top:8px" class="hint">Beat volume
      <input type="range" id="beatvol" min="0" max="1" step="0.05" value="0.8" style="flex:1"></label>
  </div></div>
  <ul class="takes" id="takes"></ul>
</section>

<section class="m-mix">
  <h2>Your mix</h2>
  <label class="file"><input type="file" name="mix" accept="audio/*,.wav,.mp3,.flac,.m4a">
    <div class="icon">🎚️</div><div><div class="t">Rough / finished mix</div><div class="s">Tap to choose · one stereo file, WAV best</div></div></label>
  <label class="file"><input type="file" name="reference" accept="audio/*,.wav,.mp3,.flac,.m4a">
    <div class="icon">⭐</div><div><div class="t">Reference song <span class="muted">(optional)</span></div><div class="s">A released track whose sound you want</div></div></label>
  <div class="row"><div><div class="lbl">Bring vocals forward</div><div class="hint">+2 dB presence on the centre (lead vocal)</div></div>
    <label class="switch"><input type="checkbox" name="vocal_lift"><span></span></label></div>
  <p class="muted" style="margin:6px 2px 0">It checks for clipping, phase, mud, harshness and silence, fixes what it can, then masters. Auto-tune needs the separate vocal (use "Vocal + beat").</p>
</section>

<section class="m-stems">
  <h2>Tracks</h2>
  <label class="file" data-for="lead"><input type="file" name="lead" accept="audio/*,.wav,.mp3,.flac,.m4a" required>
    <div class="icon">🎤</div><div><div class="t">Lead vocal</div><div class="s">Tap to choose · dry, no effects</div></div></label>
  <label class="file" data-for="beat"><input type="file" name="beat" accept="audio/*,.wav,.mp3,.flac,.m4a" required>
    <div class="icon">🥁</div><div><div class="t">Beat / instrumental</div><div class="s">Tap to choose</div></div></label>
  <label class="file" data-for="adlibs"><input type="file" name="adlibs" accept="audio/*,.wav,.mp3,.flac,.m4a" multiple>
    <div class="icon">🔥</div><div><div class="t">Ad-libs <span class="muted">(optional, several ok)</span></div><div class="s">Tap to choose</div></div></label>
  <p class="muted" style="margin:6px 2px 0">Export every track from the same start point (bar 1).</p>
</section>

<section>
  <h2>Sound</h2>
  <div class="row"><div class="stack"><div class="lbl">Style</div><select name="preset">{{PRESETS}}</select></div></div>
  <div class="row"><div class="stack"><div class="lbl">Sound like <span class="muted">(your reference library)</span></div>
    <select name="profile" id="profsel"><option value="">— no reference profile —</option>{{PROFILES}}</select>
    <div class="hint">Matches loudness, tonal balance and stereo width of the songs you taught it</div></div></div>
  <div class="row m-vox"><div class="stack"><div class="lbl">Auto-tune</div>
    <div class="seg">
      <input type="radio" name="tune" id="t0" value="off"><label for="t0">Off</label>
      <input type="radio" name="tune" id="t1" value="natural"><label for="t1">Natural</label>
      <input type="radio" name="tune" id="t2" value="" checked><label for="t2">Style</label>
      <input type="radio" name="tune" id="t3" value="hard"><label for="t3">Hard</label>
    </div><div class="hint">"Style" uses the preset's own setting · Hard = instant robotic snap</div></div></div>
  <div class="row m-vox"><div class="stack"><div class="lbl">Key</div>
    <input type="text" name="key" placeholder="Auto-detect (or e.g. F# minor)" autocomplete="off"></div></div>
  <div class="row m-vox"><div><div class="lbl">Song changes key</div><div class="hint">Detect a key per section</div></div>
    <label class="switch"><input type="checkbox" name="key_changes"><span></span></label></div>
  <div class="row m-vox"><div><div class="lbl">Flex-Tune</div><div class="hint">Keep intentional bends & runs natural</div></div>
    <label class="switch"><input type="checkbox" id="flexon"><span></span></label></div>
</section>

<section class="m-vox">
  <h2>Vocal stack</h2>
  <div class="row"><div><div class="lbl">Doubles</div><div class="hint">Two thick double-tracks, wide</div></div>
    <label class="switch"><input type="checkbox" name="doubles"><span></span></label></div>
  <div class="row"><div class="stack"><div class="lbl">Harmonies <span class="muted">(in key)</span></div>
    <div class="chips">
      <input type="checkbox" name="harmony" value="3up" id="h1"><label for="h1">3rd up</label>
      <input type="checkbox" name="harmony" value="3down" id="h2"><label for="h2">3rd down</label>
      <input type="checkbox" name="harmony" value="5up" id="h3"><label for="h3">5th up</label>
      <input type="checkbox" name="harmony" value="5down" id="h4"><label for="h4">5th down</label>
      <input type="checkbox" name="harmony" value="8up" id="h5"><label for="h5">Octave up</label>
      <input type="checkbox" name="harmony" value="8down" id="h6"><label for="h6">Octave down</label>
    </div></div></div>
  <div class="row"><div class="stack"><div class="lbl">Only on the hook</div>
    <input type="text" name="stack_at" placeholder="Everywhere (or e.g. 0:45-1:15, 2:10-2:40)" autocomplete="off"></div></div>
</section>

<section>
  <h2>Extra versions</h2>
  <div class="chips">
    <input type="checkbox" name="deliver" value="ebu-r128" id="d1"><label for="d1">Radio/TV EU · -23 LUFS</label>
    <input type="checkbox" name="deliver" value="atsc-a85" id="d2"><label for="d2">Radio/TV US · -24 LKFS</label>
    <input type="checkbox" name="deliver" value="apple" id="d3"><label for="d3">Apple Music · -16 LUFS</label>
  </div>
  <p class="muted" style="margin:10px 2px 0">Your main master is always made. These are extra files for stations or platforms that ask for them, each checked against its spec.</p>
</section>

<section>
  <details><summary>More options</summary>
    <div class="row"><div class="stack"><div class="lbl">Song name</div><input type="text" name="name" placeholder="From the vocal file name"></div></div>
    <div class="row"><div class="stack"><div class="lbl">Loudness (LUFS)</div><input type="number" name="lufs" step="0.5" placeholder="Preset default (e.g. -9 trap, -14 streaming)"></div></div>
    <div class="row m-vox"><div class="stack"><div class="lbl">Vocal level vs beat (dB)</div><input type="number" name="vocal_level" step="0.5" placeholder="Preset default"></div></div>
  </details>
</section>
</form>

<section>
  <details id="libbox"><summary>Reference library — teach it a sound</summary>
    <form id="lf">
      <label class="file"><input type="file" name="refs" accept="audio/*,.wav,.mp3,.flac,.m4a" multiple required>
        <div class="icon">📚</div><div><div class="t">Released songs you love</div><div class="s">Tap to choose · several ok · only numbers are kept</div></div></label>
      <div class="row"><div class="stack"><div class="lbl">Profile name</div>
        <input type="text" name="profile_name" placeholder="e.g. maes" autocomplete="off" required></div></div>
      <button type="submit" class="small" id="learnbtn">Learn this sound</button>
      <pre id="learnout" class="muted" style="white-space:pre-wrap;margin:10px 0 0"></pre>
    </form>
  </details>
</section>

<section id="progress">
  <h2 id="ptitle">Uploading</h2>
  <div class="bar"><i id="pbar"></i></div>
  <ol class="steps" id="steps"></ol>
  <p class="muted" id="phint">Keep Termux open while it works. A 3-minute song takes a few minutes on a phone.</p>
</section>

<section id="result">
  <h2>Done</h2>
  <ul id="diag"></ul>
  <div class="big"><div><b id="rl"></b><span>LUFS</span></div><div><b id="rt"></b><span>dBTP peak</span></div><div><b id="rk"></b><span>key</span></div></div>
  <audio id="player" controls preload="metadata"></audio>
  <ul class="checks" id="checks"></ul>
  <div class="dl" id="files"></div>
  <p class="muted" id="where"></p>
</section>
</main>
<div class="go"><button id="go" form="f" type="submit">Mix &amp; master</button></div>

<script>
const $ = s => document.querySelector(s);
const LABELS = {master_24bit: ["Master · 24-bit WAV", "upload this to your distributor"],
  master_16bit_cd: ["Master · 16-bit 44.1k WAV", "CD quality / distributors that need 16-bit"],
  mp3_preview: ["Preview · MP3 320k", "for sharing, not for stores"],
  premaster_mix: ["Mix before mastering", "for a mastering engineer"],
  vocal_stem: ["Lead vocal stem", ""], stack_stem: ["Doubles + harmonies stem", ""],
  adlib_stem: ["Ad-lib stem", ""], instrumental_stem: ["Beat stem", ""],
  "version_ebu-r128": ["Radio/TV EU · EBU R128", "-23 LUFS, -1 dBTP max"],
  "version_atsc-a85": ["Radio/TV US · ATSC A/85", "-24 LKFS, -2 dBTP max"],
  "version_apple": ["Apple Music level", "-16 LUFS, -1 dBTP max"],
  "version_streaming": ["Streaming level", "-14 LUFS, -1 dBTP max"]};
$("#lf").addEventListener("submit", async e => {
  e.preventDefault();
  const btn = $("#learnbtn"), out = $("#learnout");
  btn.disabled = true; btn.textContent = "Learning…"; out.textContent = "";
  try {
    const r = await (await fetch("/api/learn", {method: "POST", body: new FormData($("#lf"))})).json();
    if (!r.id) throw new Error(r.error || "failed");
    for (;;) {
      const j = await (await fetch("/api/jobs/" + r.id)).json();
      out.textContent = (j.log || []).map(l => l.msg).join("\n");
      if (j.status === "error") throw new Error(j.error);
      if (j.status === "done") {
        out.textContent = j.result.text;
        const sel = $("#profsel");
        if (![...sel.options].some(o => o.value === j.result.profile)) sel.add(new Option(j.result.profile, j.result.profile));
        sel.value = j.result.profile;
        break;
      }
      await new Promise(res => setTimeout(res, 1000));
    }
  } catch (err) { out.innerHTML = '<span class="err"></span>'; out.firstChild.textContent = "Couldn't learn: " + err.message; }
  btn.disabled = false; btn.textContent = "Learn this sound";
});
function setMode() {
  const mode = document.querySelector("input[name=mode]:checked").value;
  document.body.dataset.mode = mode;
  try { localStorage.setItem("sm_mode", mode); } catch (_) {}
  document.querySelector("input[name=mix]").required = mode === "mix";
  document.querySelector("input[name=lead]").required = mode === "stems";
  document.querySelector("input[name=beat]").required = mode === "stems";
}
try { const m = localStorage.getItem("sm_mode"); if (m) document.querySelector(`input[name=mode][value=${m}]`).checked = true; } catch (_) {}
document.querySelectorAll("input[name=mode]").forEach(r => r.addEventListener("change", setMode));
setMode();
const STEPS_EST = 9;
const Studio = (() => {
  let ctx = null, beatBuf = null, beatFor = null, stream = null, recording = null, info = null;
  const ls = (k, d) => { try { return localStorage.getItem(k) ?? d; } catch (_) { return d; } };
  const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch (_) {} };
  let latencyMs = +ls("sm_latency", "0");
  const session = () => ($("#session").value || "").trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  const fmt = t => { t = Math.max(0, t); return Math.floor(t / 60) + ":" + (t % 60).toFixed(1).padStart(4, "0"); };
  const parseT = v => { v = (v || "0").trim(); if (v.includes(":")) { const [m, s] = v.split(":"); return (+m) * 60 + (+s || 0); } return +v || 0; };
  const WORKLET = `class Rec extends AudioWorkletProcessor {
    constructor() { super(); this.buf = []; this.n = 0; this.start = 0; }
    process(inputs) {
      const ch = inputs[0] && inputs[0][0];
      if (ch) { if (!this.n) this.start = currentFrame; this.buf.push(ch.slice(0)); this.n += ch.length;
        if (this.n >= 4096) { this.port.postMessage({f: this.start, d: this.buf}); this.buf = []; this.n = 0; } }
      return true; } }
  registerProcessor("rec", Rec);`;
  async function audio() {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)({latencyHint: "interactive"});
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], {type: "application/javascript"})));
    }
    if (ctx.state === "suspended") await ctx.resume();
    return ctx;
  }
  async function mic() {
    const dev = $("#micsel").value;
    if (stream && stream._dev === dev) return stream;
    if (stream) stream.getTracks().forEach(t => t.stop());
    // raw voice: phones' call processing (echo cancel, noise suppression, auto gain) damages vocals
    stream = await navigator.mediaDevices.getUserMedia({audio: {deviceId: dev ? {exact: dev} : undefined,
      echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1}});
    stream._dev = dev;
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "audioinput");
    const sel = $("#micsel"), keep = sel.value;
    sel.innerHTML = '<option value="">Default microphone</option>' + devs.map((d, i) =>
      `<option value="${d.deviceId}">${(d.label || "Microphone " + (i + 1)).replace(/</g, "")}</option>`).join("");
    sel.value = keep;
    return stream;
  }
  // capture: returns {stop(): Float32Array of samples, firstFrame}
  async function capture(onLevel) {
    const c = await audio(), s = await mic();
    const src = c.createMediaStreamSource(s), node = new AudioWorkletNode(c, "rec"), sink = c.createGain();
    sink.gain.value = 0; src.connect(node); node.connect(sink); sink.connect(c.destination);
    const chunks = []; let first = null;
    node.port.onmessage = e => {
      if (first === null) first = e.data.f;
      let pk = 0; for (const b of e.data.d) { chunks.push(b); for (let i = 0; i < b.length; i += 4) pk = Math.max(pk, Math.abs(b[i])); }
      onLevel && onLevel(pk);
    };
    return { stop() {
      src.disconnect(); node.disconnect(); sink.disconnect();
      const n = chunks.reduce((a, b) => a + b.length, 0), out = new Float32Array(n);
      let o = 0; for (const b of chunks) { out.set(b, o); o += b.length; }
      return {data: out, first: first ?? 0};
    } };
  }
  function wav(data, sr) {  // 32-bit float mono WAV: no quality loss, keeps headroom
    const buf = new ArrayBuffer(44 + data.length * 4), v = new DataView(buf);
    const w = (o, str) => [...str].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
    w(0, "RIFF"); v.setUint32(4, 36 + data.length * 4, true); w(8, "WAVE"); w(12, "fmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 3, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true);
    v.setUint32(28, sr * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 32, true); w(36, "data");
    v.setUint32(40, data.length * 4, true); new Float32Array(buf, 44).set(data);
    return new Blob([buf], {type: "audio/wav"});
  }
  async function loadBeat() {
    const s_ = session(); if (!s_ || !info || !info.beat) return null;
    if (beatBuf && beatFor === s_ + info.beat.name) return beatBuf;
    const c = await audio(), raw = await (await fetch(`/studio/${s_}/beat`)).arrayBuffer();
    beatBuf = await c.decodeAudioData(raw); beatFor = s_ + info.beat.name; return beatBuf;
  }
  function render() {
    $("#latval").textContent = Math.round(latencyMs) + " ms";
    $("#beatname").textContent = info && info.beat ? `${info.beat.name} · ${fmt(info.beat.duration_s)}` : "Tap to choose (saved with the song)";
    $("#studiobeat").closest("label").classList.toggle("has", !!(info && info.beat));
    const ul = $("#takes"); ul.innerHTML = "";
    ((info && info.takes) || []).slice().sort((a, b) => a.offset_s - b.offset_s).forEach(t => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="tag ${t.role}">${t.role === "lead" ? "Lead" : "Ad-lib"}</span>
        <div class="grow"><div>${fmt(t.offset_s)} → ${fmt(t.offset_s + t.duration_s)}</div>
        ${t.warning ? `<div class="warn">${t.warning}</div>` : ""}</div>
        <button type="button" data-play="${t.id}">▶</button><button type="button" data-del="${t.id}">🗑</button>`;
      ul.appendChild(li);
    });
    if (!ul.children.length) ul.innerHTML = '<li class="muted">No takes yet</li>';
  }
  async function refresh() {
    const s_ = session(); if (!s_) { info = null; render(); return; }
    lsSet("sm_session", $("#session").value);
    info = await (await fetch(`/api/studio/${s_}`)).json(); render();
  }
  let playing = [];
  function stopPlay() { playing.forEach(n => { try { n.stop(); } catch (_) {} }); playing = []; }
  async function playTake(id) {
    stopPlay(); const t = info.takes.find(x => x.id === id), c = await audio(), beat = await loadBeat();
    const tb = await c.decodeAudioData(await (await fetch(`/studio/${session()}/take/${id}`)).arrayBuffer());
    const when = c.currentTime + 0.1, pre = Math.min(2, t.offset_s);
    const g = c.createGain(); g.gain.value = +$("#beatvol").value; g.connect(c.destination);
    if (beat) { const b = c.createBufferSource(); b.buffer = beat; b.connect(g); b.start(when, t.offset_s - pre); playing.push(b); }
    const v = c.createBufferSource(); v.buffer = tb; v.connect(c.destination); v.start(when + pre); playing.push(v);
  }
  async function record() {
    const btn = $("#recbtn");
    if (recording) {  // STOP
      const r = recording; recording = null; btn.classList.remove("on"); btn.textContent = "REC";
      r.src.stop(); clearInterval(r.timer);
      const {data, first} = r.cap.stop();
      // beat position p was heard at context time t0 + (p - playFrom); the voice answering it
      // reaches the recorder `latency` later. Keep the audio from the take's start position on.
      const startFrame = Math.round((r.t0 + (r.startAt - r.playFrom) + latencyMs / 1000) * r.sr) - first;
      const take = startFrame >= 0 ? data.subarray(startFrame) : data;
      if (take.length < r.sr * 0.3) { $("#rechint").textContent = "Too short - nothing saved."; return; }
      $("#rechint").textContent = "Saving take…";
      const fd = new FormData();
      fd.append("take", wav(take, r.sr), "take.wav"); fd.append("role", r.role);
      fd.append("offset_s", r.startAt.toFixed(3)); fd.append("latency_ms", latencyMs);
      const res = await fetch(`/api/studio/${session()}/take`, {method: "POST", body: fd});
      const j = await res.json(); if (!res.ok) { $("#rechint").textContent = j.error; return; }
      info = j; render(); const last = j.takes[j.takes.length - 1];
      $("#rechint").textContent = last.warning ? "Saved - " + last.warning : "Saved. Tap ▶ to hear it with the beat.";
      return;
    }
    if (!session()) { alert("Give the song a name first"); return; }
    stopPlay();
    try {
      const c = await audio(), beat = await loadBeat();
      if (!beat) { alert("Choose a beat first"); return; }
      const startAt = parseT($("#startat").value), playFrom = Math.max(0, startAt - 3);
      let peakHold = 0;
      const cap = await capture(pk => { peakHold = Math.max(pk, peakHold * 0.8);
        const el = $("#lvl"); el.style.width = Math.min(100, peakHold * 100) + "%"; el.classList.toggle("hot", pk > 0.95); });
      const g = c.createGain(); g.gain.value = +$("#beatvol").value; g.connect(c.destination);
      const src = c.createBufferSource(); src.buffer = beat; src.connect(g);
      const t0 = c.currentTime + 0.2; src.start(t0, playFrom);
      const role = document.querySelector("input[name=role]:checked").value;
      const timer = setInterval(() => { $("#rectime").textContent = fmt(playFrom + c.currentTime - t0); }, 100);
      recording = {cap, src, t0, startAt, playFrom, sr: c.sampleRate, role, timer};
      src.onended = () => { if (recording) record(); };
      btn.classList.add("on"); btn.textContent = "STOP";
      $("#rechint").textContent = startAt > playFrom ? `Lead-in… your part starts at ${fmt(startAt)}` : "Recording…";
    } catch (err) { alert("Microphone not available: " + err.message); }
  }
  async function calibrate() {
    if (recording) return;
    const b = $("#calib"); b.disabled = true; b.textContent = "Listening…";
    try {
      const c = await audio(), sr = c.sampleRate, n = 8, gap = 0.8;
      const cap = await capture(null);
      const click = c.createBuffer(1, Math.round(sr * 0.004), sr), d = click.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.sin(2 * Math.PI * 2000 * i / sr) * (1 - i / d.length);
      const t0 = c.currentTime + 0.4;
      for (let k = 0; k < n; k++) { const s_ = c.createBufferSource(); s_.buffer = click; s_.connect(c.destination); s_.start(t0 + k * gap); }
      await new Promise(r => setTimeout(r, (0.4 + n * gap + 0.6) * 1000));
      const {data, first} = cap.stop();
      let mx = 0; for (let i = 0; i < data.length; i++) mx = Math.max(mx, Math.abs(data[i]));
      const thr = mx * 0.35, lats = [];
      for (let k = 0; k < n; k++) {
        const a = Math.round((t0 + k * gap) * sr) - first, e = a + Math.round(gap * 0.9 * sr);
        for (let i = Math.max(0, a); i < Math.min(e, data.length); i++) if (Math.abs(data[i]) > thr) { lats.push((i - a) / sr * 1000); break; }
      }
      lats.sort((x, y) => x - y);
      const med = lats[Math.floor(lats.length / 2)], spread = lats.length ? lats[lats.length - 1] - lats[0] : 999;
      if (mx < 0.01 || lats.length < 5 || spread > 15) {
        alert("Couldn't hear the clicks clearly. Hold one earbud right on the phone's microphone, turn the volume up, and try again in a quiet room.");
      } else { latencyMs = med; lsSet("sm_latency", String(med)); render(); }
    } catch (err) { alert("Microphone not available: " + err.message); }
    b.disabled = false; b.textContent = "Calibrate";
  }
  $("#session").value = ls("sm_session", "");
  $("#session").addEventListener("change", refresh);
  $("#studiobeat").addEventListener("change", async e => {
    if (!session()) { alert("Give the song a name first"); e.target.value = ""; return; }
    const fd = new FormData(); fd.append("beat", e.target.files[0]);
    $("#beatname").textContent = "Uploading…";
    const res = await fetch(`/api/studio/${session()}/beat`, {method: "POST", body: fd}); const j = await res.json();
    if (!res.ok) { alert(j.error); } else { info = j; beatBuf = null; render(); }
  });
  $("#recbtn").addEventListener("click", record);
  $("#calib").addEventListener("click", calibrate);
  $("#latval").addEventListener("click", () => { const v = prompt("Earbud delay in ms", Math.round(latencyMs));
    if (v !== null && !isNaN(+v)) { latencyMs = +v; lsSet("sm_latency", String(latencyMs)); render(); } });
  $("#takes").addEventListener("click", async e => {
    const p = e.target.dataset.play, d = e.target.dataset.del;
    if (p) playTake(p);
    if (d && confirm("Delete this take?")) { info = await (await fetch(`/api/studio/${session()}/delete/${d}`, {method: "POST"})).json(); render(); }
  });
  refresh();
  return {session, refresh};
})();

document.querySelectorAll("label.file input").forEach(inp => inp.addEventListener("change", () => {
  const lab = inp.closest("label"), s = lab.querySelector(".s");
  const names = [...inp.files].map(f => f.name);
  lab.classList.toggle("has", names.length > 0);
  s.textContent = names.length ? names.join(", ") : "Tap to choose";
}));

$("#f").addEventListener("submit", e => {
  e.preventDefault();
  const fd = new FormData($("#f"));
  if ($("#flexon").checked) fd.set("flex", "35");
  if (document.body.dataset.mode === "studio") {
    if (!Studio.session()) { alert("Give the song a name first"); return; }
    fd.set("session", Studio.session()); fd.delete("lead"); fd.delete("beat"); fd.delete("adlibs");
  }
  $("#go").disabled = true; $("#go").textContent = "Working…";
  $("#result").style.display = "none"; $("#progress").style.display = "block";
  $("#steps").innerHTML = ""; $("#ptitle").textContent = "Uploading"; $("#pbar").style.width = "0";
  $("#progress").scrollIntoView({behavior: "smooth"});
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  xhr.upload.onprogress = ev => { if (ev.lengthComputable) $("#pbar").style.width = (ev.loaded / ev.total * 15) + "%"; };
  xhr.onload = () => {
    let r = {}; try { r = JSON.parse(xhr.responseText); } catch (_) {}
    if (xhr.status !== 200) return fail(r.error || "Upload failed");
    $("#ptitle").textContent = "Mixing & mastering"; poll(r.id);
  };
  xhr.onerror = () => fail("Lost connection to studiomix - is Termux still running?");
  xhr.send(fd);
});

function fail(msg) {
  $("#ptitle").innerHTML = '<span class="err">' + msg.replace(/</g, "&lt;") + "</span>";
  $("#go").disabled = false; $("#go").textContent = "Mix & master";
}

async function poll(id) {
  let j;
  try { j = await (await fetch("/api/jobs/" + id)).json(); } catch (_) { return setTimeout(() => poll(id), 1500); }
  const ol = $("#steps"); ol.innerHTML = "";
  (j.log || []).forEach((s, i, a) => {
    const li = document.createElement("li"); if (i === a.length - 1 && j.status === "running") li.className = "now";
    li.innerHTML = '<span class="t">' + s.t.toFixed(0) + 's</span><span class="m"></span>';
    li.querySelector(".m").textContent = s.msg; ol.appendChild(li);
  });
  $("#pbar").style.width = Math.min(95, 15 + (j.log || []).length / STEPS_EST * 80) + "%";
  if (j.status === "queued") $("#ptitle").textContent = "Waiting for the previous song to finish";
  if (j.status === "error") return fail("Something went wrong: " + j.error);
  if (j.status !== "done") return setTimeout(() => poll(id), 1000);
  $("#pbar").style.width = "100%"; $("#ptitle").textContent = "Finished in " + j.result.seconds + "s";
  show(id, j);
}

function show(id, j) {
  const r = j.result, o = r.output;
  const dg = $("#diag"); dg.innerHTML = "";
  (r.findings || []).forEach(t => { const li = document.createElement("li"); li.textContent = t; dg.appendChild(li); });
  $("#rl").textContent = o.integrated_lufs; $("#rt").textContent = o.true_peak_dbtp; $("#rk").textContent = r.key;
  const url = f => "/jobs/" + id + "/" + encodeURIComponent(f);
  $("#player").src = url(r.files.mp3_preview || r.files.master_16bit_cd);
  $("#checks").innerHTML = "";
  r.checks.forEach(c => { const li = document.createElement("li");
    li.innerHTML = '<span class="' + (c.ok ? "ok" : "warn") + '">' + (c.ok ? "✓" : "!") + "</span> ";
    li.appendChild(document.createTextNode(c.check + " — " + c.detail)); $("#checks").appendChild(li); });
  const box = $("#files"); box.innerHTML = "";
  Object.entries(r.files).forEach(([k, f]) => { const a = document.createElement("a");
    a.href = url(f); a.download = f; const [t, s] = LABELS[k] || [f, ""];
    a.innerHTML = "<div><div></div><span></span></div><div>↓</div>";
    a.querySelector("div div").textContent = t; a.querySelector("span").textContent = s; box.appendChild(a); });
  $("#where").textContent = "Also saved in: " + j.out_dir;
  $("#result").style.display = "block"; $("#result").scrollIntoView({behavior: "smooth"});
  $("#go").disabled = false; $("#go").textContent = "Mix another";
}
</script>
</body>
</html>
"""

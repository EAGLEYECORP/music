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
            if job["mode"] == "mix":
                log = master_mix(job["mix"], job["out"], job["preset"], name=job["name"],
                                 reference_path=job.get("reference"), vocal_lift_db=o.get("vocal_lift", 0.0),
                                 deliver_extra=o.get("deliver"), verbose=False, progress=progress)
            else:
                if o.get("key"):
                    pitch.parse_key(o["key"])
                check_harmonies(o.get("harmonies"))
                log = run(
                    job["lead"], job["beat"], job["out"], job["preset"], name=job["name"],
                    adlib_paths=job["adlibs"], key=o.get("key") or None, key_changes=o.get("key_changes", False),
                    stack_at=parse_time_ranges(o.get("stack_at")), verbose=False, progress=progress,
                    deliver_extra=o.get("deliver"),
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
                view = {k: job.get(k) for k in ("id", "status", "log", "error", "result", "name")}
                view["out_dir"] = str(job["out"])
            return self._json(200, view)
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
        if self.path != "/api/jobs":
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD:
            return self._json(413, {"error": "upload too large (max 1 GB)"})
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            return self._json(400, {"error": "expected a form upload"})
        fields, files = _parse_multipart(ctype, self.rfile.read(length))
        f = lambda k, d="": (fields.get(k) or [d])[0].strip()  # noqa: E731
        mode = "mix" if f("mode") == "mix" else "stems"
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
        first = files["mix" if mode == "mix" else "lead"][0][0]
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
                        "vocal_lift": 2.0 if f("vocal_lift") == "on" else 0.0},
            "started": time.time(),
        }
        if mode == "mix":
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
    return PAGE.replace("{{PRESETS}}", presets).replace("{{VERSION}}", __version__)


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
.modes { grid-template-columns: 1fr 1fr; margin-bottom: 14px; }
.modes label { padding: 12px 0; font-size: 15px; }
body.mode-mix .only-stems { display: none; }
body:not(.mode-mix) .only-mix { display: none; }
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
  <input type="radio" name="mode" id="m1" value="stems" checked><label for="m1">🎤 Vocal + beat</label>
  <input type="radio" name="mode" id="m2" value="mix"><label for="m2">🎚️ Finished mix</label>
</div>

<section class="only-mix">
  <h2>Your mix</h2>
  <label class="file"><input type="file" name="mix" accept="audio/*,.wav,.mp3,.flac,.m4a">
    <div class="icon">🎚️</div><div><div class="t">Rough / finished mix</div><div class="s">Tap to choose · one stereo file, WAV best</div></div></label>
  <label class="file"><input type="file" name="reference" accept="audio/*,.wav,.mp3,.flac,.m4a">
    <div class="icon">⭐</div><div><div class="t">Reference song <span class="muted">(optional)</span></div><div class="s">A released track whose sound you want</div></div></label>
  <div class="row"><div><div class="lbl">Bring vocals forward</div><div class="hint">+2 dB presence on the centre (lead vocal)</div></div>
    <label class="switch"><input type="checkbox" name="vocal_lift"><span></span></label></div>
  <p class="muted" style="margin:6px 2px 0">It checks for clipping, phase, mud, harshness and silence, fixes what it can, then masters. Auto-tune needs the separate vocal (use "Vocal + beat").</p>
</section>

<section class="only-stems">
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
  <div class="row only-stems"><div class="stack"><div class="lbl">Auto-tune</div>
    <div class="seg">
      <input type="radio" name="tune" id="t0" value="off"><label for="t0">Off</label>
      <input type="radio" name="tune" id="t1" value="natural"><label for="t1">Natural</label>
      <input type="radio" name="tune" id="t2" value="" checked><label for="t2">Style</label>
      <input type="radio" name="tune" id="t3" value="hard"><label for="t3">Hard</label>
    </div><div class="hint">"Style" uses the preset's own setting · Hard = instant robotic snap</div></div></div>
  <div class="row only-stems"><div class="stack"><div class="lbl">Key</div>
    <input type="text" name="key" placeholder="Auto-detect (or e.g. F# minor)" autocomplete="off"></div></div>
  <div class="row only-stems"><div><div class="lbl">Song changes key</div><div class="hint">Detect a key per section</div></div>
    <label class="switch"><input type="checkbox" name="key_changes"><span></span></label></div>
  <div class="row only-stems"><div><div class="lbl">Flex-Tune</div><div class="hint">Keep intentional bends & runs natural</div></div>
    <label class="switch"><input type="checkbox" id="flexon"><span></span></label></div>
</section>

<section class="only-stems">
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
    <div class="row only-stems"><div class="stack"><div class="lbl">Vocal level vs beat (dB)</div><input type="number" name="vocal_level" step="0.5" placeholder="Preset default"></div></div>
  </details>
</section>
</form>

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
function setMode() {
  const mix = document.querySelector("input[name=mode]:checked").value === "mix";
  document.body.classList.toggle("mode-mix", mix);
  document.querySelector("input[name=mix]").required = mix;
  document.querySelector("input[name=lead]").required = !mix;
  document.querySelector("input[name=beat]").required = !mix;
}
document.querySelectorAll("input[name=mode]").forEach(r => r.addEventListener("change", setMode));
setMode();
const STEPS_EST = 9;

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

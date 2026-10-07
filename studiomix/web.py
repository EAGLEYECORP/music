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
            if job["mode"] == "mix" and o.get("ai_remix"):
                from .engine import ai_remix

                log = ai_remix(job["mix"], job["out"], job["preset"], name=job["name"], verbose=False,
                               progress=progress, reference_path=job.get("reference"),
                               deliver_extra=o.get("deliver"), profile=prof)
            elif job["mode"] == "mix":
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
            if o.get("lyrics"):
                from .engine import add_lyrics

                src = (Path(job["out"]) / log["files"]["ai_vocals"] if "ai_vocals" in log["files"]
                       else job.get("lead"))
                if src:
                    add_lyrics(log, src, job["out"], job["name"], say=progress)
            with _jobs_lock:
                job["status"] = "done"
                job["result"] = {
                    "output": log["output"],
                    "key": log.get("key", {}).get("key", "—" if job["mode"] == "mix" and "ai_remix" not in log
                                                  else "tuning off"),
                    "findings": log.get("diagnosis", {}).get("findings", []),
                    "checks": log["delivery_check"], "files": log["files"],
                    "seconds": log["processing_seconds"],
                    "fan": log.get("previews"),
                    "notes": log.get("notes", []),
                    "engineer": job.get("engineer", []),
                    "fan_files": [v["file"] for e in (log.get("previews") or {}).get("platforms", {}).values()
                                  for k, v in e.items() if isinstance(v, dict) and "file" in v],
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

    def _send_file(self, path: Path, ctype: str, extra: dict | None = None) -> None:
        """Serve a file with HTTP range support: audio players need it to seek (and the
        listen-like-a-fan A/B switch jumps to the same moment in another file)."""
        size = path.stat().st_size
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        a, b = 0, size - 1
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                a = int(m.group(1))
                b = min(size - 1, int(m.group(2))) if m.group(2) else size - 1
            else:  # suffix range: the last N bytes
                a = max(0, size - int(m.group(2)))
            if a > b or a >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        partial = bool(m and (m.group(1) or m.group(2)))
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1))
        if partial:
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(a)
            left = b - a + 1
            while left > 0:
                chunk = f.read(min(left, 1 << 20))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _redo(self, job_id: str, ask: str) -> None:
        """Same song, same inputs, plus plain-words changes on top of the previous settings."""
        from .ai.engineer import interpret

        with _jobs_lock:
            old = _jobs.get(job_id)
        if not old or old.get("status") != "done":
            return self._json(404, {"error": "that song isn't finished (or the app was restarted)"})
        if not ask.strip():
            return self._json(400, {"error": "say what to change, e.g. \"more 808, vocal louder\""})
        preset, said, unknown = interpret(ask, old["preset"])
        if not said:
            return self._json(400, {"error": "didn't understand that - try e.g. \"more 808\", \"vocal louder\", "
                                             "\"less harsh\", \"more reverb\", \"voix plus forte\""})
        job = dict(old)
        job.update(id=uuid.uuid4().hex[:12], status="queued", log=[], preset=preset, started=time.time(),
                   result=None, options=dict(old["options"]),
                   engineer=old.get("engineer", []) + [{"ask": ask, "changes": said, "not_understood": unknown}])
        version = old.get("version", 1) + 1
        job["version"] = version
        job["out"] = Path(re.sub(r"_v\d+$", "", str(old["out"])) + f"_v{version}")
        job["name"] = re.sub(r"_v\d+$", "", old["name"]) + f"_v{version}"
        if any("master loudness" in c for c in said):
            job["options"]["lufs_set"] = True
        with _jobs_lock:
            _jobs[job["id"]] = job
        threading.Thread(target=_run_job, args=(job,), daemon=True).start()
        return self._json(200, {"id": job["id"], "changes": said, "not_understood": unknown})

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _host_ok(self) -> bool:
        """DNS-rebinding guard: only answer requests addressed to localhost or an IP address.
        A hostile web page can make its own domain point at 127.0.0.1, but the browser still
        sends that domain name as Host - so it is refused, and your takes can't be read."""
        import ipaddress

        host = (self.headers.get("Host") or "").strip().lower()
        name = host[: host.find("]") + 1] if host.startswith("[") else host.split(":")[0]
        if name == "localhost":
            return True
        try:
            ipaddress.ip_address(name.strip("[]"))
            return True
        except ValueError:
            return False

    def _origin_ok(self) -> bool:
        """CSRF guard: a request that changes something must come from this app's own page. Browsers
        always send Origin on cross-site POSTs; tools like curl send none and are allowed."""
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        from urllib.parse import urlsplit

        return origin != "null" and urlsplit(origin).netloc.lower() == (self.headers.get("Host") or "").lower()

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._json(403, {"error": "forbidden host"})
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
            return self._send_file(path, ctype)
        m = re.fullmatch(r"/jobs/([0-9a-f]{12})/([^/]+)", self.path)
        if m:
            with _jobs_lock:
                job = _jobs.get(m.group(1))
                res = (job or {}).get("result", {})
                allowed = set(res.get("files", {}).values()) | set(res.get("fan_files", []))
            from urllib.parse import unquote

            fname = unquote(m.group(2))
            if not job or fname not in allowed:
                return self._json(404, {"error": "no such file"})
            path = Path(job["out"]) / fname
            ctype = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".m4a": "audio/mp4",
                     ".json": "application/json", ".txt": "text/plain; charset=utf-8",
                     ".srt": "text/plain; charset=utf-8", ".lrc": "text/plain; charset=utf-8"}.get(path.suffix,
                                                                                             "application/octet-stream")
            return self._send_file(path, ctype, {"Content-Disposition": f'inline; filename="{fname}"'})
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok() or not self._origin_ok():
            return self._json(403, {"error": "forbidden: requests must come from the studiomix page"})
        st = re.fullmatch(r"/api/studio/([A-Za-z0-9_-]{1,40})/(beat|take|delete/([0-9a-f]{10})|update/([0-9a-f]{10}))",
                          self.path)
        redo = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/redo", self.path)
        if self.path not in ("/api/jobs", "/api/learn") and not st and not redo:
            return self._json(404, {"error": "not found"})
        if st and st.group(3):
            from . import studio

            studio.delete_take(self.jobs_dir, st.group(1), st.group(3))
            return self._json(200, studio.info(self.jobs_dir, st.group(1)))
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length <= 0 or length > MAX_UPLOAD:
            return self._json(413, {"error": "upload too large (max 1 GB)"})
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            return self._json(400, {"error": "expected a form upload"})
        fields, files = _parse_multipart(ctype, self.rfile.read(length))
        f = lambda k, d="": (fields.get(k) or [d])[0].strip()  # noqa: E731
        if redo:
            return self._redo(redo.group(1), f("ask"))
        if st:
            from . import studio

            try:
                if st.group(4):
                    return self._json(200, studio.update_take(self.jobs_dir, st.group(1), st.group(4),
                                                              {k: v[0] for k, v in fields.items()}))
                if st.group(2) == "beat":
                    if not files.get("beat"):
                        return self._json(400, {"error": "choose a beat"})
                    studio.save_beat(self.jobs_dir, st.group(1), *files["beat"][0])
                else:
                    if not files.get("take"):
                        return self._json(400, {"error": "no audio received"})
                    studio.add_take(self.jobs_dir, st.group(1), files["take"][0][1], f("role", "lead"),
                                    float(f("offset_s", "0") or 0), float(f("latency_ms", "0") or 0),
                                    group=f("group"), active=f("active", "1") != "0")
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
            if not info["beat"] or not any(t["role"] == "lead" and t.get("active", True) for t in info["takes"]):
                return self._json(400, {"error": "load a beat and record (or ★ select) at least one lead take first"})
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
        if f("punch") == "on":
            overrides["punch"] = 1.0
        harmonies = ",".join(fields.get("harmony", []))
        if harmonies:
            overrides["harmonies"] = harmonies
        try:
            preset = get_preset(preset_name, **overrides)
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        engineer = []
        if f("ask"):
            from .ai.engineer import interpret

            preset, said, unknown = interpret(f("ask"), preset)
            engineer.append({"ask": f("ask"), "changes": said, "not_understood": unknown})

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
                        "ai_remix": f("ai_remix") == "on",
                        "lyrics": f("lyrics") == "on",
                        "profile": f("profile"), "lufs_set": bool(f("lufs"))},
            "started": time.time(), "engineer": engineer,
        }
        if engineer and any("master loudness" in c for c in engineer[0]["changes"]):
            job["options"]["lufs_set"] = True  # a profile must not undo "louder"
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
    from .ai import separate as sep

    ai_ok = sep.available()
    from .ai import lyrics as lyr

    lyr_ok = lyr.available()
    return (PAGE.replace("{{PRESETS}}", presets).replace("{{VERSION}}", __version__)
            .replace("{{PROFILES}}", profs).replace("{{TUNE_JS}}", TUNE_JS)
            .replace("{{AI_HINT}}", "The AI pulls the vocal out of your song, then it is auto-tuned, re-mixed and "
                     "mastered. Takes about as long as the song on a computer." if ai_ok else
                     "Needs a computer (pip install onnxruntime) - not available on this device.")
            .replace("{{AI_DISABLED}}", "" if ai_ok else "disabled")
            .replace("{{LYR_HINT}}", "Captions for TikTok / Reels / YouTube (.srt) and synced lyrics (.lrc), "
                     "transcribed from your vocal - a draft to correct." if lyr_ok else
                     "Needs a computer (pip install sherpa-onnx) - not available on this device.")
            .replace("{{LYR_DISABLED}}", "" if lyr_ok else "disabled"))


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


# Live auto-tune for monitoring while recording (runs in the browser's audio thread). The take
# itself is saved raw; this is only what the singer hears in their earbuds. Kept free of browser
# APIs so the same code is tested in Node (tests/test_studiomix.py).
TUNE_JS = r"""
const SCALE_STEPS = {major: [0, 2, 4, 5, 7, 9, 11], minor: [0, 2, 3, 5, 7, 8, 10],
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]};
class TuneCore {
  // pitch tracker (YIN on a 4x decimated, 1 kHz low-passed copy) -> nearest note of the key
  // (with hysteresis) -> pitch-synchronous two-head delay-line shifter (~10 ms of delay)
  constructor(sr) {
    this.sr = sr; this.N = 16384; this.buf = new Float32Array(this.N); this.w = 0;
    this.dec = 4; this.dsr = sr / 4; this.dN = 1024; this.dbuf = new Float32Array(this.dN); this.dw = 0; this.dk = 0;
    const w0 = 2 * Math.PI * 1000 / sr, al = Math.sin(w0) / (2 * 0.7071), c = Math.cos(w0), a0 = 1 + al;
    this.lp = [(1 - c) / 2 / a0, (1 - c) / a0, (1 - c) / 2 / a0, -2 * c / a0, (1 - al) / a0];
    this.z1 = 0; this.z2 = 0;
    this.tmin = Math.max(2, Math.floor(this.dsr / 1100)); this.tmax = Math.ceil(this.dsr / 60); this.W = 256;
    this.d = new Float32Array(this.tmax + 2);
    this.f0 = 0; this.voiced = false; this.note = null; this.corr = 0; this.wet = 0;
    this.phase = 0; this.L = sr * 0.02; this.Lt = this.L; this.minD = Math.round(sr * 0.002);
    this.setKey(0, "chromatic"); this.speed = 0.015; this.amount = 1; this.blocks = 0; this.enabled = true;
  }
  setKey(tonic, scale) {
    this.allowed = new Array(12).fill(false);
    for (const s of SCALE_STEPS[scale] || SCALE_STEPS.chromatic) this.allowed[(tonic + s) % 12] = true;
    this.note = null;
  }
  detect() {
    const W = this.W, tmax = this.tmax, d = this.d, b = this.dbuf, M = this.dN - 1;
    // the window is the newest W samples, each compared with the one `tau` earlier: the estimate
    // describes the voice ~13 ms ago, about as late as the shifter plays it back
    const start = (this.dw - W + this.dN) & M;
    let energy = 0;
    for (let j = 0; j < W; j++) { const x = b[(start + j) & M]; energy += x * x; }
    if (energy / W < 1e-6) { this.voiced = false; return; }  // below -60 dBFS: silence
    let run = 0, best = -1;
    d[0] = 1;
    for (let tau = 1; tau <= tmax; tau++) {
      let s = 0;
      for (let j = 0; j < W; j++) { const e = b[(start + j) & M] - b[(start + j - tau + this.dN) & M]; s += e * e; }
      run += s; d[tau] = run > 0 ? s * tau / run : 1;
    }
    for (let tau = this.tmin; tau < tmax; tau++) {
      if (d[tau] < 0.15) { while (tau + 1 < tmax && d[tau + 1] < d[tau]) tau++; best = tau; break; }
    }
    if (best < 0) { this.voiced = false; return; }
    const a = d[best - 1], m = d[best], c = d[best + 1], den = a - 2 * m + c;
    const t = best + (Math.abs(den) > 1e-9 ? 0.5 * (a - c) / den : 0);
    this.f0 = this.dsr / t; this.voiced = true;
  }
  target(midi) {
    // nearest allowed note; only leave the current note when another is clearly closer
    let bestN = null, bestD = 99;
    for (let n = Math.floor(midi) - 2; n <= Math.ceil(midi) + 2; n++) {
      if (!this.allowed[((n % 12) + 12) % 12]) continue;
      const dd = Math.abs(midi - n); if (dd < bestD) { bestD = dd; bestN = n; }
    }
    if (this.note !== null && this.allowed[((this.note % 12) + 12) % 12]
        && Math.abs(midi - this.note) < bestD + 0.25 && Math.abs(midi - this.note) < 1.0) return this.note;
    this.note = bestN; return bestN;
  }
  process(inp, out) {
    const n = inp.length, sr = this.sr, M = this.N - 1, DM = this.dN - 1, lp = this.lp;
    if ((this.blocks++ & 1) === 0) this.detect();
    let want = 0, wetT = 0;
    if (this.enabled && this.voiced && this.f0 > 55 && this.f0 < 1100) {
      const midi = 69 + 12 * Math.log2(this.f0 / 440);
      want = (this.target(midi) - midi) * this.amount; wetT = 1;
      const P = sr / this.f0, m = Math.max(1, Math.ceil(0.006 * sr / P));
      this.Lt = 2 * m * P;  // the two heads sit a whole number of periods apart: no comb
    }
    const a = Math.exp(-n / (Math.max(0.001, this.speed) * sr));
    if (wetT) this.corr = a * this.corr + (1 - a) * want;
    const ratio = Math.pow(2, this.corr / 12), aw = Math.exp(-1 / (0.012 * sr)), aL = Math.exp(-1 / (0.01 * sr));
    for (let i = 0; i < n; i++) {
      const x = inp[i];
      this.buf[this.w] = x; this.w = (this.w + 1) & M;
      const y = lp[0] * x + this.z1; this.z1 = lp[1] * x - lp[3] * y + this.z2; this.z2 = lp[2] * x - lp[4] * y;
      if (++this.dk === this.dec) { this.dk = 0; this.dbuf[this.dw] = y; this.dw = (this.dw + 1) & DM; }
      this.wet = aw * this.wet + (1 - aw) * wetT;
      this.L = aL * this.L + (1 - aL) * this.Lt;
      const L = this.L;
      this.phase += (1 - ratio) / L; this.phase -= Math.floor(this.phase);
      const p2 = this.phase + 0.5 - (this.phase >= 0.5 ? 1 : 0);
      const g1 = Math.sin(Math.PI * this.phase) ** 2, g2 = 1 - g1;
      const sh = g1 * this.read(this.minD + this.phase * L) + g2 * this.read(this.minD + p2 * L);
      const dry = this.read(this.minD + 0.5 * L);
      out[i] = this.wet * sh + (1 - this.wet) * dry;
    }
  }
  read(delay) {
    const pos = this.w - 1 - delay, i = Math.floor(pos), f = pos - i, M = this.N - 1;
    const a = this.buf[i & M], b = this.buf[(i + 1) & M];
    return a + f * (b - a);
  }
}
"""


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
.notes { list-style: none; padding: 0; margin: 8px 0; }
.notes li { padding: 10px 12px; margin: 6px 0; border-radius: 12px; background: var(--chip); font-size: 14px; }
.notes li.done { background: none; padding: 2px 0; color: var(--muted); font-size: 13px; }
.notes li button { margin-top: 8px; display: block; }
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
.takes li { flex-wrap: wrap; }
.takes li.off { opacity: .45; }
.takes canvas { width: 100%; height: 34px; display: block; margin-top: 6px; border-radius: 6px; background: var(--chip); }
.takes .star { font-size: 18px; padding: 6px 10px; }
.takes .star.on { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
.edit { width: 100%; display: grid; gap: 6px; padding: 8px 0 2px; }
.edit[hidden] { display: none; }
#fan[hidden], audio[hidden] { display: none; }
.seg input:disabled + label { opacity: .35; pointer-events: none; }
.switch input:disabled + span { opacity: .35; }
.edit label { display: grid; grid-template-columns: 92px 1fr 64px; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); }
.edit output { text-align: right; font-variant-numeric: tabular-nums; color: var(--text); }
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
    <div style="display:flex;gap:8px;margin-top:8px;align-items:center">
      <input type="text" id="startat" value="0:00" style="width:92px" aria-label="start at">
      <span class="hint">→</span>
      <input type="text" id="endat" placeholder="end" style="width:92px" aria-label="end at">
      <label class="hint" style="display:flex;gap:6px;align-items:center;margin-left:auto">Loop
        <label class="switch"><input type="checkbox" id="loopon"><span></span></label></label>
    </div>
    <div class="hint">You hear 3 s of beat before the start. Loop: the section repeats and every pass is kept - ★ the best one.</div>
    <div class="rec">
      <button type="button" class="recbtn" id="recbtn">REC</button>
      <div class="meter"><div class="lvl"><i id="lvl"></i></div><div class="time" id="rectime">0:00.0</div>
        <div class="hint" id="rechint">Earbuds in, tap REC, perform, tap STOP.</div></div>
    </div>
    <label style="display:flex;gap:8px;align-items:center;margin-top:8px" class="hint">Beat volume
      <input type="range" id="beatvol" min="0" max="1" step="0.05" value="0.8" style="flex:1"></label>
  </div></div>
  <div class="row"><div><div class="lbl">Hear yourself</div>
    <div class="hint" id="monhint">Your voice in your earbuds, auto-tuned live. Wired earbuds - Bluetooth arrives too late to sing with.</div></div>
    <label class="switch"><input type="checkbox" id="monon"><span></span></label></div>
  <div id="monopts" hidden>
    <div class="seg" style="grid-template-columns:1fr 1fr 1fr">
      <input type="radio" name="montune" id="mt0" value="off"><label for="mt0">Clean</label>
      <input type="radio" name="montune" id="mt1" value="0.04" checked><label for="mt1">Natural</label>
      <input type="radio" name="montune" id="mt2" value="0.005"><label for="mt2">Hard</label>
    </div>
    <div class="row"><div class="stack"><div class="lbl">Key</div>
      <select id="monkey"><option value="auto">Auto - from the beat</option><option value="0,chromatic">Any note (chromatic)</option><option value="0,minor">C minor</option><option value="1,minor">C# minor</option><option value="2,minor">D minor</option><option value="3,minor">Eb minor</option><option value="4,minor">E minor</option><option value="5,minor">F minor</option><option value="6,minor">F# minor</option><option value="7,minor">G minor</option><option value="8,minor">Ab minor</option><option value="9,minor">A minor</option><option value="10,minor">Bb minor</option><option value="11,minor">B minor</option><option value="0,major">C major</option><option value="1,major">C# major</option><option value="2,major">D major</option><option value="3,major">Eb major</option><option value="4,major">E major</option><option value="5,major">F major</option><option value="6,major">F# major</option><option value="7,major">G major</option><option value="8,major">Ab major</option><option value="9,major">A major</option><option value="10,major">Bb major</option><option value="11,major">B major</option></select>
      <div class="hint">A key you pick here is also used for the final auto-tune. Takes are saved untouched either way.</div></div></div>
    <label style="display:flex;gap:8px;align-items:center" class="hint">Voice volume
      <input type="range" id="monvol" min="0" max="1.5" step="0.05" value="0.9" style="flex:1"></label>
  </div>
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
  <div class="row"><div><div class="lbl">AI remix ✨</div><div class="hint" id="aihint">{{AI_HINT}}</div></div>
    <label class="switch"><input type="checkbox" name="ai_remix" {{AI_DISABLED}}><span></span></label></div>
  <p class="muted" style="margin:6px 2px 0">It checks for clipping, phase, mud, harshness and silence, fixes what it can, then masters. With AI remix, the vocal is pulled out of the song first, so it gets auto-tuned and re-mixed too.</p>
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
  <div class="row"><div class="stack"><div class="lbl">Tell the engineer <span class="muted">(optional · English or French)</span></div>
    <input type="text" name="ask" placeholder="e.g. more 808, vocal a bit less harsh" autocomplete="off">
    <div class="hint">Plain words: 808, vocal louder, less harsh, more reverb, dry, hard autotune, ad-libs quieter, wider, punchier, louder, darker… · voix plus forte, trop de reverb</div></div></div>
  <div class="row"><div><div class="lbl">Lyrics → captions ✨</div><div class="hint">{{LYR_HINT}}</div></div>
    <label class="switch"><input type="checkbox" name="lyrics" {{LYR_DISABLED}}><span></span></label></div>
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
  <div class="row"><div><div class="lbl">Punch over loudness</div>
    <div class="hint">Up to 2 dB less loud master (never under -11 LUFS), harder-hitting drums. Plays just as loud on Spotify, YouTube and Apple Music - they turn every song to the same level.</div></div>
    <label class="switch"><input type="checkbox" name="punch"><span></span></label></div>
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
  <div id="fan" hidden>
    <h2 style="margin-top:14px">Listen like a fan</h2>
    <p class="muted" style="margin:0 2px 10px">Exactly how each app plays it: its loudness rule and its sound format. Switch while it plays - it stays at the same moment.</p>
    <div class="seg" style="grid-template-columns:repeat(4,1fr)">
      <input type="radio" name="fanplat" id="fp1" value="spotify" checked><label for="fp1">Spotify</label>
      <input type="radio" name="fanplat" id="fp2" value="apple"><label for="fp2">Apple</label>
      <input type="radio" name="fanplat" id="fp3" value="youtube"><label for="fp3">YouTube</label>
      <input type="radio" name="fanplat" id="fp4" value="phone"><label for="fp4">Phone</label>
    </div>
    <div class="seg" style="grid-template-columns:1fr 1fr;margin-top:8px">
      <input type="radio" name="fanab" id="fa1" value="master" checked><label for="fa1">Master</label>
      <input type="radio" name="fanab" id="fa2" value="mix"><label for="fa2" id="fanmixlbl">Before mastering</label>
    </div>
    <audio id="fanplayer" controls preload="auto" style="margin-top:10px"></audio>
    <p class="muted" id="fannote" style="margin:6px 2px 0"></p>
  </div>
  <audio id="player" controls preload="metadata"></audio>
  <div id="eng">
    <h2 style="margin-top:14px">Engineer</h2>
    <ul class="notes" id="engdone"></ul>
    <ul class="notes" id="notes"></ul>
    <div class="row"><div class="stack"><div class="lbl">Not quite? Say what to change</div>
      <input type="text" id="redoask" placeholder="e.g. more 808, voix plus forte" autocomplete="off"></div></div>
    <button type="button" class="small" id="redobtn">Redo with these changes</button>
    <p class="muted" id="redomsg" style="margin:6px 2px 0"></p>
  </div>
  <ul class="checks" id="checks"></ul>
  <div class="dl" id="files"></div>
  <p class="muted" id="where"></p>
</section>
</main>
<div class="go"><button id="go" form="f" type="submit">Mix &amp; master</button></div>

<script type="text/plain" id="tunejs">{{TUNE_JS}}</script>
<script>
const $ = s => document.querySelector(s);
const LABELS = {master_24bit: ["Master · 24-bit WAV", "upload this to your distributor"],
  ai_vocals: ["AI-separated vocal", "pulled out of your song by the AI"],
  lyrics_srt: ["Captions · SRT", "for TikTok / Reels / YouTube - a draft to correct"],
  lyrics_lrc: ["Synced lyrics · LRC", "for music players"], lyrics_txt: ["Lyrics · text", ""],
  ai_instrumental: ["AI-separated beat", "your song without the vocal"],
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
  let ctx = null, beatBuf = null, beatFor = null, stream = null, recording = null, info = null, uploading = false;
  const ls = (k, d) => { try { return localStorage.getItem(k) ?? d; } catch (_) { return d; } };
  const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch (_) {} };
  let latencyMs = +ls("sm_latency", "0");
  const session = () => ($("#session").value || "").trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  const fmt = t => { t = Math.max(0, t); return Math.floor(t / 60) + ":" + (t % 60).toFixed(1).padStart(4, "0"); };
  const parseT = v => { v = (v || "0").trim(); if (v.includes(":")) { const [m, s] = v.split(":"); return (+m) * 60 + (+s || 0); } return +v || 0; };
  const WORKLET = document.getElementById("tunejs").textContent + `
  class Tune extends AudioWorkletProcessor {
    constructor() { super(); this.core = new TuneCore(sampleRate);
      this.port.onmessage = e => { const m = e.data;
        if (m.key) this.core.setKey(m.key[0], m.key[1]);
        if (m.speed !== undefined) this.core.speed = m.speed;
        if (m.enabled !== undefined) this.core.enabled = m.enabled; }; }
    process(inputs, outputs) { const i = inputs[0] && inputs[0][0], o = outputs[0] && outputs[0][0];
      if (i && o) this.core.process(i, o); return true; } }
  registerProcessor("tune", Tune);
  class Rec extends AudioWorkletProcessor {
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
  // live monitoring: mic -> tuner -> earbuds (the recorder taps the raw mic separately)
  let mon = null;
  function monKey() {
    const v = $("#monkey").value, b = info && info.beat;
    if (v === "auto") return b && b.scale ? [b.tonic, b.scale] : [0, "chromatic"];
    const [t, sc] = v.split(","); return [+t, sc];
  }
  function monSettings() {
    if (!mon) return;
    const tune = document.querySelector("input[name=montune]:checked").value;
    mon.node.port.postMessage({key: monKey(), enabled: tune !== "off", speed: tune === "off" ? 0.04 : +tune});
    mon.gain.gain.value = +$("#monvol").value;
  }
  async function monitor(on) {
    if (mon) { mon.src.disconnect(); mon.node.disconnect(); mon.gain.disconnect(); mon = null; }
    $("#monopts").hidden = !on;
    if (!on) return;
    try {
      const c = await audio(), s = await mic();
      const src = c.createMediaStreamSource(s), node = new AudioWorkletNode(c, "tune", {outputChannelCount: [1]});
      const gain = c.createGain(); src.connect(node); node.connect(gain); gain.connect(c.destination);
      mon = {src, node, gain}; monSettings();
    } catch (err) { $("#monon").checked = false; $("#monopts").hidden = true; alert("Microphone not available: " + err.message); }
  }
  function monHint() {
    const b = info && info.beat, k = $("#monkey").value === "auto" && b ? ` Beat key: ${b.key}.` : "";
    $("#monhint").textContent = (latencyMs > 100
      ? `Your earbuds lag ${Math.round(latencyMs)} ms - that's Bluetooth. Hearing yourself that late is confusing; use wired earbuds for this.`
      : "Your voice in your earbuds, auto-tuned live. Wired earbuds - Bluetooth arrives too late to sing with.") + k;
    $("#monhint").classList.toggle("warn", latencyMs > 100);
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
    monHint(); monSettings();
    $("#beatname").textContent = info && info.beat ? `${info.beat.name} · ${fmt(info.beat.duration_s)}` : "Tap to choose (saved with the song)";
    $("#studiobeat").closest("label").classList.toggle("has", !!(info && info.beat));
    const ul = $("#takes"); ul.innerHTML = "";
    const takes = ((info && info.takes) || []).slice().sort((a, b) => a.offset_s - b.offset_s || a.created.localeCompare(b.created));
    const passNo = {}, groupCount = {};
    takes.forEach(t => { if (t.group) groupCount[t.group] = (groupCount[t.group] || 0) + 1; });
    takes.forEach(t => {
      const on = t.active !== false, li = document.createElement("li");
      if (t.group) passNo[t.group] = (passNo[t.group] || 0) + 1;
      li.className = on ? "" : "off";
      const ts = t.trim_start_s || 0, te = t.trim_end_s || 0;
      li.innerHTML = `<button type="button" class="star ${on ? "on" : ""}" data-star="${t.id}" title="use this take">★</button>
        <span class="tag ${t.role}">${t.role === "lead" ? "Lead" : "Ad-lib"}</span>
        <div class="grow"><div>${fmt(t.offset_s + ts)} → ${fmt(t.offset_s + t.duration_s - te)}
          ${t.group ? `<span class="muted"> · pass ${passNo[t.group]}/${groupCount[t.group]}</span>` : ""}</div>
        ${t.warning ? `<div class="warn">${t.warning}</div>` : ""}</div>
        <button type="button" data-play="${t.id}">▶</button><button type="button" data-edit="${t.id}">✎</button>
        <button type="button" data-del="${t.id}">🗑</button>
        <canvas data-wave="${t.id}" width="600" height="34"></canvas>
        <div class="edit" data-panel="${t.id}" hidden>
          <label>Trim start <input type="range" name="trim_start_s" min="0" max="${t.duration_s}" step="0.05" value="${ts}"><output>${ts.toFixed(2)} s</output></label>
          <label>Trim end <input type="range" name="trim_end_s" min="0" max="${t.duration_s}" step="0.05" value="${te}"><output>${te.toFixed(2)} s</output></label>
          <label>Nudge <input type="range" name="nudge_ms" min="-150" max="150" step="5" value="${t.nudge_ms || 0}"><output>${t.nudge_ms || 0} ms</output></label>
          <label>Volume <input type="range" name="gain_db" min="-12" max="12" step="0.5" value="${t.gain_db || 0}"><output>${t.gain_db || 0} dB</output></label>
          <button type="button" class="mini" data-save="${t.id}">Save edit</button>
        </div>`;
      ul.appendChild(li);
      drawWave(t);
    });
    if (!ul.children.length) ul.innerHTML = '<li class="muted">No takes yet</li>';
  }
  async function refresh() {
    const s_ = session(); if (!s_) { info = null; render(); return; }
    lsSet("sm_session", $("#session").value);
    info = await (await fetch(`/api/studio/${s_}`)).json(); render();
  }
  const waves = {};
  async function drawWave(t) {
    const cv = document.querySelector(`canvas[data-wave="${t.id}"]`); if (!cv) return;
    try {
      if (!waves[t.id]) {
        const c = await audio(), b = await c.decodeAudioData(await (await fetch(`/studio/${session()}/take/${t.id}`)).arrayBuffer());
        const d = b.getChannelData(0), n = cv.width, step = Math.max(1, Math.floor(d.length / n)), pk = new Float32Array(n);
        for (let i = 0; i < n; i++) { let m = 0; for (let j = i * step; j < Math.min(d.length, (i + 1) * step); j += 4) m = Math.max(m, Math.abs(d[j])); pk[i] = m; }
        waves[t.id] = pk;
      }
      const pk = waves[t.id], g = cv.getContext("2d"), W = cv.width, H = cv.height;
      const css = getComputedStyle(document.documentElement);
      g.clearRect(0, 0, W, H);
      const a = (t.trim_start_s || 0) / t.duration_s * W, b = W - (t.trim_end_s || 0) / t.duration_s * W;
      // drawn relative to the take's own peak (up to 10x): a normal phone take is quiet in absolute
      // terms and would look like a flat line; the "very quiet" warning covers real level problems
      const top = Math.max(0.1, ...pk);
      for (let i = 0; i < W; i++) {
        const h = Math.max(1, pk[i] / top * H * 0.95);
        g.fillStyle = (i >= a && i <= b) ? css.getPropertyValue("--accent") : css.getPropertyValue("--line");
        g.fillRect(i, (H - h) / 2, 1, h);
      }
    } catch (_) {}
  }
  let playing = [];
  function stopPlay() { playing.forEach(n => { try { n.stop(); } catch (_) {} }); playing = []; }
  async function playTake(id) {
    stopPlay(); const t = info.takes.find(x => x.id === id), c = await audio(), beat = await loadBeat();
    const tb = await c.decodeAudioData(await (await fetch(`/studio/${session()}/take/${id}`)).arrayBuffer());
    const ts = t.trim_start_s || 0, te = t.trim_end_s || 0, nudge = (t.nudge_ms || 0) / 1000;
    const at = Math.max(0, t.offset_s + ts + nudge), pre = Math.min(2, at), when = c.currentTime + 0.1;
    const g = c.createGain(); g.gain.value = +$("#beatvol").value; g.connect(c.destination);
    if (beat) { const b = c.createBufferSource(); b.buffer = beat; b.connect(g); b.start(when, at - pre); playing.push(b); }
    const vg = c.createGain(); vg.gain.value = Math.pow(10, (t.gain_db || 0) / 20); vg.connect(c.destination);
    const v = c.createBufferSource(); v.buffer = tb; v.connect(vg);
    v.start(when + pre, ts, Math.max(0.05, t.duration_s - ts - te)); playing.push(v);
  }
  async function record() {
    const btn = $("#recbtn");
    if (recording) {  // STOP
      const r = recording; recording = null; btn.classList.remove("on"); btn.textContent = "REC";
      r.src.stop(); clearInterval(r.timer);
      const {data, first} = r.cap.stop();
      // beat position p was heard at context time t0 + (p - playFrom); the voice answering it
      // reaches the recorder `latency` later. Keep the audio from the take's start position on.
      const base = (r.t0 + (r.startAt - r.playFrom) + latencyMs / 1000) * r.sr - first;
      let passes = [];
      if (r.loopLen) {  // one take per pass of the loop
        const L = Math.round(r.loopLen * r.sr);
        for (let k = 0; ; k++) {
          const a = Math.round(base + k * L); if (a >= data.length) break;
          const seg = data.subarray(Math.max(0, a), Math.min(data.length, a + L));
          if (seg.length >= Math.max(r.sr * 0.3, L * 0.3)) passes.push(seg);
        }
      } else {
        const a = Math.round(base); passes = [a >= 0 ? data.subarray(a) : data];
      }
      passes = passes.filter(p => p.length >= r.sr * 0.3);
      if (!passes.length) { $("#rechint").textContent = "Too short - nothing saved."; return; }
      const group = r.loopLen ? Math.random().toString(36).slice(2, 10) : "";
      let j = null;
      for (let k = 0; k < passes.length; k++) {
        $("#rechint").textContent = passes.length > 1 ? `Saving pass ${k + 1}/${passes.length}…` : "Saving take…";
        const fd = new FormData();
        fd.append("take", wav(passes[k], r.sr), "take.wav"); fd.append("role", r.role);
        fd.append("offset_s", r.startAt.toFixed(3)); fd.append("latency_ms", latencyMs);
        fd.append("group", group);
        // in a loop, the last complete pass is used by default (you usually get better each time)
        const full = !r.loopLen || passes[k].length >= Math.round(r.loopLen * r.sr) * 0.95;
        const lastFull = passes.map(p => !r.loopLen || p.length >= Math.round(r.loopLen * r.sr) * 0.95).lastIndexOf(true);
        fd.append("active", (!r.loopLen || k === (lastFull >= 0 ? lastFull : passes.length - 1)) ? "1" : "0");
        const res = await fetch(`/api/studio/${session()}/take`, {method: "POST", body: fd});
        j = await res.json(); if (!res.ok) { $("#rechint").textContent = j.error; return; }
      }
      info = j; render(); const last = j.takes[j.takes.length - 1];
      $("#rechint").textContent = last.warning ? "Saved - " + last.warning
        : passes.length > 1 ? `Saved ${passes.length} passes - ★ the best one.` : "Saved. Tap ▶ to hear it with the beat.";
      return;
    }
    if (!session()) { alert("Give the song a name first"); return; }
    stopPlay();
    try {
      const c = await audio(), beat = await loadBeat();
      if (!beat) { alert("Choose a beat first"); return; }
      const startAt = parseT($("#startat").value), playFrom = Math.max(0, startAt - 3);
      const endAt = parseT($("#endat").value), looping = $("#loopon").checked;
      if (looping && !(endAt > startAt + 1)) { alert("For loop recording set an end time after the start"); return; }
      let peakHold = 0;
      const cap = await capture(pk => { peakHold = Math.max(pk, peakHold * 0.8);
        const el = $("#lvl"); el.style.width = Math.min(100, peakHold * 100) + "%"; el.classList.toggle("hot", pk > 0.95); });
      const g = c.createGain(); g.gain.value = +$("#beatvol").value; g.connect(c.destination);
      const src = c.createBufferSource(); src.buffer = beat; src.connect(g);
      if (looping) { src.loop = true; src.loopStart = startAt; src.loopEnd = Math.min(endAt, beat.duration); }
      const t0 = c.currentTime + 0.2; src.start(t0, playFrom);
      const role = document.querySelector("input[name=role]:checked").value;
      const loopLen = looping ? Math.min(endAt, beat.duration) - startAt : 0;
      const timer = setInterval(() => {
        let pos = playFrom + c.currentTime - t0, pass = "";
        if (loopLen && pos > startAt) { const k = Math.floor((pos - startAt) / loopLen); pos = startAt + (pos - startAt) % loopLen; pass = `  ·  pass ${k + 1}`; }
        $("#rectime").textContent = fmt(pos) + pass; }, 100);
      recording = {cap, src, t0, startAt, playFrom, sr: c.sampleRate, role, timer, loopLen};
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
    $("#beatname").textContent = "Uploading…"; uploading = true; $("#recbtn").disabled = true;
    try {
      const res = await fetch(`/api/studio/${session()}/beat`, {method: "POST", body: fd}); const j = await res.json();
      if (!res.ok) { alert(j.error); } else { info = j; beatBuf = null; }
    } finally { uploading = false; $("#recbtn").disabled = false; render(); }
  });
  $("#recbtn").addEventListener("click", record);
  $("#monon").addEventListener("change", e => monitor(e.target.checked));
  document.querySelectorAll("input[name=montune]").forEach(r => r.addEventListener("change", monSettings));
  $("#monvol").addEventListener("input", monSettings);
  $("#monkey").addEventListener("change", () => {
    const v = $("#monkey").value, k = document.querySelector("input[name=key]");
    if (k && (k.value === "" || k.dataset.fromStudio)) {  // also tune the final mix in this key
      k.value = v === "auto" ? "" : $("#monkey").selectedOptions[0].textContent.replace("Any note (chromatic)", "chromatic");
      k.dataset.fromStudio = v === "auto" ? "" : "1";
    }
    monSettings(); monHint();
  });
  $("#micsel").addEventListener("change", () => { if (mon) monitor(true); });
  $("#calib").addEventListener("click", calibrate);
  $("#latval").addEventListener("click", () => { const v = prompt("Earbud delay in ms", Math.round(latencyMs));
    if (v !== null && !isNaN(+v)) { latencyMs = +v; lsSet("sm_latency", String(latencyMs)); render(); } });
  $("#takes").addEventListener("input", e => {
    const o = e.target.parentElement.querySelector("output"); if (!o) return;
    const u = {trim_start_s: " s", trim_end_s: " s", nudge_ms: " ms", gain_db: " dB"}[e.target.name];
    o.textContent = (+e.target.value).toFixed(e.target.name.startsWith("trim") ? 2 : 1).replace(/\.0$/, "") + u;
  });
  async function updateTake(id, fields) {
    const fd = new FormData(); Object.entries(fields).forEach(([k, v]) => fd.append(k, v));
    const res = await fetch(`/api/studio/${session()}/update/${id}`, {method: "POST", body: fd});
    const j = await res.json(); if (!res.ok) { alert(j.error); return; } info = j; render();
  }
  $("#takes").addEventListener("click", async e => {
    const p = e.target.dataset.play, d = e.target.dataset.del;
    const st = e.target.dataset.star, ed = e.target.dataset.edit, sv = e.target.dataset.save;
    if (st) { const t = info.takes.find(x => x.id === st); updateTake(st, {active: t.active === false ? "1" : "0"}); }
    if (ed) { const pn = document.querySelector(`[data-panel="${ed}"]`); pn.hidden = !pn.hidden; }
    if (sv) {
      const pn = document.querySelector(`[data-panel="${sv}"]`), f = {};
      pn.querySelectorAll("input").forEach(i => f[i.name] = i.value); updateTake(sv, f);
    }
    if (p) playTake(p);
    if (d && confirm("Delete this take?")) { info = await (await fetch(`/api/studio/${session()}/delete/${d}`, {method: "POST"})).json(); render(); }
  });
  refresh();
  return {session, refresh};
})();

document.querySelectorAll("label.file input:not(#studiobeat)").forEach(inp => inp.addEventListener("change", () => {
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
  $("#ptitle").textContent = j.status === "queued" ? "Waiting for the previous song to finish" : "Mixing & mastering";
  if (j.status === "error") return fail("Something went wrong: " + j.error);
  if (j.status !== "done") return setTimeout(() => poll(id), 1000);
  $("#pbar").style.width = "100%"; $("#ptitle").textContent = "Finished in " + j.result.seconds + "s";
  show(id, j);
}

let fanData = null, fanUrl = null;
function fanShow(fan, url) {
  fanData = fan; fanUrl = url;
  $("#fan").hidden = !fan; $("#player").hidden = !!fan;
  if (!fan) return;
  $("#fanmixlbl").textContent = fan.mix_label || "Before mastering";
  // Apple and the phone preview use AAC, like those apps; a few browsers (e.g. Chromium on Linux)
  // can't play it - say so instead of a silent button
  const aac = $("#fanplayer").canPlayType('audio/mp4; codecs="mp4a.40.2"') !== "";
  ["fp2", "fp4"].forEach(id => { $("#" + id).disabled = !aac; });
  if (!aac && ["apple", "phone"].includes(document.querySelector("input[name=fanplat]:checked").value)) $("#fp1").checked = true;
  fanData.aacNote = aac ? "" : " (Apple and Phone previews are AAC files that this browser can't play - Chrome on Android and Safari can.)";
  fanPick(false);
}
function fanPick(keepTime) {
  if (!fanData) return;
  const plat = document.querySelector("input[name=fanplat]:checked").value;
  const e = fanData.platforms[plat], hasMix = !!e.mix;
  $("#fa2").disabled = !hasMix;
  if (!hasMix) $("#fa1").checked = true;
  const which = document.querySelector("input[name=fanab]:checked").value, v = e[which];
  const p = $("#fanplayer"), t = p.currentTime, wasPlaying = !p.paused;
  p.src = fanUrl(v.file);
  if (keepTime) p.addEventListener("loadedmetadata", () => { p.currentTime = t; if (wasPlaying) p.play().catch(() => {}); }, {once: true});
  const say = (x, who) => `${who} plays at ${x.plays_at_lufs} LUFS` + (x.gain_db < 0 ? ` (turned down ${-x.gain_db} dB)` :
    x.gain_db > 0 ? ` (turned up ${x.gain_db} dB)` : "");
  let note = `On ${e.label}: ` + say(e.master, "the master");
  if (hasMix) {
    const d = (e.master.plays_at_lufs - e.mix.plays_at_lufs).toFixed(1);
    note += `; ${(fanData.mix_label || "before mastering").toLowerCase()}: ${e.mix.plays_at_lufs} LUFS` +
      (d > 0.4 ? ` - ${d} dB quieter, it can't be turned up further without clipping.` : ".");
  } else note += ".";
  if (plat === "phone") note += " Phone speakers play no deep bass - check the 808 still reads.";
  $("#fannote").textContent = note + (fanData.aacNote || "");
}
document.querySelectorAll("input[name=fanplat],input[name=fanab]").forEach(r => r.addEventListener("change", () => fanPick(true)));
let lastJob = null;
function engShow(id, r) {
  lastJob = id;
  const done = $("#engdone"); done.innerHTML = "";
  (r.engineer || []).forEach(e => {
    e.changes.forEach(c => { const li = document.createElement("li"); li.className = "done"; li.textContent = "✓ " + c; done.appendChild(li); });
    (e.not_understood || []).forEach(u => { const li = document.createElement("li"); li.className = "done";
      li.textContent = "? didn't understand \u201c" + u + "\u201d"; done.appendChild(li); });
  });
  const ul = $("#notes"); ul.innerHTML = "";
  (r.notes || []).forEach(n => {
    const li = document.createElement("li"), b = document.createElement("button");
    li.textContent = n.text; b.type = "button"; b.className = "mini"; b.textContent = "Fix: \u201c" + n.ask + "\u201d";
    b.addEventListener("click", () => { const box = $("#redoask");
      if (!box.value.includes(n.ask)) box.value = (box.value ? box.value + ", " : "") + n.ask; b.disabled = true; });
    li.appendChild(b); ul.appendChild(li);
  });
  if (!(r.notes || []).length) { const li = document.createElement("li"); li.className = "done"; li.textContent = "No problems found that need fixing."; ul.appendChild(li); }
  $("#redoask").value = ""; $("#redomsg").textContent = "";
}
$("#redobtn").addEventListener("click", async () => {
  const ask = $("#redoask").value.trim(); if (!ask || !lastJob) { $("#redomsg").textContent = "Type what to change first."; return; }
  const fd = new FormData(); fd.append("ask", ask);
  const res = await fetch("/api/jobs/" + lastJob + "/redo", {method: "POST", body: fd}); const r = await res.json();
  if (!res.ok) { $("#redomsg").textContent = r.error; return; }
  $("#result").style.display = "none"; $("#progress").style.display = "block";
  $("#steps").innerHTML = ""; $("#ptitle").textContent = "Redoing: " + r.changes.length + " change(s)"; $("#pbar").style.width = "10%";
  $("#progress").scrollIntoView({behavior: "smooth"}); $("#go").disabled = true;
  poll(r.id);
});
function show(id, j) {
  const r = j.result, o = r.output;
  const dg = $("#diag"); dg.innerHTML = "";
  (r.findings || []).forEach(t => { const li = document.createElement("li"); li.textContent = t; dg.appendChild(li); });
  $("#rl").textContent = o.integrated_lufs; $("#rt").textContent = o.true_peak_dbtp; $("#rk").textContent = r.key;
  const url = f => "/jobs/" + id + "/" + encodeURIComponent(f);
  $("#player").src = url(r.files.mp3_preview || r.files.master_16bit_cd);
  fanShow(r.fan, url);
  engShow(id, r);
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

"""`studiomix doctor`: check this device can make a release-ready master, and how fast.

Checks the install (Python, numpy, scipy, pyloudnorm, ffmpeg), phone storage access and free
memory, then mixes and masters a short generated song and times it - so on a phone you know
before your first real session whether everything works and how long a full song will take.
"""

from __future__ import annotations

import os
import platform
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

GB_PER_MINUTE = 0.45  # measured: a 3.5-minute song with ad-libs, doubles and harmony peaks at ~1.8 GB


def _mem_available_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return None


def _peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 / (1024 if sys.platform == "darwin" else 1)  # bytes on macOS, KB elsewhere


def _demo_audio(d: Path, seconds: float) -> tuple[Path, Path]:
    """A vocal and a beat to mix: the repo's demo generator when present, else a simple stand-in."""
    import numpy as np

    from . import audio_io

    tools = Path(__file__).resolve().parents[1] / "tools"
    try:
        sys.path.insert(0, str(tools))
        import make_demo

        v, b = make_demo.vocal(48000, seconds), make_demo.beat(44100, seconds)
    except Exception:
        sr = 48000
        t = np.arange(int(sr * seconds)) / sr
        f = 220.0 * 2 ** (np.floor(t / 0.5) % 5 * 2 / 12)
        v = (np.sin(2 * np.pi * np.cumsum(f) / sr) * (0.3 + 0.2 * np.sin(2 * np.pi * 0.5 * t)))[None]
        tb = np.arange(int(44100 * seconds)) / 44100
        b = 0.4 * np.sin(2 * np.pi * 55 * tb) * np.exp(-((tb % 0.65) * 8))
        b = np.vstack([b, b])
    finally:
        if sys.path and sys.path[0] == str(tools):
            sys.path.pop(0)
    audio_io.write_wav(d / "vocal.wav", v, 48000)
    audio_io.write_wav(d / "beat.wav", b, 44100, 16)
    return d / "vocal.wav", d / "beat.wav"


def main(argv: list[str]) -> int:
    quick = "--quick" in argv
    ok_all = True
    lines: list[str] = []

    def report(ok: bool | None, what: str, detail: str = "") -> None:
        nonlocal ok_all
        mark = {True: "PASS", False: "FAIL", None: "NOTE"}[ok]
        if ok is False:
            ok_all = False
        line = f"  [{mark}] {what}" + (f": {detail}" if detail else "")
        lines.append(line)
        print(line, flush=True)

    termux = "com.termux" in os.environ.get("PREFIX", "")
    print(f"studiomix doctor - {'Android (Termux)' if termux else platform.system()} {platform.machine()}")
    v = sys.version_info
    report(v >= (3, 10), "Python", f"{v.major}.{v.minor}.{v.micro}" + ("" if v >= (3, 10) else " (need 3.10+)"))
    for mod, need in (("numpy", True), ("scipy", True), ("pyloudnorm", True)):
        try:
            m = __import__(mod)
            report(True, mod, getattr(m, "__version__", "installed"))
        except Exception as e:
            fix = "pkg install python-" + mod if termux and mod != "pyloudnorm" else "pip install " + mod
            report(False, mod, f"missing ({e.__class__.__name__}) - fix: {fix}")
    if not ok_all:
        print("\nFix the items above, then run `studiomix doctor` again.")
        return 1
    from .dsp import _jit

    report(None, "numba", "found - faster dynamics" if _jit.HAS_NUMBA else "not installed - fine, the built-in fallback is used")
    ff = shutil.which("ffmpeg")
    report(bool(ff), "ffmpeg", "found - MP3/M4A input, MP3 previews and the second loudness meter" if ff else
           "missing - WAV still works, but install it for MP3/M4A and the independent meter: "
           + ("pkg install ffmpeg" if termux else "see ffmpeg.org"))
    if ff:
        from . import previews

        report(None if not previews.available() else True, "listen-like-a-fan previews",
               "ready (Ogg Vorbis, Opus, AAC encoders found)" if previews.available() else
               "this ffmpeg lacks the Vorbis/Opus/AAC encoders - masters are unaffected, previews are skipped")
    if termux:
        dl = Path.home() / "storage" / "downloads"
        report(dl.is_dir(), "phone storage", f"{dl} -> Downloads" if dl.is_dir() else
               "no access yet - run: termux-setup-storage (and accept the popup)")

    mem = _mem_available_gb()
    if mem is not None:
        need = GB_PER_MINUTE * 4 + 0.3
        report(mem >= need or None, "free memory",
               f"{mem:.1f} GB available; a 4-minute song with every feature needs ~{need:.1f} GB"
               + ("" if mem >= need else " - close other apps first (songs up to "
                  f"~{max(1.0, (mem - 0.3) / GB_PER_MINUTE):.0f} min will fit now)"))

    seconds = 10.0 if quick else 20.0
    print(f"\nmixing + mastering a {seconds:.0f}-second test song (auto-tune, doubles, harmony, master)...", flush=True)
    from .engine import run
    from .presets import get_preset

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        vocal, beat = _demo_audio(d, seconds)
        t0 = time.time()
        try:
            log = run(vocal, beat, d / "out", get_preset("trap", doubles=True, harmonies="3up"),
                      name="doctor", verbose=False)
        except Exception as e:  # a crash here is exactly what the doctor is for
            report(False, "test song", f"{e.__class__.__name__}: {e}")
            return 1
        took = time.time() - t0
        files = sorted(p.name for p in (d / "out").iterdir())
    o = log["output"]
    report(True, "test song", f"{o['integrated_lufs']} LUFS, {o['true_peak_dbtp']} dBTP in {took:.0f} s")
    for c in log["delivery_check"]:
        if not c["ok"] and not c["check"].startswith("Dynamics"):
            report(False, c["check"], c["detail"])
    per_min = took / seconds * 60.0 * 1.2  # measured: full songs run ~20% slower per second than the short test
    report(None, "speed", f"a 3-minute song takes about {per_min * 3 / 60:.0f} min on this device "
                          f"(peak memory so far {_peak_rss_gb():.1f} GB)")
    if "doctor_master_preview_320k.mp3" not in files:
        report(None, "mp3 preview", "skipped (no ffmpeg) - the WAV masters are what you upload anyway")

    print("\n" + ("All good - start the app with: studiomix serve" if ok_all else
                  "Something needs fixing - see FAIL above."))
    return 0 if ok_all else 1

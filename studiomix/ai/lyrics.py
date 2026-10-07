"""Lyrics -> timed captions (.srt for video, .lrc for players, .txt) from a vocal, or from a whole
song with AI separation first.

Speech recognition is OpenAI's Whisper run through sherpa-onnx (onnxruntime, CPU, int8). The
vocal is cut into phrases where it is actually sung (not fixed blocks), so each caption line
starts and ends with the singing. Measured on real vocals: Whisper "small" transcribes sung
English well (a word wrong here and there); fast French rap comes out as a draft to correct.
"base" is faster but failed on sung English, so "small" is the default. Treat the result as a
draft to edit, not finished lyrics.

    pip install sherpa-onnx
    studiomix lyrics vocal.wav --lang en          # or a full song: --song
"""

from __future__ import annotations

import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-{}.tar.bz2"
SIZES = {"tiny": "~100 MB", "base": "~160 MB", "small": "~375 MB"}


def available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
        return True
    except Exception:
        return False


def fetch(size: str, say=print) -> Path:
    """Folder with the int8 encoder/decoder and tokens of a Whisper model (downloaded once)."""
    from .separate import model_dir

    d = model_dir() / f"whisper-{size}"
    need = [f"{size}-encoder.int8.onnx", f"{size}-decoder.int8.onnx", f"{size}-tokens.txt"]
    if all((d / n).exists() for n in need):
        return d
    say(f"downloading the Whisper {size} model ({SIZES.get(size, '')} - once)")
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=d) as tmp:
        arc = Path(tmp) / "m.tar.bz2"
        with urllib.request.urlopen(URL.format(size), timeout=60) as r, open(arc, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        with tarfile.open(arc, "r:bz2") as t:
            for m in t.getmembers():  # only the three files we use, never paths outside `tmp`
                if Path(m.name).name in need and m.isfile():
                    m.name = Path(m.name).name
                    t.extract(m, tmp)
        for n in need:
            os.replace(Path(tmp) / n, d / n)
    return d


def phrases(vocal: np.ndarray, sr: int, max_s: float = 20.0) -> list[tuple[int, int]]:
    """Sample ranges where the vocal is sung: short gaps (< 0.6 s) bridged, long phrases split at
    their quietest moment so each fits Whisper's 30 s window with room to spare."""
    from ..dsp import analysis

    act = analysis.activity_mask(vocal, sr)
    act = act[0] if isinstance(act, tuple) else act
    hop = int(0.01 * sr)
    a = act[::hop]
    gap = int(0.6 / 0.01)
    runs, start, quiet = [], None, 0
    for i, on in enumerate(list(a) + [False] * (gap + 1)):
        if on:
            start = i if start is None else start
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet > gap:
                runs.append((start * hop, (i - quiet + 1) * hop))
                start, quiet = None, 0
    mono = np.mean(vocal, axis=0)
    out = []
    for s0, s1 in runs:
        s0, s1 = max(0, s0 - int(0.15 * sr)), min(len(mono), s1 + int(0.25 * sr))
        if s1 - s0 < int(0.4 * sr):
            continue
        while s1 - s0 > max_s * sr:  # split at the quietest 100 ms between 40 % and 100 % of max_s
            lo, hi = s0 + int(0.4 * max_s * sr), s0 + int(max_s * sr)
            e = np.convolve(mono[lo:hi] ** 2, np.ones(int(0.1 * sr)), "valid")
            cut = lo + int(np.argmin(e)) + int(0.05 * sr)
            out.append((s0, cut))
            s0 = cut
        out.append((s0, s1))
    return out


def _clean(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]|\([^)]*music[^)]*\)|♪", " ", text, flags=re.I)  # [Music], ♪ ...
    return re.sub(r"\s+", " ", text).strip()


def transcribe(vocal: np.ndarray, sr: int, lang: str = "", size: str = "small", say=print,
               threads: int | None = None) -> list[dict]:
    """[{start, end, text}, ...] for each sung phrase."""
    import sherpa_onnx

    from .. import audio_io

    if not available():
        raise RuntimeError("lyrics need sherpa-onnx: pip install sherpa-onnx (on a computer)")
    d = fetch(size, say)
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(d / f"{size}-encoder.int8.onnx"), decoder=str(d / f"{size}-decoder.int8.onnx"),
        tokens=str(d / f"{size}-tokens.txt"), language=lang, task="transcribe",
        num_threads=threads or max(1, min(4, os.cpu_count() or 1)))
    mono16 = audio_io.resample(np.mean(vocal, axis=0, keepdims=True), sr, 16000)[0].astype(np.float32)
    lines = []
    for s0, s1 in phrases(vocal, sr):
        a, b = int(s0 * 16000 / sr), int(s1 * 16000 / sr)
        st = rec.create_stream()
        st.accept_waveform(16000, mono16[a:b])
        rec.decode_stream(st)
        text = _clean(st.result.text)
        if text:
            lines += _caption_lines(text, s0 / sr, s1 / sr)
    return lines


def _caption_lines(text: str, t0: float, t1: float, width: int = 42) -> list[dict]:
    """Split a phrase's text into caption-sized lines (<= `width` characters, at word boundaries).
    Whisper gives no word times here, so the phrase's time is shared by length: each phrase starts
    and ends with the singing; inside a long phrase the line changes are approximate."""
    # Whisper starts each sung line with a capital letter: break there first ("I", "I'm"... aside)
    words, segs = text.split(), [[]]
    for w in words:
        if len(segs[-1]) >= 3 and w[:1].isupper() and not re.match(r"^I(\b|'|’)", w):
            segs.append([])
        segs[-1].append(w)
    out = []
    for seg in segs:  # balanced wrap: n equal-ish lines rather than a full line plus an orphan
        line = " ".join(seg)
        n = -(-len(line) // width)
        target, cur = len(line) / n, ""
        for w in seg:
            if cur and len(cur) + 1 + len(w) > target + 6:
                out.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            out.append(cur)
    total = sum(len(x) + 1 for x in out)
    lines, t = [], t0
    for x in out:
        dt = (t1 - t0) * (len(x) + 1) / total
        lines.append({"start": round(t, 2), "end": round(t + dt, 2), "text": x})
        t += dt
    return lines


def _ts(t: float, sep: str = ",") -> str:
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int(round((s % 1) * 1000)):03d}"


def write(lines: list[dict], base: Path) -> dict[str, Path]:
    """base.srt (captions for video), base.lrc (synced lyrics for players), base.txt."""
    base = Path(base)
    srt = "\n".join(f"{i + 1}\n{_ts(ln['start'])} --> {_ts(ln['end'])}\n{ln['text']}\n" for i, ln in enumerate(lines))
    lrc = "\n".join(f"[{int(ln['start'] // 60):02d}:{ln['start'] % 60:05.2f}]{ln['text']}" for ln in lines)
    files = {"srt": base.with_suffix(".srt"), "lrc": base.with_suffix(".lrc"), "txt": base.with_suffix(".txt")}
    files["srt"].write_text(srt, encoding="utf-8")
    files["lrc"].write_text(lrc + "\n", encoding="utf-8")
    files["txt"].write_text("\n".join(ln["text"] for ln in lines) + "\n", encoding="utf-8")
    return files

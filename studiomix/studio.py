"""Phone studio sessions: a beat plus recorded takes, stored on the device.

Each song is a folder `<jobs_dir>/studio/<session>/` with the beat, `takes/*.wav` and
`takes.json`. Takes arrive from the browser already latency-corrected and tagged with the beat
position they start at, so building the lead / ad-lib tracks is just placing them on a timeline.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from . import audio_io

ROLES = ("lead", "adlib")
_lock = threading.Lock()


def safe_session(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", (name or "").strip().lower()).strip("-")[:40]
    if not s:
        raise ValueError("give the song a name (letters or numbers)")
    return s


def session_dir(root: Path, session: str) -> Path:
    d = Path(root) / "studio" / safe_session(session)
    (d / "takes").mkdir(parents=True, exist_ok=True)
    return d


def _meta(d: Path) -> dict:
    p = d / "session.json"
    return json.loads(p.read_text()) if p.exists() else {"beat": None, "takes": []}


def _save_meta(d: Path, meta: dict) -> None:
    tmp = d / "session.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(d / "session.json")


def info(root: Path, session: str) -> dict:
    d = session_dir(root, session)
    with _lock:
        meta = _meta(d)
    return {"session": d.name, "beat": meta["beat"], "takes": meta["takes"]}


def save_beat(root: Path, session: str, filename: str, data: bytes) -> dict:
    d = session_dir(root, session)
    ext = Path(filename).suffix.lower() or ".wav"
    if ext not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aif", ".aiff", ".opus"}:
        ext = ".wav"
    for old in d.glob("beat.*"):
        old.unlink()
    path = d / f"beat{ext}"
    path.write_bytes(data)
    x, sr = audio_io.load(path)  # validate it decodes
    with _lock:
        meta = _meta(d)
        meta["beat"] = {"file": path.name, "name": Path(filename).name, "duration_s": round(x.shape[-1] / sr, 2)}
        _save_meta(d, meta)
    return meta["beat"]


def add_take(root: Path, session: str, data: bytes, role: str, offset_s: float, latency_ms: float,
             group: str = "", active: bool = True) -> dict:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    d = session_dir(root, session)
    tid = uuid.uuid4().hex[:10]
    path = d / "takes" / f"{tid}.wav"
    path.write_bytes(data)
    x, sr = audio_io.load(path)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    take = {"id": tid, "role": role, "offset_s": round(max(0.0, float(offset_s)), 3),
            "duration_s": round(x.shape[-1] / sr, 2), "sr": sr, "latency_ms": round(float(latency_ms), 1),
            "peak_dbfs": round(20 * np.log10(peak + 1e-12), 1), "created": time.strftime("%H:%M:%S"),
            # comping / editing
            "active": bool(active), "group": re.sub(r"[^0-9a-z]", "", group.lower())[:16],
            "trim_start_s": 0.0, "trim_end_s": 0.0, "gain_db": 0.0, "nudge_ms": 0.0}
    if peak >= 0.999:
        take["warning"] = "clipped - move the phone a bit further away or sing softer"
    elif peak < 10 ** (-40 / 20):
        take["warning"] = "very quiet - is the right microphone selected?"
    with _lock:
        meta = _meta(d)
        if take["group"] and take["active"]:  # one active pass per loop recording
            for t in meta["takes"]:
                if t.get("group") == take["group"]:
                    t["active"] = False
        meta["takes"].append(take)
        _save_meta(d, meta)
    return take


EDITABLE = {"active": bool, "trim_start_s": float, "trim_end_s": float, "gain_db": float, "nudge_ms": float}
LIMITS = {"gain_db": (-24.0, 12.0), "nudge_ms": (-500.0, 500.0)}


def update_take(root: Path, session: str, tid: str, changes: dict) -> dict:
    """Comp/edit a take: use it or not, trim its start/end, nudge its timing, change its level.
    Using a pass of a loop recording sets the other passes of that loop aside."""
    d = session_dir(root, session)
    with _lock:
        meta = _meta(d)
        take = next((t for t in meta["takes"] if t["id"] == tid), None)
        if take is None:
            raise ValueError("no such take")
        for k, conv in EDITABLE.items():
            if k not in changes:
                continue
            v = changes[k]
            v = (str(v).lower() in ("1", "true", "on", "yes")) if conv is bool else float(v)
            if k in LIMITS:
                v = float(np.clip(v, *LIMITS[k]))
            if k in ("trim_start_s", "trim_end_s"):
                v = float(np.clip(v, 0.0, max(0.0, take["duration_s"] - 0.1)))
            take[k] = v
        if take["trim_start_s"] + take["trim_end_s"] > take["duration_s"] - 0.1:
            take["trim_end_s"] = max(0.0, take["duration_s"] - 0.1 - take["trim_start_s"])
        if take.get("active") and take.get("group"):
            for t in meta["takes"]:
                if t is not take and t.get("group") == take["group"]:
                    t["active"] = False
        _save_meta(d, meta)
    return info(root, session)


def delete_take(root: Path, session: str, tid: str) -> None:
    d = session_dir(root, session)
    with _lock:
        meta = _meta(d)
        meta["takes"] = [t for t in meta["takes"] if t["id"] != tid]
        _save_meta(d, meta)
    p = d / "takes" / f"{re.sub(r'[^0-9a-f]', '', tid)}.wav"
    if p.exists():
        p.unlink()


def file_path(root: Path, session: str, kind: str, tid: str | None = None) -> Path | None:
    d = session_dir(root, session)
    meta = _meta(d)
    if kind == "beat":
        return d / meta["beat"]["file"] if meta["beat"] else None
    if kind == "take" and tid and any(t["id"] == tid for t in meta["takes"]):
        return d / "takes" / f"{tid}.wav"
    return None


def build_tracks(root: Path, session: str) -> dict:
    """Lay the takes on the beat's timeline. Returns {'beat': path, 'lead': path, 'adlib': path?}."""
    d = session_dir(root, session)
    meta = _meta(d)
    if not meta["beat"]:
        raise ValueError("upload a beat first")
    if not any(t["role"] == "lead" and t.get("active", True) for t in meta["takes"]):
        raise ValueError("record (or ★ select) at least one lead take")
    beat_path = d / meta["beat"]["file"]
    out = {"beat": beat_path}
    sr = 48000
    for role in ROLES:
        takes = [t for t in meta["takes"] if t["role"] == role and t.get("active", True)]
        if not takes:
            continue
        clips = []
        for t in takes:
            x, tsr = audio_io.load(d / "takes" / f"{t['id']}.wav")
            x = np.mean(x, axis=0, keepdims=True)
            a = int(round(t.get("trim_start_s", 0.0) * tsr))
            b = x.shape[-1] - int(round(t.get("trim_end_s", 0.0) * tsr))
            x = x[:, a:max(a + 1, b)] * 10 ** (t.get("gain_db", 0.0) / 20)
            fade = min(x.shape[-1] // 2, int(0.01 * tsr))  # 10 ms fades: trimmed edges never click
            if fade > 0:
                ramp = np.linspace(0.0, 1.0, fade)
                x[:, :fade] *= ramp
                x[:, -fade:] *= ramp[::-1]
            start = t["offset_s"] + t.get("trim_start_s", 0.0) + t.get("nudge_ms", 0.0) / 1000.0
            if start < 0:  # nudged before the song start
                x = x[:, int(round(-start * tsr)):]
                start = 0.0
            clips.append((start, audio_io.resample(x, tsr, sr)))
        n = max(int(o * sr) + c.shape[-1] for o, c in clips)
        track = np.zeros((1, n))
        for o, c in clips:
            a = int(round(o * sr))
            track[:, a:a + c.shape[-1]] += c
        peak = np.max(np.abs(track))
        if peak > 0.99:  # overlapping takes summed past full scale: scale (the chain re-levels anyway)
            track *= 0.99 / peak
        p = d / f"{role}_track.wav"
        audio_io.write_wav(p, track, sr, 32)
        out[role] = p
    return out

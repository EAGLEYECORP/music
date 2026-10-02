"""Reference profiles: learn the sound of released songs you like, then master toward it.

    studiomix learn refs/*.mp3 --name maes      # analyse references -> saved profile
    studiomix vocal.wav beat.wav --profile maes
    studiomix master mix.wav --profile maes

A profile stores what can be measured reliably even from lossy files: integrated loudness,
loudness range, the long-term tonal curve (below each file's lossy cutoff) and the stereo width
per frequency band. Audio is never stored - only these numbers.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import audio_io
from .dsp import analysis, filters, pitch
from .presets import Preset

# 1/6-octave grid the tonal curve is stored on
GRID = 20.0 * 2.0 ** (np.arange(0, 10.0 * 6 + 1) / 6.0)  # 20 Hz .. 20.5 kHz
GRID = GRID[GRID <= 20000]
WIDTH_BANDS = [(120, 500), (500, 2000), (2000, 8000), (8000, 16000)]


def profile_dir() -> Path:
    return Path(os.environ.get("STUDIOMIX_PROFILES", Path.home() / ".studiomix" / "profiles")).expanduser()


def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip().lower()).strip("-")
    if not s:
        raise ValueError("profile name must contain letters or numbers")
    return s[:40]


def _band(x: np.ndarray, sr: int, a: float, b: float) -> np.ndarray:
    return filters.bandpass(x, sr, a, min(b, sr * 0.45), order=2)


PROFILE_VERSION = 2


def side_mid_db(x: np.ndarray, sr: int, a: float, b: float) -> float:
    """Side-to-mid energy ratio (dB) strictly inside [a, b) Hz, measured in the frequency domain
    (a time-domain band-pass lets loud bass leak in and hides the band's real width)."""
    from scipy import signal as _sig

    mid, side = 0.5 * (x[0] + x[1]), 0.5 * (x[0] - x[1])
    n = int(min(8192, 2 ** int(np.log2(max(len(mid), 256)))))
    f, pm = _sig.welch(mid, sr, nperseg=n)
    _, ps = _sig.welch(side, sr, nperseg=n)
    sel = (f >= a) & (f < b)
    return float(10 * np.log10((ps[sel].sum() + 1e-20) / (pm[sel].sum() + 1e-20)))


def song_body(x: np.ndarray, sr: int, drop_lu: float = 12.0) -> tuple[int, int]:
    """Sample range of the song itself: skips video intros/outros, skits and silence - everything
    before the first and after the last moment the music is within `drop_lu` of its loud parts."""
    st = analysis.short_term_lufs(x, sr, 3.0, 1.0)
    if len(st) < 8:
        return 0, x.shape[-1]
    loud = np.percentile(st, 90)
    idx = np.flatnonzero(st > loud - drop_lu)
    a = int(idx[0]) * sr
    b = min(x.shape[-1], (int(idx[-1]) + 3) * sr)
    return a, b


def analyse(path) -> dict:
    """Measure one reference track (only the song body - intros/outros of videos are skipped)."""
    x, sr0 = audio_io.load(path)
    sr = audio_io.working_rate(sr0)
    x = audio_io.resample(x, sr0, sr)
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    x = x[:2]
    total = x.shape[-1] / sr
    a, b = song_body(x, sr)
    x = x[:, a:b]
    m = analysis.measure(x, sr)
    f, db = filters.ltas_db(np.mean(x, axis=0), sr)
    db = filters.fractional_octave_smooth(f, db, 1 / 3)
    # lossy cutoff: where the spectrum falls 45 dB below the 1-2 kHz level for good
    ref = float(np.mean(db[(f > 1000) & (f < 2000)]))
    dead = (f > 10000) & (db < ref - 45)
    cutoff = float(f[np.argmax(dead)]) if dead.any() else float(sr / 2)
    curve = np.interp(GRID, f, db)
    curve -= np.median(curve[(GRID >= 200) & (GRID <= 5000)])  # shape only, not level
    curve[GRID > min(cutoff * 0.95, 16500.0)] = np.nan
    tonic, mode, conf = pitch.detect_key(pitch.chroma(x, sr))
    tempo = analysis.estimate_tempo(x, sr)
    lo = filters.lowpass(x, sr, 120.0, order=4)
    return {
        "file": Path(path).name,
        "analysed": f"{int(a / sr) // 60}:{int(a / sr) % 60:02d}-{int(b / sr) // 60}:{int(b / sr) % 60:02d}"
                    f" of {int(total) // 60}:{int(total) % 60:02d}",
        "integrated_lufs": m["integrated_lufs"],
        "true_peak_dbtp": m["true_peak_dbtp"],
        "loudness_range_lu": m["loudness_range_lu"],
        "plr_db": m["plr_db"],
        "lossy_cutoff_hz": round(cutoff),
        "curve_db": [None if not np.isfinite(v) else round(float(v), 2) for v in curve],
        "side_mid_db": {f"{a}-{b}": round(side_mid_db(x, sr, a, b), 2) for a, b in WIDTH_BANDS},
        "low_end_correlation": round(analysis.stereo_correlation(lo), 3),
        "key": pitch.key_name(tonic, mode),
        "tempo_bpm": round(tempo, 1) if tempo else None,
    }


def learn(paths, name: str, progress=None) -> dict:
    """Analyse references and save (or extend) a profile. Returns the profile."""
    name = _safe(name)
    pdir = profile_dir()
    pdir.mkdir(parents=True, exist_ok=True)
    existing = load(name) if (pdir / f"{name}.json").exists() else None
    tracks = list(existing["tracks"]) if existing else []
    known = {t["file"] for t in tracks}
    for p in paths:
        if progress:
            progress(f"analysing {Path(p).name}")
        t = analyse(p)
        if t["file"] in known:  # re-learning the same file replaces it
            tracks = [o for o in tracks if o["file"] != t["file"]]
        tracks.append(t)
    prof = summarise(name, tracks)
    (pdir / f"{name}.json").write_text(json.dumps(prof, indent=2))
    return prof


def summarise(name: str, tracks: list[dict]) -> dict:
    curves = np.array([[np.nan if v is None else v for v in t["curve_db"]] for t in tracks], dtype=float)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # bands above every file's lossy cutoff
        curve = np.nanmean(curves, axis=0)
    # above every file's cutoff: continue with the average slope of the last octave measured
    valid = np.isfinite(curve)
    if not valid.all() and valid.sum() > 6:
        last = np.flatnonzero(valid)[-1]
        slope = (curve[last] - curve[last - 6])  # dB per octave (6 grid points)
        for i in range(last + 1, len(curve)):
            curve[i] = curve[last] + slope * (i - last) / 6.0
    widths = {k: float(np.median([t["side_mid_db"][k] for t in tracks])) for k in tracks[0]["side_mid_db"]}
    return {
        "name": name,
        "version": PROFILE_VERSION,
        "updated": time.strftime("%Y-%m-%d %H:%M"),
        "tracks": tracks,
        "target_lufs": round(float(np.median([t["integrated_lufs"] for t in tracks])), 2),
        "loudness_range_lu": round(float(np.median([t["loudness_range_lu"] for t in tracks])), 2),
        "low_end_correlation": round(float(np.median([t["low_end_correlation"] for t in tracks])), 3),
        "grid_hz": [round(float(g), 1) for g in GRID],
        "curve_db": [round(float(v), 2) for v in curve],
        "side_mid_db": {k: round(v, 2) for k, v in widths.items()},
    }


def load(name: str) -> dict:
    p = profile_dir() / f"{_safe(name)}.json"
    if not p.exists():
        avail = ", ".join(list_profiles()) or "none yet - create one with: studiomix learn REFS... --name NAME"
        raise KeyError(f"no profile '{name}' (available: {avail})")
    prof = json.loads(p.read_text())
    if prof.get("version", 1) < PROFILE_VERSION:
        raise KeyError(f"profile '{name}' was made by an older studiomix - run `studiomix learn` again "
                       "with the same songs to update it")
    return prof


def list_profiles() -> list[str]:
    d = profile_dir()
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


def apply(prof: dict, preset: Preset, keep_loudness: bool = False) -> Preset:
    """Preset tuned toward a profile: its loudness (kept inside safe limits) and how mono its
    low end is. The tonal curve and widths are applied in the master chain."""
    changes = {}
    if not keep_loudness:
        changes["target_lufs"] = float(np.clip(prof["target_lufs"], -16.0, -7.5))
    changes["master_bass_mono_hz"] = 120.0 if prof["low_end_correlation"] > 0.85 else 80.0
    return replace(preset, **changes)


def describe(prof: dict) -> str:
    lines = [f"profile '{prof['name']}' - {len(prof['tracks'])} reference(s), updated {prof['updated']}",
             f"  loudness {prof['target_lufs']} LUFS (median), LRA {prof['loudness_range_lu']} LU, "
             f"low-end correlation {prof['low_end_correlation']}",
             "  width (side vs mid): " + ", ".join(f"{k} Hz {v:+.1f} dB" for k, v in prof["side_mid_db"].items())]
    g, c = np.array(prof["grid_hz"]), np.array(prof["curve_db"])
    rel = c - filters.slope_target(g, -4.5)  # vs. the average commercial-mix tilt
    rel -= np.median(rel[(g >= 200) & (g <= 5000)])
    marks = [(60, "sub"), (150, "bass"), (350, "low-mid"), (1000, "mid"), (3000, "presence"), (8000, "brilliance"),
             (14000, "air")]
    lines.append("  tone vs typical commercial mix (dB): "
                 + "  ".join(f"{lbl} {np.interp(f_, g, rel):+.1f}" for f_, lbl in marks))
    for t in prof["tracks"]:
        lines.append(f"   - {t['file'][:40]:40s} {t['integrated_lufs']:6.2f} LUFS  {t['true_peak_dbtp']:+5.2f} dBTP  "
                     f"LRA {t['loudness_range_lu']:4.1f}  {t['key']}  {t['tempo_bpm']} BPM  cutoff {t['lossy_cutoff_hz']} Hz"
                     + (f"  (analysed {t['analysed']})" if t.get("analysed") else ""))
    return "\n".join(lines)

"""Listen like a fan: the master (and your unmastered mix) the way each platform plays it.

Every streaming service turns songs to its own loudness and re-encodes them with its own codec.
These previews apply each service's published loudness rule and its codec, so you hear what a
fan hears - and the same for the mix you started from, played by the same rules:

  Spotify       -14 LUFS; turns quiet songs up only as far as -1 dBTP allows   Ogg Vorbis ~160k
  Apple Music   -16 LUFS (Sound Check), same headroom rule                     AAC 256k (ffmpeg's)
  YouTube       -14 LUFS; only ever turns songs down                           Opus ~128k
  Phone speaker -14 LUFS turned down only, mono, a small speaker's response    AAC 128k
                (TikTok / Reels / Shorts on the phone)

Your mix is compared on Spotify and the phone speaker. The master is also encoded at full level
with Spotify's codec (fans with normalisation switched off) and decoded again: lossy encoders
push peaks up, past 0 dBFS clips on playback. That is the real
reason loud masters need true-peak headroom, measured instead of assumed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import audio_io
from .dsp import analysis, dynamics, filters

PLATFORMS = {
    # key: (label, target LUFS, may turn up, codec args, extension)
    "spotify": ("Spotify", -14.0, True, ["-c:a", "libvorbis", "-q:a", "5"], "ogg"),
    "apple": ("Apple Music", -16.0, True, ["-c:a", "aac", "-b:a", "256k"], "m4a"),
    "youtube": ("YouTube", -14.0, False, ["-c:a", "libopus", "-b:a", "128k"], "ogg"),
    # mono (it is a phone speaker); ffmpeg's fast AAC coder - its artifacts hide under the speaker's
    # response, and it halves the encoding time on a phone
    "phone": ("Phone speaker", -14.0, False, ["-c:a", "aac", "-aac_coder", "fast", "-b:a", "96k"], "m4a"),
}
CODEC_CHECKS = {
    # what a fan with normalisation off gets: the master itself through Spotify's codec. (ffmpeg's
    # own AAC encoder is not used for this: measured, its overshoot doesn't even follow the master's
    # ceiling - lowering the ceiling made it worse - so it would only judge the encoder, not you.)
    "Ogg Vorbis 320k (Spotify Very High)": (["-c:a", "libvorbis", "-q:a", "9"], "ogg"),
    "Ogg Vorbis 160k (Spotify Normal)": (["-c:a", "libvorbis", "-q:a", "5"], "ogg"),
}
# the before/after comparison is made where it matters most: Spotify and the phone speaker
COMPARE_MIX_ON = ("spotify", "phone")


def available() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    except OSError:
        return False
    return all(c in enc for c in ("libvorbis", "libopus", " aac "))


def _encode(x: np.ndarray, sr: int, path: Path, codec: list[str]) -> None:
    """Pipe float audio straight into ffmpeg (no temporary WAV on the phone's storage)."""
    data = np.ascontiguousarray(x.T.astype("<f4")).tobytes()
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(x.shape[0]),
                        "-i", "pipe:0", *codec, str(path)], input=data, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace").strip().splitlines()[-1:] or "ffmpeg failed")


def phone_speaker(x: np.ndarray, sr: int) -> np.ndarray:
    """A phone's loudspeaker: mono (1, n), nothing below ~300 Hz, a peaky upper-mid, rolled-off top."""
    m = np.mean(x, axis=0, keepdims=True)
    m = filters.highpass(m, sr, 320.0, order=4)
    m = filters.highpass(m, sr, 180.0, order=2)
    m = m + 0.6 * filters.bandpass(m, sr, 1800.0, 4200.0, order=2)
    return filters.lowpass(m, sr, 11000.0, order=2)  # mono


def playback_gain_db(lufs: float, tp: float, target: float, may_raise: bool) -> float:
    g = target - lufs
    if g > 0:
        g = min(g, -1.0 - tp) if may_raise else 0.0  # never turned up into clipping
    return g


def make(out_dir: Path, name: str, master: np.ndarray, mix: np.ndarray, sr: int,
         platforms: list[str] | None = None, master_lufs_tp: tuple[float, float] | None = None) -> dict:
    """Write the fan previews next to the master. Returns the log entry (files, levels, codec checks)."""
    out = {"platforms": {}, "codec_checks": []}
    sources = {"master": master, "mix": np.asarray(mix, dtype=np.float64)}
    meas = {k: (analysis.integrated_lufs(v, sr), dynamics.true_peak_db(v))
            for k, v in sources.items() if not (k == "master" and master_lufs_tp)}
    if master_lufs_tp:
        meas["master"] = master_lufs_tp
    jobs = []  # (audio, path, codec): encoded in parallel - each is its own ffmpeg process
    for key in platforms or list(PLATFORMS):
        label, target, may_raise, codec, ext = PLATFORMS[key]
        entry = {"label": label, "target_lufs": target}
        for which, x in sources.items():
            if which == "mix" and key not in COMPARE_MIX_ON:
                continue
            lufs, tp = meas[which]
            g = playback_gain_db(lufs, tp, target, may_raise)
            y = x * 10 ** (g / 20)
            if key == "phone":
                y = phone_speaker(y, sr)
            fn = f"{name}_fan_{key}_{which}.{ext}"
            jobs.append((np.clip(y, -1.0, 1.0).astype(np.float32), out_dir / fn, codec))
            entry[which] = {"file": fn, "plays_at_lufs": round(lufs + g, 1), "gain_db": round(g, 1)}
        out["platforms"][key] = entry
    checks = [(label, out_dir / f".codec_check_{i}.{ext}", codec)
              for i, (label, (codec, ext)) in enumerate(CODEC_CHECKS.items())]
    # the whole song: on a limited master every peak sits at the ceiling, so where the codec
    # overshoots can't be predicted from the waveform (measured: excerpts around the biggest
    # peaks read -0.1 dBFS where the full song reaches +0.6)
    full = master.astype(np.float32)
    jobs += [(full, p, codec) for _label, p, codec in checks]
    try:
        # slowest first (AAC, then the long-window Vorbis checks), so the workers finish together
        jobs.sort(key=lambda j: (0 if "aac" in j[2] and "fast" not in j[2] else 1, -j[0].size))
        with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as pool:
            for f in [pool.submit(_encode, y, sr, p, c) for y, p, c in jobs]:
                f.result()
        # normalisation off: the master through the codec at full level, decoded again
        for label, p, _codec in checks:
            dec, _ = audio_io.load(p)
            out["codec_checks"].append({"codec": label,
                                        "decoded_peak_dbfs": round(float(20 * np.log10(np.max(np.abs(dec)) + 1e-12)), 2)})
    finally:
        for _label, p, _codec in checks:
            p.unlink(missing_ok=True)
    return out

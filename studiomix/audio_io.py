"""Reading, resampling and writing audio files."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def load(path: str | Path) -> tuple[np.ndarray, int]:
    """Load any common audio file as float64 (channels, samples)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
        return data.T.copy(), int(sr)
    except Exception:
        pass
    try:
        from pedalboard.io import AudioFile

        with AudioFile(str(path)) as f:
            data = f.read(f.frames)
            return data.astype(np.float64), int(f.samplerate)
    except Exception:
        pass
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "decoded.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:a", "pcm_f32le", str(wav)],
                           check=True)
            data, sr = sf.read(str(wav), dtype="float64", always_2d=True)
            return data.T.copy(), int(sr)
    raise RuntimeError(f"could not decode {path} (install ffmpeg for more formats)")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    frac = Fraction(sr_out, sr_in).limit_denominator(1000)
    return signal.resample_poly(x, frac.numerator, frac.denominator, axis=-1)


def working_rate(*rates: int) -> int:
    """Highest input rate, but never below 44.1 kHz."""
    return max(44100, *rates)


def write_wav(path: str | Path, x: np.ndarray, sr: int, bits: int = 24) -> None:
    subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}[bits]
    sf.write(str(path), np.clip(x, -1.0, 1.0).T, sr, subtype=subtype)


def write_wav16_dithered(path: str | Path, x: np.ndarray, sr: int, seed: int = 0) -> None:
    """16-bit with TPDF dither (the industry standard for a 16-bit / CD-quality delivery)."""
    rng = np.random.default_rng(seed)
    lsb = 1.0 / 32768.0
    tpdf = (rng.random(x.shape) - rng.random(x.shape)) * lsb
    q = np.round((x + tpdf) * 32767.0)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    sf.write(str(path), q.T, sr, subtype="PCM_16")


def write_mp3(path: str | Path, x: np.ndarray, sr: int, bitrate: str = "320k") -> bool:
    """MP3 preview (for sharing, never for distribution). Returns False if no encoder is available."""
    try:
        from pedalboard.io import AudioFile

        with AudioFile(str(path), "w", sr, x.shape[0], quality=bitrate.replace("k", "")) as f:
            f.write(x.astype(np.float32))
        return True
    except Exception:
        pass
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "tmp.wav"
            write_wav(wav, x, sr, 24)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame",
                            "-b:a", bitrate, str(path)], check=True)
            return True
    return False

"""Reading, resampling and writing audio files.

Only numpy/scipy are required. WAV is read and written natively; other formats are decoded
with soundfile (libsndfile) if it is installed, otherwise with ffmpeg (`pkg install ffmpeg`
on Termux).
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        sr, data = wavfile.read(str(path))
    if data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    elif data.dtype == np.int16:
        x = data / 32768.0
    elif data.dtype == np.int32:  # scipy returns 24-bit audio left-justified in int32
        x = data / 2147483648.0
    else:
        x = data.astype(np.float64)
    x = x.reshape(len(x), -1).T
    return np.ascontiguousarray(x, dtype=np.float64), int(sr)


def _ffmpeg_decode(path: Path) -> tuple[np.ndarray, int]:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "decoded.wav"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn", "-c:a", "pcm_f32le", str(wav)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not wav.exists():
            msg = (r.stderr or "").strip().splitlines()
            raise RuntimeError(f"could not decode {path.name}: {msg[-1] if msg else 'not an audio file'}")
        return _read_wav(wav)


def load(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as float64 (channels, samples)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    errors = []
    try:
        return _read_wav(path)
    except Exception as e:
        errors.append(e)
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
        return np.ascontiguousarray(data.T), int(sr)
    except Exception as e:
        errors.append(e)
    if shutil.which("ffmpeg"):
        return _ffmpeg_decode(path)
    raise RuntimeError(f"could not decode {path.name}: install ffmpeg to open this format ({errors[-1]})")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    frac = Fraction(sr_out, sr_in).limit_denominator(1000)
    return signal.resample_poly(x, frac.numerator, frac.denominator, axis=-1)


def working_rate(*rates: int) -> int:
    """Highest input rate, but never below 44.1 kHz."""
    return max(44100, *rates)


def _write_pcm(path: str | Path, ints: np.ndarray, sr: int, bits: int) -> None:
    """Write interleaved integer samples (channels, n) as a PCM WAV file."""
    ch, n = ints.shape
    inter = ints.T.reshape(-1)
    if bits == 16:
        payload = inter.astype("<i2").tobytes()
    elif bits == 24:
        b = inter.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3]
        payload = b.tobytes()
    else:
        raise ValueError(bits)
    block = ch * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, sr * block, block, bits)
    header += b"data" + struct.pack("<I", len(payload))
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)


def write_wav(path: str | Path, x: np.ndarray, sr: int, bits: int = 24) -> None:
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    if bits == 32:
        wavfile.write(str(path), sr, x.T.astype(np.float32))
        return
    full = 2 ** (bits - 1)
    _write_pcm(path, np.clip(np.round(x * (full - 1)), -full, full - 1).astype(np.int64), sr, bits)


def write_wav16_dithered(path: str | Path, x: np.ndarray, sr: int, seed: int = 0) -> None:
    """16-bit with TPDF dither (the industry standard for a 16-bit / CD-quality delivery)."""
    rng = np.random.default_rng(seed)
    lsb = 1.0 / 32768.0
    tpdf = (rng.random(x.shape) - rng.random(x.shape)) * lsb
    q = np.clip(np.round((x + tpdf) * 32767.0), -32768, 32767).astype(np.int64)
    _write_pcm(path, q, sr, 16)


def write_mp3(path: str | Path, x: np.ndarray, sr: int, bitrate: str = "320k") -> bool:
    """MP3 preview (for sharing, never for distribution). Returns False if no encoder is available."""
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "tmp.wav"
            write_wav(wav, x, sr, 24)
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame",
                                "-b:a", bitrate, str(path)])
            if r.returncode == 0:
                return True
    try:
        from pedalboard.io import AudioFile

        with AudioFile(str(path), "w", sr, x.shape[0], quality=bitrate.replace("k", "")) as f:
            f.write(x.astype(np.float32))
        return True
    except Exception:
        return False

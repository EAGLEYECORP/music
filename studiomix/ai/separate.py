"""AI source separation: pull the vocal out of a finished song (or a beat out of a song).

Runs MDX-Net models (from the open Ultimate Vocal Remover model collection) with onnxruntime
on the CPU - no PyTorch, no GPU. The spectrogram front end (STFT 6144 / hop 1024, 3072 bins x
256 frames per chunk, overlapping chunks, optional sign-flip averaging) reproduces the reference
implementation in numpy.

    pip install onnxruntime        # once
    studiomix separate song.wav    # -> song_vocals.wav + song_instrumental.wav

Models are downloaded on first use (~65 MB each) into ~/.studiomix/models (STUDIOMIX_MODELS).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

MODEL_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{}.onnx"


@dataclass(frozen=True)
class MdxModel:
    file: str          # model name in the UVR collection
    stem: str          # what the network outputs: "vocals" | "instrumental" | ...
    other: str         # name of the remainder (mix - stem)
    compensate: float  # level correction of the network's output
    n_fft: int = 6144
    dim_f: int = 3072
    dim_t: int = 256
    hop: int = 1024
    about: str = ""


MODELS = {
    "vocals": MdxModel("Kim_Vocal_2", "vocals", "instrumental", 1.009,
                       about="lead + backing vocals out of a full song"),
    "instrumental": MdxModel("UVR-MDX-NET-Inst_HQ_3", "instrumental", "vocals", 1.022,
                             about="the cleanest beat / karaoke version"),
    # this network outputs the reverb itself; the dry voice is what's left
    "dereverb": MdxModel("Reverb_HQ_By_FoxJoy", "reverb", "dry", 1.0,
                         about="removes room / added reverb from a vocal"),
}


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def model_dir() -> Path:
    return Path(os.environ.get("STUDIOMIX_MODELS", Path.home() / ".studiomix" / "models")).expanduser()


def fetch(model: MdxModel, say=print) -> Path:
    """Path of the model file, downloading it on first use (atomically: no half files)."""
    d = model_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{model.file}.onnx"
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    say(f"downloading the {model.file} model (~65 MB, once)")
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".part")
    os.close(fd)
    try:
        with urllib.request.urlopen(MODEL_URL.format(model.file), timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        if Path(tmp).stat().st_size < 1_000_000:
            raise RuntimeError("download incomplete")
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


class _Mdx:
    def __init__(self, model: MdxModel, path: Path, threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or max(1, min(4, os.cpu_count() or 1))
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        shape = self.sess.get_inputs()[0].shape  # [batch, 4, dim_f, dim_t]: trust the file over the table
        if isinstance(shape[2], int) and isinstance(shape[3], int):
            model = replace(model, dim_f=shape[2], dim_t=shape[3])
        self.m = model
        self.n_bins = model.n_fft // 2 + 1
        self.chunk = model.hop * (model.dim_t - 1)
        n = model.n_fft
        self.win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)  # periodic Hann

    # torch.stft(center=True, reflect pad) / torch.istft equivalents for (batch, 2, chunk) audio
    def _stft(self, x: np.ndarray) -> np.ndarray:
        m, half = self.m, self.m.n_fft // 2
        xp = np.pad(x, ((0, 0), (0, 0), (half, half)), mode="reflect")
        idx = np.arange(m.dim_t)[:, None] * m.hop + np.arange(m.n_fft)[None, :]
        spec = np.fft.rfft(xp[..., idx] * self.win, axis=-1)          # (b, 2, t, bins)
        spec = spec.transpose(0, 1, 3, 2)[:, :, : m.dim_f, :]          # (b, 2, f, t)
        out = np.empty((x.shape[0], 4, m.dim_f, m.dim_t), np.float32)
        out[:, 0::2], out[:, 1::2] = spec.real, spec.imag               # [L re, L im, R re, R im]
        return out

    def _istft(self, s: np.ndarray) -> np.ndarray:
        m, half = self.m, self.m.n_fft // 2
        c = s[:, 0::2] + 1j * s[:, 1::2]                                # (b, 2, f, t)
        c = np.pad(c, ((0, 0), (0, 0), (0, self.n_bins - m.dim_f), (0, 0)))
        frames = np.fft.irfft(c.transpose(0, 1, 3, 2), n=m.n_fft, axis=-1) * self.win
        total = m.n_fft + m.hop * (m.dim_t - 1)
        y = np.zeros(s.shape[:1] + (2, total), np.float64)
        env = np.zeros(total)
        for t in range(m.dim_t):
            a = t * m.hop
            y[..., a:a + m.n_fft] += frames[:, :, t]
            env[a:a + m.n_fft] += self.win ** 2
        y /= np.maximum(env, 1e-8)
        return y[..., half: half + self.chunk]

    def run(self, mix: np.ndarray, progress=None, fast: bool = True) -> np.ndarray:
        """mix (2, n) at 44.1 kHz -> the model's stem (2, n)."""
        m = self.m
        trim = m.n_fft // 2
        gen = self.chunk - 2 * trim
        n = mix.shape[-1]
        pad = gen - n % gen
        x = np.concatenate([np.zeros((2, trim)), mix, np.zeros((2, pad + trim))], axis=1)
        starts = list(range(0, x.shape[-1] - self.chunk + 1, gen))
        out = np.zeros((2, len(starts) * gen))
        for k, a in enumerate(starts):
            seg = x[None, :, a:a + self.chunk].astype(np.float32)
            spec = self._stft(seg)
            pred = self.sess.run(None, {"input": spec})[0]
            # the sign-flip pair of the reference implementation: measured 0.05 dB SDR better on
            # real music for twice the time, so it is the optional "best" mode
            if not fast:
                pred = 0.5 * (pred - self.sess.run(None, {"input": -spec})[0])
            out[:, k * gen:(k + 1) * gen] = self._istft(pred)[0, :, trim:trim + gen]
            if progress:
                progress((k + 1) / len(starts))
        return out[:, :n] * m.compensate


def separate(x: np.ndarray, sr: int, kind: str = "vocals", say=print, progress=None,
             fast: bool = True) -> dict[str, np.ndarray]:
    """Split a (channels, n) signal. Returns {stem: audio, other: mix - stem} at the input rate."""
    from .. import audio_io
    from ..dsp import effects

    if not available():
        raise RuntimeError("AI separation needs onnxruntime: pip install onnxruntime "
                           "(on a computer - Termux usually has no build of it)")
    model = MODELS[kind]
    eng = _Mdx(model, fetch(model, say))
    stereo = effects.to_stereo(x)
    work = audio_io.resample(stereo, sr, 44100)  # the models are trained at 44.1 kHz
    stem = audio_io.resample(eng.run(work, progress, fast), 44100, sr)[:, : stereo.shape[-1]]
    if stem.shape[-1] < stereo.shape[-1]:
        stem = np.pad(stem, ((0, 0), (0, stereo.shape[-1] - stem.shape[-1])))
    return {model.stem: stem, model.other: stereo - stem}

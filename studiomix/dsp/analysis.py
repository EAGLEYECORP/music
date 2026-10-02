"""Measurement: loudness (BS.1770), loudness range, true peak, activity, tempo."""

from __future__ import annotations

import warnings

import numpy as np
import pyloudnorm
from scipy import signal

from .dynamics import true_peak_db
from .filters import biquad_sos

EPS = 1e-12


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """ITU-R BS.1770-4 integrated loudness. Silent input returns -70."""
    meter = pyloudnorm.Meter(sr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = meter.integrated_loudness(x.T)
    return float(v) if np.isfinite(v) else -70.0


def _k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    sos = np.vstack([
        biquad_sos("highshelf", 1681.97, sr, 0.7072, 3.9998),
        biquad_sos("highpass", 38.135, sr, 0.5003),
    ])
    return signal.sosfilt(sos, x, axis=-1)


def short_term_lufs(x: np.ndarray, sr: int, window_s: float = 3.0, hop_s: float = 1.0) -> np.ndarray:
    """Short-term loudness (3 s windows by default) in LUFS."""
    k = _k_weight(x, sr)
    p = np.sum(k * k, axis=0)  # sum over channels (L/R weights are 1.0)
    win, hop = int(window_s * sr), int(hop_s * sr)
    if len(p) < win:
        return np.array([-0.691 + 10 * np.log10(np.mean(p) + EPS)])
    csum = np.concatenate([[0.0], np.cumsum(p)])
    starts = np.arange(0, len(p) - win + 1, hop)
    ms = (csum[starts + win] - csum[starts]) / win
    return -0.691 + 10.0 * np.log10(ms + EPS)


def loudness_range(x: np.ndarray, sr: int) -> float:
    """EBU R128 LRA (EBU Tech 3342): spread between the 10th and 95th percentile of gated short-term loudness."""
    st = short_term_lufs(x, sr)
    st = st[st > -70.0]
    if len(st) < 2:
        return 0.0
    rel = 10.0 * np.log10(np.mean(10.0 ** (st / 10.0))) - 20.0
    st = st[st > rel]
    if len(st) < 2:
        return 0.0
    return float(np.percentile(st, 95) - np.percentile(st, 10))


def sample_peak_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(np.max(np.abs(x)) + EPS))


def stereo_correlation(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 1.0
    l, r = x[0], x[1]
    d = np.sqrt(np.sum(l * l) * np.sum(r * r))
    return float(np.sum(l * r) / d) if d > 0 else 1.0


def frame_rms_db(x: np.ndarray, sr: int, frame_ms: float = 50.0) -> np.ndarray:
    hop = max(1, int(sr * frame_ms / 1000))
    p = np.mean(x * x, axis=0)
    n = len(p) // hop
    if n == 0:
        return np.array([10 * np.log10(np.mean(p) + EPS)])
    return 10.0 * np.log10(p[: n * hop].reshape(n, hop).mean(axis=1) + EPS)


def activity_mask(x: np.ndarray, sr: int, frame_ms: float = 50.0, margin_db: float = 30.0):
    """Per-sample boolean mask of where a source (e.g. vocal) is actually performing.

    A frame is active when it is louder than both an estimated noise floor + 10 dB and
    the loud passages - `margin_db`.
    """
    hop = max(1, int(sr * frame_ms / 1000))
    f = frame_rms_db(x, sr, frame_ms)
    # ignore digital silence (e.g. gaps between takes placed on a timeline): it says nothing about
    # the room's noise floor and would make every bit of real noise look like singing
    real = f[f > -100.0]
    if len(real) < 10:
        real = f
    loud = np.percentile(real, 95)
    floor = np.percentile(real, 10)
    thr = max(floor + 10.0, loud - margin_db, -80.0)
    frames = f > thr
    # close tiny gaps (consonants, breaths) with a short dilation
    k = max(1, int(150 / frame_ms))
    frames = np.convolve(frames.astype(float), np.ones(2 * k + 1), mode="same") > 0
    mask = np.repeat(frames, hop)
    if len(mask) < x.shape[-1]:
        mask = np.concatenate([mask, np.full(x.shape[-1] - len(mask), frames[-1] if len(frames) else False)])
    return mask[: x.shape[-1]], float(floor)


def estimate_tempo(x: np.ndarray, sr: int, lo_bpm: float = 70.0, hi_bpm: float = 180.0) -> float | None:
    """Rough tempo estimate from a spectral-flux onset envelope (used to sync vocal delays)."""
    mono = np.mean(x, axis=0)
    if len(mono) < sr * 8:
        return None
    hop = 512
    _, _, z = signal.stft(mono, sr, nperseg=2048, noverlap=2048 - hop, boundary=None, padded=False)
    mag = np.log1p(np.abs(z) * 100.0)
    flux = np.maximum(np.diff(mag, axis=1), 0.0).sum(axis=0)
    flux = flux - signal.medfilt(flux, 31)
    flux = np.maximum(flux, 0.0)
    if not np.any(flux):
        return None
    flux = flux - flux.mean()
    ac = signal.correlate(flux, flux, mode="full", method="fft")[len(flux) - 1:]
    fps = sr / hop
    lags = np.arange(len(ac))
    bpm = np.where(lags > 0, 60.0 * fps / np.maximum(lags, 1), 0)
    valid = (bpm >= lo_bpm) & (bpm <= hi_bpm)
    if not valid.any():
        return None
    best = lags[valid][np.argmax(ac[valid])]
    # parabolic interpolation for sub-frame precision
    if 1 <= best < len(ac) - 1:
        a, b, c = ac[best - 1], ac[best], ac[best + 1]
        denom = a - 2 * b + c
        best = best + (0.5 * (a - c) / denom if denom != 0 else 0.0)
    return float(60.0 * fps / best)


def measure(x: np.ndarray, sr: int) -> dict:
    lufs = integrated_lufs(x, sr)
    tp = true_peak_db(x)
    st = short_term_lufs(x, sr)
    return {
        "integrated_lufs": round(lufs, 2),
        "true_peak_dbtp": round(tp, 2),
        "sample_peak_dbfs": round(sample_peak_db(x), 2),
        "loudness_range_lu": round(loudness_range(x, sr), 2),
        "max_short_term_lufs": round(float(np.max(st)), 2),
        "plr_db": round(tp - lufs, 2),
        "stereo_correlation": round(stereo_correlation(x), 3),
        "duration_s": round(x.shape[-1] / sr, 2),
    }


def ffmpeg_ebur128(path) -> dict | None:
    """Independent second opinion from ffmpeg's EBU R128 meter (integrated, LRA, true peak).

    Returns None when ffmpeg isn't installed. ffmpeg's true peak uses its own 4x oversampler,
    so small (< ~0.1 dB) differences from ours are normal.
    """
    import re
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return None
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                        "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
                       capture_output=True, text=True)
    summary = r.stderr[r.stderr.rfind("Summary:"):]
    vals = {}
    for key, pat in (("integrated_lufs", r"I:\s*(-?[\d.]+|-inf) LUFS"), ("loudness_range_lu", r"LRA:\s*(-?[\d.]+) LU"),
                     ("true_peak_dbtp", r"Peak:\s*(-?[\d.]+|-inf) dBFS")):
        m = re.search(pat, summary)
        if m:
            vals[key] = float(m.group(1)) if m.group(1) != "-inf" else -float("inf")
    return vals or None

"""Filters: RBJ biquads, Linkwitz-Riley crossovers and linear-phase matching EQ.

All audio is shaped (channels, samples), float64.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

EPS = 1e-12


def biquad_sos(kind: str, f0: float, sr: int, q: float = 0.7071, gain_db: float = 0.0) -> np.ndarray:
    """Return a single second-order section (RBJ Audio EQ Cookbook)."""
    f0 = float(np.clip(f0, 10.0, sr * 0.49))
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)

    if kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sq = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sq)]
        a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sq]
    elif kind == "highshelf":
        sq = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sq)]
        a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sq]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        raise ValueError(f"unknown biquad kind: {kind}")

    b = np.asarray(b) / a[0]
    a = np.asarray(a) / a[0]
    return np.concatenate([b, a])[None, :]


def apply_sos(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return signal.sosfilt(sos, x, axis=-1)


def eq(x: np.ndarray, sr: int, kind: str, f0: float, gain_db: float = 0.0, q: float = 0.7071) -> np.ndarray:
    if kind in ("peak", "lowshelf", "highshelf") and abs(gain_db) < 1e-3:
        return x
    return apply_sos(x, biquad_sos(kind, f0, sr, q, gain_db))


def highpass(x: np.ndarray, sr: int, f: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, min(f, sr * 0.45), "highpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x, axis=-1)


def lowpass(x: np.ndarray, sr: int, f: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, min(f, sr * 0.45), "lowpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x, axis=-1)


def bandpass(x: np.ndarray, sr: int, f_lo: float, f_hi: float, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, [f_lo, min(f_hi, sr * 0.45)], "bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x, axis=-1)


# ---------------------------------------------------------------- crossovers

def _lr4(sr: int, f: float):
    lp = signal.butter(2, f, "lowpass", fs=sr, output="sos")
    hp = signal.butter(2, f, "highpass", fs=sr, output="sos")
    return np.vstack([lp, lp]), np.vstack([hp, hp])


def lr4_split(x: np.ndarray, sr: int, f: float):
    """Linkwitz-Riley 24 dB/oct split. low + high is an allpass of x (flat magnitude)."""
    lp, hp = _lr4(sr, f)
    return signal.sosfilt(lp, x, axis=-1), signal.sosfilt(hp, x, axis=-1)


def lr4_allpass(x: np.ndarray, sr: int, f: float) -> np.ndarray:
    lo, hi = lr4_split(x, sr, f)
    return lo + hi


def three_band_split(x: np.ndarray, sr: int, f1: float, f2: float):
    """Phase-coherent 3-band split; low + mid + high has flat magnitude response."""
    low, rest = lr4_split(x, sr, f1)
    mid, high = lr4_split(rest, sr, f2)
    low = lr4_allpass(low, sr, f2)  # align low band phase with the mid/high split
    return low, mid, high


# ---------------------------------------------------------------- spectral matching

def ltas_db(mono: np.ndarray, sr: int, nfft: int = 8192):
    """Long-term average spectrum in dB."""
    nfft = int(min(nfft, max(256, 2 ** int(np.log2(max(len(mono), 256))))))
    f, p = signal.welch(mono, sr, nperseg=nfft, noverlap=nfft // 2, window="hann")
    return f, 10.0 * np.log10(p + EPS)


def fractional_octave_smooth(freqs: np.ndarray, db: np.ndarray, frac: float = 1 / 3) -> np.ndarray:
    """Smooth a spectrum (in dB) by averaging power over a fractional-octave window."""
    power = 10.0 ** (db / 10.0)
    csum = np.concatenate([[0.0], np.cumsum(power)])
    df = freqs[1] - freqs[0]
    half = 2.0 ** (frac / 2.0)
    lo = np.clip(np.floor(freqs / half / df).astype(int), 0, len(freqs) - 1)
    hi = np.clip(np.ceil(freqs * half / df).astype(int), 0, len(freqs) - 1)
    hi = np.maximum(hi, lo)
    avg = (csum[hi + 1] - csum[lo]) / (hi - lo + 1)
    return 10.0 * np.log10(avg + EPS)


def match_eq_fir(freqs: np.ndarray, gain_db: np.ndarray, sr: int, ntaps: int = 4097) -> np.ndarray:
    """Linear-phase FIR that realises the given gain curve."""
    nyq = sr / 2.0
    f = np.clip(freqs, 0, nyq)
    g = 10.0 ** (gain_db / 20.0)
    if f[0] > 0:
        f = np.concatenate([[0.0], f])
        g = np.concatenate([[g[0]], g])
    if f[-1] < nyq:
        f = np.concatenate([f, [nyq]])
        g = np.concatenate([g, [g[-1]]])
    f, idx = np.unique(f, return_index=True)
    g = g[idx]
    return signal.firwin2(ntaps, f / nyq, g, window="blackmanharris")


def apply_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply a linear-phase FIR with latency compensation (output length == input length)."""
    delay = (len(h) - 1) // 2
    y = signal.oaconvolve(x, h[None, :], mode="full", axes=-1)
    return y[:, delay:delay + x.shape[-1]]


def correction_curve(
    freqs: np.ndarray,
    measured_db: np.ndarray,
    target_db: np.ndarray,
    strength: float,
    max_db: float,
    f_lo: float,
    f_hi: float,
    smooth_oct: float = 1 / 2,
    max_boost_db: float | None = None,
) -> np.ndarray:
    """Smoothed, clamped, level-independent EQ curve pushing `measured` toward `target`."""
    band = (freqs >= f_lo) & (freqs <= f_hi)
    diff = target_db - measured_db
    diff = diff - np.median(diff[band])  # only shape matters, not absolute level
    diff = np.clip(diff * strength, -max_db, max_db if max_boost_db is None else max_boost_db)
    # fade the correction out smoothly outside [f_lo, f_hi]
    w = np.ones_like(freqs)
    with np.errstate(divide="ignore"):
        lf = np.log2(np.maximum(freqs, 1.0))
    w = np.where(freqs < f_lo, np.clip(1 - (np.log2(f_lo) - lf), 0, 1), w)
    w = np.where(freqs > f_hi, np.clip(1 - (lf - np.log2(f_hi)), 0, 1), w)
    return _smooth_lin(freqs, diff * w, smooth_oct)


def _smooth_lin(freqs: np.ndarray, db: np.ndarray, frac: float = 1 / 2) -> np.ndarray:
    """Fractional-octave smoothing of a curve that is already in dB (averages dB, not power)."""
    csum = np.concatenate([[0.0], np.cumsum(db)])
    df = freqs[1] - freqs[0]
    half = 2.0 ** (frac / 2.0)
    lo = np.clip(np.floor(freqs / half / df).astype(int), 0, len(freqs) - 1)
    hi = np.clip(np.ceil(freqs * half / df).astype(int), 0, len(freqs) - 1)
    hi = np.maximum(hi, lo)
    return (csum[hi + 1] - csum[lo]) / (hi - lo + 1)


def slope_target(freqs: np.ndarray, db_per_octave: float, pivot: float = 1000.0) -> np.ndarray:
    return db_per_octave * np.log2(np.maximum(freqs, 20.0) / pivot)

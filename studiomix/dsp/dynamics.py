"""Dynamics: compressor, expander, de-esser, soft clipper and true-peak limiter."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from ._jit import HAS_NUMBA, kernel
from .filters import bandpass, highpass

EPS = 1e-12


def coef(ms: float, sr: int) -> float:
    """One-pole smoothing coefficient for a time constant in milliseconds."""
    return float(np.exp(-1.0 / (max(ms, 1e-3) * 1e-3 * sr)))


def db_to_lin(db):
    return 10.0 ** (np.asarray(db) / 20.0)


def detector_db(x: np.ndarray, sr: int, mode: str = "rms", window_ms: float = 5.0) -> np.ndarray:
    """Per-sample level in dB of a (channels, n) signal, linked across channels."""
    if mode == "rms":
        p = np.mean(x * x, axis=0)
        c = coef(window_ms, sr)
        p = signal.lfilter([1.0 - c], [1.0, -c], p)
        return 10.0 * np.log10(np.maximum(p, 0.0) * 2.0 + EPS)  # +3 dB: sine RMS reads as its peak
    return 20.0 * np.log10(np.max(np.abs(x), axis=0) + EPS)


@kernel
def _compressor_gain(det_db, thr, ratio, knee, att, rel, max_gr):
    n = len(det_db)
    out = np.empty(n)
    slope = 1.0 / ratio - 1.0
    g = 0.0
    for i in range(n):
        over = det_db[i] - thr
        if knee > 0.0 and -knee < 2.0 * over < knee:
            t = slope * (over + knee / 2.0) ** 2 / (2.0 * knee)
        elif over > 0.0:
            t = slope * over
        else:
            t = 0.0
        if t < -max_gr:
            t = -max_gr
        if t < g:
            g = att * g + (1.0 - att) * t
        else:
            g = rel * g + (1.0 - rel) * t
        out[i] = g
    return out


@kernel
def _expander_gain(det_db, thr, ratio, range_db, att, rel):
    n = len(det_db)
    out = np.empty(n)
    g = 0.0
    for i in range(n):
        under = det_db[i] - thr
        t = (ratio - 1.0) * under if under < 0.0 else 0.0
        if t < -range_db:
            t = -range_db
        if t > g:  # opening: fast
            g = att * g + (1.0 - att) * t
        else:  # closing: slow
            g = rel * g + (1.0 - rel) * t
        out[i] = g
    return out


@kernel
def _release_follow(g_db, rel):
    n = len(g_db)
    out = np.empty(n)
    g = 0.0
    for i in range(n):
        t = g_db[i]
        if t < g:
            g = t
        else:
            g = rel * g + (1.0 - rel) * t
        out[i] = g
    return out


# Without numba the kernels run in plain Python on a decimated control signal (one value per
# CONTROL_BLOCK samples, ~0.3 ms at 48 kHz) and the resulting gain curve is interpolated back.
CONTROL_BLOCK = 16


def _blocks(x: np.ndarray, reduce):
    n = len(x)
    nb = -(-n // CONTROL_BLOCK)
    pad = nb * CONTROL_BLOCK - n
    xb = np.pad(x, (0, pad), mode="edge").reshape(nb, CONTROL_BLOCK)
    return reduce(xb, axis=1)


def _unblock(g: np.ndarray, n: int) -> np.ndarray:
    centers = np.arange(len(g)) * CONTROL_BLOCK + (CONTROL_BLOCK - 1) / 2.0
    return np.interp(np.arange(n), centers, g)


def compressor_gain(det_db: np.ndarray, sr: int, thr: float, ratio: float, knee: float, attack_ms: float,
                    release_ms: float, max_gr: float) -> np.ndarray:
    """Gain reduction (dB, <= 0) for a detector signal."""
    if HAS_NUMBA:
        return _compressor_gain(det_db, float(thr), float(ratio), float(knee), coef(attack_ms, sr),
                                coef(release_ms, sr), float(max_gr))
    csr = sr / CONTROL_BLOCK
    g = _compressor_gain(_blocks(det_db, np.max).tolist(), float(thr), float(ratio), float(knee),
                         coef(attack_ms, csr), coef(release_ms, csr), float(max_gr))
    return _unblock(g, len(det_db))


def expander_gain(det_db: np.ndarray, sr: int, thr: float, ratio: float, range_db: float, attack_ms: float,
                  release_ms: float) -> np.ndarray:
    if HAS_NUMBA:
        return _expander_gain(det_db, float(thr), float(ratio), float(range_db), coef(attack_ms, sr),
                              coef(release_ms, sr))
    csr = sr / CONTROL_BLOCK
    g = _expander_gain(_blocks(det_db, np.max).tolist(), float(thr), float(ratio), float(range_db),
                       coef(attack_ms, csr), coef(release_ms, csr))
    return _unblock(g, len(det_db))


def release_follow(g_db: np.ndarray, sr: int, release_ms: float) -> np.ndarray:
    """Instant attack / smooth release on a gain curve; never exceeds the input gain."""
    if HAS_NUMBA:
        return _release_follow(g_db, coef(release_ms, sr))
    g = _release_follow(_blocks(g_db, np.min).tolist(), coef(release_ms, sr / CONTROL_BLOCK))
    return np.minimum(_unblock(g, len(g_db)), g_db)


def compress(
    x: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    knee_db: float = 6.0,
    makeup_db: float = 0.0,
    sidechain: np.ndarray | None = None,
    detector: str = "rms",
    rms_ms: float = 5.0,
    max_gr_db: float = 40.0,
):
    """Feed-forward compressor. Returns (output, gain_reduction_db per sample)."""
    key = x if sidechain is None else sidechain
    det = detector_db(key, sr, detector, rms_ms)
    gr = compressor_gain(det, sr, threshold_db, ratio, knee_db, attack_ms, release_ms, max_gr_db)
    return x * db_to_lin(gr + makeup_db)[None, :], gr


def expand(x: np.ndarray, sr: int, threshold_db: float, ratio: float = 2.0, range_db: float = 10.0,
           attack_ms: float = 2.0, release_ms: float = 120.0):
    """Downward expander (soft gate) for cleaning noise between phrases."""
    det = detector_db(x, sr, "rms", 10.0)
    gr = expander_gain(det, sr, threshold_db, ratio, range_db, attack_ms, release_ms)
    return x * db_to_lin(gr)[None, :], gr


def deess(x: np.ndarray, sr: int, active: np.ndarray, split_hz: float = 4500.0, max_reduction_db: float = 8.0,
          sensitivity_db: float = 3.0):
    """Split-band de-esser. Only the band above `split_hz` is attenuated.

    The threshold adapts to the material: sibilant energy that sticks out more than
    `sensitivity_db` above the typical (75th percentile) sibilance level is reduced.
    """
    high = highpass(x, sr, split_hz, order=2)
    low = x - high  # exact complementary split
    key = bandpass(x, sr, 5000.0, min(10000.0, sr * 0.45), order=2)
    det = detector_db(key, sr, "rms", 1.0)
    act = det[active] if active is not None and active.any() else det
    thr = float(np.percentile(act, 75)) + sensitivity_db
    gr = compressor_gain(det, sr, thr, 5.0, 4.0, 0.5, 60.0, max_reduction_db)
    return low + high * db_to_lin(gr)[None, :], gr


# ---------------------------------------------------------------- limiting

def _refined_peaks(up: np.ndarray, refine_from: float = 0.0) -> np.ndarray:
    """|signal| with each local extremum replaced by the vertex of the parabola through it and its
    two neighbours. A 4x-oversampled grid can still miss the top of a fast peak by ~0.15 dB; the
    parabola recovers it (worst-case error < 0.02 dB), at a fraction of the cost of 8x.
    Only points above `refine_from` are refined (refinement never lifts a value by more than
    ~0.2 dB, so callers that care about the top only pass a threshold just below it)."""
    out = np.abs(up)
    if out.shape[-1] < 3:
        return out
    idx = np.nonzero(out[..., 1:-1] > refine_from)
    if not idx[0].size:
        return out
    j = idx[-1] + 1
    lead = idx[:-1]
    a0, a1, a2 = up[(*lead, j - 1)], up[(*lead, j)], up[(*lead, j + 1)]
    den = a0 - 2.0 * a1 + a2
    is_ext = ((a1 >= a0) & (a1 >= a2)) | ((a1 <= a0) & (a1 <= a2))
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = a1 - (a2 - a0) ** 2 / (8.0 * den)
    vertex = np.where(is_ext & (np.abs(den) > 1e-12), vertex, a1)
    out[(*lead, j)] = np.maximum(out[(*lead, j)], np.abs(vertex))
    return out


_REFINE_MARGIN = 10.0 ** (-0.5 / 20.0)  # refine everything within 0.5 dB of the level that matters


# Oversampling a whole song at once costs ~640 MB for 3.5 min of stereo at 48 kHz (x4). Work in
# blocks instead; each block is padded with real neighbouring samples, longer than the resampling
# filter reaches, so the result is identical to processing everything at once.
_BLOCK = 1 << 18  # ~5 s at 48 kHz
_PAD = 128


def _os_blocks(n: int):
    for s0 in range(0, n, _BLOCK):
        a, b = max(0, s0 - _PAD), min(n, s0 + _BLOCK + _PAD)
        yield s0, min(n, s0 + _BLOCK), a, b


def true_peak_envelope(x: np.ndarray, oversample: int = 4, exact_above: float = 0.0) -> np.ndarray:
    """Per-sample true-peak magnitude (max over the oversampled signal, all channels).
    Values below `exact_above` (linear) may read up to ~0.2 dB low - pass the level you act on."""
    n = x.shape[-1]
    out = np.empty(n)
    for s0, s1, a, b in _os_blocks(n):
        up = signal.resample_poly(x[:, a:b], oversample, 1, axis=-1)
        m = np.max(_refined_peaks(up, exact_above * _REFINE_MARGIN), axis=0)
        m = m[: (b - a) * oversample].reshape(b - a, oversample).max(axis=1)
        out[s0:s1] = m[s0 - a: s1 - a]
    return np.maximum(out, np.roll(out, -1))  # inter-sample peaks straddle two samples


def true_peak_db(x: np.ndarray, oversample: int = 4) -> float:
    """ITU-R BS.1770-style true peak (dBTP): 4x oversampling plus parabolic peak refinement."""
    peak = 0.0
    for _s0, _s1, a, b in _os_blocks(x.shape[-1]):
        up = signal.resample_poly(x[:, a:b], oversample, 1, axis=-1)
        if up.size:
            floor = max(peak, float(np.max(np.abs(up)))) * _REFINE_MARGIN
            peak = max(peak, float(np.max(_refined_peaks(up, floor))))
    return float(20.0 * np.log10(peak + EPS))


def soft_clip(x: np.ndarray, ceiling_db: float, knee_db: float = 3.0, oversample: int = 4) -> np.ndarray:
    """Oversampled soft clipper: linear below (ceiling - knee), tanh-saturating up to ceiling."""
    t = 10.0 ** (ceiling_db / 20.0)
    k = t * 10.0 ** (-knee_db / 20.0)
    if np.max(np.abs(x)) < k * 0.5:  # far below the knee even allowing for inter-sample peaks
        return x
    out = x.copy()
    for s0, s1, a, b in _os_blocks(x.shape[-1]):
        seg = x[:, a:b]
        if np.max(np.abs(seg)) < k * 0.5:
            continue
        up = signal.resample_poly(seg, oversample, 1, axis=-1)
        over = np.abs(up) > k
        if not over.any():
            continue
        av = np.abs(up[over])
        up[over] = np.sign(up[over]) * (k + (t - k) * np.tanh((av - k) / (t - k)))
        down = signal.resample_poly(up, 1, oversample, axis=-1)[:, : b - a]
        out[:, s0:s1] = down[:, s0 - a: s1 - a]
    return out


def limit(x: np.ndarray, sr: int, ceiling_db: float = -1.0, lookahead_ms: float = 1.5,
          release_ms: float = 80.0, oversample: int = 4):
    """Look-ahead true-peak brickwall limiter. Returns (output, gain_reduction_db)."""
    c = 10.0 ** (ceiling_db / 20.0)
    env = true_peak_envelope(x, oversample, exact_above=c)
    req = np.minimum(1.0, c / np.maximum(env, EPS))
    a = max(1, int(lookahead_ms * 1e-3 * sr))
    width = 2 * a + 1
    # min over a window then a box average of the same width: the smoothed gain is
    # guaranteed to be <= the required gain at every sample (no overs), with a
    # smooth attack ramp starting `a` samples before each peak.
    g = minimum_filter1d(req, width, mode="nearest")
    g = uniform_filter1d(g, width, mode="nearest")
    g_db = release_follow(20.0 * np.log10(np.maximum(g, EPS)), sr, release_ms)
    y = x * db_to_lin(g_db)[None, :]
    # safety net for residual inter-sample overs created by the gain modulation itself. True peak
    # is homogeneous in gain (oversampling and peak refinement are linear), so one exact trim
    # lands it 0.01 dB under the ceiling - no need to measure again.
    tp = true_peak_db(y, oversample)
    if tp > ceiling_db:
        y *= 10.0 ** ((ceiling_db - tp - 0.01) / 20.0)
    return y, g_db

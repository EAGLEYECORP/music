"""Time-based and colour effects: reverb, ping-pong delay, saturation, stereo imaging."""

from __future__ import annotations

import numpy as np
import pedalboard
from scipy import signal

from .filters import highpass, lowpass, lr4_split


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 1:
        return np.vstack([x, x])
    return x[:2]


def reverb(x: np.ndarray, sr: int, room_size: float = 0.55, damping: float = 0.5, width: float = 1.0,
           predelay_ms: float = 30.0, hp_hz: float = 350.0, lp_hz: float = 7500.0) -> np.ndarray:
    """100% wet stereo reverb return with pre-delay and return EQ (like an aux send)."""
    src = to_stereo(x).astype(np.float32)
    pad = int(sr * predelay_ms / 1000)
    tail = int(sr * 4.0)
    src = np.pad(src, ((0, 0), (pad, tail)))
    board = pedalboard.Pedalboard([
        pedalboard.Reverb(room_size=room_size, damping=damping, wet_level=1.0, dry_level=0.0, width=width),
    ])
    wet = board(src, sr).astype(np.float64)[:, : x.shape[-1]]
    wet = highpass(wet, sr, hp_hz, order=2)
    return lowpass(wet, sr, lp_hz, order=2)


def ping_pong_delay(x: np.ndarray, sr: int, delay_s: float, feedback: float = 0.3, taps: int = 6,
                    hp_hz: float = 500.0, lp_hz: float = 5000.0) -> np.ndarray:
    """100% wet ping-pong delay return; each repeat alternates sides and is darker."""
    mono = np.mean(x, axis=0)
    n = len(mono)
    d = max(1, int(delay_s * sr))
    out = np.zeros((2, n))
    tap = highpass(mono[None, :], sr, hp_hz, order=2)[0]
    lp = signal.butter(1, lp_hz, "lowpass", fs=sr, output="sos")
    for k in range(1, taps + 1):
        shift = k * d
        if shift >= n:
            break
        tap = signal.sosfilt(lp, tap)  # successive repeats lose top end
        ch = (k - 1) % 2
        out[ch, shift:] += (feedback ** (k - 1)) * tap[: n - shift]
    return out


def saturate(x: np.ndarray, drive_db: float = 6.0, mix: float = 0.2) -> np.ndarray:
    """Parallel 2x-oversampled tanh saturation for harmonic warmth."""
    if mix <= 0:
        return x
    g = 10.0 ** (drive_db / 20.0)
    up = signal.resample_poly(x, 2, 1, axis=-1)
    peak = np.max(np.abs(up)) + 1e-12
    sat = np.tanh(up / peak * g) / np.tanh(g) * peak
    sat = signal.resample_poly(sat, 1, 2, axis=-1)[:, : x.shape[-1]]
    # level-match the saturated path so `mix` changes colour, not loudness
    rms_x = np.sqrt(np.mean(x * x)) + 1e-12
    rms_s = np.sqrt(np.mean(sat * sat)) + 1e-12
    return (1 - mix) * x + mix * sat * (rms_x / rms_s)


def stereo_image(x: np.ndarray, sr: int, mono_below_hz: float = 120.0, high_width: float = 1.1,
                 high_split_hz: float = 3000.0) -> np.ndarray:
    """Mono the low end (punch + vinyl/club safety) and gently widen the highs via M/S."""
    if x.shape[0] < 2:
        return x
    mid = 0.5 * (x[0] + x[1])
    side = 0.5 * (x[0] - x[1])
    side = side[None, :]
    if mono_below_hz > 0:
        _, side = lr4_split(side, sr, mono_below_hz)
    if abs(high_width - 1.0) > 1e-3:
        lo, hi = lr4_split(side, sr, high_split_hz)
        side = lo + hi * high_width
    side = side[0]
    return np.vstack([mid + side, mid - side])

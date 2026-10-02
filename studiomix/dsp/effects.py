"""Time-based and colour effects: reverb, ping-pong delay, saturation, stereo imaging."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from .filters import highpass, lowpass, lr4_split


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 1:
        return np.vstack([x, x])
    return x[:2]


def reverb_ir(sr: int, decay_s: float, damping: float = 0.5, width: float = 1.0, seed: int = 7) -> np.ndarray:
    """Synthesise a stereo plate/hall impulse response: sparse early reflections into a dense
    exponentially decaying tail whose highs die faster than its lows (air/surface absorption)."""
    rng = np.random.default_rng(seed)
    n = int(sr * min(decay_s * 1.1, 6.0))
    t = np.arange(n) / sr
    ir = np.zeros((2, n))
    # three bands with their own RT60
    bands = [("lowpass", 500.0, 1.15), ("bandpass", (500.0, 4000.0), 1.0), ("highpass", 4000.0, 1.0 - 0.65 * damping)]
    for kind, f, rt_scale in bands:
        noise = rng.standard_normal((2, n))
        sos = signal.butter(2, f, kind, fs=sr, output="sos")
        env = 10.0 ** (-3.0 * t / max(decay_s * rt_scale, 0.05))
        ir += signal.sosfilt(sos, noise, axis=-1) * env
    # diffuse tail builds up over the first ~20 ms
    ir *= np.minimum(1.0, t / 0.02)[None, :]
    # early reflections, alternating sides
    ref = np.sqrt(np.mean(ir[:, : int(sr * 0.1)] ** 2))
    for k in range(10):
        i = int(sr * rng.uniform(0.004, 0.06))
        ir[k % 2, i] += 5.0 * ref * (1 - k / 12) * rng.choice([-1, 1])
    mid = 0.5 * (ir[0] + ir[1])
    side = 0.5 * (ir[0] - ir[1]) * width
    ir = np.vstack([mid + side, mid - side])
    return ir / np.sqrt(np.sum(ir ** 2) / 2)  # unit energy per channel: wet ~ dry level


def reverb(x: np.ndarray, sr: int, room_size: float = 0.55, damping: float = 0.5, width: float = 1.0,
           predelay_ms: float = 30.0, hp_hz: float = 350.0, lp_hz: float = 7500.0) -> np.ndarray:
    """100% wet stereo reverb return with pre-delay and return EQ (like an aux send)."""
    decay = 0.5 + 2.6 * float(np.clip(room_size, 0.0, 1.0))
    ir = reverb_ir(sr, decay, damping, width)
    pad = int(sr * predelay_ms / 1000)
    src = np.mean(x, axis=0)
    src = highpass(src[None, :], sr, hp_hz, order=2)[0]
    wet = np.vstack([signal.oaconvolve(src, ir[c])[: len(src)] for c in range(2)])
    wet = np.pad(wet, ((0, 0), (pad, 0)))[:, : x.shape[-1]]
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


def pan_curve(n: int, sr: int, segments: list[tuple[int, int, float]], smooth_ms: float = 30.0) -> np.ndarray:
    """Per-sample pan position (-1 = left, +1 = right) from (start, end, pan) segments."""
    pan = np.zeros(n)
    for a, b, p in segments:
        pan[a:b] = p
    return uniform_filter1d(pan, max(1, int(sr * smooth_ms / 1000)), mode="nearest")


def apply_pan(mono: np.ndarray, pan: np.ndarray | float) -> np.ndarray:
    """Constant-power panning of a mono signal; centre gives unity gain on both sides."""
    theta = (np.asarray(pan) + 1.0) * np.pi / 4.0
    g = np.sqrt(2.0)
    return np.vstack([mono * g * np.cos(theta), mono * g * np.sin(theta)])


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

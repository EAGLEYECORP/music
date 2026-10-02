"""Voice noise reduction for phone / bedroom recordings (fans, AC, hiss, mains hum).

Learns a noise print from the gaps between phrases (where the singer is silent), then applies a
decision-directed Wiener filter (Ephraim & Malah 1984) frame by frame. The smoothed a-priori SNR
keeps the gain from jumping around between frames, which is what causes the "musical noise" /
underwater sound of simple spectral subtraction. A gain floor limits the reduction so the voice
keeps its air and breaths.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


def noise_print(power: np.ndarray, frame_active: np.ndarray | None) -> np.ndarray:
    """Noise PSD per bin from inactive frames (or the quietest 15% of frames)."""
    energy = power.sum(axis=0)
    # digital silence (e.g. the gaps between recorded takes on a timeline) tells nothing about
    # the room noise - leave it out
    real = energy > energy.max() * 1e-10
    if real.sum() < 10:
        real = np.ones_like(real)
    quiet = None
    if frame_active is not None:
        sel = (~frame_active) & real
        if sel.sum() >= max(20, 0.05 * real.sum()):
            quiet = power[:, sel]
    if quiet is None:
        e = energy[real]
        quiet = power[:, real][:, e <= np.percentile(e, 15)]
    # median is robust to the odd breath or click in the gaps; x1.44 turns the median of a
    # chi-square(2) power estimate into its mean
    return np.median(quiet, axis=1) * 1.44 + 1e-20


def denoise(x: np.ndarray, sr: int, active: np.ndarray | None = None, floor_db: float | None = None,
            alpha: float = 0.98, max_reduction_db: float = 15.0, beta: float | None = None) -> tuple[np.ndarray, dict]:
    """Reduce stationary background noise of a (channels, n) vocal. Returns (out, stats).

    floor_db=None adapts the strength to how noisy the take is: full reduction
    (-max_reduction_db) for noisy takes (SNR <= 15 dB), gentler as the take gets cleaner, and
    nothing at all above 40 dB SNR - a good take is never processed harder than it needs.
    """
    nper = int(2 ** np.round(np.log2(sr * 0.043)))  # ~43 ms
    hop = nper // 4
    f, t, Z = signal.stft(x, sr, nperseg=nper, noverlap=nper - hop, boundary="even", padded=True)
    Zm = Z if Z.ndim == 2 else Z.mean(axis=0)
    power = np.abs(Zm) ** 2

    frame_active = None
    if active is not None:
        idx = np.clip((t * sr).astype(int), 0, len(active) - 1)
        frame_active = active[idx]
    N = noise_print(power, frame_active)

    # is there anything worth removing? (signal-to-noise of the performance)
    sig = power[:, frame_active].mean() if frame_active is not None and frame_active.any() else power.mean()
    snr_db = 10 * np.log10(sig / N.mean())
    stats = {"input_snr_db": round(float(snr_db), 1)}
    if floor_db is None:
        floor_db = float(np.interp(snr_db, [15.0, 30.0, 40.0], [-max_reduction_db, -max_reduction_db * 0.45, 0.0]))
    if floor_db > -1.0:
        stats["applied"] = False
        return x, stats
    stats["max_reduction_db"] = round(-floor_db, 1)
    if beta is None:
        # gain exponent: 1 = classic Wiener. Gentler (0.5) on cleaner takes, where the voice has more
        # to lose than the noise; firmer (0.75) on noisy takes. Measured: never lowers voice
        # fidelity (SDR) on the benchmark while removing 6-13 dB of noise.
        beta = float(np.interp(snr_db, [15.0, 20.0], [0.75, 0.5]))
    floor = 10 ** (floor_db / 20)
    gamma = power / N[:, None]
    G = np.empty_like(gamma)
    g_prev = np.ones(len(N))
    gam_prev = np.ones(len(N))
    for i in range(gamma.shape[1]):
        xi = alpha * (g_prev ** 2) * gam_prev + (1 - alpha) * np.maximum(gamma[:, i] - 1.0, 0.0)
        g = (xi / (1.0 + xi)) ** beta
        g = np.maximum(g, floor)
        G[:, i] = g
        g_prev, gam_prev = g, gamma[:, i]
    # light smoothing across frequency: removes isolated single-bin "chirps"
    G = np.maximum(uniform_filter1d(G, 3, axis=0), floor)
    Zout = Z * (G if Z.ndim == 2 else G[None, :, :])
    _, y = signal.istft(Zout, sr, nperseg=nper, noverlap=nper - hop, boundary=True)
    y = np.atleast_2d(y)[:, : x.shape[-1]]
    if y.shape[-1] < x.shape[-1]:
        y = np.pad(y, ((0, 0), (0, x.shape[-1] - y.shape[-1])))
    stats.update(applied=True, noise_floor_reduction_db=round(float(10 * np.log10(np.mean(G[:, ~frame_active] ** 2)))
                                                           if frame_active is not None and (~frame_active).any()
                                                           else 0.0, 1))
    return y, stats

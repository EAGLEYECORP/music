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


_BLOCK_FRAMES = 1024  # frames per processing block: bounded memory however long the song is


class _Stft:
    """Hann-window STFT computed and resynthesised block by block (75% overlap, weighted
    overlap-add). The same transform as scipy's stft/istft pair, without ever holding the whole
    song's spectrogram - a 4-minute vocal stays at a few MB instead of gigabytes."""

    def __init__(self, x: np.ndarray, nper: int, hop: int):
        self.nper, self.hop, self.n = nper, hop, x.shape[-1]
        half = nper // 2
        xp = np.pad(x, ((0, 0), (half, half)), mode="reflect" if x.shape[-1] > half else "constant")
        extra = (-(xp.shape[-1] - nper)) % hop
        self.xp = np.pad(xp, ((0, 0), (0, extra)))
        self.frames = (self.xp.shape[-1] - nper) // hop + 1
        self.win = signal.get_window("hann", nper)

    def blocks(self):
        for i0 in range(0, self.frames, _BLOCK_FRAMES):
            yield i0, min(self.frames, i0 + _BLOCK_FRAMES)

    def spectrum(self, i0: int, i1: int) -> np.ndarray:
        """(channels, bins, frames) spectrum of frames [i0, i1)."""
        seg = self.xp[:, i0 * self.hop: (i1 - 1) * self.hop + self.nper]
        fr = np.lib.stride_tricks.sliding_window_view(seg, self.nper, axis=-1)[:, :: self.hop]
        return np.fft.rfft(fr * self.win, axis=-1).transpose(0, 2, 1)

    def overlap_add(self, out: np.ndarray, norm: np.ndarray, i0: int, Z: np.ndarray) -> None:
        fr = np.fft.irfft(Z.transpose(0, 2, 1), self.nper, axis=-1) * self.win
        w2 = self.win ** 2
        for k in range(fr.shape[1]):
            a = (i0 + k) * self.hop
            out[:, a: a + self.nper] += fr[:, k]
            norm[a: a + self.nper] += w2

    def frame_times(self, sr: int) -> np.ndarray:
        return np.arange(self.frames) * self.hop / sr


def denoise(x: np.ndarray, sr: int, active: np.ndarray | None = None, floor_db: float | None = None,
            alpha: float = 0.98, max_reduction_db: float = 15.0, beta: float | None = None) -> tuple[np.ndarray, dict]:
    """Reduce stationary background noise of a (channels, n) vocal. Returns (out, stats).

    floor_db=None adapts the strength to how noisy the take is: full reduction
    (-max_reduction_db) for noisy takes (SNR <= 15 dB), gentler as the take gets cleaner, and
    nothing at all above 40 dB SNR - a good take is never processed harder than it needs.
    """
    nper = int(2 ** np.round(np.log2(sr * 0.043)))  # ~43 ms
    hop = nper // 4
    st = _Stft(np.atleast_2d(x), nper, hop)
    # pass 1: the (channel-averaged) power spectrogram, float32 - enough for the noise print
    power = np.empty((nper // 2 + 1, st.frames), dtype=np.float32)
    for i0, i1 in st.blocks():
        power[:, i0:i1] = np.abs(st.spectrum(i0, i1).mean(axis=0)) ** 2

    frame_active = None
    if active is not None:
        idx = np.clip((st.frame_times(sr) * sr).astype(int), 0, len(active) - 1)
        frame_active = active[idx]
    N = noise_print(power, frame_active).astype(np.float64)

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

    # pass 2: decision-directed Wiener gain frame by frame, applied and resynthesised per block
    out = np.zeros(st.xp.shape)
    norm = np.zeros(st.xp.shape[-1])
    g_prev = np.ones(len(N))
    gam_prev = np.ones(len(N))
    quiet_g2, quiet_n = 0.0, 0
    for i0, i1 in st.blocks():
        gamma = power[:, i0:i1] / N[:, None]
        G = np.empty_like(gamma, dtype=np.float64)
        for i in range(gamma.shape[1]):
            xi = alpha * (g_prev ** 2) * gam_prev + (1 - alpha) * np.maximum(gamma[:, i] - 1.0, 0.0)
            g = np.maximum((xi / (1.0 + xi)) ** beta, floor)
            G[:, i] = g
            g_prev, gam_prev = g, gamma[:, i]
        # light smoothing across frequency: removes isolated single-bin "chirps"
        G = np.maximum(uniform_filter1d(G, 3, axis=0), floor)
        if frame_active is not None:
            q = ~frame_active[i0:i1]
            quiet_g2 += float(np.sum(G[:, q] ** 2))
            quiet_n += int(q.sum()) * G.shape[0]
        st.overlap_add(out, norm, i0, st.spectrum(i0, i1) * G[None])
    half = nper // 2
    y = out[:, half: half + x.shape[-1]] / np.maximum(norm[half: half + x.shape[-1]], 1e-10)
    stats.update(applied=True, noise_floor_reduction_db=round(float(10 * np.log10(quiet_g2 / quiet_n)), 1)
                 if quiet_n else 0.0)
    return y.reshape(x.shape) if x.ndim == 1 else y, stats

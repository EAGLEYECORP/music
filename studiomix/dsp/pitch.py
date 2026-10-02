"""Pitch correction ("auto-tune").

Pipeline
  1. YIN pitch tracking (de Cheveigne & Kawahara 2002), vectorised with FFTs.
  2. Key / scale detection from the beat's chroma + the singer's own pitch histogram
     (Krumhansl-Schmuckler key profiles), or a user-supplied key.
  3. Note decisions with hysteresis (no warbling between two notes), then a target pitch
     curve shaped by retune speed, humanize (sustained notes keep their vibrato) and amount.
  4. TD-PSOLA resynthesis: pitch-synchronous grains are re-spaced to the new period. Grains
     keep their original shape, so formants (the singer's timbre) are preserved - no chipmunk.
"""

from __future__ import annotations

import re

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.ndimage import median_filter

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic-minor": [0, 2, 3, 5, 7, 8, 11],
    "major-pentatonic": [0, 2, 4, 7, 9],
    "minor-pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "chromatic": list(range(12)),
}

# Krumhansl-Kessler probe-tone profiles
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(np.asarray(f) / 440.0)


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m) - 69.0) / 12.0)


# ------------------------------------------------------------------ pitch tracking

def _cmnd_candidates(x: np.ndarray, sr: int, fmin: float, fmax: float, hop: int, max_cands: int = 6):
    """YIN cumulative-mean-normalised difference, reduced to its local-minimum candidates.

    Returns per-frame arrays (nfr, max_cands) of candidate periods (samples, sub-sample precise)
    and their CMND values, in increasing-period order, plus frame RMS (dB).
    """
    W = int(2 ** np.ceil(np.log2(sr * 0.04)))  # ~40 ms analysis window
    tau_max = min(W // 2 - 2, int(sr / fmin))
    tau_min = max(2, int(sr / fmax))
    I = W - tau_max - 1  # integration length
    xp = np.pad(x, (W // 2, W // 2 + hop))
    frames = sliding_window_view(xp, W)[::hop]
    nfr = frames.shape[0]
    nfft = int(2 ** np.ceil(np.log2(W + I)))
    cand_tau = np.full((nfr, max_cands), np.nan)
    cand_val = np.full((nfr, max_cands), np.inf)
    rms = np.zeros(nfr)
    taus = np.arange(tau_max + 2)
    for s in range(0, nfr, 1024):
        fr = frames[s:s + 1024].astype(np.float64)
        n = len(fr)
        rms[s:s + n] = np.sqrt(np.mean(fr[:, W // 2 - hop: W // 2 + hop] ** 2, axis=1))
        A = np.fft.rfft(fr[:, :I], nfft)
        B = np.fft.rfft(fr, nfft)
        corr = np.fft.irfft(np.conj(A) * B, nfft)[:, : tau_max + 2]
        c = np.concatenate([np.zeros((n, 1)), np.cumsum(fr * fr, axis=1)], axis=1)
        d = np.maximum(c[:, I][:, None] + (c[:, taus + I] - c[:, taus]) - 2.0 * corr, 0.0)
        d[:, 0] = 0.0
        cs = np.cumsum(d[:, 1:], axis=1)
        cmnd = np.ones_like(d)
        cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cs, 1e-12)
        cmnd[cs[:, -1] < 1e-12, 1:] = 1.0  # digital silence

        r = cmnd[:, tau_min - 1: tau_max + 2]
        mid = r[:, 1:-1]
        is_min = (mid < r[:, :-2]) & (mid <= r[:, 2:]) & (mid < 0.6)
        rows, cols = np.nonzero(is_min)
        first = np.searchsorted(rows, np.arange(n))
        rank = np.arange(len(rows)) - first[rows]
        keep = rank < max_cands
        rows, cols, rank = rows[keep], cols[keep], rank[keep]
        t = cols + tau_min
        l, c0, rr = cmnd[rows, t - 1], cmnd[rows, t], cmnd[rows, t + 1]
        den = l - 2 * c0 + rr
        with np.errstate(divide="ignore", invalid="ignore"):
            sh = np.where(np.abs(den) > 1e-12, 0.5 * (l - rr) / den, 0.0)
        sh = np.clip(sh, -1, 1)
        cand_tau[s + rows, rank] = t + sh
        cand_val[s + rows, rank] = np.maximum(c0 - 0.25 * (l - rr) * sh, 0.0)
    return cand_tau, cand_val, 20.0 * np.log10(rms + 1e-12)


BETA_B = 8.0
UV_LOUD_SCALE = 0.12  # how much a loud frame discounts the 'unvoiced' hypothesis  # threshold prior Beta(2, BETA_B): mean 2 / (2 + BETA_B)


def _candidate_probs(cand_val: np.ndarray) -> np.ndarray:
    """pYIN (Mauch & Dixon 2014): run YIN under a Beta(2, 18) distribution of thresholds; each
    candidate's probability is the share of thresholds for which it is the *first* dip below
    threshold. This favours the true period over its multiples (octave-down errors)."""
    from scipy.special import betainc

    cdf = lambda v: betainc(2.0, BETA_B, np.clip(v, 0.0, 1.0))  # noqa: E731
    prefix = np.minimum.accumulate(cand_val, axis=1)
    prev = np.concatenate([np.full((len(cand_val), 1), np.inf), prefix[:, :-1]], axis=1)
    hi = np.where(np.isinf(prev), 1.0, cdf(np.where(np.isinf(prev), 1.0, prev)))
    lo = cdf(np.where(np.isinf(cand_val), 1.0, cand_val))
    return np.where(cand_val < prev, np.maximum(hi - lo, 0.0), 0.0)


def _viterbi(cand_midi: np.ndarray, probs: np.ndarray, gate: np.ndarray, uv_scale: np.ndarray,
             jump_cost: float = 0.5, switch_cost: float = 2.5) -> np.ndarray:
    """Most likely path through (candidates + unvoiced) over time. Returns chosen index per frame
    (K = unvoiced)."""
    nfr, K = probs.shape
    voiced_p = probs.sum(axis=1)
    em = np.empty((nfr, K + 1))
    em[:, :K] = -np.log(np.maximum(probs, 1e-9))
    em[:, :K][~np.isfinite(cand_midi)] = np.inf
    em[:, K] = -np.log(np.clip((1.0 - voiced_p) * uv_scale, 0.02, 1.0))
    em[~gate, :K] = np.inf  # too quiet to be the singer
    back = np.zeros((nfr, K + 1), dtype=np.int32)
    cost = em[0].copy()
    m_prev = cand_midi[0]
    for i in range(1, nfr):
        m = cand_midi[i]
        trans = np.empty((K + 1, K + 1))  # [from, to]
        with np.errstate(invalid="ignore"):
            jump = np.abs(m_prev[:, None] - m[None, :]) * jump_cost
        trans[:K, :K] = np.where(np.isfinite(jump), jump, np.inf)
        trans[:K, K] = switch_cost
        trans[K, :K] = switch_cost
        trans[K, K] = 0.0
        tot = cost[:, None] + trans
        back[i] = np.argmin(tot, axis=0)
        cost = tot[back[i], np.arange(K + 1)] + em[i]
        m_prev = m
    path = np.empty(nfr, dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for i in range(nfr - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def yin(x: np.ndarray, sr: int, fmin: float = 65.0, fmax: float = 1100.0, hop: int | None = None):
    """Probabilistic YIN f0 tracker with Viterbi smoothing.

    Returns (frame_times_s, f0_hz (nan = unvoiced), aperiodicity, frame_rms_db, hop).
    """
    hop = hop or int(round(sr * 0.005))
    # analyse below ~2.2 kHz: the fundamental and low harmonics carry the pitch, while breath,
    # rasp and sibilance (which make YIN call voiced frames unvoiced) mostly live above
    lp = signal.sosfiltfilt(signal.butter(4, 1200.0, fs=sr, output="sos"), x)
    cand_tau, cand_val, rms_db = _cmnd_candidates(lp, sr, fmin, fmax, hop)
    probs = _candidate_probs(cand_val)
    cand_midi = np.where(np.isfinite(cand_tau), hz_to_midi(sr / np.where(np.isfinite(cand_tau), cand_tau, 1.0)),
                         np.nan)
    loud = np.percentile(rms_db, 98)
    gate = (rms_db > loud - 45.0) & (rms_db > -75.0)
    # on an isolated vocal, energy below 1.2 kHz is strong evidence of a sung vowel ("s"/"sh"
    # consonants have almost none there): breathy-but-loud frames lean voiced
    uv_scale = np.interp(rms_db - loud, [-30.0, -8.0], [1.0, UV_LOUD_SCALE])
    path = _viterbi(cand_midi, probs, gate, uv_scale)
    K = cand_tau.shape[1]
    rows = np.arange(len(path))
    voiced = path < K
    pick = np.minimum(path, K - 1)
    f0 = np.where(voiced, sr / np.where(voiced, cand_tau[rows, pick], 1.0), np.nan)
    aper = np.where(voiced, cand_val[rows, pick], 1.0)
    times = np.arange(len(path)) * hop / sr
    return times, f0, aper, rms_db, hop


def track(x: np.ndarray, sr: int, fmin: float = 65.0, fmax: float = 1100.0):
    """Pitch track with voicing decisions and cleanup. Returns dict of per-frame arrays."""
    times, f0, aper, rms_db, hop = yin(x, sr, fmin, fmax)
    voiced = np.isfinite(f0)
    midi = np.where(voiced, hz_to_midi(np.where(voiced, f0, 1.0)), np.nan)
    # drop voiced islands shorter than 40 ms (usually consonant noise)
    min_len = int(0.04 * sr / hop)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(np.int8), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < min_len:
            voiced[a:b] = False
    midi[~voiced] = np.nan
    return {"times": times, "midi": midi, "voiced": voiced, "hop": hop, "aper": aper}


# ------------------------------------------------------------------ key detection

def chroma(x: np.ndarray, sr: int) -> np.ndarray:
    """12-bin pitch-class energy profile of a (channels, n) or mono signal."""
    mono = np.mean(x, axis=0) if x.ndim == 2 else x
    dec = max(1, sr // 11025)  # 55-2000 Hz needs nothing above ~5.5 kHz
    mono = signal.resample_poly(mono, 1, dec)
    fs = sr / dec
    f, _, z = signal.stft(mono, fs, nperseg=8192, noverlap=6144)  # ~1.3 Hz bins: resolves low notes
    mag = np.abs(z)
    # keep tonal energy only (median-filter harmonic/percussive split): drums smear chroma
    harm = median_filter(mag, size=(1, 9), mode="nearest")
    perc = median_filter(mag, size=(9, 1), mode="nearest")
    mag = np.where(harm >= perc, mag, 0.0)
    keep = (f >= 55.0) & (f <= 2000.0)
    m = hz_to_midi(f[keep])
    pcs = np.round(m).astype(int) % 12
    centre = np.clip(1.0 - 2.0 * np.abs(m - np.round(m)), 0.0, 1.0)  # bins between notes count less
    prof = np.zeros(12)
    np.add.at(prof, pcs, mag[keep].sum(axis=1) * centre)
    return prof / (prof.sum() + 1e-12)


def pitch_class_histogram(midi: np.ndarray) -> np.ndarray:
    m = midi[np.isfinite(midi)]
    h = np.bincount(np.round(m).astype(int) % 12, minlength=12).astype(float) if len(m) else np.zeros(12)
    return h / (h.sum() + 1e-12)


def detect_key(profile: np.ndarray, melody: np.ndarray | None = None) -> tuple[int, str, float]:
    """Best matching (tonic pitch class, 'major'|'minor', confidence).

    `profile` is the beat's chroma (the harmony - most reliable). `melody` is the singer's
    pitch-class histogram: it votes through Krumhansl correlation and through how much of the
    melody falls inside each candidate scale. Confidence is the winner's beat correlation.
    """
    best = (0, "major", -2.0, -9.0)
    for tonic in range(12):
        for mode, prof in (("major", _KK_MAJOR), ("minor", _KK_MINOR)):
            r = float(np.corrcoef(profile, np.roll(prof, tonic))[0, 1])
            score = r
            if melody is not None and melody.sum() > 0:
                in_scale = melody[[(tonic + i) % 12 for i in SCALES[mode]]].sum()
                score += 0.15 * float(np.corrcoef(melody, np.roll(prof, tonic))[0, 1]) + 1.0 * in_scale
            if score > best[3]:
                best = (tonic, mode, r, score)
    return best[:3]


def parse_key(text: str) -> tuple[int, str]:
    """'F# minor', 'Bbm', 'c major', 'A min', 'Eb' -> (pitch class, scale)."""
    t = text.strip()
    m = re.match(r"^([A-Ga-g])([#b]?)\s*(.*)$", t)
    if not m:
        raise ValueError(f"can't read key '{text}' (try e.g. 'F# minor' or 'Bb major')")
    name = m.group(1).upper() + m.group(2)
    name = _FLATS.get(name, name)
    rest = m.group(3).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {"": "major", "maj": "major", "m": "minor", "min": "minor", "minor": "minor", "major": "major"}
    scale = aliases.get(rest, rest)
    if scale not in SCALES:
        raise ValueError(f"unknown scale '{rest}'. choose from: {', '.join(SCALES)}")
    return NOTE_NAMES.index(name), scale


def key_name(tonic: int, scale: str) -> str:
    return f"{NOTE_NAMES[tonic]} {scale}"


# ------------------------------------------------------------------ correction curve

def _note_path(m: np.ndarray, allowed: np.ndarray, sigma: float, switch_cost: float):
    """Viterbi over the scale notes near a phrase. Returns (candidate notes, index per frame)."""
    lo, hi = int(np.floor(np.min(m))) - 2, int(np.ceil(np.max(m))) + 2
    notes = np.array([k for k in range(lo, hi + 1) if k % 12 in allowed], dtype=float)
    em = (m[:, None] - notes[None, :]) ** 2 / (2 * sigma ** 2)
    K = len(notes)
    cost = em[0].copy()
    back = np.zeros((len(m), K), dtype=np.int32)
    for i in range(1, len(m)):
        best_prev = int(np.argmin(cost))
        stay = cost
        switch = cost[best_prev] + switch_cost
        use_switch = switch < stay
        back[i] = np.where(use_switch, best_prev, np.arange(K))
        cost = np.where(use_switch, switch, stay) + em[i]
    path = np.empty(len(m), dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for i in range(len(m) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return notes, path


def target_curve(trk: dict, sr: int, tonic: int, scale: str, retune_ms: float, humanize: float,
                 amount: float = 1.0, hysteresis: float = 0.2) -> np.ndarray:
    """Corrected pitch (MIDI) per frame; NaN where unvoiced."""
    midi, voiced = trk["midi"], trk["voiced"]
    hop_s = trk["hop"] / sr
    allowed = np.array(sorted({(tonic + i) % 12 for i in SCALES[scale]}))
    n = len(midi)

    # slow pitch (note centre) vs. fast detail (vibrato, scoops) per voiced segment
    b, a = signal.butter(2, 4.0, fs=1.0 / hop_s)
    smooth = np.full(n, np.nan)
    v = voiced.astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], v, [0]])))
    for s0, s1 in zip(edges[::2], edges[1::2]):
        seg = midi[s0:s1]
        smooth[s0:s1] = signal.filtfilt(b, a, seg, padlen=min(len(seg) - 1, 9)) if len(seg) > 10 else np.median(seg)

    # note decisions: per voiced phrase, the most likely sequence of scale notes (Viterbi), using the
    # whole phrase - so a scoop or a fall at the edge of a note can't choose the wrong note
    target = np.full(n, np.nan)
    since = np.zeros(n)
    for s0, s1 in zip(edges[::2], edges[1::2]):
        notes, path = _note_path(smooth[s0:s1], allowed, sigma=0.5, switch_cost=8.0 + 4.0 * hysteresis)
        seg_target = notes[path]
        target[s0:s1] = seg_target
        onset = 0
        for j in range(s1 - s0):
            if j > 0 and seg_target[j] != seg_target[j - 1]:
                onset = j
            since[s0 + j] = (j - onset) * hop_s

    tau = retune_ms / 1000.0
    decay = np.exp(-since / tau) if tau > 0 else np.zeros(n)
    dev_slow = smooth - target           # how far off the note centre the singer is
    detail = midi - smooth               # vibrato / micro-movement
    # retune: the note centre is pulled to the target with a time constant of `retune_ms`
    # humanize: vibrato and expression on held notes survive (0 = robotic, 1 = all kept)
    out = target + dev_slow * decay + detail * (decay + (1.0 - decay) * humanize)
    out = midi + amount * (out - midi)
    return np.where(voiced, out, np.nan)


# ------------------------------------------------------------------ PSOLA resynthesis

def _pitch_marks(x: np.ndarray, sr: int, f0_frames: np.ndarray, hop: int):
    """Analysis marks: one per pitch period on voiced audio (aligned to waveform peaks),
    every ~5 ms on unvoiced audio. Returns (marks, half-grain lengths)."""
    n = len(x)
    lp = signal.sosfiltfilt(signal.butter(2, 1000.0, fs=sr, output="sos"), x)
    uv_period = int(sr * 0.005)
    marks, periods = [], []
    pos = 0
    nfr = len(f0_frames)
    while pos < n:
        fi = min(nfr - 1, pos // hop)
        f = f0_frames[fi]
        if np.isfinite(f) and f > 0:
            P = int(round(sr / f))
            if marks and periods[-1] != uv_period:
                # snap to the strongest peak near where the next period should start
                w = max(1, P // 4)
                lo, hi = max(0, pos - w), min(n, pos + w + 1)
                pos = lo + int(np.argmax(lp[lo:hi]))
            else:
                lo, hi = pos, min(n, pos + P)
                pos = lo + int(np.argmax(lp[lo:hi])) if hi > lo else pos
        else:
            P = uv_period
        if marks and pos <= marks[-1]:
            pos = marks[-1] + 1
        marks.append(pos)
        periods.append(P)
        pos += P
    return np.asarray(marks), np.asarray(periods)


def _cubic(xp: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Catmull-Rom interpolation of xp at fractional positions (exact at integer positions)."""
    i = np.floor(pos).astype(np.int64)
    mu = pos - i
    p0, p1, p2, p3 = xp[i - 1], xp[i], xp[i + 1], xp[i + 2]
    return p1 + 0.5 * mu * (p2 - p0 + mu * (2 * p0 - 5 * p1 + 4 * p2 - p3 + mu * (3 * (p1 - p2) + p3 - p0)))


def psola(x: np.ndarray, sr: int, f0_frames: np.ndarray, ratio_frames: np.ndarray, hop: int) -> np.ndarray:
    """Time-domain PSOLA pitch shift of a mono signal by a per-frame ratio (1 = unchanged).

    Synthesis grains are spaced by the singer's *exact* (fractional) period divided by the ratio
    and placed with sub-sample precision, so the new pitch is exact even for high voices where a
    one-sample rounding of the period would be tens of cents.
    """
    n = len(x)
    marks, periods = _pitch_marks(x, sr, f0_frames, hop)
    maxp = int(periods.max()) + 4
    xp = np.pad(x, (maxp, maxp))
    y = np.zeros(n + 2 * maxp)
    wsum = np.zeros(n + 2 * maxp)
    nfr = len(ratio_frames)
    ts = float(marks[0])
    k = 0
    while ts < n:
        # nearest analysis mark to the synthesis time
        while k + 1 < len(marks) and abs(marks[k + 1] - ts) <= abs(marks[k] - ts):
            k += 1
        m, P = int(marks[k]), int(periods[k])
        t0 = int(np.floor(ts))
        frac = ts - t0
        j = np.arange(t0 - P + 1, t0 + P + 1)          # output samples covered by this grain
        u = j - ts                                      # position relative to the grain centre
        w = 0.5 * (1.0 + np.cos(np.pi * u / P))         # continuous Hann, centred on ts
        if frac == 0.0:
            g = xp[m - P + 1 + maxp: m + P + 1 + maxp]
        else:
            g = _cubic(xp, m + u + maxp)
        y[j + maxp] += g * w
        wsum[j + maxp] += w
        fi = min(nfr - 1, max(0, t0) // hop)
        r = ratio_frames[fi]
        if abs(r - 1.0) < 1e-4 and k + 1 < len(marks):
            ts = float(marks[k + 1])  # nothing to shift: lock onto the analysis marks (bit-transparent)
            k += 1
        else:
            f = f0_frames[fi]
            period = sr / f if np.isfinite(f) and f > 0 else float(P)
            ts += period / r
    y = y[maxp: maxp + n]
    wsum = wsum[maxp: maxp + n]
    # grain coverage is always >= ~0.7 inside the signal (|shift| <= 3 semitones); only the very
    # edges fall below that, and there the original audio is used
    return np.where(wsum >= 0.5, y / np.maximum(wsum, 1e-9), x)


def autotune(x: np.ndarray, sr: int, tonic: int, scale: str, retune_ms: float, humanize: float, amount: float,
             trk: dict | None = None) -> tuple[np.ndarray, dict]:
    """Pitch-correct a (channels, n) vocal. Returns (tuned, stats)."""
    mono = np.mean(x, axis=0)
    trk = trk or track(mono, sr)
    out_midi = target_curve(trk, sr, tonic, scale, retune_ms, humanize, amount)
    shift = np.where(trk["voiced"], out_midi - trk["midi"], 0.0)
    shift = np.clip(np.nan_to_num(shift), -3.0, 3.0)
    ratio = 2.0 ** (shift / 12.0)
    f0 = np.where(trk["voiced"], midi_to_hz(np.nan_to_num(trk["midi"], nan=60.0)), np.nan)
    tuned = np.vstack([psola(ch, sr, f0, ratio, trk["hop"]) for ch in x])
    v = trk["voiced"]
    before = trk["midi"][v] - np.round(trk["midi"][v])
    after = out_midi[v] - np.round(out_midi[v])
    stats = {
        "voiced_percent": round(float(100.0 * v.mean()), 1),
        "avg_correction_cents": round(float(np.mean(np.abs(shift[v])) * 100), 1) if v.any() else 0.0,
        "avg_off_pitch_cents_before": round(float(np.mean(np.abs(before)) * 100), 1) if v.any() else 0.0,
        "avg_off_pitch_cents_after": round(float(np.mean(np.abs(after)) * 100), 1) if v.any() else 0.0,
    }
    return tuned, stats

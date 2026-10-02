"""Objective benchmark for the pitch corrector.

    python tools/bench_tune.py

Synthetic singers with known ground truth are rendered twice: as sung (out of tune, scoops,
drift, vibrato, breath) and *ideally tuned* (same voice, same breath noise, perfect pitch). The
corrector's output is compared against both:

  track_gross_%   frames where the pitch tracker is off by > 50 cents (octave errors etc.)
  track_med_c     median tracker error on voiced frames (cents)
  note_err_c      mean |cents| between each tuned note's centre and its scale note
  note_worst_c    worst note
  breath_err_db   harmonic-to-noise ratio of output minus ideal on steady notes; > 0 means the
                  breath was turned into buzz (classic PSOLA artifact), 0 is perfect
  breath_in_db    same for the untouched input (vibrato/scoops smear harmonics: not 0 either)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studiomix.dsp import pitch  # noqa: E402

SR = 48000
A_MINOR = [9, 11, 0, 2, 4, 5, 7]
VOWELS = {"a": (800, 1150, 2900), "e": (400, 2000, 2600), "o": (450, 800, 2830), "i": (300, 2300, 3000)}


def singer(base_midi: float, seconds: float, seed: int, breath: float, retune_target: bool):
    """Render (audio, true_f0_per_sample, note_table). If retune_target, the pitch is the perfect
    note centre: no detune, no scoop, no drift, no vibrato (the ideal hard-tune result)."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    f0 = np.zeros(n)
    amp = np.zeros(n)
    vowel_track = []
    notes = []
    t = 0.3
    allowed = [m for m in range(int(base_midi) - 5, int(base_midi) + 9) if m % 12 in A_MINOR]
    while t < seconds - 0.8:
        dur = rng.uniform(0.35, 0.9)
        a, b = int(t * SR), int((t + dur) * SR)
        note = allowed[rng.integers(len(allowed))]
        tt = np.arange(b - a) / SR
        # always draw the same random numbers so the sung and ideal renders are the same song
        detune = rng.uniform(-0.40, 0.40)  # beyond ~45 cents the intended note is ambiguous
        has_scoop, scoop_depth = rng.random() < 0.6, rng.uniform(0.3, 1.0)
        drift_amt, vib_rate = rng.uniform(-0.15, 0.15), rng.uniform(5, 6.5)
        if retune_target:
            m = np.full(b - a, float(note))
        else:
            scoop = -scoop_depth * np.exp(-tt / 0.06) if has_scoop else 0.0
            drift = drift_amt * tt / dur
            vib = 0.35 * np.sin(2 * np.pi * vib_rate * tt) * np.clip((tt - 0.25) / 0.2, 0, 1)
            m = note + detune + scoop + drift + vib
        f0[a:b] = pitch.midi_to_hz(m)
        amp[a:b] = np.minimum(1, tt / 0.02) * np.minimum(1, (dur - tt) / 0.04)
        vowel_track.append((a, b, "aeoi"[rng.integers(4)]))
        notes.append((a, b, note))
        t += dur + rng.uniform(0.08, 0.3)

    # glottal pulse train (band-limited by 2x oversampled saw) through formants, plus breath noise
    phase = 2 * np.pi * np.cumsum(f0) / SR
    src = signal.sawtooth(phase, 0.05) * amp
    out = np.zeros(n)
    for a, b, vw in vowel_track:
        seg = src[max(0, a - 200):b]
        y = np.zeros(len(seg))
        for fc, bw in zip(VOWELS[vw], (90, 110, 150)):
            bb, aa = signal.iirpeak(fc, fc / bw, fs=SR)
            y += signal.lfilter(bb, aa, seg)
        out[max(0, a - 200):b] += y
    # unvoiced consonants ("s", "sh", "h") in some gaps: must NOT be detected as pitched
    cons_rng = np.random.default_rng(seed + 2000)
    cons = np.zeros(n)
    for (a0, b0, _), (a1, _, _) in zip(notes[:-1], notes[1:]):
        if a1 - b0 > int(0.06 * SR) and cons_rng.random() < 0.7:
            m_ = min(int(0.07 * SR), a1 - b0)
            band = [(4000, 10000), (2000, 6000), (300, 6000)][cons_rng.integers(3)]
            burst = signal.sosfilt(signal.butter(2, band, "bandpass", fs=SR, output="sos"),
                                   cons_rng.standard_normal(m_)) * np.hanning(m_)
            cons[b0:b0 + m_] += burst * 0.15
    noise_rng = np.random.default_rng(seed + 1000)  # same breath noise in sung and ideal render
    nz = signal.sosfilt(signal.butter(2, [250, 10000], "bandpass", fs=SR, output="sos"), noise_rng.standard_normal(n))
    out = out * 0.3  # fixed gain (not peak-normalised) so sung and ideal renders match in level
    out += breath * nz * amp * 0.5 + cons
    return out, f0, notes


def lsd(a: np.ndarray, b: np.ndarray, voiced: np.ndarray) -> float:
    """Log-spectral distance (dB) over voiced frames, 80 Hz - 8 kHz, 1/6-octave smoothed."""
    f, _, A = signal.stft(a, SR, nperseg=2048, noverlap=1536)
    _, _, B = signal.stft(b, SR, nperseg=2048, noverlap=1536)
    hop = 512
    idx = np.clip(np.arange(A.shape[1]) * hop, 0, len(voiced) - 1)
    fr = voiced[idx]
    band = (f > 80) & (f < 8000)
    pa = np.abs(A[band][:, fr]) ** 2 + 1e-10
    pb = np.abs(B[band][:, fr]) ** 2 + 1e-10
    k = 5
    ker = np.ones(k) / k
    pa = signal.lfilter(ker, 1, pa, axis=0)
    pb = signal.lfilter(ker, 1, pb, axis=0)
    d = 10 * np.log10(pa / pb)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=0))))


def roughness(x: np.ndarray, voiced: np.ndarray) -> float:
    """Mean YIN aperiodicity (full band) over the steady voiced frames: grain/phase artifacts
    and jitter raise it. Compare against the ideal render of the same voice."""
    _, cv, _, _ = pitch._cmnd_candidates(x, SR, 65.0, 1100.0, 240)
    best = np.min(cv, axis=1)
    idx = np.minimum(np.arange(len(best)) * 240, len(voiced) - 1)
    v = voiced[idx]
    return float(np.mean(np.minimum(best[v], 1.0)))


def breath_error(out: np.ndarray, ideal: np.ndarray, notes) -> float:
    """dB difference in harmonic-to-noise ratio between output and ideal on steady notes.

    Energy within +-15% of f0 around each harmonic (1-6 kHz region, where breath lives) vs. the
    energy between harmonics. Classic PSOLA periodises breath, which raises HNR (positive error).
    """
    errs = []
    for a, b, note in notes:
        a2, b2 = a + int(0.3 * (b - a)), b - int(0.15 * (b - a))
        if b2 - a2 < 4096:
            continue
        f0 = float(pitch.midi_to_hz(note))
        hn = []
        for x in (out, ideal):
            f, p = signal.welch(x[a2:b2], SR, nperseg=4096)
            sel = (f > 1000) & (f < 6000)
            k = f[sel] / f0
            d = np.abs(k - np.round(k))
            hn.append(10 * np.log10(p[sel][d < 0.15].sum() / (p[sel][d > 0.3].sum() + 1e-20) + 1e-20))
        errs.append(hn[0] - hn[1])
    return float(np.mean(errs)) if errs else 0.0


def run_case(name: str, base_midi: float, breath: float, seed: int) -> dict:
    sung, f0_true, notes = singer(base_midi, 12.0, seed, breath, False)
    ideal, _, _ = singer(base_midi, 12.0, seed, breath, True)

    trk = pitch.track(sung, SR)
    frame_idx = np.minimum((trk["times"] * SR).astype(int), len(f0_true) - 1)
    truth = f0_true[frame_idx]
    tv = truth > 0
    est = trk["midi"]
    both = tv & trk["voiced"]
    err = np.abs(est[both] - pitch.hz_to_midi(truth[both])) * 100
    gross = 100 * np.mean(err > 50) if len(err) else 100.0
    # consonant frames (no pitch in the truth, but audible) wrongly reported as pitched
    frame_rms = np.array([np.sqrt(np.mean(sung[max(0, i - 240):i + 240] ** 2)) for i in frame_idx])
    audible_unvoiced = (~tv) & (frame_rms > 1e-3)
    false_v = 100 * np.mean(trk["voiced"][audible_unvoiced]) if audible_unvoiced.any() else 0.0
    missed = 100 * np.mean(~trk["voiced"][tv])

    t0 = time.time()
    out, _ = pitch.autotune(sung[None, :], SR, 9, "minor", retune_ms=0, humanize=0, amount=1.0)
    dt = time.time() - t0
    out = out[0]

    ot = pitch.track(out, SR)
    errs = []
    for a, b, note in notes:
        # judge the steady middle of each note
        fa, fb = int((a + 0.35 * (b - a)) / ot["hop"]), int((b - 0.15 * (b - a)) / ot["hop"])
        seg = ot["midi"][fa:fb]
        seg = seg[np.isfinite(seg)]
        if len(seg) > 5:
            errs.append(abs(np.median(seg) - note) * 100)
    # steady part of every note (skip onsets/ends where even the ideal render is not periodic)
    steady = np.zeros(len(f0_true), dtype=bool)
    for a, b, _ in notes:
        steady[a + int(0.1 * (b - a)) + 2400: b - 2400] = True
    return {
        "case": name,
        "track_gross_%": round(gross, 2),
        "track_miss_%": round(missed, 1),
        "false_voice_%": round(false_v, 1),
        "track_med_c": round(float(np.median(err)), 1) if len(err) else None,
        "note_err_c": round(float(np.mean(errs)), 2) if errs else 999.0,
        "note_worst_c": round(float(np.max(errs)), 1) if errs else 999.0,
        "notes_ok_%": round(100 * float(np.mean(np.array(errs) < 10)) * len(errs) / len(notes), 1) if errs else 0.0,
        "breath_err_db": round(breath_error(out, ideal, notes), 2),
        "breath_in_db": round(breath_error(sung, ideal, notes), 2),
        "sec": round(dt, 2),
    }


CASES = [
    ("bass  (A2-ish)", 45.0, 0.05, 1),
    ("tenor (A3-ish)", 57.0, 0.05, 2),
    ("alto  (E4-ish)", 64.0, 0.08, 3),
    ("sopr. (C5-ish)", 72.0, 0.08, 4),
    ("high  (E5-C6)", 76.0, 0.05, 5),
    ("breathy tenor", 57.0, 0.17, 6),
    ("whisper-ish alto", 64.0, 0.28, 7),
]


def main() -> None:
    rows = [run_case(*c) for c in CASES]
    keys = list(rows[0])
    print("  ".join(f"{k:>14s}" for k in keys))
    for r in rows:
        print("  ".join(f"{str(r[k]):>14s}" for k in keys))
    for k in ("track_gross_%", "track_miss_%", "false_voice_%", "notes_ok_%", "note_err_c", "breath_err_db"):
        print(f"mean {k}: {np.mean([r[k] for r in rows]):.2f}")


if __name__ == "__main__":
    main()

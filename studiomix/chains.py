"""The three processing chains: lead vocal, instrumental and master bus."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .dsp import analysis, dynamics, effects, filters
from .presets import Preset

EPS = 1e-12


def _rms_db(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    p = np.mean(x * x, axis=0)
    if mask is not None and mask.any():
        p = p[mask]
    return float(10.0 * np.log10(np.mean(p) + EPS))


def _gain(x: np.ndarray, db: float) -> np.ndarray:
    return x * 10.0 ** (db / 20.0)


def _tonal_correction(x, sr, target_db_fn, strength, max_db, f_lo, f_hi, smooth_oct, mask=None, ntaps=4097,
                      max_boost_db=None):
    """Measure the long-term spectrum and apply a gentle linear-phase corrective EQ."""
    mono = np.mean(x, axis=0)
    if mask is not None and mask.sum() > sr:
        mono = mono[mask]
    freqs, meas = filters.ltas_db(mono, sr)
    meas = filters.fractional_octave_smooth(freqs, meas, 1 / 3)
    target = target_db_fn(freqs)
    curve = filters.correction_curve(freqs, meas, target, strength, max_db, f_lo, f_hi, smooth_oct, max_boost_db)
    h = filters.match_eq_fir(freqs, curve, sr, ntaps)
    # report the correction at a few landmark frequencies
    marks = {f"{int(f)}Hz": round(float(np.interp(f, freqs, curve)), 2)
             for f in (60, 150, 400, 1000, 3000, 6000, 12000) if f < sr / 2}
    return filters.apply_fir(x, h), marks


# ------------------------------------------------------------------ vocal

def vocal_chain(v: np.ndarray, sr: int, p: Preset, log: dict) -> tuple[np.ndarray, np.ndarray]:
    """Clean, tone and control the lead vocal. Returns (dry processed vocal, activity mask)."""
    # stereo vocal files are almost always a mono performance: fold to mono unless truly stereo
    if v.shape[0] >= 2:
        corr = analysis.stereo_correlation(v)
        log["input_stereo_correlation"] = round(corr, 3)
        if corr > 0.8:
            v = np.mean(v[:2], axis=0, keepdims=True)
        else:
            v = v[:2]

    active, _ = analysis.activity_mask(v, sr)
    log["active_seconds"] = round(active.sum() / sr, 1)
    if active.sum() < sr * 0.5:
        raise ValueError("the vocal file looks silent - check the file")

    # 1. normalise the performance to a fixed working level so every threshold below is meaningful
    v = _gain(v, -20.0 - _rms_db(v, active))

    # 2. rumble / plosive / handling-noise removal
    v = filters.highpass(v, sr, p.vocal_hpf_hz, order=4)

    # 3. soft gate (downward expander) for room noise between phrases
    frames = analysis.frame_rms_db(v, sr, 50.0)
    floor = float(np.percentile(frames, 10))
    if floor < -45.0:
        v, _ = dynamics.expand(v, sr, threshold_db=floor + 10.0, ratio=2.0, range_db=10.0)
        log["noise_floor_db"] = round(floor, 1)

    # 4. adaptive corrective EQ toward a balanced lead-vocal spectrum (cut generously, boost carefully)
    if p.vocal_auto_eq_strength > 0:
        v, marks = _tonal_correction(
            v, sr, lambda f: filters.slope_target(f, -3.5), p.vocal_auto_eq_strength, 6.0,
            150.0, 10000.0, 1 / 2, mask=active, max_boost_db=3.0,
        )
        log["auto_eq_db"] = marks

    # 5. subtractive + character EQ
    v = filters.eq(v, sr, "peak", 300.0, p.vocal_mud_cut_db, 1.0)
    v = filters.eq(v, sr, "peak", 800.0, p.vocal_boxy_cut_db, 1.4)
    v = filters.eq(v, sr, "peak", 3500.0, p.vocal_presence_db, 0.9)
    v = filters.eq(v, sr, "highshelf", 12000.0, p.vocal_air_db, 0.7)

    # 6. two-stage compression: fast peak catcher into a slow, smooth leveler
    amt = p.vocal_comp_amount
    if amt > 0:
        v, gr1 = dynamics.compress(v, sr, -12.0, 1 + 3.0 * amt, 3.0, 60.0, knee_db=6.0)
        v, gr2 = dynamics.compress(v, sr, -24.0, 1 + 1.5 * amt, 20.0, 220.0, knee_db=10.0)
        log["compression_avg_gr_db"] = round(float(np.mean((gr1 + gr2)[active])), 2)
        log["compression_max_gr_db"] = round(float(np.min(gr1 + gr2)), 2)

    # 7. de-ess after compression + brightening (both exaggerate sibilance)
    if p.vocal_deess_db > 0:
        v, gr = dynamics.deess(v, sr, active, max_reduction_db=p.vocal_deess_db)
        log["deess_max_gr_db"] = round(float(np.min(gr)), 2)

    # 8. harmonic warmth
    v = effects.saturate(v, drive_db=8.0, mix=p.vocal_saturation)
    return v, active


def vocal_rider(v: np.ndarray, inst: np.ndarray, sr: int, active: np.ndarray, max_db: float, log: dict) -> np.ndarray:
    """Automatic fader riding: keep the vocal at a constant level *relative to the beat*.

    When the chorus gets louder the vocal comes up with it; in sparse sections it comes down.
    """
    if max_db <= 0:
        return v
    hop = int(0.1 * sr)
    win = int(0.4 * sr)
    n = v.shape[-1] // hop
    if n < 10:
        return v

    def st(x):
        p = np.mean(x * x, axis=0)
        c = np.concatenate([[0.0], np.cumsum(p)])
        idx = np.arange(n) * hop
        lo = np.maximum(idx - win // 2, 0)
        hi = np.minimum(idx + win // 2, len(p))
        return 10 * np.log10((c[hi] - c[lo]) / np.maximum(hi - lo, 1) + EPS)

    inst_bal = filters.bandpass(inst, sr, 200.0, 6000.0)  # what actually masks a vocal
    rel = st(v) - st(inst_bal)
    act = active[np.arange(n) * hop]
    if act.sum() < 5:
        return v
    target = np.median(rel[act])
    corr = np.clip(target - rel, -max_db, max_db) * 0.6
    # hold the last value through silences, then zero-phase smooth (offline: no lag)
    corr = np.where(act, corr, np.nan)
    idx = np.where(~np.isnan(corr), np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    corr = np.nan_to_num(corr[idx])
    b, a = signal.butter(1, 1.5, fs=10.0)  # ~1.5 Hz smoothing on a 10 Hz control signal
    corr = signal.filtfilt(b, a, corr)
    gain = np.interp(np.arange(v.shape[-1]), np.arange(n) * hop + hop // 2, corr)
    log["rider_range_db"] = [round(float(corr.min()), 2), round(float(corr.max()), 2)]
    return v * 10.0 ** (gain / 20.0)[None, :]


def vocal_effects(v: np.ndarray, sr: int, p: Preset, tempo: float | None, log: dict) -> np.ndarray:
    """Reverb + delay sends, ducked by the dry vocal so the words stay clear."""
    dry = effects.to_stereo(v)
    wet = np.zeros_like(dry)
    if p.vocal_reverb > 0:
        wet += p.vocal_reverb * effects.reverb(v, sr, room_size=p.vocal_reverb_size, damping=0.5)
    if p.vocal_delay > 0:
        beat = 60.0 / tempo if tempo else 0.5
        delay_s = beat * 4 * p.vocal_delay_note
        while delay_s > 0.7:
            delay_s /= 2
        log["delay_ms"] = round(delay_s * 1000, 1)
        wet += p.vocal_delay * effects.ping_pong_delay(v, sr, delay_s, feedback=0.35)
    if not np.any(wet):
        return dry
    wet, _ = dynamics.compress(wet, sr, -26.0, 3.0, 10.0, 250.0, knee_db=6.0, sidechain=dry, max_gr_db=6.0)
    return dry + wet


# ------------------------------------------------------------------ instrumental

def instrumental_chain(inst: np.ndarray, vocal_dry: np.ndarray, sr: int, active: np.ndarray, p: Preset,
                       log: dict) -> np.ndarray:
    inst = effects.to_stereo(inst)
    inst = filters.highpass(inst, sr, p.inst_hpf_hz, order=4)
    if p.inst_carve_db > 0:
        # dynamic EQ: dip the vocal's presence range in the beat only while the vocal is singing
        band = filters.bandpass(inst, sr, 1500.0, 5000.0, order=2)
        key = filters.bandpass(vocal_dry, sr, 1000.0, 5000.0, order=2)
        det = dynamics.detector_db(key, sr, "rms", 10.0)
        thr = float(np.percentile(det[active], 10)) if active.any() else -40.0
        gr = dynamics._compressor_gain(det, thr, 3.0, 6.0, dynamics.coef(10.0, sr), dynamics.coef(180.0, sr),
                                       float(p.inst_carve_db))
        g = 10.0 ** (gr / 20.0)
        inst = inst + (g - 1.0)[None, :] * band
        log["carve_avg_db"] = round(float(np.mean(gr[active])) if active.any() else 0.0, 2)
    return inst


# ------------------------------------------------------------------ master

def master_chain(mix: np.ndarray, sr: int, p: Preset, log: dict, reference: np.ndarray | None = None) -> np.ndarray:
    """Tonal balance, multiband + glue compression and stereo imaging (pre-limiter)."""
    x = mix

    # 1. tonal balance: toward a reference track if given, else a commercial-mix spectral tilt
    if reference is not None:
        rf, rdb = filters.ltas_db(np.mean(reference, axis=0), sr)
        rdb = filters.fractional_octave_smooth(rf, rdb, 1 / 3)
        x, marks = _tonal_correction(x, sr, lambda f: np.interp(f, rf, rdb), 0.7, 6.0, 30.0, 16000.0, 1 / 2)
        log["tonal_match"] = "reference"
    elif p.master_tonal_strength > 0:
        x, marks = _tonal_correction(x, sr, lambda f: filters.slope_target(f, p.master_tilt_db_oct),
                                     p.master_tonal_strength, 3.0, 60.0, 14000.0, 1.0)
        log["tonal_match"] = "target-tilt"
    else:
        marks = {}
    log["tonal_eq_db"] = marks

    # 2. multiband compression: tames boomy lows / harsh highs independently, keeps tone
    if p.master_mb_amount > 0:
        bands = filters.three_band_split(x, sr, 120.0, 4000.0)
        settings = [(0.8, 25.0, 160.0), (0.5, 12.0, 110.0), (0.8, 5.0, 70.0)]
        out = np.zeros_like(x)
        mb_log = []
        for band, (r, att, rel) in zip(bands, settings):
            det = dynamics.detector_db(band, sr, "rms", 10.0)
            thr = float(np.percentile(det, 85))
            y, gr = dynamics.compress(band, sr, thr, 1 + 2 * r * p.master_mb_amount, att, rel,
                                      knee_db=6.0, max_gr_db=4.0)
            y *= np.sqrt(np.mean(band ** 2) / (np.mean(y ** 2) + EPS))  # keep the band's average level
            out += y
            mb_log.append(round(float(np.mean(gr)), 2))
        x = out
        log["multiband_avg_gr_db"] = dict(zip(["low", "mid", "high"], mb_log))

    # 3. bus "glue" compression
    if p.master_glue_ratio > 1:
        det = dynamics.detector_db(x, sr, "rms", 10.0)
        thr = float(np.percentile(det, 90)) - 2.0
        x, gr = dynamics.compress(x, sr, thr, p.master_glue_ratio, 30.0, 150.0, knee_db=6.0, max_gr_db=4.0)
        log["glue_avg_gr_db"] = round(float(np.mean(gr)), 2)

    # 4. stereo image: mono bass, slightly wider top
    x = effects.stereo_image(x, sr, p.master_bass_mono_hz, p.master_width)
    return x


def finalize_loudness(x: np.ndarray, sr: int, target_lufs: float, ceiling_dbtp: float, clip_knee_db: float,
                      release_ms: float, log: dict) -> np.ndarray:
    """Drive into soft clipper + true-peak limiter until integrated loudness hits the target."""
    gain = target_lufs - analysis.integrated_lufs(x, sr)
    y, gr = x, np.zeros(x.shape[-1])
    for _ in range(6):
        y = _gain(x, gain)
        if clip_knee_db > 0:
            y = dynamics.soft_clip(y, ceiling_dbtp + clip_knee_db, knee_db=3.0)
        y, gr = dynamics.limit(y, sr, ceiling_dbtp, lookahead_ms=1.5, release_ms=release_ms)
        err = target_lufs - analysis.integrated_lufs(y, sr)
        if abs(err) < 0.1:
            break
        gain += err * 1.1  # limiting eats part of every boost; overshoot slightly to converge
    log["limiter_max_gr_db"] = round(float(np.min(gr)), 2)
    log["limiter_avg_gr_db"] = round(float(np.mean(gr)), 2)
    return y


def fade_edges(x: np.ndarray, sr: int, fade_in_ms: float = 5.0, fade_out_ms: float = 30.0,
               silence_db: float = -75.0) -> np.ndarray:
    """Trim trailing silence (after effect tails) and apply click-free micro fades."""
    env = np.max(np.abs(x), axis=0)
    loud = np.nonzero(env > 10 ** (silence_db / 20))[0]
    if len(loud):
        end = min(x.shape[-1], loud[-1] + int(0.25 * sr))
        x = x[:, :end].copy()
    n_in, n_out = int(sr * fade_in_ms / 1000), int(sr * fade_out_ms / 1000)
    if n_in:
        x[:, :n_in] *= np.linspace(0, 1, n_in) ** 2
    if n_out:
        x[:, -n_out:] *= np.linspace(1, 0, n_out) ** 2
    return x

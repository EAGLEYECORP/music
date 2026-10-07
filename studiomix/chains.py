"""The three processing chains: lead vocal, instrumental and master bus."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .dsp import analysis, dynamics, effects, filters, pitch
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

def vocal_chain(v: np.ndarray, sr: int, p: Preset, log: dict, key=None, label: str = "vocal",
                trk: dict | None = None, capture: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Clean, tune, tone and control a vocal. Returns (dry processed vocal, activity mask)."""
    # stereo vocal files are almost always a mono performance: fold to mono unless truly stereo
    if v.shape[0] >= 2:
        corr = analysis.stereo_correlation(v)
        log["input_stereo_correlation"] = round(corr, 3)
        if corr > 0.8:
            v = np.mean(v[:2], axis=0, keepdims=True)
        else:
            v = v[:2]

    if p.vocal_ai_dereverb:
        v = _ai_dereverb(v, sr, log)

    active, _ = analysis.activity_mask(v, sr)
    log["active_seconds"] = round(active.sum() / sr, 1)
    if active.sum() < sr * 0.5:
        raise ValueError(f"the {label} file looks silent - check the file")

    # 1. normalise the performance to a fixed working level so every threshold below is meaningful
    v = _gain(v, -20.0 - _rms_db(v, active))

    # 2. rumble / plosive / handling-noise removal
    v = filters.highpass(v, sr, p.vocal_hpf_hz, order=4)

    # 2b. background noise (phone / bedroom recordings): learned from the gaps between phrases
    if p.vocal_denoise:
        from .dsp.denoise import denoise

        v, nstats = denoise(v, sr, active)
        log["denoise"] = nstats

    # 3. pitch correction, on the clean raw voice (before compression/saturation colour it)
    if key is not None and p.tune_amount > 0:
        sections = isinstance(key, list)  # key per song section
        tonic, scale = (key, None) if sections else key
        v, stats = pitch.autotune(v, sr, tonic, scale, p.tune_retune_ms, p.tune_humanize, p.tune_amount,
                                   trk=trk, flex_cents=p.tune_flex_cents, capture=capture)
        log["autotune"] = {"key": "per section" if sections else pitch.key_name(*key),
                           "retune_ms": p.tune_retune_ms, "humanize": p.tune_humanize,
                           **({"flex_cents": p.tune_flex_cents} if p.tune_flex_cents > 0 else {}), **stats}

    # 3b. soft gate (downward expander) for room noise between phrases
    frames = analysis.frame_rms_db(v, sr, 50.0)
    frames = frames[frames > -100.0] if (frames > -100.0).sum() >= 10 else frames  # skip digital silence
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


def pan_phrases(v: np.ndarray, sr: int, active: np.ndarray, width: float, start_left: bool = True) -> np.ndarray:
    """Pan each ad-lib phrase alternately left/right (the classic hip-hop ad-lib spread)."""
    mono = np.mean(v, axis=0)
    if width <= 0:
        return np.vstack([mono, mono])
    edges = np.flatnonzero(np.diff(np.concatenate([[0], active.astype(np.int8), [0]])))
    starts = list(edges[::2])
    segments = []
    side = -1.0 if start_left else 1.0
    # each phrase holds its side until the next phrase starts (no swing back through centre in the tail)
    for i, a in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(mono)
        segments.append((0 if i == 0 else a, end, side * width))
        side = -side
    return effects.apply_pan(mono, effects.pan_curve(len(mono), sr, segments))


def section_mask(n: int, sr: int, ranges: list[tuple[float, float]] | None, fade_ms: float = 40.0) -> np.ndarray:
    """0..1 gain curve that is 1 inside the given (start_s, end_s) ranges (all 1s if none)."""
    if not ranges:
        return np.ones(n)
    m = np.zeros(n)
    for a, b in ranges:
        m[max(0, int(a * sr)):min(n, int(b * sr))] = 1.0
    from scipy.ndimage import uniform_filter1d

    # running average (linear time; a plain convolution would be O(n * k) on a full song)
    return np.clip(uniform_filter1d(m, max(1, int(sr * fade_ms / 1000)), mode="constant"), 0.0, 1.0)


def build_stack(lead: np.ndarray, sr: int, capture: dict, p: Preset, active: np.ndarray,
                ranges: list[tuple[float, float]] | None, log: dict) -> np.ndarray | None:
    """Doubles and harmonies made from the processed, tuned lead. Returns a dry stereo bus."""
    voices = []
    if p.doubles:
        for i, pan in enumerate((-0.8, 0.8)):
            voices.append(("double", effects.apply_pan(pitch.double(lead, sr, capture, seed=11 + i)[0], pan),
                           p.doubles_db))
    ivs = [s_.strip() for s_ in p.harmonies.split(",") if s_.strip()]
    if ivs and capture.get("key") is not None:
        for i, iv in enumerate(ivs):
            pan = (-0.55 if i % 2 == 0 else 0.55) if len(ivs) > 1 else 0.0
            h = pitch.harmony(lead, sr, capture, iv)[0]
            voices.append((f"harmony {iv}", effects.apply_pan(h, pan), p.harmony_db))
    elif ivs:
        log["harmony_note"] = "harmonies need auto-tune on (they follow the song key)"
    if not voices:
        return None
    mask = section_mask(lead.shape[-1], sr, ranges)
    on = active & (mask > 0.5)
    if on.sum() < sr * 0.2:
        on = active
    ref = np.sqrt(np.mean(np.mean(lead, axis=0)[on] ** 2)) + 1e-12
    bus = np.zeros((2, lead.shape[-1]))
    for name, v, level_db in voices:
        v = filters.highpass(v, sr, 160.0, order=2)  # stacks stay out of the lead's chest
        rms = np.sqrt(np.mean(np.mean(v, axis=0)[on] ** 2)) + 1e-12
        bus += v * (ref / rms) * 10 ** (level_db / 20)
    log["stack_voices"] = [n for n, _, _ in voices]
    if ranges:
        log["stack_sections_s"] = [[round(a, 1), round(b, 1)] for a, b in ranges]
    return bus * mask[None, :]


# ------------------------------------------------------------------ instrumental

def _ai_dereverb(v: np.ndarray, sr: int, log: dict) -> np.ndarray:
    """Remove the recording room's reverb (bedroom, bathroom) with the AI de-reverb model; the
    chain's own reverb is then the only space on the vocal. Skipped (and said) without the AI."""
    from .ai import separate as sep

    if not sep.available():
        log["ai_dereverb"] = "needs onnxruntime (pip install onnxruntime) - skipped"
        return v
    try:
        dry = sep.separate(v, sr, "dereverb", say=lambda m: None)["dry"]
    except Exception as e:
        log["ai_dereverb"] = f"skipped: {e}"
        return v
    log["ai_dereverb"] = "room echo removed (AI)"
    return np.mean(dry, axis=0, keepdims=True) if v.shape[0] == 1 else dry[: v.shape[0]]


def _rebalance_beat(inst: np.ndarray, sr: int, p: Preset, log: dict) -> np.ndarray:
    """Turn the 808 / the drums of the beat up or down. With the AI installed the beat is split
    into drums / bass / other and the real stems are moved; without it the 808 falls back to a
    low shelf at the same amount, and drums can't be moved (said in the log)."""
    from .ai import separate as sep

    if sep.available():
        try:
            st = sep.split_beat(inst, sr, say=lambda m: None)
            g = lambda db: 10 ** (db / 20)  # noqa: E731
            log["beat_split"] = {"bass_db": p.inst_bass_db, "drums_db": p.inst_drums_db}
            return st["other"] + st["bass"] * g(p.inst_bass_db) + st["drums"] * g(p.inst_drums_db)
        except Exception as e:  # no network for the first download, etc.
            log["beat_split_error"] = str(e)
    if p.inst_bass_db:
        inst = signal.sosfilt(filters.biquad_sos("lowshelf", 90.0, sr, 0.7071, p.inst_bass_db), inst, axis=-1)
        log["bass_shelf_db"] = p.inst_bass_db
    if p.inst_drums_db:
        log["drums_level"] = "needs the AI beat split (pip install onnxruntime) - skipped"
    return inst


def bass_harmonics(x: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Overtones of the sub-bass (mono): a phone speaker plays nothing below ~300 Hz, so an 808
    vanishes on it - but the ear rebuilds a low note from its overtones (the "missing
    fundamental"). Saturating the sub creates them; only the 150 Hz - 1.5 kHz part is added."""
    sub = filters.lowpass(np.mean(x, axis=0, keepdims=True), sr, 120.0, order=4)
    peak = float(np.max(np.abs(sub))) + 1e-12
    shaped = np.tanh(4.0 * sub / peak) + 0.6 * (np.abs(sub) / peak)   # odd + even harmonics
    h = filters.lowpass(filters.highpass(shaped, sr, 160.0, order=4), sr, 1500.0, order=2)
    rms_sub, rms_h = np.sqrt(np.mean(sub ** 2)) + 1e-12, np.sqrt(np.mean(h ** 2)) + 1e-12
    return (h * (0.5 * amount * rms_sub / rms_h))[0]


def instrumental_chain(inst: np.ndarray, vocal_dry: np.ndarray, sr: int, active: np.ndarray, p: Preset,
                       log: dict) -> np.ndarray:
    inst = effects.to_stereo(inst)
    inst = filters.highpass(inst, sr, p.inst_hpf_hz, order=4)
    if p.inst_bass_db or p.inst_drums_db:
        inst = _rebalance_beat(inst, sr, p, log)
    if p.inst_low_db:
        inst = signal.sosfilt(filters.biquad_sos("lowshelf", 90.0, sr, 0.7071, p.inst_low_db), inst, axis=-1)
        log["low_shelf_db"] = p.inst_low_db
    if p.bass_harmonics > 0:
        inst = inst + bass_harmonics(inst, sr, p.bass_harmonics)[None, :]
        log["bass_harmonics"] = p.bass_harmonics
    if p.inst_carve_db > 0:
        # dynamic EQ: dip the vocal's presence range in the beat only while the vocal is singing
        band = filters.bandpass(inst, sr, 1500.0, 5000.0, order=2)
        key = filters.bandpass(vocal_dry, sr, 1000.0, 5000.0, order=2)
        det = dynamics.detector_db(key, sr, "rms", 10.0)
        thr = float(np.percentile(det[active], 10)) if active.any() else -40.0
        gr = dynamics.compressor_gain(det, sr, thr, 3.0, 6.0, 10.0, 180.0, p.inst_carve_db)
        g = 10.0 ** (gr / 20.0)
        inst = inst + (g - 1.0)[None, :] * band
        log["carve_avg_db"] = round(float(np.mean(gr[active])) if active.any() else 0.0, 2)
    return inst


# ------------------------------------------------------------------ finished / rough mix

def diagnose_mix(x: np.ndarray, sr: int, src_path=None) -> dict:
    """What a mastering engineer checks first. Returns {'findings': [...], 'metrics': {...}}."""
    findings: list[str] = []
    mono = np.mean(x, axis=0)
    peak = np.max(np.abs(x))
    # clipping: runs of 3+ consecutive samples at (almost) the same full-scale value
    hot = np.abs(x) >= 0.999 * peak
    runs = 0
    if peak > 0.98:
        for ch in hot:
            d = np.diff(np.concatenate([[0], ch.astype(np.int8), [0]]))
            runs += int(np.sum((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) >= 3))
    if runs:
        findings.append(f"the bounce is clipped in {runs} places - re-export the mix ~3-6 dB quieter if you can "
                        "(clipping can't be fully undone; the master will hide it as well as possible)")
    if src_path is not None and str(src_path).lower().endswith((".mp3", ".m4a", ".aac", ".ogg", ".opus")):
        findings.append("the mix is a lossy file (mp3/aac): a WAV export of the mix will master cleaner")
    dc = float(np.max(np.abs(np.mean(x, axis=1))))
    if dc > 1e-3:
        findings.append(f"DC offset ({20 * np.log10(dc):.0f} dBFS) - removed")
    corr = analysis.stereo_correlation(x)
    lo = filters.lowpass(x, sr, 150.0, order=4)
    lo_corr = analysis.stereo_correlation(lo)
    if lo_corr < 0.5:
        findings.append(f"low end is partly out of phase between left and right (correlation {lo_corr:.2f}) - "
                        "bass would vanish on phones/mono - the bass is recovered and made mono")
    if corr < 0.2:
        findings.append(f"very wide / phasey mix (correlation {corr:.2f}) - check it in mono")
    # tonal balance against the commercial tilt, in four broad regions
    f, db = filters.ltas_db(mono, sr)
    db = filters.fractional_octave_smooth(f, db, 1.0)
    rel = db - filters.slope_target(f, -4.5)
    ref = np.median(rel[(f > 200) & (f < 5000)])

    def band(lo_, hi_):
        return float(np.mean(rel[(f >= lo_) & (f < hi_)]) - ref)

    tone = {"sub_lows_40_120": band(40, 120), "mud_200_500": band(200, 500),
            "harsh_2k_5k": band(2000, 5000), "air_8k_14k": band(8000, 14000)}
    if tone["sub_lows_40_120"] > 5:
        findings.append("low end is heavy/boomy - tamed by the tonal balance and the low-band compressor")
    elif tone["sub_lows_40_120"] < -5:
        findings.append("low end is thin - gently lifted")
    if tone["mud_200_500"] > 3:
        findings.append("muddy low-mids (200-500 Hz) - cleaned up")
    if tone["harsh_2k_5k"] > 3:
        findings.append("harsh upper-mids (2-5 kHz) - dynamic anti-harshness applied")
    if tone["air_8k_14k"] < -6:
        findings.append("dull top end - air added")
    head = np.argmax(np.max(np.abs(x), axis=0) > 10 ** (-60 / 20)) / sr
    if head > 1.0:
        findings.append(f"{head:.1f} s of silence before the music starts - trimmed to 0.2 s")
    lufs = analysis.integrated_lufs(x, sr)
    if not findings:
        findings.append("clean mix - no problems found; mastering only")
    return {"findings": findings, "metrics": {
        "integrated_lufs": round(lufs, 2), "sample_peak_dbfs": round(20 * np.log10(peak + EPS), 2),
        "stereo_correlation": round(corr, 3), "low_end_correlation": round(lo_corr, 3), "clipped_spots": runs,
        "tone_vs_commercial_db": {k: round(v, 1) for k, v in tone.items()}, "lead_in_silence_s": round(float(head), 2),
    }}


def repair_mix(x: np.ndarray, sr: int, p: Preset, diag: dict, vocal_lift_db: float, log: dict) -> np.ndarray:
    """Mix-level fixes before mastering. Vocal work happens on the Mid (centre) channel, where the
    lead vocal of a hip-hop/pop mix sits - the beat's stereo content is left alone."""
    x = x - np.mean(x, axis=1, keepdims=True)  # DC
    x = filters.highpass(x, sr, 25.0, order=4)  # subsonic rumble
    lead_in = diag["metrics"]["lead_in_silence_s"]
    if lead_in > 1.0:
        x = x[:, int((lead_in - 0.2) * sr):]
    mid = 0.5 * (x[0] + x[1])
    side = 0.5 * (x[0] - x[1])
    if diag["metrics"]["low_end_correlation"] < 0.0:
        # out-of-phase bass lives in the SIDE signal; folding the lows to mono (as the master
        # does) would delete it. Recover it: the low band of whichever of mid/side carries more
        # bass becomes the mono low end.
        mid_lo, mid_hi = filters.lr4_split(mid[None, :], sr, 150.0)
        side_lo, side_hi = filters.lr4_split(side[None, :], sr, 150.0)
        if np.mean(side_lo ** 2) > np.mean(mid_lo ** 2):
            mid = (side_lo + mid_hi)[0]
            side = side_hi[0]
            log["low_end_phase_fix"] = "recovered out-of-phase bass from the side channel"
    m = mid[None, :]
    active, _ = analysis.activity_mask(m, sr)
    # sibilance and harshness live on the vocal, i.e. in the centre
    m, gr_s = dynamics.deess(m, sr, active, max_reduction_db=4.0, sensitivity_db=4.0)
    band = filters.bandpass(m, sr, 2000.0, 5000.0, order=2)
    det = dynamics.detector_db(band, sr, "rms", 5.0)
    thr = float(np.percentile(det[active], 80)) if active.any() else float(np.percentile(det, 80))
    gr_h = dynamics.compressor_gain(det, sr, thr, 3.0, 6.0, 3.0, 80.0, 3.0)
    m = m + (10 ** (gr_h / 20) - 1.0)[None, :] * band
    log["centre_deess_max_db"] = round(float(np.min(gr_s)), 2)
    log["anti_harsh_max_db"] = round(float(np.min(gr_h)), 2)
    if vocal_lift_db:
        # presence + intelligibility in the centre only: brings the lead forward without
        # touching the beat's wide elements
        m = filters.eq(m, sr, "peak", 2800.0, vocal_lift_db, 0.8)
        m = filters.eq(m, sr, "peak", 250.0, -0.5 * vocal_lift_db, 1.0)
        log["vocal_lift_db"] = vocal_lift_db
    mid = m[0]
    return np.vstack([mid + side, mid - side])


# ------------------------------------------------------------------ master

def match_width(x: np.ndarray, sr: int, targets: dict, strength: float = 0.8, max_db: float = 6.0,
                passes: int = 3) -> dict:
    """Move each band's side (stereo) level toward a profile's side/mid ratio, in place.

    Gains are applied to the side signal with a linear-phase FIR (exact band gains, no phase
    fighting), re-measured after every pass. Below 120 Hz nothing changes - the low end stays mono -
    and a band that is nearly mono (side < -18 dB) is widened by at most 6 dB in total: boosting the
    side of mono material only amplifies leftovers and sounds phasey.
    """
    from .profiles import side_mid_db

    bands = []
    for key, target in targets.items():
        a, b = (float(v) for v in key.split("-"))
        if a >= 120 and b <= sr * 0.45:
            bands.append((key, a, b, float(target)))
    if not bands:
        return {}
    mid = 0.5 * (x[0] + x[1])
    side = 0.5 * (x[0] - x[1])
    total = {k: 0.0 for k, *_ in bands}
    start = {k: side_mid_db(x, sr, a, b) for k, a, b, _ in bands}
    for _ in range(passes):
        cur_x = np.vstack([mid + side, mid - side])
        pts_f, pts_g = [20.0, 112.0], [0.0, 0.0]
        moved = False
        for k, a, b, target in bands:
            cur = side_mid_db(cur_x, sr, a, b)
            g = (target - cur) * strength
            lim = 6.0 if start[k] < -18.0 else max_db + 3.0
            g = float(np.clip(total[k] + g, -(max_db + 3.0), lim) - total[k])
            if abs(g) >= 0.2:
                moved = True
            total[k] += g
            # flat across the band (short transitions at the edges), so the whole band gets it
            pts_f += [a * 1.06, b / 1.06]
            pts_g += [g, g]
        if not moved:
            break
        pts_f.append(sr / 2)
        pts_g.append(pts_g[-1])
        freqs = np.linspace(0, sr / 2, 4097)
        curve = np.interp(np.log2(np.maximum(freqs, 20.0)), np.log2(pts_f), pts_g)
        side = filters.apply_fir(side[None, :], filters.match_eq_fir(freqs, curve, sr, 2049))[0]
    x[0], x[1] = mid + side, mid - side
    return {k: round(v, 2) for k, v in total.items() if abs(v) >= 0.2}


def master_chain(mix: np.ndarray, sr: int, p: Preset, log: dict, reference: np.ndarray | None = None,
                 profile: dict | None = None) -> np.ndarray:
    """Tonal balance, multiband + glue compression and stereo imaging (pre-limiter)."""
    x = mix

    # 1. tonal balance: toward a learned profile, a reference track, or a commercial-mix tilt
    if profile is not None:
        g = np.array(profile["grid_hz"])
        c = np.array(profile["curve_db"])
        hi = min(16000.0, 0.95 * min(t["lossy_cutoff_hz"] for t in profile["tracks"]))
        x, marks = _tonal_correction(x, sr, lambda f: np.interp(np.log2(np.maximum(f, 20.0)), np.log2(g), c),
                                     0.6, 4.0, 30.0, hi, 1 / 2)
        log["tonal_match"] = f"profile '{profile['name']}' ({len(profile['tracks'])} refs)"
    elif reference is not None:
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
        settings = [(0.8, 25.0, 160.0), (0.5, 12.0, 110.0), (0.8, 5.0, 70.0)]
        out = np.zeros_like(x)
        mb_log = []

        def bands():  # filters.three_band_split, one band at a time (a phone holds 3 fewer copies)
            low, rest = filters.lr4_split(x, sr, 120.0)
            yield filters.lr4_allpass(low, sr, 4000.0)
            del low
            mid, high = filters.lr4_split(rest, sr, 4000.0)
            del rest
            yield mid
            del mid
            yield high

        for band, (r, att, rel) in zip(bands(), settings):
            det = dynamics.detector_db(band, sr, "rms", 10.0)
            thr = float(np.percentile(det, 85))
            y, gr = dynamics.compress(band, sr, thr, 1 + 2 * r * p.master_mb_amount, att, rel,
                                      knee_db=6.0, max_gr_db=4.0)
            y *= np.sqrt(np.mean(band ** 2) / (np.mean(y ** 2) + EPS))  # keep the band's average level
            out += y
            mb_log.append(round(float(np.mean(gr)), 2))
            del band, y, gr, det
        x = out
        log["multiband_avg_gr_db"] = dict(zip(["low", "mid", "high"], mb_log))

    # 3. bus "glue" compression
    if p.master_glue_ratio > 1:
        det = dynamics.detector_db(x, sr, "rms", 10.0)
        thr = float(np.percentile(det, 90)) - 2.0
        x, gr = dynamics.compress(x, sr, thr, p.master_glue_ratio, 30.0, 150.0, knee_db=6.0, max_gr_db=4.0)
        log["glue_avg_gr_db"] = round(float(np.mean(gr)), 2)

    # 4. stereo image: mono bass, slightly wider top (or the profile's width per band)
    x = effects.stereo_image(x, sr, p.master_bass_mono_hz, 1.0 if profile is not None else p.master_width)
    if profile is not None:
        # closed loop: re-measure after the compressors and correct what is left, the way a final
        # mastering EQ is set - so the result really lands near the reference's tonal balance
        g = np.array(profile["grid_hz"])
        c = np.array(profile["curve_db"])
        hi = min(16000.0, 0.95 * min(t["lossy_cutoff_hz"] for t in profile["tracks"]))
        x, marks2 = _tonal_correction(x, sr, lambda f: np.interp(np.log2(np.maximum(f, 20.0)), np.log2(g), c),
                                      0.85, 6.0, 30.0, hi, 1.0)
        log["tonal_eq_pass2_db"] = marks2
        log["width_match_db"] = match_width(x, sr, profile["side_mid_db"])
    return x


def finalize_loudness(x: np.ndarray, sr: int, target_lufs: float, ceiling_dbtp: float, clip_knee_db: float,
                      release_ms: float, log: dict, start_gain_db: float | None = None) -> np.ndarray:
    """Drive into soft clipper + true-peak limiter until integrated loudness hits the target.

    Loudness after limiting is monotonic in the drive gain, so a secant search (aimed a hair
    above the target) converges in 2-4 passes. The closest render at or just above the target
    is then trimmed *down* to land exactly on it: the result is within 0.03 LU of the target,
    never louder, and turning down can only lower the true peak.
    The final drive gain is stored in log["drive_db"] for reuse at other rates.
    """
    def render(gain):
        y = _gain(x, gain)
        if clip_knee_db > 0:
            y = dynamics.soft_clip(y, ceiling_dbtp + clip_knee_db, knee_db=3.0)
        y, gr = dynamics.limit(y, sr, ceiling_dbtp, lookahead_ms=1.5, release_ms=release_ms)
        return y, gr, analysis.integrated_lufs(y, sr)

    aim = target_lufs + 0.04
    # keep only two candidate renders (closest at/above the target, and the loudest below it):
    # holding every trial would cost ~150 MB per minute of stereo audio, too much on a phone
    best_above = best_below = None

    def keep(r):
        nonlocal best_above, best_below
        if r[0] >= target_lufs:
            if best_above is None or r[0] < best_above[0]:
                best_above = r
        elif best_below is None or r[0] > best_below[0]:
            best_below = r

    g0 = start_gain_db if start_gain_db is not None else target_lufs - analysis.integrated_lufs(x, sr)
    y, gr, l0 = render(g0)
    keep((l0, g0, y, gr))
    del y, gr
    g1 = g0 + (aim - l0) * 1.3  # limiting eats part of every boost
    for _ in range(6):
        if best_above is not None and best_above[0] - target_lufs <= 0.15:
            break
        y, gr, l1 = render(g1)
        keep((l1, g1, y, gr))
        del y, gr
        if best_above is not None and best_below is not None:
            # bracketed: interpolate between the closest renders on either side (a secant through
            # two nearby points can see a tiny slope and leap far past the target)
            (la, ga), (lb, gb) = best_above[:2], best_below[:2]
            frac = float(np.clip((aim - lb) / max(la - lb, 1e-6), 0.1, 0.9))
            g0, l0, g1 = g1, l1, gb + frac * (ga - gb)
            continue
        slope = (l1 - l0) / (g1 - g0) if abs(g1 - g0) > 1e-6 else 1.0
        slope = float(np.clip(slope, 0.25, 1.0))
        g0, l0 = g1, l1
        g1 = g1 + float(np.clip((aim - l1) / slope, -6.0, 6.0))
    lufs, gain, y, gr = best_above if best_above is not None else best_below
    if lufs > target_lufs:
        y = _gain(y, target_lufs - lufs)  # final exact trim (down only)
    log["drive_db"] = round(float(gain), 3)
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

"""End-to-end pipeline: load -> mix -> master -> export + delivery report."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import audio_io, chains
from . import previews as previews_mod
from .dsp import analysis, effects, pitch
from .presets import DELIVERY_PROFILES, Preset, adlib_preset

EPS = 1e-12
TAIL_S = 3.0  # room for reverb / delay tails after the last note


def _align(x: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
    out = np.zeros((x.shape[0], length))
    if offset >= 0:
        n = min(x.shape[-1], length - offset)
        out[:, offset:offset + n] = x[:, :n]
    else:
        n = min(x.shape[-1] + offset, length)
        out[:, :n] = x[:, -offset:-offset + n]
    return out


def _say(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"  - {msg}", flush=True)


def run(
    vocal_path: str | Path,
    inst_path: str | Path,
    out_dir: str | Path,
    preset: Preset,
    name: str | None = None,
    reference_path: str | Path | None = None,
    vocal_offset_ms: float = 0.0,
    ceiling_overridden: bool = False,
    export_stems: bool = True,
    verbose: bool = True,
    adlib_paths: list[str | Path] | None = None,
    key: str | None = None,
    key_changes: bool = False,
    stack_at: list[tuple[float, float]] | None = None,
    progress=None,
    deliver_extra: list[str] | None = None,
    profile: dict | None = None,
    previews: bool = True,
) -> dict:
    t0 = time.time()

    def say(msg: str) -> None:
        if progress is not None:
            progress(msg)
        _say(verbose, msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or Path(vocal_path).stem
    adlib_paths = list(adlib_paths or [])
    log: dict = {"preset": preset.name, "vocal": {}, "adlibs": [], "instrumental": {}, "master": {}}

    # ---------------------------------------------------------------- load
    say("loading audio")
    v, sr_v = audio_io.load(vocal_path)
    b, sr_b = audio_io.load(inst_path)
    adl = [audio_io.load(pth) for pth in adlib_paths]
    sr = audio_io.working_rate(sr_v, sr_b, *[s_ for _, s_ in adl])
    v = audio_io.resample(v, sr_v, sr)
    b = audio_io.resample(b, sr_b, sr)
    adl = [audio_io.resample(x, s_, sr) for x, s_ in adl]
    log["sample_rate"] = sr
    log["input"] = {
        "vocal": {"file": str(vocal_path), "sr": sr_v, "channels": v.shape[0], **analysis.measure(effects.to_stereo(v), sr)},
        "instrumental": {"file": str(inst_path), "sr": sr_b, "channels": b.shape[0], **analysis.measure(effects.to_stereo(b), sr)},
    }
    off = int(vocal_offset_ms * sr / 1000)
    length = max([v.shape[-1] + off, b.shape[-1]] + [x.shape[-1] + off for x in adl]) + int(TAIL_S * sr)
    v = _align(v, length, off)
    b = _align(b, length)
    adl = [_align(x, length, off) for x in adl]

    reference = None
    if reference_path:
        r, sr_r = audio_io.load(reference_path)
        reference = effects.to_stereo(audio_io.resample(r, sr_r, sr))
        log["input"]["reference"] = {"file": str(reference_path), **analysis.measure(reference, sr)}

    # ---------------------------------------------------------------- key
    tune_key, trk = None, None
    if preset.tune_amount > 0:
        if key:
            tune_key = pitch.parse_key(key)
            log["key"] = {"key": pitch.key_name(*tune_key), "source": "user"}
        else:
            say("detecting the song key (beat chroma + vocal melody)")
            trk = pitch.track(np.mean(v, axis=0), sr)
            tonic, mode, conf = pitch.detect_key(pitch.chroma(b, sr), pitch.pitch_class_histogram(trk["midi"]))
            sections = pitch.detect_key_sections(b, sr, trk) if key_changes else []
            if key_changes and len(sections) > 1:
                tune_key = [(a_, b_, t_, m_) for a_, b_, t_, m_, _c in sections]
                log["key"] = {"key": " -> ".join(pitch.key_name(t_, m_) for _a, _b, t_, m_, _c in sections),
                              "source": "detected per section",
                              "sections": [{"from_s": round(a_, 1), "to_s": round(b_, 1),
                                            "key": pitch.key_name(t_, m_), "confidence": c_}
                                           for a_, b_, t_, m_, c_ in sections]}
            elif conf >= 0.5:
                tune_key = (tonic, mode)
                log["key"] = {"key": pitch.key_name(tonic, mode), "source": "detected", "confidence": round(conf, 2)}
            else:
                # not sure about the key: snapping to the nearest semitone is always safe
                tune_key = (0, "chromatic")
                log["key"] = {"key": "chromatic", "source": "fallback", "confidence": round(conf, 2),
                              "best_guess": pitch.key_name(tonic, mode),
                              "note": "key detection unsure - pass --key for scale-aware tuning"}
        say(f"key: {log['key']['key']}")

    # ---------------------------------------------------------------- vocals
    say("lead vocal: cleanup, auto-tune, auto-EQ, compression, de-essing, saturation")
    capture: dict = {}
    vocal, active = chains.vocal_chain(v, sr, preset, log["vocal"], tune_key, "vocal", trk, capture)
    del v, trk  # (a 4-minute song is ~150 MB per stereo copy: on a phone every copy counts)

    stack = None
    if preset.doubles or preset.harmonies:
        say("vocal stack: " + ", ".join(
            (["doubles"] if preset.doubles else []) + ([f"harmonies {preset.harmonies}"] if preset.harmonies else [])))
        if not capture:  # tuning off: doubles still work from the plain pitch track
            t_ = pitch.track(np.mean(vocal, axis=0), sr)
            capture = {"trk": t_, "out_midi": t_["midi"], "key": None}
        log["stack"] = {}
        stack = chains.build_stack(vocal, sr, capture, preset, active, stack_at, log["stack"])
    del capture

    ad_dry = []
    ap = adlib_preset(preset)
    for i, (x, pth) in enumerate(zip(adl, adlib_paths)):
        say(f"ad-libs {i + 1}/{len(adl)}: same chain, thinner + more compressed, panned per phrase")
        alog: dict = {"file": str(pth)}
        dry, act = chains.vocal_chain(x, sr, ap, alog, tune_key, "ad-lib")
        dry = chains.pan_phrases(dry, sr, act, preset.adlib_pan, start_left=(i % 2 == 0))
        ad_dry.append(dry)
        log["adlibs"].append(alog)
        adl[i] = x = dry = act = None
    del adl

    # ---------------------------------------------------------------- instrumental
    say("instrumental: subsonic filter, vocal-keyed presence carve")
    key_sig = vocal if not ad_dry else effects.to_stereo(vocal) + 0.5 * sum(ad_dry)
    inst = chains.instrumental_chain(b, key_sig, sr, active, preset, log["instrumental"])
    del b, key_sig
    inst = inst * 10 ** ((-18.0 - analysis.integrated_lufs(inst, sr)) / 20)  # common gain staging

    # ---------------------------------------------------------------- balance + ride
    say("balancing vocals against the beat + automatic vocal riding")
    vocal = chains.vocal_rider(vocal, inst, sr, active, preset.vocal_rider_db, log["vocal"])
    tempo = analysis.estimate_tempo(inst, sr)
    log["instrumental"]["tempo_bpm"] = round(tempo, 1) if tempo else None
    lead_bus = chains.vocal_effects(vocal, sr, preset, tempo, log["vocal"])
    del vocal
    i_lufs = analysis.integrated_lufs(inst, sr)
    lead_target = i_lufs + preset.vocal_balance_db
    lead_gain = 10 ** ((lead_target - analysis.integrated_lufs(lead_bus, sr)) / 20)
    lead_bus = lead_bus * lead_gain
    log["vocal"]["balance_lu_vs_inst"] = preset.vocal_balance_db

    has_stack = stack is not None
    if has_stack:
        # same gain as the lead, wetter (stacks sit behind the lead in the space)
        sp = replace(preset, vocal_reverb=preset.vocal_reverb * 1.5 + 0.04, vocal_delay=preset.vocal_delay * 0.5)
        stack_bus = chains.vocal_effects(stack, sr, sp, tempo, {}) * lead_gain
        del stack
    else:
        stack_bus = np.zeros_like(lead_bus)

    adlib_bus = np.zeros_like(lead_bus)
    has_adlibs = bool(ad_dry)
    for i, alog in enumerate(log["adlibs"]):
        bus = chains.vocal_effects(ad_dry[i], sr, ap, tempo, alog)
        ad_dry[i] = None
        adlib_bus += bus * 10 ** ((lead_target + preset.adlib_level_db - analysis.integrated_lufs(bus, sr)) / 20)
        alog["level_lu_vs_lead"] = preset.adlib_level_db
        del bus
    del ad_dry

    mix = inst + lead_bus
    mix += stack_bus
    mix += adlib_bus
    headroom = 10 ** ((-6.0 - analysis.sample_peak_db(mix)) / 20)  # classic premaster: peaks at -6 dBFS
    mix *= headroom
    # the stems and the premaster are only needed again when they are written out: park them on
    # disk (memory-mapped) so a phone has that RAM free for mastering
    spill = Path(tempfile.mkdtemp(prefix=".studiomix-", dir=out_dir))
    try:
        stems = []
        if export_stems:
            for stem_name, bus_, keep in [("vocal_stem", lead_bus, True), ("stack_stem", stack_bus, has_stack),
                                          ("adlib_stem", adlib_bus, has_adlibs), ("instrumental_stem", inst, True)]:
                if keep:
                    bus_ *= headroom
                    stems.append((stem_name, _park(bus_, spill / f"{stem_name}.npy")))
        lead_bus = adlib_bus = stack_bus = inst = bus_ = None

        # ---------------------------------------------------------------- master
        say("master: tonal balance, multiband + glue compression, stereo image")
        pre = chains.master_chain(mix, sr, preset, log["master"], reference, profile)
        premaster = _park(mix, spill / "premaster.npy")
        del mix
        return deliver(pre, premaster, sr, preset, name, out_dir, log, say, t0, ceiling_overridden, stems,
                       deliver_extra, fan_mix=premaster if previews else None, fan_mix_label="Before mastering")
    finally:
        stems = premaster = None
        shutil.rmtree(spill, ignore_errors=True)


def _park(x: np.ndarray, path: Path) -> np.ndarray:
    """Move a song-length array to disk and return a read-only memory map of it (float32 keeps
    full 24-bit resolution at half the size)."""
    np.save(path, x.astype(np.float32))
    return np.load(path, mmap_mode="r")


def master_mix(
    mix_path: str | Path,
    out_dir: str | Path,
    preset: Preset,
    name: str | None = None,
    reference_path: str | Path | None = None,
    vocal_lift_db: float = 0.0,
    ceiling_overridden: bool = False,
    deliver_extra: list[str] | None = None,
    verbose: bool = True,
    progress=None,
    profile: dict | None = None,
    previews: bool = True,
) -> dict:
    """Finished or rough stereo mix -> diagnosed, repaired, mastered and verified release files."""
    t0 = time.time()

    def say(msg: str) -> None:
        if progress is not None:
            progress(msg)
        _say(verbose, msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or Path(mix_path).stem
    log: dict = {"mode": "mastering a finished mix", "preset": preset.name, "mix_repair": {}, "master": {}}

    say("loading the mix")
    x, sr0 = audio_io.load(mix_path)
    sr = audio_io.working_rate(sr0)
    x = effects.to_stereo(audio_io.resample(x, sr0, sr))
    uploaded = x if previews else None  # "your mix" for the listen-like-a-fan comparison
    log["sample_rate"] = sr
    log["input"] = {"mix": {"file": str(mix_path), "sr": sr0, "channels": x.shape[0], **analysis.measure(x, sr)}}
    reference = None
    if reference_path:
        r, sr_r = audio_io.load(reference_path)
        reference = effects.to_stereo(audio_io.resample(r, sr_r, sr))
        log["input"]["reference"] = {"file": str(reference_path), **analysis.measure(reference, sr)}

    say("diagnosing the mix (clipping, phase, tone, headroom)")
    diag = chains.diagnose_mix(x, sr, mix_path)
    log["diagnosis"] = diag

    say("repairing: DC/rumble, centre de-essing, anti-harshness" + (", vocal lift" if vocal_lift_db else ""))
    x = chains.repair_mix(x, sr, preset, diag, vocal_lift_db, log["mix_repair"])
    x = np.pad(x, ((0, 0), (0, int(1.0 * sr))))  # room for the fade-out
    x *= 10 ** ((-6.0 - analysis.sample_peak_db(x)) / 20)  # premaster level: peaks at -6 dBFS

    say("master: tonal balance, multiband + glue compression, stereo image")
    pre = chains.master_chain(x, sr, preset, log["master"], reference, profile)
    return deliver(pre, x, sr, preset, name, out_dir, log, say, t0, ceiling_overridden, None, deliver_extra,
                   fan_mix=uploaded, fan_mix_label="Your mix")


def deliver(pre: np.ndarray, premaster: np.ndarray, sr: int, preset: Preset, name: str, out_dir: Path, log: dict,
            say, t0: float, ceiling_overridden: bool = False, stems: list | None = None,
            extra: list[str] | None = None, fan_mix: np.ndarray | None = None,
            fan_mix_label: str = "Before mastering") -> dict:
    """Loudness + limiting, every export, extra delivery versions, and verification of the files
    as written. Shared by the vocal+beat pipeline and the finished-mix mastering pipeline."""
    if preset.punch > 0 and preset.target_lufs > -11.0:
        # Measured: at a fixed loudness and ceiling, no clipper/limiter/transient trick buys back
        # hit contrast - the drum peaks sit at the ceiling and the loudness fixes the body. What
        # does: a little less loudness. Streaming plays every master at -14 LUFS, so the quieter
        # master plays just as loud there, and hits harder.
        louder = preset.target_lufs
        preset = replace(preset, target_lufs=round(max(-11.0, louder - 2.0 * preset.punch), 1))
        log["master"]["punch"] = {"amount": preset.punch, "loudness_from": louder, "loudness_to": preset.target_lufs}
        say(f"punch: loudness {louder} -> {preset.target_lufs} LUFS (same volume on streaming, harder-hitting drums)")
    ceiling = preset.ceiling_dbtp
    if preset.target_lufs > -14.0 and not ceiling_overridden and ceiling > -2.0:
        # Spotify: masters louder than -14 LUFS should peak below -2 dBTP to survive lossy encoding
        ceiling = -2.0
    log["master"]["ceiling_dbtp"] = ceiling
    log["master"]["target_lufs"] = preset.target_lufs

    say(f"master: loudness to {preset.target_lufs} LUFS, true-peak ceiling {ceiling} dBTP")
    master = chains.finalize_loudness(pre, sr, preset.target_lufs, ceiling, preset.master_clip_knee_db,
                                      preset.limiter_release_ms, log["master"])
    master = chains.fade_edges(master, sr)
    final_len = master.shape[-1]

    # ---------------------------------------------------------------- export
    say("exporting")
    files = {}
    p24 = out_dir / f"{name}_master_24bit_{sr / 1000:g}k.wav"
    audio_io.write_wav(p24, master, sr, 24)
    files["master_24bit"] = p24.name

    if sr == 44100:
        m44 = master
    else:
        # limit the 44.1 kHz delivery natively: resampling a limited master re-creates overs
        m44 = chains.finalize_loudness(audio_io.resample(pre, sr, 44100), 44100, preset.target_lufs, ceiling,
                                       preset.master_clip_knee_db, preset.limiter_release_ms, {},
                                       start_gain_db=log["master"]["drive_db"])
        m44 = chains.fade_edges(m44, 44100)
    p16 = out_dir / f"{name}_master_16bit_44.1k.wav"
    audio_io.write_wav16_dithered(p16, m44, 44100)
    files["master_16bit_cd"] = p16.name

    pmp3 = out_dir / f"{name}_master_preview_320k.mp3"
    if audio_io.write_mp3(pmp3, m44, 44100):
        files["mp3_preview"] = pmp3.name

    ppre = out_dir / f"{name}_premaster_mix_24bit.wav"
    audio_io.write_wav(ppre, premaster[:, :final_len], sr, 24)
    files["premaster_mix"] = ppre.name

    for stem_name, stem in stems or []:
        pth = out_dir / f"{name}_{stem_name}_24bit.wav"
        audio_io.write_wav(pth, stem[:, :final_len], sr, 24)
        files[stem_name] = pth.name

    # ---------------------------------------------------------------- extra delivery versions
    versions = {}
    for prof in extra or []:
        label, lufs, tp_max, tol = DELIVERY_PROFILES[prof]
        say(f"delivery version: {label} ({lufs} LUFS, <= {tp_max} dBTP)")
        # gentle: no clipper, slower release - broadcast versions keep the dynamics
        y = chains.finalize_loudness(pre, sr, lufs, tp_max, 0.0, max(preset.limiter_release_ms, 120.0), {})
        y = chains.fade_edges(y, sr)
        pth = out_dir / f"{name}_{prof}_{abs(lufs):g}LUFS_24bit_{sr / 1000:g}k.wav"
        audio_io.write_wav(pth, y, sr, 24)
        files[f"version_{prof}"] = pth.name
        versions[prof] = (label, lufs, tp_max, tol)

    final = analysis.measure(master, sr)

    # ---------------------------------------------------------------- listen like a fan
    fan = None
    if fan_mix is not None and previews_mod.available():
        say("listen like a fan: the master and your mix the way Spotify, Apple Music, YouTube and a phone play them")
        try:
            fan = previews_mod.make(out_dir, name, master, np.asarray(fan_mix)[:, :final_len], sr,
                                    master_lufs_tp=(final["integrated_lufs"], final["true_peak_dbtp"]))
            fan["mix_label"] = fan_mix_label
            log["previews"] = fan
        except Exception as e:  # previews are a bonus: never lose a finished master over them
            log["previews_error"] = str(e)

    # ---------------------------------------------------------------- report
    say("verifying the delivered files (our meter + ffmpeg EBU R128)")
    deliverables = {}
    for key_ in ["master_24bit", "master_16bit_cd", "mp3_preview"] + [f"version_{p_}" for p_ in versions]:
        if key_ in files:
            pth = out_dir / files[key_]
            dec, dsr = audio_io.load(pth)  # what is actually on disk: dither, rate, encoding
            ours = analysis.measure(dec, dsr)
            deliverables[key_] = {"file": files[key_], "ours": ours, "ffmpeg": analysis.ffmpeg_ebur128(pth)}
    log["output"] = final
    log["deliverables"] = deliverables
    log["files"] = files
    log["settings"] = asdict(preset)
    main = {k: v for k, v in deliverables.items() if not k.startswith("version_")}
    checks = delivery_check(final, ceiling, preset.target_lufs, main)
    for prof, (label, lufs, tp_max, tol) in versions.items():
        d = deliverables[f"version_{prof}"]
        o_, f_ = d["ours"], d.get("ffmpeg") or {}
        tp = max(o_["true_peak_dbtp"], f_.get("true_peak_dbtp", -99.0) - 0.05)
        li = [o_["integrated_lufs"]] + ([f_["integrated_lufs"]] if "integrated_lufs" in f_ else [])
        ok = all(abs(v - lufs) <= tol for v in li) and tp <= tp_max + 0.005
        checks.insert(3, {"check": f"{label} version", "ok": ok,
                          "detail": f"{o_['integrated_lufs']:.2f} LUFS (spec {lufs} +-{tol}), {o_['true_peak_dbtp']:.2f} dBTP "
                                    f"(max {tp_max}), LRA {o_['loudness_range_lu']} LU"})
    if fan and fan["codec_checks"]:
        # judged on Spotify's 320k stream (premium listeners are the ones who switch normalisation
        # off); the 160k stream overshoots more and depends on the material - reported, with context
        hi, lo = fan["codec_checks"][0], fan["codec_checks"][-1]
        ok = hi["decoded_peak_dbfs"] <= 0.0
        detail = ", ".join(f"{c['codec']}: decoded peak {c['decoded_peak_dbfs']:+.2f} dBFS" for c in fan["codec_checks"])
        if not ok:
            detail += " - clips after encoding: lower the ceiling (--ceiling -2.5) or the loudness"
        elif lo["decoded_peak_dbfs"] > 0.0:
            detail += (" - the 160k stream overshoots; players decode in floating point and scale by the volume,"
                       " so it only clips at full volume on fixed-point outputs")
        checks.append({"check": "Survives Spotify's encoder at full level (normalisation off)", "ok": ok,
                       "detail": detail})
    log["delivery_check"] = checks
    log["processing_seconds"] = round(time.time() - t0, 1)

    (out_dir / f"{name}_report.json").write_text(json.dumps(log, indent=2, default=float))
    (out_dir / f"{name}_report.txt").write_text(format_report(name, log))
    return log


def delivery_check(m: dict, ceiling: float, target: float | None = None,
                   deliverables: dict | None = None) -> list[dict]:
    """Release checks, judged on the files as written (both WAVs, both meters) when available."""
    lufs = m["integrated_lufs"]
    wavs = {k: d for k, d in (deliverables or {}).items() if k != "mp3_preview"}
    if wavs:
        tp_ours = max(d["ours"]["true_peak_dbtp"] for d in wavs.values())
        sp = max(d["ours"]["sample_peak_dbfs"] for d in wavs.values())
        ff = [d["ffmpeg"] for d in wavs.values() if d.get("ffmpeg")]
        tp_ff = max((f["true_peak_dbtp"] for f in ff), default=None)
        lufs_all = [d["ours"]["integrated_lufs"] for d in wavs.values()] + [f["integrated_lufs"] for f in ff]
    else:
        tp_ours, sp, tp_ff, ff, lufs_all = m["true_peak_dbtp"], m["sample_peak_dbfs"], None, [], [lufs]
    tp = tp_ours if tp_ff is None else max(tp_ours, tp_ff - 0.05)  # ffmpeg prints 0.1 dB steps
    where = "24-bit + 16-bit files" if len(wavs) > 1 else "master"
    checks = [
        {"check": "True peak within ceiling", "ok": tp <= ceiling + 0.005,
         "detail": f"{tp_ours:.2f} dBTP" + (f" (ffmpeg {tp_ff:.1f})" if tp_ff is not None else "")
                   + f", ceiling {ceiling} - {where}"},
        {"check": "Spotify true-peak rule (<= -1 dBTP; <= -2 dBTP if louder than -14 LUFS)",
         "ok": tp <= (-2.0 if lufs > -14.0 else -1.0) + 0.005, "detail": f"{tp_ours:.2f} dBTP @ {lufs} LUFS"},
        {"check": "No clipping", "ok": sp < 0.0, "detail": f"sample peak {sp:.2f} dBFS"},
    ]
    if target is not None:
        worst = max(abs(v - target) for v in lufs_all)
        checks.append({"check": "Loudness on target (+-0.1 LU, every file, both meters)", "ok": worst <= 0.1 + 0.05 * bool(ff),
                       "detail": ", ".join(f"{v:.2f}" for v in lufs_all) + f" LUFS (target {target})"})
    if ff:
        ours_i = [d["ours"]["integrated_lufs"] for d in wavs.values() if d.get("ffmpeg")]
        diff = max(abs(o - f["integrated_lufs"]) for o, f in zip(ours_i, ff))
        checks.append({"check": "Independent meter agrees (ffmpeg EBU R128)", "ok": diff <= 0.15,
                       "detail": f"loudness within {diff:.2f} LU of ours"})
    checks += [
        {"check": "Mono compatible (stereo correlation > 0)", "ok": m["stereo_correlation"] > 0.0,
         "detail": f"{m['stereo_correlation']}"},
        {"check": "Dynamics (PLR >= 7 dB; loud trap masters often sit at 6-7)", "ok": m["plr_db"] >= 7.0,
         "detail": f"PLR {m['plr_db']} dB" + ("" if m["plr_db"] >= 7.0 else
                                                " - for more punch, trade a little loudness: Punch option (--punch)")},
    ]
    turn_down = max(0.0, lufs + 14.0)
    checks.append({
        "check": "Streaming normalisation", "ok": True,
        "detail": (f"Spotify/YouTube/Tidal will turn this down ~{turn_down:.1f} dB to -14 LUFS"
                   if turn_down > 0.05 else "Plays at native level on Spotify (-14 LUFS)"),
    })
    return checks


def format_report(name: str, log: dict) -> str:
    o = log["output"]
    head = f"Preset: {log['preset']}    Sample rate: {log['sample_rate']} Hz"
    if "instrumental" in log:
        head += (f"    Tempo: {log['instrumental'].get('tempo_bpm') or 'n/a'} BPM    "
                 f"Key: {log.get('key', {}).get('key', 'tuning off')}")
    lines = [
        f"STUDIOMIX MASTER REPORT - {name}" + (f"  ({log['mode']})" if log.get("mode") else ""),
        "=" * 60,
        head,
        "",
        "FINAL MASTER",
        f"  Integrated loudness : {o['integrated_lufs']} LUFS   (target {log['master']['target_lufs']})",
        f"  True peak           : {o['true_peak_dbtp']} dBTP   (ceiling {log['master']['ceiling_dbtp']})",
        *([f"  Punch               : loudness {log['master']['punch']['loudness_from']} -> "
           f"{log['master']['punch']['loudness_to']} LUFS for harder-hitting drums (same volume on streaming)"]
          if log["master"].get("punch") else []),
        f"  Loudness range      : {o['loudness_range_lu']} LU",
        f"  Peak-to-loudness    : {o['plr_db']} dB",
        f"  Stereo correlation  : {o['stereo_correlation']}",
        f"  Duration            : {o['duration_s']} s",
        "",
        "INPUTS",
    ]
    for k, v in log["input"].items():
        lines.append(f"  {k:12s} : {v['integrated_lufs']} LUFS, peak {v['true_peak_dbtp']} dBTP")
    if log.get("diagnosis"):
        lines += ["", "MIX DIAGNOSIS"] + [f"  - {f_}" for f_ in log["diagnosis"]["findings"]]
    lines += ["", "WHAT WAS DONE"]
    sections = [(k, log[k]) for k in ("mix_repair", "vocal") if log.get(k)]
    sections += [("stack", log["stack"])] if log.get("stack") else []
    sections += [(f"ad-lib {i + 1}", a_) for i, a_ in enumerate(log.get("adlibs", []))]
    sections += [(k, log[k]) for k in ("instrumental", "master") if k in log]
    for section, entries in sections:
        for k, v in entries.items():
            lines.append(f"  {section:12s} {k:24s} {v}")
    if log.get("key", {}).get("note"):
        lines.append(f"  NOTE: {log['key']['note']}")
    fan = log.get("previews")
    if fan:
        lines += ["", "LISTEN LIKE A FAN (each app's loudness rule + its codec)"]
        for e in fan["platforms"].values():
            mix = (f"   | {fan['mix_label'].lower()}: {e['mix']['plays_at_lufs']} LUFS ({e['mix']['gain_db']:+.1f} dB)"
                   if "mix" in e else "")
            lines.append(f"  {e['label']:13s} master {e['master']['plays_at_lufs']} LUFS ({e['master']['gain_db']:+.1f} dB){mix}")
        for c in fan["codec_checks"]:
            lines.append(f"  normalisation off, {c['codec']}: decoded peak {c['decoded_peak_dbfs']:+.2f} dBFS")
    if log.get("deliverables"):
        lines += ["", "DELIVERED FILES (decoded from disk)        ours: LUFS / dBTP / LRA      ffmpeg: LUFS / dBTP"]
        for d in log["deliverables"].values():
            o_, f_ = d["ours"], d.get("ffmpeg") or {}
            ff_txt = (f"{f_.get('integrated_lufs', float('nan')):6.1f} / {f_.get('true_peak_dbtp', float('nan')):5.1f}"
                      if f_ else "n/a (install ffmpeg)")
            lines.append(f"  {d['file'][:40]:40s} {o_['integrated_lufs']:6.2f} / {o_['true_peak_dbtp']:5.2f} / "
                         f"{o_['loudness_range_lu']:4.1f}      {ff_txt}")
        lines.append("  (the MP3 is a preview: lossy encoding adds ~0.5-1 dB of peaks - never upload it)")
    lines += ["", "DELIVERY CHECKS"]
    for c in log["delivery_check"]:
        lines.append(f"  [{'PASS' if c['ok'] else 'WARN'}] {c['check']}: {c['detail']}")
    lines += ["", "FILES"]
    for k, v in log["files"].items():
        lines.append(f"  {k:20s} {v}")
    lines += ["", "Upload the 24-bit WAV (or 16-bit if your distributor requires it) - never the MP3.",
              "Broadcast versions (EBU R128 / ATSC A/85) are for radio/TV stations that ask for them.", ""]
    return "\n".join(lines)

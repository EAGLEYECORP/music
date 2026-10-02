"""End-to-end pipeline: load -> mix -> master -> export + delivery report."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import audio_io, chains
from .dsp import analysis, effects, pitch
from .presets import Preset, adlib_preset

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

    stack = None
    if preset.doubles or preset.harmonies:
        say("vocal stack: " + ", ".join(
            (["doubles"] if preset.doubles else []) + ([f"harmonies {preset.harmonies}"] if preset.harmonies else [])))
        if not capture:  # tuning off: doubles still work from the plain pitch track
            t_ = pitch.track(np.mean(vocal, axis=0), sr)
            capture = {"trk": t_, "out_midi": t_["midi"], "key": None}
        log["stack"] = {}
        stack = chains.build_stack(vocal, sr, capture, preset, active, stack_at, log["stack"])

    ad_dry = []
    ap = adlib_preset(preset)
    for i, (x, pth) in enumerate(zip(adl, adlib_paths)):
        say(f"ad-libs {i + 1}/{len(adl)}: same chain, thinner + more compressed, panned per phrase")
        alog: dict = {"file": str(pth)}
        dry, act = chains.vocal_chain(x, sr, ap, alog, tune_key, "ad-lib")
        dry = chains.pan_phrases(dry, sr, act, preset.adlib_pan, start_left=(i % 2 == 0))
        ad_dry.append(dry)
        log["adlibs"].append(alog)

    # ---------------------------------------------------------------- instrumental
    say("instrumental: subsonic filter, vocal-keyed presence carve")
    key_sig = vocal if not ad_dry else effects.to_stereo(vocal) + 0.5 * sum(ad_dry)
    inst = chains.instrumental_chain(b, key_sig, sr, active, preset, log["instrumental"])
    inst = inst * 10 ** ((-18.0 - analysis.integrated_lufs(inst, sr)) / 20)  # common gain staging

    # ---------------------------------------------------------------- balance + ride
    say("balancing vocals against the beat + automatic vocal riding")
    vocal = chains.vocal_rider(vocal, inst, sr, active, preset.vocal_rider_db, log["vocal"])
    tempo = analysis.estimate_tempo(inst, sr)
    log["instrumental"]["tempo_bpm"] = round(tempo, 1) if tempo else None
    lead_bus = chains.vocal_effects(vocal, sr, preset, tempo, log["vocal"])
    i_lufs = analysis.integrated_lufs(inst, sr)
    lead_target = i_lufs + preset.vocal_balance_db
    lead_gain = 10 ** ((lead_target - analysis.integrated_lufs(lead_bus, sr)) / 20)
    lead_bus = lead_bus * lead_gain
    log["vocal"]["balance_lu_vs_inst"] = preset.vocal_balance_db

    if stack is not None:
        # same gain as the lead, wetter (stacks sit behind the lead in the space)
        sp = replace(preset, vocal_reverb=preset.vocal_reverb * 1.5 + 0.04, vocal_delay=preset.vocal_delay * 0.5)
        stack_bus = chains.vocal_effects(stack, sr, sp, tempo, {}) * lead_gain
    else:
        stack_bus = np.zeros_like(lead_bus)

    adlib_bus = np.zeros_like(lead_bus)
    for dry, alog in zip(ad_dry, log["adlibs"]):
        bus = chains.vocal_effects(dry, sr, ap, tempo, alog)
        adlib_bus += bus * 10 ** ((lead_target + preset.adlib_level_db - analysis.integrated_lufs(bus, sr)) / 20)
        alog["level_lu_vs_lead"] = preset.adlib_level_db
    vocal_bus = lead_bus + stack_bus + adlib_bus

    mix = inst + vocal_bus
    headroom = -6.0 - analysis.sample_peak_db(mix)  # classic premaster: peaks at -6 dBFS
    mix *= 10 ** (headroom / 20)
    vocal_bus *= 10 ** (headroom / 20)
    lead_bus *= 10 ** (headroom / 20)
    adlib_bus *= 10 ** (headroom / 20)
    stack_bus *= 10 ** (headroom / 20)
    inst *= 10 ** (headroom / 20)

    # ---------------------------------------------------------------- master
    say("master: tonal balance, multiband + glue compression, stereo image")
    pre = chains.master_chain(mix, sr, preset, log["master"], reference)

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
    audio_io.write_wav(ppre, mix[:, :final_len], sr, 24)
    files["premaster_mix"] = ppre.name

    if export_stems:
        stems = [("vocal_stem", lead_bus), ("instrumental_stem", inst)]
        if ad_dry:
            stems.insert(1, ("adlib_stem", adlib_bus))
        if stack is not None:
            stems.insert(1, ("stack_stem", stack_bus))
        for stem_name, stem in stems:
            pth = out_dir / f"{name}_{stem_name}_24bit.wav"
            audio_io.write_wav(pth, stem[:, :final_len], sr, 24)
            files[stem_name] = pth.name

    # ---------------------------------------------------------------- report
    say("verifying the delivered files (our meter + ffmpeg EBU R128)")
    final = analysis.measure(master, sr)
    deliverables = {}
    for key_ in ("master_24bit", "master_16bit_cd", "mp3_preview"):
        if key_ in files:
            pth = out_dir / files[key_]
            dec, dsr = audio_io.load(pth)  # what is actually on disk: dither, rate, encoding
            ours = analysis.measure(dec, dsr)
            deliverables[key_] = {"file": files[key_], "ours": ours, "ffmpeg": analysis.ffmpeg_ebur128(pth)}
    log["output"] = final
    log["deliverables"] = deliverables
    log["files"] = files
    log["settings"] = asdict(preset)
    log["delivery_check"] = delivery_check(final, ceiling, preset.target_lufs, deliverables)
    log["processing_seconds"] = round(time.time() - t0, 1)

    (out_dir / f"{name}_report.json").write_text(json.dumps(log, indent=2))
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
         "detail": f"PLR {m['plr_db']} dB"},
    ]
    turn_down = max(0.0, lufs + 14.0)
    checks.append({
        "check": "Streaming normalisation", "ok": True,
        "detail": (f"Spotify/YouTube/Tidal will turn this down ~{turn_down:.1f} dB to -14 LUFS"
                   if turn_down > 0.05 else "Plays at native level on Spotify (-14 LUFS)"),
    })
    return checks


def format_report(name: str, log: dict) -> str:
    o, i = log["output"], log["input"]
    lines = [
        f"STUDIOMIX MASTER REPORT - {name}",
        "=" * 60,
        f"Preset: {log['preset']}    Sample rate: {log['sample_rate']} Hz    "
        f"Tempo: {log['instrumental'].get('tempo_bpm') or 'n/a'} BPM    "
        f"Key: {log.get('key', {}).get('key', 'tuning off')}",
        "",
        "FINAL MASTER",
        f"  Integrated loudness : {o['integrated_lufs']} LUFS   (target {log['master']['target_lufs']})",
        f"  True peak           : {o['true_peak_dbtp']} dBTP   (ceiling {log['master']['ceiling_dbtp']})",
        f"  Loudness range      : {o['loudness_range_lu']} LU",
        f"  Peak-to-loudness    : {o['plr_db']} dB",
        f"  Stereo correlation  : {o['stereo_correlation']}",
        f"  Duration            : {o['duration_s']} s",
        "",
        "INPUTS",
        f"  Vocal        : {i['vocal']['integrated_lufs']} LUFS, peak {i['vocal']['true_peak_dbtp']} dBTP",
        f"  Instrumental : {i['instrumental']['integrated_lufs']} LUFS, peak {i['instrumental']['true_peak_dbtp']} dBTP",
        "",
        "WHAT WAS DONE",
    ]
    sections = [("vocal", log["vocal"])] + ([("stack", log["stack"])] if log.get("stack") else [])
    sections += [(f"ad-lib {i + 1}", a) for i, a in enumerate(log["adlibs"])]
    sections += [("instrumental", log["instrumental"]), ("master", log["master"])]
    for section, entries in sections:
        for k, v in entries.items():
            lines.append(f"  {section:12s} {k:24s} {v}")
    if log.get("key", {}).get("note"):
        lines.append(f"  NOTE: {log['key']['note']}")
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
    lines += ["", "Upload the 24-bit WAV (or 16-bit if your distributor requires it) - never the MP3.", ""]
    return "\n".join(lines)

"""End-to-end pipeline: load -> mix -> master -> export + delivery report."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import audio_io, chains
from .dsp import analysis, effects
from .presets import Preset

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
) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or Path(vocal_path).stem
    log: dict = {"preset": preset.name, "vocal": {}, "instrumental": {}, "master": {}}

    # ---------------------------------------------------------------- load
    _say(verbose, "loading audio")
    v, sr_v = audio_io.load(vocal_path)
    b, sr_b = audio_io.load(inst_path)
    sr = audio_io.working_rate(sr_v, sr_b)
    v = audio_io.resample(v, sr_v, sr)
    b = audio_io.resample(b, sr_b, sr)
    log["sample_rate"] = sr
    log["input"] = {
        "vocal": {"file": str(vocal_path), "sr": sr_v, "channels": v.shape[0], **analysis.measure(effects.to_stereo(v), sr)},
        "instrumental": {"file": str(inst_path), "sr": sr_b, "channels": b.shape[0], **analysis.measure(effects.to_stereo(b), sr)},
    }
    length = max(v.shape[-1] + int(vocal_offset_ms * sr / 1000), b.shape[-1]) + int(TAIL_S * sr)
    v = _align(v, length, int(vocal_offset_ms * sr / 1000))
    b = _align(b, length)

    reference = None
    if reference_path:
        r, sr_r = audio_io.load(reference_path)
        reference = effects.to_stereo(audio_io.resample(r, sr_r, sr))
        log["input"]["reference"] = {"file": str(reference_path), **analysis.measure(reference, sr)}

    # ---------------------------------------------------------------- vocal
    _say(verbose, "vocal: cleanup, auto-EQ, compression, de-essing, saturation")
    vocal, active = chains.vocal_chain(v, sr, preset, log["vocal"])

    # ---------------------------------------------------------------- instrumental
    _say(verbose, "instrumental: subsonic filter, vocal-keyed presence carve")
    inst = chains.instrumental_chain(b, vocal, sr, active, preset, log["instrumental"])
    inst = inst * 10 ** ((-18.0 - analysis.integrated_lufs(inst, sr)) / 20)  # common gain staging

    # ---------------------------------------------------------------- balance + ride
    _say(verbose, "balancing vocal against the beat + automatic vocal riding")
    vocal = chains.vocal_rider(vocal, inst, sr, active, preset.vocal_rider_db, log["vocal"])
    tempo = analysis.estimate_tempo(inst, sr)
    log["instrumental"]["tempo_bpm"] = round(tempo, 1) if tempo else None
    vocal_bus = chains.vocal_effects(vocal, sr, preset, tempo, log["vocal"])
    v_lufs = analysis.integrated_lufs(vocal_bus, sr)
    i_lufs = analysis.integrated_lufs(inst, sr)
    vgain = (i_lufs + preset.vocal_balance_db) - v_lufs
    vocal_bus = vocal_bus * 10 ** (vgain / 20)
    log["vocal"]["balance_lu_vs_inst"] = preset.vocal_balance_db

    mix = inst + vocal_bus
    headroom = -6.0 - analysis.sample_peak_db(mix)  # classic premaster: peaks at -6 dBFS
    mix *= 10 ** (headroom / 20)
    vocal_bus *= 10 ** (headroom / 20)
    inst *= 10 ** (headroom / 20)

    # ---------------------------------------------------------------- master
    _say(verbose, "master: tonal balance, multiband + glue compression, stereo image")
    pre = chains.master_chain(mix, sr, preset, log["master"], reference)

    ceiling = preset.ceiling_dbtp
    if preset.target_lufs > -14.0 and not ceiling_overridden and ceiling > -2.0:
        # Spotify: masters louder than -14 LUFS should peak below -2 dBTP to survive lossy encoding
        ceiling = -2.0
    log["master"]["ceiling_dbtp"] = ceiling
    log["master"]["target_lufs"] = preset.target_lufs

    _say(verbose, f"master: loudness to {preset.target_lufs} LUFS, true-peak ceiling {ceiling} dBTP")
    master = chains.finalize_loudness(pre, sr, preset.target_lufs, ceiling, preset.master_clip_knee_db,
                                      preset.limiter_release_ms, log["master"])
    master = chains.fade_edges(master, sr)
    final_len = master.shape[-1]

    # ---------------------------------------------------------------- export
    _say(verbose, "exporting")
    files = {}
    p24 = out_dir / f"{name}_master_24bit_{sr / 1000:g}k.wav"
    audio_io.write_wav(p24, master, sr, 24)
    files["master_24bit"] = p24.name

    if sr == 44100:
        m44 = master
    else:
        # limit the 44.1 kHz delivery natively: resampling a limited master re-creates overs
        m44 = chains.finalize_loudness(audio_io.resample(pre, sr, 44100), 44100, preset.target_lufs, ceiling,
                                       preset.master_clip_knee_db, preset.limiter_release_ms, {})
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
        for key, stem in (("vocal_stem", vocal_bus), ("instrumental_stem", inst)):
            pth = out_dir / f"{name}_{key}_24bit.wav"
            audio_io.write_wav(pth, stem[:, :final_len], sr, 24)
            files[key] = pth.name

    # ---------------------------------------------------------------- report
    final = analysis.measure(master, sr)
    log["output"] = final
    log["files"] = files
    log["settings"] = asdict(preset)
    log["delivery_check"] = delivery_check(final, ceiling)
    log["processing_seconds"] = round(time.time() - t0, 1)

    (out_dir / f"{name}_report.json").write_text(json.dumps(log, indent=2))
    (out_dir / f"{name}_report.txt").write_text(format_report(name, log))
    return log


def delivery_check(m: dict, ceiling: float) -> list[dict]:
    lufs, tp = m["integrated_lufs"], m["true_peak_dbtp"]
    checks = [
        {"check": "True peak within ceiling", "ok": tp <= ceiling + 0.05,
         "detail": f"{tp} dBTP (ceiling {ceiling})"},
        {"check": "Spotify true-peak rule (<= -1 dBTP; <= -2 dBTP if louder than -14 LUFS)",
         "ok": tp <= (-2.0 if lufs > -14.0 else -1.0) + 0.05, "detail": f"{tp} dBTP @ {lufs} LUFS"},
        {"check": "No clipping", "ok": m["sample_peak_dbfs"] < 0.0, "detail": f"sample peak {m['sample_peak_dbfs']} dBFS"},
        {"check": "Mono compatible (stereo correlation > 0)", "ok": m["stereo_correlation"] > 0.0,
         "detail": f"{m['stereo_correlation']}"},
        {"check": "Healthy dynamics (PLR >= 7 dB)", "ok": m["plr_db"] >= 7.0, "detail": f"PLR {m['plr_db']} dB"},
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
        f"Tempo: {log['instrumental'].get('tempo_bpm') or 'n/a'} BPM",
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
    for section in ("vocal", "instrumental", "master"):
        for k, v in log[section].items():
            lines.append(f"  {section:12s} {k:24s} {v}")
    lines += ["", "DELIVERY CHECKS"]
    for c in log["delivery_check"]:
        lines.append(f"  [{'PASS' if c['ok'] else 'WARN'}] {c['check']}: {c['detail']}")
    lines += ["", "FILES"]
    for k, v in log["files"].items():
        lines.append(f"  {k:20s} {v}")
    lines += ["", "Upload the 24-bit WAV (or 16-bit if your distributor requires it) - never the MP3.", ""]
    return "\n".join(lines)

"""Genre presets. Every field can be overridden from the command line."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Preset:
    name: str
    description: str

    # ---- vocal chain
    vocal_hpf_hz: float = 90.0
    vocal_mud_cut_db: float = -2.5      # around 300 Hz
    vocal_boxy_cut_db: float = -1.5     # around 800 Hz
    vocal_presence_db: float = 2.0      # around 3.5 kHz
    vocal_air_db: float = 3.0           # shelf at 12 kHz
    vocal_auto_eq_strength: float = 0.5
    vocal_comp_amount: float = 1.0      # scales both compressor stages (0 = off, 1.5 = heavy)
    vocal_deess_db: float = 8.0         # maximum de-esser reduction
    vocal_saturation: float = 0.15      # parallel saturation mix
    vocal_reverb: float = 0.12          # reverb send level (linear, 0-1)
    vocal_reverb_size: float = 0.55
    vocal_delay: float = 0.06           # delay send level (linear, 0-1)
    vocal_delay_note: float = 0.25      # in bars of 4/4: 0.25 = quarter note, 0.125 = eighth
    vocal_rider_db: float = 3.0         # max vocal-rider correction vs. the beat
    vocal_balance_db: float = 0.5       # vocal loudness relative to instrumental (LU)

    # ---- instrumental
    inst_hpf_hz: float = 25.0
    inst_carve_db: float = 3.0          # max dip in the 1.5-5 kHz band while the vocal sings

    # ---- master
    master_tonal_strength: float = 0.4
    master_tilt_db_oct: float = -4.5    # target spectral slope of a balanced commercial mix
    master_mb_amount: float = 1.0       # multiband compression amount
    master_glue_ratio: float = 2.0
    master_bass_mono_hz: float = 120.0
    master_width: float = 1.1
    master_clip_knee_db: float = 2.0    # soft-clip headroom above the limiter ceiling (0 = off)
    target_lufs: float = -12.0
    ceiling_dbtp: float = -1.0
    limiter_release_ms: float = 80.0


PRESETS: dict[str, Preset] = {
    p.name: p
    for p in [
        Preset(
            name="pop",
            description="Bright, upfront vocal; polished and competitive (-11 LUFS).",
            vocal_presence_db=2.5, vocal_air_db=3.5, vocal_balance_db=1.0,
            target_lufs=-11.0, ceiling_dbtp=-1.0,
        ),
        Preset(
            name="hiphop",
            description="Hip-hop / trap: loud, dry and in-your-face vocal, heavy low end (-9 LUFS).",
            vocal_hpf_hz=100.0, vocal_comp_amount=1.3, vocal_saturation=0.25,
            vocal_reverb=0.06, vocal_reverb_size=0.4, vocal_delay=0.08, vocal_delay_note=0.125,
            vocal_balance_db=1.5, inst_carve_db=3.5, master_tilt_db_oct=-5.0,
            master_bass_mono_hz=150.0, master_width=1.05, master_clip_knee_db=2.5,
            target_lufs=-9.0, limiter_release_ms=60.0,
        ),
        Preset(
            name="rnb",
            description="Smooth, warm vocal with lush space (-11 LUFS).",
            vocal_presence_db=1.5, vocal_air_db=3.0, vocal_saturation=0.2,
            vocal_reverb=0.18, vocal_reverb_size=0.7, vocal_delay=0.08,
            vocal_balance_db=0.5, master_width=1.15, target_lufs=-11.0,
        ),
        Preset(
            name="rock",
            description="Vocal sits inside dense guitars and drums (-10 LUFS).",
            vocal_hpf_hz=100.0, vocal_presence_db=3.0, vocal_comp_amount=1.2, vocal_saturation=0.25,
            vocal_reverb=0.1, vocal_balance_db=-0.5, inst_carve_db=4.0,
            master_tilt_db_oct=-4.0, target_lufs=-10.0,
        ),
        Preset(
            name="acoustic",
            description="Natural, dynamic singer-songwriter / ballad (-14 LUFS).",
            vocal_hpf_hz=75.0, vocal_comp_amount=0.7, vocal_saturation=0.05,
            vocal_reverb=0.16, vocal_reverb_size=0.65, vocal_delay=0.03,
            vocal_balance_db=1.0, inst_carve_db=2.0, master_mb_amount=0.6, master_glue_ratio=1.5,
            master_clip_knee_db=0.0, target_lufs=-14.0, limiter_release_ms=120.0,
        ),
        Preset(
            name="streaming",
            description="Exactly Spotify/YouTube/Tidal reference loudness: -14 LUFS, -1 dBTP, max dynamics.",
            master_clip_knee_db=0.0, target_lufs=-14.0, ceiling_dbtp=-1.0, limiter_release_ms=100.0,
        ),
    ]
}


def get_preset(preset_name: str, **overrides) -> Preset:
    if preset_name not in PRESETS:
        raise KeyError(f"unknown preset '{preset_name}'. choose from: {', '.join(PRESETS)}")
    valid = {f.name for f in fields(Preset)} - {"name", "description"}
    clean = {k: v for k, v in overrides.items() if v is not None and k in valid}
    return replace(PRESETS[preset_name], **clean)

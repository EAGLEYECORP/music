"""Genre presets. Every field can be overridden from the command line."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Preset:
    name: str
    description: str

    # ---- vocal chain
    vocal_hpf_hz: float = 90.0
    vocal_denoise: bool = True          # adaptive background-noise reduction (fans, AC, hiss, hum)
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

    # ---- pitch correction (applied to lead and ad-libs)
    tune_amount: float = 1.0            # 0 = off, 1 = full correction
    tune_retune_ms: float = 25.0        # 0 = instant/robotic (T-Pain), 20-40 = modern pop, 80+ = natural
    tune_humanize: float = 0.35         # how much vibrato / expression survives on held notes (0-1)
    tune_flex_cents: float = 0.0        # Flex-Tune: leave deviations beyond this alone (0 = correct all)

    # ---- vocal stacks (built from the tuned lead)
    doubles: bool = False               # two double-tracked copies of the lead, panned wide
    doubles_db: float = -7.0            # each double vs. the lead
    harmonies: str = ""                 # comma list of intervals: 3up,3down,4up,5up,5down,6down,8up,8down
    harmony_db: float = -9.0            # each harmony voice vs. the lead

    # ---- ad-libs
    adlib_level_db: float = -5.0        # ad-lib loudness relative to the lead (LU)
    adlib_pan: float = 0.5              # phrases alternate between this far left / right (0 = centre)
    adlib_reverb: float = 0.18
    adlib_delay: float = 0.14

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
            tune_retune_ms=30.0, tune_humanize=0.4,
            target_lufs=-11.0, ceiling_dbtp=-1.0,
        ),
        Preset(
            name="hiphop",
            description="Hip-hop / trap: loud, dry and in-your-face vocal, heavy low end (-9 LUFS).",
            vocal_hpf_hz=100.0, vocal_comp_amount=1.3, vocal_saturation=0.25,
            vocal_reverb=0.06, vocal_reverb_size=0.4, vocal_delay=0.08, vocal_delay_note=0.125,
            vocal_balance_db=1.5, inst_carve_db=3.5, master_tilt_db_oct=-5.0,
            tune_retune_ms=10.0, tune_humanize=0.15, adlib_level_db=-4.0, adlib_pan=0.55,
            master_bass_mono_hz=150.0, master_width=1.05, master_clip_knee_db=2.5,
            target_lufs=-9.0, limiter_release_ms=60.0,
        ),
        Preset(
            name="trap",
            description="Trap / melodic rap: hard robotic auto-tune, loud, wide ad-libs (-8.5 LUFS).",
            vocal_hpf_hz=110.0, vocal_comp_amount=1.4, vocal_saturation=0.25, vocal_air_db=4.0,
            vocal_reverb=0.08, vocal_reverb_size=0.5, vocal_delay=0.1, vocal_delay_note=0.125,
            vocal_balance_db=1.0, inst_carve_db=3.5, master_tilt_db_oct=-5.0,
            tune_retune_ms=0.0, tune_humanize=0.0, adlib_level_db=-3.5, adlib_pan=0.7, adlib_delay=0.18,
            master_bass_mono_hz=150.0, master_width=1.1, master_clip_knee_db=3.0,
            target_lufs=-8.5, limiter_release_ms=50.0,
        ),
        Preset(
            name="rnb",
            description="Smooth, warm vocal with lush space (-11 LUFS).",
            vocal_presence_db=1.5, vocal_air_db=3.0, vocal_saturation=0.2,
            vocal_reverb=0.18, vocal_reverb_size=0.7, vocal_delay=0.08,
            vocal_balance_db=0.5, master_width=1.15, target_lufs=-11.0,
            tune_retune_ms=30.0, tune_humanize=0.5, adlib_level_db=-6.0, adlib_reverb=0.24,
        ),
        Preset(
            name="rock",
            description="Vocal sits inside dense guitars and drums (-10 LUFS).",
            vocal_hpf_hz=100.0, vocal_presence_db=3.0, vocal_comp_amount=1.2, vocal_saturation=0.25,
            vocal_reverb=0.1, vocal_balance_db=-0.5, inst_carve_db=4.0,
            master_tilt_db_oct=-4.0, target_lufs=-10.0,
            tune_amount=0.8, tune_retune_ms=70.0, tune_humanize=0.8,
        ),
        Preset(
            name="acoustic",
            description="Natural, dynamic singer-songwriter / ballad (-14 LUFS).",
            vocal_hpf_hz=75.0, vocal_comp_amount=0.7, vocal_saturation=0.05,
            vocal_reverb=0.16, vocal_reverb_size=0.65, vocal_delay=0.03,
            vocal_balance_db=1.0, inst_carve_db=2.0, master_mb_amount=0.6, master_glue_ratio=1.5,
            master_clip_knee_db=0.0, target_lufs=-14.0, limiter_release_ms=120.0,
            tune_amount=0.7, tune_retune_ms=90.0, tune_humanize=0.85,
        ),
        Preset(
            name="streaming",
            description="Exactly Spotify/YouTube/Tidal reference loudness: -14 LUFS, -1 dBTP, max dynamics.",
            master_clip_knee_db=0.0, target_lufs=-14.0, ceiling_dbtp=-1.0, limiter_release_ms=100.0,
        ),
    ]
}


# Extra delivery versions rendered from the same pre-limiter master. (label, LUFS, max dBTP, tolerance LU)
DELIVERY_PROFILES = {
    "ebu-r128": ("EBU R128 broadcast (EU TV/radio)", -23.0, -1.0, 0.5),
    "atsc-a85": ("ATSC A/85 broadcast (US TV/radio)", -24.0, -2.0, 2.0),
    "apple": ("Apple Music Sound Check level", -16.0, -1.0, 0.5),
    "streaming": ("Spotify/YouTube reference level", -14.0, -1.0, 0.5),
}

TUNE_STYLES = {
    # name: (retune_ms, humanize, amount)
    "off": (0.0, 0.0, 0.0),
    "natural": (80.0, 0.8, 0.8),
    "pop": (25.0, 0.4, 1.0),
    "hard": (0.0, 0.0, 1.0),
}


def adlib_preset(p: Preset) -> Preset:
    """Ad-libs: thinner, more compressed and more effected than the lead."""
    return replace(
        p,
        vocal_hpf_hz=max(p.vocal_hpf_hz, 150.0),
        vocal_mud_cut_db=p.vocal_mud_cut_db - 1.5,
        vocal_presence_db=p.vocal_presence_db + 1.0,
        vocal_comp_amount=p.vocal_comp_amount * 1.3,
        vocal_saturation=min(0.5, p.vocal_saturation + 0.1),
        vocal_reverb=p.adlib_reverb,
        vocal_delay=p.adlib_delay,
    )


def get_preset(preset_name: str, **overrides) -> Preset:
    if preset_name not in PRESETS:
        raise KeyError(f"unknown preset '{preset_name}'. choose from: {', '.join(PRESETS)}")
    valid = {f.name for f in fields(Preset)} - {"name", "description"}
    clean = {k: v for k, v in overrides.items() if v is not None and k in valid}
    for k, (lo, hi, what) in SAFE_RANGES.items():
        v = clean.get(k)
        if v is not None and not (lo <= v <= hi):
            raise ValueError(f"{what} must be between {lo:g} and {hi:g} (got {v:g})")
    return replace(PRESETS[preset_name], **clean)


# values outside these are mistakes (a ceiling above 0 dBTP clips every platform's decoder;
# -40 LUFS is inaudible), so they are refused rather than rendered
SAFE_RANGES = {
    "target_lufs": (-30.0, -5.0, "loudness target (LUFS)"),
    "ceiling_dbtp": (-6.0, 0.0, "true-peak ceiling (dBTP)"),
    "vocal_balance_db": (-12.0, 12.0, "vocal level (dB)"),
    "tune_amount": (0.0, 1.0, "tune amount"),
    "tune_flex_cents": (0.0, 100.0, "flex (cents)"),
    "master_width": (0.0, 2.0, "width"),
}

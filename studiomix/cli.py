"""Command line interface: `studiomix VOCAL INSTRUMENTAL [options]`."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .presets import PRESETS, TUNE_STYLES, get_preset


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="studiomix",
        description="Mix a vocal over an instrumental and master it for Spotify, Apple Music, YouTube & co.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="presets:\n" + "\n".join(f"  {p.name:10s} {p.description}" for p in PRESETS.values()),
    )
    ap.add_argument("vocal", help="lead vocal file (wav; mp3/flac/m4a... with ffmpeg installed)")
    ap.add_argument("instrumental", help="instrumental / beat file")
    ap.add_argument("-a", "--adlibs", nargs="+", default=[], metavar="FILE",
                    help="one or more ad-lib / hype-vocal files (same start point as the lead)")
    ap.add_argument("-o", "--out", default="out", help="output folder (default: ./out)")
    ap.add_argument("-p", "--preset", default="pop", choices=list(PRESETS), help="genre preset (default: pop)")
    ap.add_argument("-n", "--name", help="song name used for output files (default: vocal file name)")
    ap.add_argument("-r", "--reference", help="a commercial song to match tonal balance against")
    ap.add_argument("--offset-ms", type=float, default=0.0,
                    help="shift the vocal later (+) or earlier (-) relative to the beat, in milliseconds")
    ap.add_argument("--no-stems", action="store_true", help="don't export the processed vocal/instrumental stems")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--version", action="version", version=f"studiomix {__version__}")

    t = ap.add_argument_group("auto-tune")
    t.add_argument("--tune", choices=list(TUNE_STYLES),
                   help="hard = instant robotic snap, pop = fast but smooth, natural = transparent, off")
    t.add_argument("--key", help="song key, e.g. 'F# minor', 'Bb major', 'C minor-pentatonic' "
                                 "(default: detected from the beat + vocal)")
    t.add_argument("--retune-ms", dest="tune_retune_ms", type=float,
                   help="retune speed in ms (0 = robotic, 20-40 = modern pop, 80+ = natural)")
    t.add_argument("--humanize", dest="tune_humanize", type=float,
                   help="0-1: how much natural vibrato survives on held notes")
    t.add_argument("--tune-amount", dest="tune_amount", type=float, help="0-1 correction strength")

    g = ap.add_argument_group("fine tuning (override the preset)")
    g.add_argument("--lufs", dest="target_lufs", type=float, help="integrated loudness target, e.g. -14, -11, -9")
    g.add_argument("--ceiling", dest="ceiling_dbtp", type=float, help="true-peak ceiling in dBTP, e.g. -1")
    g.add_argument("--vocal-level", dest="vocal_balance_db", type=float,
                   help="vocal loudness vs. beat in dB (+ = louder vocal)")
    g.add_argument("--reverb", dest="vocal_reverb", type=float, help="vocal reverb amount 0-1")
    g.add_argument("--delay", dest="vocal_delay", type=float, help="vocal delay amount 0-1")
    g.add_argument("--air", dest="vocal_air_db", type=float, help="vocal high-end air boost in dB")
    g.add_argument("--presence", dest="vocal_presence_db", type=float, help="vocal 3.5 kHz presence boost in dB")
    g.add_argument("--vocal-comp", dest="vocal_comp_amount", type=float, help="vocal compression amount (0-2)")
    g.add_argument("--deess", dest="vocal_deess_db", type=float, help="max de-esser reduction in dB (0 = off)")
    g.add_argument("--carve", dest="inst_carve_db", type=float,
                   help="how much the beat ducks in the vocal range while singing, in dB")
    g.add_argument("--adlib-level", dest="adlib_level_db", type=float,
                   help="ad-lib loudness vs. the lead vocal in dB (default about -4)")
    g.add_argument("--adlib-pan", dest="adlib_pan", type=float, help="ad-lib left/right spread 0-1")
    g.add_argument("--width", dest="master_width", type=float, help="stereo width of the highs (1 = unchanged)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {}
    if args.tune:
        retune, humanize, amount = TUNE_STYLES[args.tune]
        overrides.update(tune_retune_ms=retune, tune_humanize=humanize, tune_amount=amount)
    overrides.update({k: v for k, v in vars(args).items() if v is not None})
    try:
        preset = get_preset(args.preset, **overrides)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    from pathlib import Path

    from .dsp import pitch
    from .engine import run  # heavy imports after argument parsing

    for f in [args.vocal, args.instrumental, *args.adlibs] + ([args.reference] if args.reference else []):
        if not Path(f).is_file():
            print(f"error: file not found: {f}", file=sys.stderr)
            return 1
    if args.key:
        try:
            pitch.parse_key(args.key)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    if not args.quiet:
        print(f"studiomix {__version__} - preset '{preset.name}'")
    try:
        log = run(
            args.vocal, args.instrumental, args.out, preset,
            name=args.name, reference_path=args.reference, vocal_offset_ms=args.offset_ms,
            ceiling_overridden=args.ceiling_dbtp is not None, export_stems=not args.no_stems,
            verbose=not args.quiet, adlib_paths=args.adlibs, key=args.key,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    o = log["output"]
    print(f"\ndone in {log['processing_seconds']}s -> {args.out}/")
    print(f"  {o['integrated_lufs']} LUFS | {o['true_peak_dbtp']} dBTP | LRA {o['loudness_range_lu']} LU"
          f" | key {log.get('key', {}).get('key', 'tuning off')}")
    for c in log["delivery_check"]:
        print(f"  [{'PASS' if c['ok'] else 'WARN'}] {c['check']}: {c['detail']}")
    for v in log["files"].values():
        print(f"  {v}")
    return 0

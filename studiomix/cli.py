"""Command line interface: `studiomix VOCAL INSTRUMENTAL [options]`."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .presets import DELIVERY_PROFILES, PRESETS, TUNE_STYLES, get_preset


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="studiomix",
        usage="studiomix VOCAL INSTRUMENTAL [options]\n"
              "       studiomix master MIX [options]                (master a finished/rough mix)\n"
              "       studiomix learn REFS... --name NAME           (learn a reference profile)\n"
              "       studiomix serve [--host HOST] [--port PORT]   (phone/browser app)",
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
    ap.add_argument("--no-previews", action="store_true",
                    help="skip the listen-like-a-fan previews (Spotify / Apple Music / YouTube / phone speaker)")
    ap.add_argument("--lyrics", nargs="?", const="auto", metavar="LANG",
                    help="also write captions (.srt), synced lyrics (.lrc) and text from the vocal "
                         "(AI, needs sherpa-onnx; LANG e.g. en, fr - default: detect)")
    ap.add_argument("--ask", metavar="WORDS",
                    help='tell the engineer in plain words (English or French), e.g. '
                         '"more 808, vocal a bit less harsh" or "voix plus forte, trop de reverb"')
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
    t.add_argument("--flex", dest="tune_flex_cents", type=float, metavar="CENTS",
                   help="Flex-Tune: only correct notes within CENTS of the target; bigger bends, falls "
                        "and blue notes are left alone (e.g. 35; default off)")
    t.add_argument("--key-changes", action="store_true",
                   help="detect a key per song section (for songs that modulate) instead of one key")

    ap.add_argument("--deliver", metavar="VERSIONS",
                    help="extra verified versions: " + ", ".join(f"{k} ({v[1]} LUFS)" for k, v in DELIVERY_PROFILES.items()))
    ap.add_argument("--profile", metavar="NAME",
                    help="master toward a learned reference profile (see: studiomix learn / studiomix profiles)")

    st = ap.add_argument_group("vocal stack (built from the tuned lead)")
    st.add_argument("--doubles", action="store_const", const=True, default=None,
                    help="add two double-tracked copies of the lead, panned wide")
    st.add_argument("--doubles-level", dest="doubles_db", type=float, help="each double vs. the lead in dB (-7)")
    st.add_argument("--harmony", dest="harmonies", metavar="INTERVALS",
                    help="harmony voices in key, comma separated: 3up,3down,4up,5up,5down,6down,8up,8down")
    st.add_argument("--harmony-level", dest="harmony_db", type=float, help="each harmony vs. the lead in dB (-9)")
    st.add_argument("--stack-at", metavar="RANGES",
                    help="only stack in these parts, e.g. '0:45-1:15,2:10-2:40' (hooks); default: everywhere")

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
    g.add_argument("--no-denoise", dest="vocal_denoise", action="store_const", const=False, default=None,
                   help="turn off background-noise reduction on the vocals")
    g.add_argument("--carve", dest="inst_carve_db", type=float,
                   help="how much the beat ducks in the vocal range while singing, in dB")
    g.add_argument("--adlib-level", dest="adlib_level_db", type=float,
                   help="ad-lib loudness vs. the lead vocal in dB (default about -4)")
    g.add_argument("--adlib-pan", dest="adlib_pan", type=float, help="ad-lib left/right spread 0-1")
    g.add_argument("--width", dest="master_width", type=float, help="stereo width of the highs (1 = unchanged)")
    g.add_argument("--punch", type=float, nargs="?", const=1.0, metavar="0-1",
                   help="trade up to 2 LU of loudness (never below -11 LUFS) for harder-hitting drums; "
                        "plays just as loud on streaming services")
    return ap


def parse_time_ranges(text: str | None) -> list[tuple[float, float]] | None:
    """'0:45-1:15, 130-160' -> [(45.0, 75.0), (130.0, 160.0)]."""
    if not text:
        return None

    def t(x: str) -> float:
        x = x.strip()
        if ":" in x:
            m, sec = x.split(":", 1)
            return int(m) * 60 + float(sec)
        return float(x)

    out = []
    for part in text.split(","):
        if part.strip():
            a, b = part.split("-", 1)
            a_, b_ = t(a), t(b)
            if b_ <= a_:
                raise ValueError(f"bad range '{part.strip()}': end must be after start")
            out.append((a_, b_))
    return out or None


def check_harmonies(text: str | None) -> None:
    from .dsp.pitch import HARMONY_STEPS

    for iv in (text or "").split(","):
        if iv.strip() and iv.strip() not in HARMONY_STEPS:
            raise ValueError(f"unknown harmony '{iv.strip()}'. choose from: {', '.join(HARMONY_STEPS)}")


def parse_deliver(text: str | None) -> list[str]:
    out = [v.strip().lower() for v in (text or "").split(",") if v.strip()]
    for v in out:
        if v not in DELIVERY_PROFILES:
            raise ValueError(f"unknown delivery version '{v}'. choose from: {', '.join(DELIVERY_PROFILES)}")
    return out


def _ask(text: str | None, preset):
    """Apply plain-words requests and say exactly what they changed."""
    if not text:
        return preset
    from .ai.engineer import EXAMPLES, interpret

    preset, said, unknown = interpret(text, preset)
    for line in said:
        print(f"  engineer: {line}")
    for u in unknown:
        print(f"  engineer: didn't understand \"{u}\" - try e.g.: {EXAMPLES}")
    return preset


def _print_summary(log: dict, out: str) -> None:
    o = log["output"]
    print(f"\ndone in {log['processing_seconds']}s -> {out}/")
    key = log.get("key", {}).get("key")
    print(f"  {o['integrated_lufs']} LUFS | {o['true_peak_dbtp']} dBTP | LRA {o['loudness_range_lu']} LU"
          + (f" | key {key}" if key else ""))
    for f_ in log.get("diagnosis", {}).get("findings", []):
        print(f"  * {f_}")
    for c in log["delivery_check"]:
        print(f"  [{'PASS' if c['ok'] else 'WARN'}] {c['check']}: {c['detail']}")
    for v in log["files"].values():
        print(f"  {v}")
    for n in log.get("notes", []):
        print(f"  NOTE: {n['text']}\n        to fix it, add: --ask \"{n['ask']}\"")
    fan = log.get("previews")
    if fan:
        print("  listen like a fan (how each app plays it):")
        for e in fan["platforms"].values():
            mix = f"   {fan['mix_label'].lower()}: {e['mix']['file']} ({e['mix']['plays_at_lufs']} LUFS)" if "mix" in e else ""
            print(f"    {e['label']:13s} {e['master']['file']} ({e['master']['plays_at_lufs']} LUFS){mix}")


def master_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="studiomix master",
        description="Turn a finished or rough stereo mix into verified, release-ready masters. "
                    "With --ai-remix the AI pulls the vocal out first, so it is re-tuned and re-mixed too.",
        epilog="presets:\n" + "\n".join(f"  {p.name:10s} {p.description}" for p in PRESETS.values()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("mix", help="your stereo mix (wav best; mp3/m4a with ffmpeg)")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("-p", "--preset", default="pop", choices=list(PRESETS))
    ap.add_argument("-n", "--name")
    ap.add_argument("-r", "--reference", help="a released song to match the tone of")
    ap.add_argument("--vocal-lift", type=float, default=0.0, metavar="DB",
                    help="bring the (centre) vocal forward by DB, e.g. 2")
    ap.add_argument("--lufs", dest="target_lufs", type=float)
    ap.add_argument("--ceiling", dest="ceiling_dbtp", type=float)
    ap.add_argument("--width", dest="master_width", type=float)
    ap.add_argument("--punch", type=float, nargs="?", const=1.0, metavar="0-1",
                    help="trade up to 2 LU of loudness for punch (never below -11 LUFS)")
    ap.add_argument("--deliver", metavar="VERSIONS",
                    help="extra verified versions: " + ", ".join(DELIVERY_PROFILES))
    ap.add_argument("--profile", metavar="NAME", help="master toward a learned reference profile")
    ap.add_argument("--no-previews", action="store_true", help="skip the listen-like-a-fan previews")
    ap.add_argument("--ask", metavar="WORDS", help='plain-words changes, e.g. "louder, wider, 808 on phones"')
    ap.add_argument("--lyrics", nargs="?", const="auto", metavar="LANG",
                    help="with --ai-remix: captions + synced lyrics from the AI-separated vocal")
    ap.add_argument("--ai-remix", action="store_true",
                    help="AI: pull the vocal and the beat apart, then re-tune, re-mix and master "
                         "(needs onnxruntime; about real time on a 4-core computer)")
    ap.add_argument("--key", help="song key for the re-tune with --ai-remix, e.g. 'F# minor'")
    ap.add_argument("--best", action="store_true", help="with --ai-remix: slower, marginally cleaner separation")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    from pathlib import Path

    from .engine import master_mix

    try:
        extra = parse_deliver(a.deliver)
        for f_ in [a.mix] + ([a.reference] if a.reference else []):
            if not Path(f_).is_file():
                raise FileNotFoundError(f"file not found: {f_}")
        preset = get_preset(a.preset, **{k: v for k, v in vars(a).items() if v is not None})
        prof = None
        if a.profile:
            from . import profiles

            prof = profiles.load(a.profile)
            preset = profiles.apply(prof, preset, keep_loudness=a.target_lufs is not None)
        preset = _ask(a.ask, preset)
        if a.ai_remix:
            from .engine import ai_remix

            log = ai_remix(a.mix, a.out, preset, name=a.name, verbose=not a.quiet, best=a.best,
                           reference_path=a.reference, ceiling_overridden=a.ceiling_dbtp is not None,
                           deliver_extra=extra, profile=prof, previews=not a.no_previews, key=a.key)
            if a.lyrics:
                from .engine import add_lyrics

                name = a.name or Path(a.mix).stem
                add_lyrics(log, Path(a.out) / log["files"]["ai_vocals"], a.out, name,
                           "" if a.lyrics == "auto" else a.lyrics, say=lambda m: print(f"  - {m}"))
        else:
            log = master_mix(a.mix, a.out, preset, name=a.name, reference_path=a.reference,
                             vocal_lift_db=a.vocal_lift, ceiling_overridden=a.ceiling_dbtp is not None,
                             deliver_extra=extra, verbose=not a.quiet, profile=prof,
                             previews=not a.no_previews)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    _print_summary(log, a.out)
    return 0


def separate_main(argv: list[str]) -> int:
    from .ai import separate as sep

    ap = argparse.ArgumentParser(
        prog="studiomix separate",
        description="AI source separation (MDX-Net models, CPU, downloaded once).\n"
                    + "\n".join(f"  {k:13s} {m.about}" for k, m in sep.MODELS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="songs (or a vocal, for --what dereverb)")
    ap.add_argument("--what", choices=list(sep.MODELS), default="vocals",
                    help="vocals: vocal + instrumental (default); instrumental: cleanest beat; "
                         "dereverb: dry vocal + its reverb")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--best", action="store_true", help="twice as slow, ~0.05 dB cleaner (measured)")
    a = ap.parse_args(argv)
    from pathlib import Path

    from . import audio_io

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        for f_ in a.files:
            x, sr = audio_io.load(f_)
            print(f"{Path(f_).name}: separating ({a.what})...", flush=True)
            for stem, y in sep.separate(x, sr, a.what, fast=not a.best).items():
                p = out / f"{Path(f_).stem}_{stem}.wav"
                audio_io.write_wav(p, y, sr, 24)
                print(f"  {p}")
    except (FileNotFoundError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def lyrics_main(argv: list[str]) -> int:
    from .ai import lyrics

    ap = argparse.ArgumentParser(prog="studiomix lyrics",
                                 description="AI lyrics -> captions (.srt), synced lyrics (.lrc) and text. "
                                             "Best from the vocal on its own; --song separates it first.")
    ap.add_argument("file", help="a vocal (or a full song with --song)")
    ap.add_argument("--song", action="store_true", help="the file is a full song: pull the vocal out first (AI)")
    ap.add_argument("--lang", default="", help="language, e.g. en, fr (default: detect)")
    ap.add_argument("--model", default="small", choices=list(lyrics.SIZES),
                    help="small (default, measured best on sung vocals) / base / tiny (faster, worse)")
    ap.add_argument("-o", "--out", default="out")
    a = ap.parse_args(argv)
    from pathlib import Path

    from . import audio_io

    try:
        x, sr = audio_io.load(a.file)
        if a.song:
            from .ai import separate as sep

            print("pulling the vocal out (AI)...", flush=True)
            x = sep.separate(x, sr, "vocals")["vocals"]
        lines = lyrics.transcribe(x, sr, a.lang, a.model)
    except (FileNotFoundError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = lyrics.write(lines, out / f"{Path(a.file).stem}_lyrics")
    for ln in lines:
        print(f"  [{int(ln['start'] // 60)}:{ln['start'] % 60:05.2f}] {ln['text']}")
    print("a draft to correct - edit the .srt / .lrc in any text editor:")
    for p in files.values():
        print(f"  {p}")
    return 0


def learn_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="studiomix learn",
                                 description="Learn the sound of released songs you like into a reusable profile. "
                                             "Run it again with more songs to add them to the same profile.")
    ap.add_argument("refs", nargs="+", help="reference songs (wav/mp3/m4a...)")
    ap.add_argument("-n", "--name", required=True, help="profile name, e.g. maes")
    a = ap.parse_args(argv)
    from pathlib import Path

    from . import profiles

    missing = [r for r in a.refs if not Path(r).is_file()]
    if missing:
        print(f"error: file not found: {missing[0]}", file=sys.stderr)
        return 1
    try:
        prof = profiles.learn(a.refs, a.name, progress=lambda m: print(f"  - {m}", flush=True))
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(profiles.describe(prof))
    print(f"saved to {profiles.profile_dir() / (prof['name'] + '.json')}")
    print(f"use it with:  --profile {prof['name']}")
    return 0


def profiles_main(argv: list[str]) -> int:
    from . import profiles

    names = profiles.list_profiles()
    if not names:
        print("no profiles yet - create one with: studiomix learn REFS... --name NAME")
        return 0
    for n in (argv or names):
        try:
            print(profiles.describe(profiles.load(n)))
        except KeyError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "learn":
        return learn_main(argv[1:])
    if argv and argv[0] == "profiles":
        return profiles_main(argv[1:])
    if argv and argv[0] == "serve":
        from .web import serve_main

        return serve_main(argv[1:])
    if argv and argv[0] == "master":
        return master_main(argv[1:])
    if argv and argv[0] == "lyrics":
        return lyrics_main(argv[1:])
    if argv and argv[0] == "separate":
        return separate_main(argv[1:])
    if argv and argv[0] == "doctor":
        from .doctor import main as doctor_main

        return doctor_main(argv[1:])
    args = build_parser().parse_args(argv)
    overrides = {}
    if args.tune:
        retune, humanize, amount = TUNE_STYLES[args.tune]
        overrides.update(tune_retune_ms=retune, tune_humanize=humanize, tune_amount=amount)
    overrides.update({k: v for k, v in vars(args).items() if v is not None})
    try:
        preset = get_preset(args.preset, **overrides)
    except (KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    from pathlib import Path

    from .dsp import pitch
    from .engine import run  # heavy imports after argument parsing

    for f in [args.vocal, args.instrumental, *args.adlibs] + ([args.reference] if args.reference else []):
        if not Path(f).is_file():
            print(f"error: file not found: {f}", file=sys.stderr)
            return 1
    try:
        if args.key:
            pitch.parse_key(args.key)
        check_harmonies(args.harmonies)
        stack_at = parse_time_ranges(args.stack_at)
        extra = parse_deliver(args.deliver)
        prof = None
        if args.profile:
            from . import profiles

            prof = profiles.load(args.profile)
            preset = profiles.apply(prof, preset, keep_loudness=args.target_lufs is not None)
        preset = _ask(args.ask, preset)
    except (ValueError, KeyError) as e:
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
            key_changes=args.key_changes, stack_at=stack_at, deliver_extra=extra, profile=prof,
            previews=not args.no_previews,
        )
        if args.lyrics:
            from .engine import add_lyrics

            add_lyrics(log, args.vocal, args.out, args.name or Path(args.vocal).stem,
                       "" if args.lyrics == "auto" else args.lyrics, say=lambda m: print(f"  - {m}"))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    _print_summary(log, args.out)
    return 0

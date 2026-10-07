"""The engineer: say what you want in plain words (English or French), get the settings.

    "more 808, vocal a bit less harsh, way more reverb"
    "voix plus forte, trop de reverb, 808 sur téléphone"

Each phrase is matched to the mix setting a human engineer would reach for, scaled by how
strongly it was said ("a bit" / "un peu" = half, "way more" / "beaucoup" = one and a half), and
complaints are turned around ("too much reverb" / "trop de reverb" = less reverb). It is a
fixed, readable rulebook - no model, works offline on a phone - and every change is listed back
in plain words, so nothing happens behind your back. Phrases it doesn't understand are named.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

from ..presets import Preset, SAFE_RANGES


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c)).replace("’", "'")


SOFT = r"\b(a bit|a little|slightly|bit|little|un peu|legerement|un poil|petit peu)\b"
HARD = r"\b(much|a lot|way|really|very|super|beaucoup|vraiment|tres|encore|grave|max|carrement)\b"
LESS = r"\b(less|lower|down|quieter|softer|reduce|cut|remove|moins|baisse|baisser|reduis|reduire|enleve|retire)\b"
MORE = r"\b(more|louder|up|boost|bigger|add|plus|monte|monter|augmente|ajoute|fort|forte)\b"
TOO = r"\b(too|trop)\b"
NOT_ENOUGH = r"\b(not enough|pas assez|missing|manque)\b"


@dataclass(frozen=True)
class Rule:
    pattern: str            # what the phrase is about
    field: str              # preset field it moves
    step: float             # change for "more" (negative for "less"), absolute when `absolute`
    label: str              # how the change is described
    absolute: bool = False  # set to `step` instead of adding it
    unit: str = " dB"
    default_dir: int = 1    # direction when the phrase says neither more nor less
    problem: bool = False   # the words name a problem ("harsh"): "less harsh", "too harsh" = fix it


# order matters: the first rule whose pattern matches a phrase wins
RULES: list[Rule] = [
    Rule(r"(808|bass|basse|sub)\b.*\b(phone|telephone|portable|speaker|haut-parleur)|"
         r"(phone|telephone|portable).*\b(808|bass|basse)", "bass_harmonics", 0.5,
         "808 audible on phone speakers (bass harmonics)", unit=""),
    Rule(r"\b(808|bass|basse|sub|low end|bas)\b", "inst_bass_db", 2.5, "808 / bass level"),
    Rule(r"\b(kick|drums?|batterie|snare|caisse claire|grosse caisse|hi-?hats?|charley|percs?|percussions?)\b",
         "inst_drums_db", 2.0, "drums level"),
    Rule(r"\b(ad-?libs?|adlibs?)\b", "adlib_level_db", 2.0, "ad-lib level"),
    # whole-mix tone (before the vocal rules, so "the song sounds muffled" is about the mix)
    Rule(r"\b(mix|song|master|morceau|track|overall|global|son|instru\w*|beat|prod)\b.*\b(muffled|dull|etouffe\w*|terne|sourd\w*)\b|\b(muffled|dull|etouffe\w*|terne|sourd\w*)\b.*\b(mix|song|master|morceau|track|overall|global|son|instru\w*|beat|prod)\b",
         "master_tilt_db_oct", 0.5, "overall brightness", unit=" dB/oct", problem=True),
    Rule(r"\b(mix|song|master|morceau|track|overall|global|son|instru\w*|beat|prod)\b.*\b(bright\w*|brillant\w*|clair\w*)\b|\b(bright\w*|brillant\w*|clair\w*)\b.*\b(mix|song|master|morceau|track|overall|global|son|instru\w*|beat|prod)\b",
         "master_tilt_db_oct", 0.5, "overall brightness", unit=" dB/oct"),
    Rule(r"\b(dark\w*|warm\w*|sombre\w*|chaud\w*)\b", "master_tilt_db_oct", -0.5, "overall brightness",
         unit=" dB/oct"),
    Rule(r"\b(harsh|aggressive|agressive?|sibilan\w*|sifflant\w*|piercing|percant\w*|ess|les s)\b",
         "vocal_deess_db", 3.0, "de-essing on the vocal", problem=True),
    Rule(r"\b(dull|muffled|etouffe\w*|sourd\w*|terne)\b", "vocal_air_db", 1.5, "vocal air / brightness",
         problem=True),
    Rule(r"\b(air|bright|brillant\w*|clair\w*)\b", "vocal_air_db", 1.5, "vocal air / brightness"),
    Rule(r"\b(room|bedroom|bathroom|chambre|piece|salle de bain)\b.*\b(echo|reverb\w*|sound|son|resonne)\b|"
         r"\b(echo|reverb\w*|resonne)\b.*\b(room|bedroom|chambre|piece)\b", "vocal_ai_dereverb", 1.0,
         "AI: room echo removed from the vocal", absolute=True, unit=""),
    Rule(r"\b(dry|drier|sec|seche|plus sec)\b", "vocal_reverb", -0.06, "vocal reverb", unit=""),
    Rule(r"\b(reverb|reverbe|reverberation|space|espace|room|salle)\b", "vocal_reverb", 0.06,
         "vocal reverb", unit=""),
    Rule(r"\b(delay|echo|echos)\b", "vocal_delay", 0.05, "vocal delay", unit=""),
    Rule(r"\b(hard|dur|fort)\b.*\b(tune|autotune|auto-tune)\b|\b(tune|autotune|auto-tune)\b.*\b(hard|dur)\b|"
         r"\b(robot\w*|t-?pain)\b", "tune_retune_ms", 0.0,
         "auto-tune: hard, instant snap", absolute=True, unit=" ms"),
    Rule(r"\b(natural|naturel\w*)\b.*\b(tune|autotune|auto-tune)|\b(tune|autotune|auto-tune)\b.*\b(natural|naturel\w*)",
         "tune_retune_ms", 80.0, "auto-tune: natural, slow", absolute=True, unit=" ms"),
    Rule(r"\b(no|sans|pas d'|without|off)\b.*\b(tune|autotune|auto-tune)", "tune_amount", 0.0,
         "auto-tune off", absolute=True, unit=""),
    Rule(r"\b(tune|autotune|auto-tune|justesse)\b", "tune_amount", 0.15, "auto-tune strength", unit=""),
    Rule(r"\b(punch\w*|patate|hit harder|tape plus|qui tape)\b", "punch", 1.0,
         "punch over loudness", absolute=True, unit=""),
    Rule(r"\b(wide|wider|large|stereo|width|largeur)\b", "master_width", 0.15, "stereo width", unit=""),
    Rule(r"\b(narrow|mono|etroit\w*)\b", "master_width", -0.15, "stereo width", unit="", default_dir=1),
    Rule(r"\b(compress\w*|squash\w*|controle|control)\b", "vocal_comp_amount", 0.3, "vocal compression", unit=""),
    Rule(r"\b(dynamic|dynamique|breath|respire)\b", "vocal_comp_amount", -0.3, "vocal compression", unit="",
         default_dir=1),
    Rule(r"\b(double\w*|thick\w*|epais\w*|fat|gras)\b", "doubles", 1.0, "double-tracked lead", absolute=True,
         unit=""),
    Rule(r"\b(noise|noisy|bruit|souffle|hiss|fan|ventilo)\b", "vocal_denoise", 1.0, "noise reduction on",
         absolute=True, unit=""),
    Rule(r"\b(vocal|vocals|voice|voix|vox|lead|chant)\b", "vocal_balance_db", 1.5, "lead vocal vs. beat"),
    Rule(r"\b(loud\w*|quiet\w*|radio|volume|fort)\b", "target_lufs", 1.0, "master loudness", unit=" LU"),
]


def _strength(t: str) -> float:
    return 0.5 if re.search(SOFT, t) else 1.5 if re.search(HARD, t) else 1.0


def _direction(t: str, rule: Rule) -> int:
    less, more = re.search(LESS, t), re.search(MORE, t)
    if rule.problem:  # "harsh", "less harsh", "too harsh" -> fix; "not harsh enough" / "more aggressive" -> back off
        return -1 if re.search(NOT_ENOUGH, t) or (more and not less and not re.search(TOO, t)) else 1
    if re.search(NOT_ENOUGH, t):
        return 1
    if re.search(TOO, t):  # "too much reverb", "voix trop forte"
        return -1
    if less and not more:
        return -1
    if more and not less:
        return 1
    if less and more:  # both words: whichever comes first decides
        return -1 if less.start() < more.start() else 1
    return rule.default_dir


def interpret(text: str, preset: Preset) -> tuple[Preset, list[str], list[str]]:
    """Apply plain-words requests to a preset. Returns (new preset, changes made, phrases not understood)."""
    changes: dict = {}
    said: list[str] = []
    unknown: list[str] = []
    for raw in re.split(r"[,;.!\n]+|\b(?:and|et|puis|then|also|aussi|but|mais)\b", text):
        phrase = raw.strip() if raw else ""
        if not phrase:
            continue
        t = _norm(phrase)
        rule = next((r for r in RULES if re.search(r.pattern, t)), None)
        if rule is None:
            unknown.append(phrase)
            continue
        cur = changes.get(rule.field, getattr(preset, rule.field))
        if rule.absolute:
            if rule.field == "tune_retune_ms" and rule.step == 0.0:
                changes.update(tune_amount=1.0, tune_humanize=0.0)
            new = rule.step if not isinstance(cur, bool) else bool(rule.step)
        else:
            k = _strength(t) * _direction(t, rule)
            new = cur + rule.step * k
            if rule.field == "vocal_deess_db" and k > 0:  # harsh: also less presence boost
                changes["vocal_presence_db"] = changes.get("vocal_presence_db", preset.vocal_presence_db) - 1.0 * k
            if rule.field == "vocal_air_db" and rule.label.startswith("vocal air") and re.search(
                    r"dull|muffled|etouffe|sourd|terne", t):
                changes["vocal_presence_db"] = changes.get("vocal_presence_db", preset.vocal_presence_db) + 1.0 * k
        if rule.field in SAFE_RANGES:
            lo, hi, _ = SAFE_RANGES[rule.field]
            new = min(hi, max(lo, new))
        if rule.field in ("vocal_reverb", "vocal_delay", "bass_harmonics"):
            new = min(1.0, max(0.0, new))
        if rule.field == "target_lufs":
            new = min(-7.5, new)
        if isinstance(new, float):
            new = round(new, 3)
        changes[rule.field] = new
        if isinstance(new, bool) or rule.absolute:
            said.append(f"{rule.label}  ← \"{phrase}\"")
        else:
            delta = new - cur
            said.append(f"{rule.label}: {'+' if delta >= 0 else ''}{round(delta, 2)}{rule.unit} "
                        f"(now {round(new, 2)}{rule.unit})  ← \"{phrase}\"")
    return replace(preset, **changes), said, unknown


EXAMPLES = ("more 808 · 808 on phones · vocal louder · less harsh · more reverb · dry · hard autotune · "
            "natural autotune · ad-libs quieter · wider · punchier · louder · darker · doubles · "
            "voix plus forte · trop de reverb · un peu plus de 808")

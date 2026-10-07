"""Engineer notes: what a mix engineer would point out after hearing the result.

Every note is measured on the files that were written (master and stems), says where in the song
it happens, and comes with the plain-words request that fixes it (see engineer.py), so a note can
be applied with one tap. Only problems that can be measured reliably are reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import audio_io
from ..dsp import analysis, filters


def _t(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def _ranges(mask: np.ndarray, hop_s: float, min_len: int = 2) -> list[tuple[float, float]]:
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start * hop_s, i * hop_s))
            start = None
    return out


def phone_bass(master: np.ndarray, sr: int) -> tuple[float, float]:
    """(share of the sub-bass in the full-range master, share of the sub-bass in what a phone
    speaker plays), dB. A phone plays nothing near an 808's fundamental: a bass-heavy song can
    have most of its low end there and still have it 35-40 dB under everything else on a phone."""
    from ..previews import phone_speaker

    sub = filters.lowpass(master, sr, 120.0, order=4)
    full = 10 * np.log10(np.sum(sub ** 2) / (np.sum(master ** 2) + 1e-20) + 1e-20)
    ps, pm = phone_speaker(sub, sr), phone_speaker(master, sr)
    phone = 10 * np.log10(np.sum(ps ** 2) / (np.sum(pm ** 2) + 1e-20) + 1e-20)
    return float(full), float(phone)


def make(out_dir: Path, files: dict, preset, output: dict) -> list[dict]:
    notes: list[dict] = []
    master, sr = audio_io.load(Path(out_dir) / files["master_24bit"])

    # 1. the 808 on phone speakers. The fix (bass harmonics) can't be re-measured on the finished
    # master - the overtones it adds sit where vocals and chords also are - so the note is about the
    # physical fact (where the low end sits) and stays quiet once the fix is on. Measured where it
    # can be isolated, on a beat: the 808's share of what a phone plays goes from -35.6 to -12.9 dB.
    full, phone = phone_bass(master, sr)
    if full > -10.0 and phone < -20.0 and preset.bass_harmonics < 0.3:
        notes.append({"text": f"Most of your low end is sub-bass, which a phone speaker can't play: there it sits "
                              f"{-phone:.0f} dB under everything else, so the 808 vanishes on TikTok / Reels. Bass "
                              f"harmonics add the 808's overtones, which a phone can play.", "ask": "808 on phones"})

    # 2. the lead vocal sinking under the beat in places (3 s loudness, 1 s steps)
    if "vocal_stem" in files and "instrumental_stem" in files:
        v, _ = audio_io.load(Path(out_dir) / files["vocal_stem"])
        b, _ = audio_io.load(Path(out_dir) / files["instrumental_stem"])
        n = min(v.shape[-1], b.shape[-1])
        sv, sb = analysis.short_term_lufs(v[:, :n], sr, 3.0, 1.0), analysis.short_term_lufs(b[:, :n], sr, 3.0, 1.0)
        m = min(len(sv), len(sb))
        sv, sb = sv[:m], sb[:m]
        sing = sv > max(np.percentile(sv, 90) - 15.0, -50.0)
        if sing.sum() >= 6:
            diff = sv - sb
            med = float(np.median(diff[sing]))
            low = sing & (diff < med - 3.0)
            spots = _ranges(low, 1.0, 3)
            if spots:
                worst = min(spots, key=lambda r: float(np.min(diff[int(r[0]):int(r[1])])))
                drop = med - float(np.min(diff[int(worst[0]):int(worst[1])]))
                where = ", ".join(f"{_t(a)}-{_t(b_)}" for a, b_ in spots[:3])
                notes.append({"text": f"The vocal sinks under the beat at {where} (up to {drop:.1f} dB lower than in the "
                                      f"rest of the song) - the beat gets busier there.", "ask": "vocal louder"})

        # 3. ad-libs covering the lead
        if "adlib_stem" in files:
            a, _ = audio_io.load(Path(out_dir) / files["adlib_stem"])
            k = min(a.shape[-1], n)
            sa = analysis.short_term_lufs(a[:, :k], sr, 0.4, 0.2)
            sl = analysis.short_term_lufs(v[:, :k], sr, 0.4, 0.2)
            j = min(len(sa), len(sl))
            over = (sa[:j] > sl[:j] - 1.0) & (sl[:j] > np.percentile(sl[:j], 90) - 15.0)
            spots = _ranges(over, 0.2, 5)
            secs = sum(b_ - a_ for a_, b_ in spots)
            if secs >= 2.0:
                notes.append({"text": f"The ad-libs cover the lead for {secs:.0f} s (first at {_t(spots[0][0])}) - "
                                      f"the lead's words get lost there.", "ask": "ad-libs quieter"})

    # 4. dynamics
    if output.get("plr_db", 99) < 7.0 and preset.punch == 0:
        notes.append({"text": f"The master is very squashed (PLR {output['plr_db']} dB). Streaming plays every song "
                              f"at the same level, so trading 2 dB of loudness for punch costs nothing there.",
                      "ask": "punchier"})
    return notes

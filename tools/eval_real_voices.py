"""Evaluate the pitch tracker and tuner on REAL voices (complements tools/bench_tune.py).

    pip install librosa praat-parselmouth        # reference trackers (dev only)
    python tools/eval_real_voices.py [workdir]

Fetches a few CC-BY recordings from the librosa example-data repository (LibriSpeech speech,
a solo trumpet) with git, then:

1. Tracker vs. consensus: frames where librosa's pYIN and Praat agree (within 50 cents) are
   treated as truth; reports agreement, voicing recall and false voicing.
2. Real-voice retune: each voiced phrase of real speech is re-pitched to a scale note with
   +-40 cents error and vibrato (a real timbre singing out of tune), then hard/pop tuned.
   Results are measured with Praat (not our own tracker) against the intended notes.

Audio is downloaded to the work dir only; it is never committed (licences: see the .toml files).
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studiomix import audio_io  # noqa: E402
from studiomix.dsp import pitch  # noqa: E402

warnings.filterwarnings("ignore")
SR = 48000
FILES = ["198-209-0000", "3436-172162-0000", "5703-47212-0000", "sorohanro_-_solo-trumpet-06"]
A_MINOR = [9, 11, 0, 2, 4, 5, 7]


def fetch(work: Path) -> dict[str, np.ndarray]:
    repo = work / "librosa-data"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout",
                        "https://github.com/librosa/data", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "--no-cone"]
                   + [f"/audio/{f}.hq.ogg" for f in FILES] + [f"/audio/{f}.toml" for f in FILES], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "HEAD"], check=True)
    out = {}
    for f in FILES:
        x, sr = audio_io.load(repo / "audio" / f"{f}.hq.ogg")
        out[f] = audio_io.resample(x.mean(0)[None], sr, SR)[0]
    return out


def references(x: np.ndarray, t: np.ndarray):
    import librosa
    import parselmouth

    y = librosa.resample(x, orig_sr=SR, target_sr=22050)
    f0, vflag, _ = librosa.pyin(y, fmin=65, fmax=1100, sr=22050, frame_length=2048, hop_length=256)
    tl = librosa.times_like(f0, sr=22050, hop_length=256)
    lp = np.interp(t, tl, np.where(vflag, pitch.hz_to_midi(np.nan_to_num(f0, nan=1)), np.nan), left=np.nan, right=np.nan)
    lpv = np.interp(t, tl, vflag.astype(float)) > 0.5
    pr_t, pr_m = praat(x)
    pr = np.interp(t, pr_t, pr_m, left=np.nan, right=np.nan)
    prv = np.isfinite(pr)
    return lp, lpv, pr, prv


def praat(x: np.ndarray):
    import parselmouth

    pp = parselmouth.Sound(x, sampling_frequency=SR).to_pitch(time_step=0.005, pitch_floor=65, pitch_ceiling=1100)
    f = pp.selected_array["frequency"]
    return pp.xs(), np.where(f > 0, pitch.hz_to_midi(np.maximum(f, 1)), np.nan)


def tracker_eval(name: str, x: np.ndarray) -> None:
    trk = pitch.track(x, SR)
    lp, lpv, pr, prv = references(x, trk["times"])
    cons = lpv & prv & (np.abs(lp - pr) < 0.5)
    err = np.abs(trk["midi"] - (lp + pr) / 2)
    mv = trk["voiced"]
    print(f"  {name[:28]:28s} within 50c {np.mean((err < 0.5)[cons & mv]) * 100:5.1f}%  "
          f"recall {np.mean(mv[cons]) * 100:5.1f}%  false-voiced {np.mean(mv[~lpv & ~prv]) * 100:4.1f}%  "
          f"| pYIN~Praat agree {np.mean((np.abs(lp - pr) < 0.5)[lpv & prv]) * 100:5.1f}%")


def retune_eval(name: str, x: np.ndarray) -> None:
    trk = pitch.track(x, SR)
    rng = np.random.default_rng(1)
    med = np.nanmedian(trk["midi"])
    avail = [m for m in range(int(med) - 4, int(med) + 6) if m % 12 in A_MINOR]
    v = trk["voiced"]
    edges = np.flatnonzero(np.diff(np.concatenate([[0], v.astype(np.int8), [0]])))
    sung_m = np.full(len(v), np.nan)
    intended = {}
    for a, b in zip(edges[::2], edges[1::2]):
        note = avail[rng.integers(len(avail))]
        tt = np.arange(b - a) * trk["hop"] / SR
        sung_m[a:b] = note + rng.uniform(-0.4, 0.4) + 0.3 * np.sin(2 * np.pi * 5.5 * tt) * np.clip((tt - 0.2) / 0.2, 0, 1)
        intended[a] = (b, note)
    ratio = np.clip(np.nan_to_num(np.where(v, 2 ** ((sung_m - trk["midi"]) / 12), 1.0), nan=1.0), 0.67, 1.5)
    f0 = np.where(v, pitch.midi_to_hz(np.nan_to_num(trk["midi"], nan=60)), np.nan)
    sung = pitch.psola(x, SR, f0, ratio, trk["hop"])

    def errors(sig):
        tp, pm = praat(sig)
        res = {}
        for a, (b, note) in intended.items():
            if b - a < 30:
                continue
            t0, t1 = (a + 0.3 * (b - a)) * trk["hop"] / SR, (b - 0.2 * (b - a)) * trk["hop"] / SR
            seg = pm[(tp >= t0) & (tp <= t1)]
            seg = seg[np.isfinite(seg)]
            if len(seg) > 5:
                res[a] = (np.median(seg) - note) * 100
        return res

    e_in = errors(sung)
    for style, rt, hz in (("hard", 0, 0.0), ("pop", 25, 0.4)):
        out, _ = pitch.autotune(sung[None], SR, 9, "minor", rt, hz, 1.0)
        e_out = errors(out[0])
        valid = [a for a in e_in if abs(e_in[a]) < 50 and a in e_out]  # where the melody was really imposed
        ei = np.abs([e_in[a] for a in valid])
        eo = np.abs([e_out[a] for a in valid])
        print(f"  {name[:16]:16s} {style:4s} notes {len(valid):3d} | in tune (<10c): before {np.mean(ei < 10) * 100:5.1f}% "
              f"after {np.mean(eo < 10) * 100:5.1f}% | mean err {ei.mean():5.1f}c -> {eo.mean():5.1f}c, worst {eo.max():5.1f}c")


def main() -> None:
    work = Path(sys.argv[1] if len(sys.argv) > 1 else "eval_data")
    work.mkdir(parents=True, exist_ok=True)
    audio = fetch(work)
    print("1. tracker vs. pYIN+Praat consensus (real recordings)")
    for name, x in audio.items():
        tracker_eval(name, x)
    print("2. real voice sung out of tune -> tuned (measured with Praat)")
    for name in FILES[:3]:
        retune_eval(name, audio[name])


if __name__ == "__main__":
    main()

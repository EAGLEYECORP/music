"""Generate a synthetic vocal + instrumental pair for trying out / testing studiomix.

    python tools/make_demo.py demo/

The 'vocal' is a formant-synthesised singer with deliberately amateur problems: uneven phrase
levels, harsh sibilants, low-frequency rumble and room noise. The beat is a simple 92 BPM loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def beat(sr: int = 44100, seconds: float = 30.0, bpm: float = 92.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    out = np.zeros((2, n))
    spb = 60.0 / bpm
    t = np.arange(int(sr * 0.5)) / sr

    kick = np.sin(2 * np.pi * (45 + 120 * np.exp(-t * 30)) * t) * np.exp(-t * 7)
    snare = (rng.standard_normal(len(t)) * np.exp(-t * 18) * 0.5
             + np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25) * 0.5)
    snare = signal.sosfilt(signal.butter(2, 900, "highpass", fs=sr, output="sos"), snare)
    hat = signal.sosfilt(signal.butter(4, 7000, "highpass", fs=sr, output="sos"),
                         rng.standard_normal(len(t)) * np.exp(-t * 60))

    def place(sample, time, gain, pan=0.0):
        i = int(time * sr)
        if i >= n:
            return
        m = min(len(sample), n - i)
        out[0, i:i + m] += sample[:m] * gain * (1 - pan)
        out[1, i:i + m] += sample[:m] * gain * (1 + pan)

    chords = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (196.0, 246.94, 293.66), (164.81, 196.0, 246.94)]
    bass_notes = [55.0, 43.65, 49.0, 41.2]
    beats = int(seconds / spb)
    for b in range(beats):
        time = b * spb
        if b % 4 in (0, 2) or (b % 8 == 7):
            place(kick, time, 0.9)
        if b % 4 in (1, 3):
            place(snare, time, 0.55)
        for h in range(2):
            place(hat, time + h * spb / 2, 0.18 if h == 0 else 0.12, pan=0.3)
        if b % 4 == 0:
            bar = (b // 4) % 4
            dur = int(spb * 4 * sr)
            tt = np.arange(dur) / sr
            env = np.minimum(1, tt * 20) * np.exp(-tt * 0.4)
            pad = sum(signal.sawtooth(2 * np.pi * f * tt * (1 + d)) for f in chords[bar] for d in (-0.003, 0.003))
            pad = signal.sosfilt(signal.butter(2, 2500, "lowpass", fs=sr, output="sos"), pad) * env * 0.05
            bass = np.sin(2 * np.pi * bass_notes[bar] * tt) * np.minimum(1, tt * 50) * 0.35
            i = int(time * sr)
            m = min(dur, n - i)
            out[0, i:i + m] += pad[:m] * 1.2 + bass[:m]
            out[1, i:i + m] += pad[:m] * 0.8 + bass[:m]
    return out / np.max(np.abs(out)) * 0.7


def vocal(sr: int = 48000, seconds: float = 30.0, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    out = np.zeros(n)
    formants = {"a": (800, 1150, 2900), "e": (400, 2000, 2600), "o": (450, 800, 2830), "i": (300, 2300, 3000)}
    notes = [220.0, 246.94, 261.63, 293.66, 329.63, 293.66, 261.63, 246.94]
    phrase_gain = [1.0, 0.35, 1.4, 0.6, 1.1, 0.45]  # very uneven performance
    t = 1.5
    k = 0
    while t < seconds - 4:
        g = phrase_gain[k % len(phrase_gain)]
        for _ in range(6):
            dur = rng.uniform(0.25, 0.6)
            m = int(dur * sr)
            tt = np.arange(m) / sr
            f0 = notes[rng.integers(len(notes))] * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * tt))
            src = signal.sawtooth(2 * np.pi * np.cumsum(f0) / sr, 0.1)
            v = np.zeros(m)
            for f, bw in zip(formants["aeoi"[rng.integers(4)]], (80, 100, 140)):
                b, a = signal.iirpeak(f, f / bw, fs=sr)
                v += signal.lfilter(b, a, src)
            v *= np.minimum(1, tt * 25) * np.minimum(1, (dur - tt) * 15)
            i = int(t * sr)
            m = min(m, n - i)
            out[i:i + m] += v[:m] * g * 0.12
            if rng.random() < 0.4:  # harsh "s"
                s = signal.sosfilt(signal.butter(4, [5500, 9500], "bandpass", fs=sr, output="sos"),
                                   rng.standard_normal(int(0.12 * sr)))
                s *= np.hanning(len(s))
                j = i + m
                k_s = max(0, min(len(s), n - j))
                out[j:j + k_s] += s[:k_s] * 0.25 * g
            t += dur + 0.12
        t += 0.8
        k += 1
    rumble = signal.sosfilt(signal.butter(2, 60, "lowpass", fs=sr, output="sos"), rng.standard_normal(n)) * 0.05
    hiss = rng.standard_normal(n) * 0.0015
    out = out + rumble + hiss
    return (out / np.max(np.abs(out)) * 0.5)[None, :]


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "demo")
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / "demo_vocal.wav", vocal().T, 48000, subtype="PCM_24")
    sf.write(d / "demo_beat.wav", beat().T, 44100, subtype="PCM_16")
    print(f"wrote {d}/demo_vocal.wav and {d}/demo_beat.wav")


if __name__ == "__main__":
    main()

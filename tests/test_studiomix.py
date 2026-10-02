import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import make_demo  # noqa: E402
from studiomix import audio_io, chains  # noqa: E402
from studiomix.dsp import analysis, dynamics, filters, pitch  # noqa: E402
from studiomix.presets import PRESETS, get_preset  # noqa: E402

SR = 48000


def noise(seconds=5.0, level=0.1, ch=2, seed=0):
    return np.random.default_rng(seed).standard_normal((ch, int(SR * seconds))) * level


def sung(freq_hz, sr=SR):
    """Formant-filtered sawtooth 'voice' following a per-sample frequency curve."""
    src = signal.sawtooth(2 * np.pi * np.cumsum(freq_hz) / sr, 0.1)
    v = 0
    for fc, bw in ((700, 80), (1200, 100), (2700, 140)):
        b, a = signal.iirpeak(fc, fc / bw, fs=sr)
        v = v + signal.lfilter(b, a, src)
    return v / np.max(np.abs(v)) * 0.5


# ------------------------------------------------------------------ dsp


def test_three_band_split_is_magnitude_flat():
    imp = np.zeros((1, 8192))
    imp[0, 0] = 1.0
    low, mid, high = filters.three_band_split(imp, SR, 120.0, 4000.0)
    mag = np.abs(np.fft.rfft((low + mid + high)[0]))
    assert np.max(np.abs(20 * np.log10(mag[1:]))) < 0.05


def test_limiter_respects_true_peak_ceiling():
    x = noise(level=0.5)
    x[:, 10000:10010] = 3.0  # nasty spikes
    y, gr = dynamics.limit(x, SR, ceiling_db=-1.0)
    assert dynamics.true_peak_db(y) <= -1.0
    assert gr.min() < -6.0


def test_compressor_static_ratio():
    t = np.arange(SR * 2) / SR
    x = np.vstack([np.sin(2 * np.pi * 1000 * t)] * 2) * 10 ** (-6 / 20)  # sine peaking at -6 dBFS
    y, gr = dynamics.compress(x, SR, threshold_db=-20.0, ratio=4.0, attack_ms=1, release_ms=50, knee_db=0)
    # detector reads a sine at its peak level: 14 dB over -> 10.5 dB reduction
    assert gr[-1000:].mean() == pytest.approx(-10.5, abs=0.3)


def test_finalize_hits_loudness_target():
    x = noise(10.0, 0.05)
    for target in (-14.0, -9.0):
        y = chains.finalize_loudness(x, SR, target, -1.0, 2.0, 80.0, {})
        assert analysis.integrated_lufs(y, SR) == pytest.approx(target, abs=0.15)
        assert dynamics.true_peak_db(y) <= -1.0


def test_lufs_reference_sine():
    t = np.arange(SR * 5) / SR
    x = np.vstack([np.sin(2 * np.pi * 997 * t)] * 2) * 10 ** (-23 / 20)
    # BS.1770: a stereo 997 Hz sine at -23 dBFS peak per channel reads about -23 LUFS
    assert analysis.integrated_lufs(x, SR) == pytest.approx(-23.0, abs=0.3)


# ------------------------------------------------------------------ file io


@pytest.mark.parametrize("bits,tol", [(16, 2 / 32768), (24, 2 / 8388608)])
def test_wav_roundtrip(tmp_path, bits, tol):
    x = noise(1.0, 0.2)
    p = tmp_path / "x.wav"
    audio_io.write_wav(p, x, SR, bits)
    y, sr = audio_io.load(p)
    assert sr == SR and y.shape == x.shape
    assert np.max(np.abs(y - x)) < tol


def test_dithered_16bit_roundtrip(tmp_path):
    x = noise(1.0, 0.2)
    p = tmp_path / "x.wav"
    audio_io.write_wav16_dithered(p, x, SR)
    y, sr = audio_io.load(p)
    assert sr == SR
    assert np.max(np.abs(y - x)) < 3 / 32768


def test_wav_readable_by_soundfile(tmp_path):
    sf = pytest.importorskip("soundfile")
    p = tmp_path / "x.wav"
    audio_io.write_wav(p, noise(0.5, 0.2), SR, 24)
    assert sf.info(str(p)).subtype == "PCM_24"


# ------------------------------------------------------------------ pitch


def test_psola_is_transparent_without_correction():
    t = np.arange(SR * 2) / SR
    x = sung(220 * (1 + 0.004 * np.sin(2 * np.pi * 5 * t)))
    trk = pitch.track(x, SR)
    f0 = np.where(trk["voiced"], pitch.midi_to_hz(np.nan_to_num(trk["midi"], nan=60)), np.nan)
    y = pitch.psola(x, SR, f0, np.ones(len(f0)), trk["hop"])
    assert np.max(np.abs(y - x)) < 1e-9


def test_autotune_snaps_to_scale():
    t = np.arange(int(SR * 3)) / SR
    # A3 sung 35 cents sharp, then a note 40 cents above D4
    f = np.where(t < 1.5, 220 * 2 ** (0.35 / 12), 293.66 * 2 ** (0.4 / 12))
    y, stats = pitch.autotune(sung(f)[None, :], SR, 0, "major", retune_ms=0, humanize=0, amount=1.0)
    out = pitch.track(y[0], SR)
    assert np.nanmedian(out["midi"][20:250]) == pytest.approx(57.0, abs=0.05)  # A3
    assert np.nanmedian(out["midi"][350:550]) == pytest.approx(62.0, abs=0.05)  # D4
    assert stats["avg_correction_cents"] > 30


def test_humanize_keeps_vibrato_hard_tune_removes_it():
    t = np.arange(SR * 3) / SR
    x = sung(220 * 2 ** (0.3 / 12) * 2 ** (0.5 * np.sin(2 * np.pi * 5.5 * t) / 12))  # +-50 cent vibrato
    spread = {}
    for name, humanize in (("hard", 0.0), ("human", 1.0)):
        y, _ = pitch.autotune(x[None, :], SR, 9, "minor", retune_ms=0, humanize=humanize, amount=1.0)
        m = pitch.track(y[0], SR)["midi"][50:-50]
        spread[name] = np.nanstd(m)
        assert np.nanmedian(m) == pytest.approx(57.0, abs=0.1)
    assert spread["hard"] < 0.1 < spread["human"]


@pytest.mark.parametrize("text,expected", [
    ("F# minor", (6, "minor")), ("Bbm", (10, "minor")), ("c major", (0, "major")),
    ("Eb", (3, "major")), ("A min", (9, "minor")), ("G minor-pentatonic", (7, "minor-pentatonic")),
])
def test_parse_key(text, expected):
    assert pitch.parse_key(text) == expected


def test_parse_key_rejects_garbage():
    with pytest.raises(ValueError):
        pitch.parse_key("H dorian-ish")


def test_detects_key_of_demo_beat():
    tonic, mode, conf = pitch.detect_key(pitch.chroma(make_demo.beat(44100, 20.0), 44100))
    assert (tonic, mode) in {(9, "minor"), (0, "major")}  # A minor / relative C major: same notes
    assert conf > 0.6


# ------------------------------------------------------------------ mix features


def test_adlib_phrases_alternate_sides():
    n = SR * 6
    v = np.zeros((1, n))
    active = np.zeros(n, dtype=bool)
    for k in range(4):
        a = int(SR * (0.5 + 1.4 * k))
        v[0, a:a + SR // 2] = noise(0.5, 0.2, 1, seed=k)[0]
        active[a:a + SR // 2] = True
    st = chains.pan_phrases(v, SR, active, 0.6)
    sides = []
    for k in range(4):
        a = int(SR * (0.5 + 1.4 * k)) + SR // 4
        seg = st[:, a:a + 1000]
        sides.append(np.sign(np.sum(seg[1] ** 2) - np.sum(seg[0] ** 2)))
    assert sides == [-1, 1, -1, 1]


def test_preset_overrides():
    p = get_preset("hiphop", target_lufs=-12.0, vocal_reverb=None)
    assert p.target_lufs == -12.0 and p.vocal_reverb == PRESETS["hiphop"].vocal_reverb
    with pytest.raises(KeyError):
        get_preset("polka")


# ------------------------------------------------------------------ end to end


@pytest.fixture(scope="module")
def demo_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    audio_io.write_wav(d / "v.wav", make_demo.vocal(48000, 12.0), 48000)
    audio_io.write_wav(d / "a.wav", make_demo.vocal(48000, 12.0, seed=5, adlib=True), 48000)
    audio_io.write_wav(d / "b.wav", make_demo.beat(44100, 12.0), 44100, 16)
    return d


def test_end_to_end_with_adlibs(demo_files, tmp_path):
    from studiomix.engine import run

    d = demo_files
    log = run(d / "v.wav", d / "b.wav", tmp_path / "out", get_preset("pop"), name="song",
              adlib_paths=[d / "a.wav"], verbose=False)
    out = log["output"]
    assert out["integrated_lufs"] == pytest.approx(-11.0, abs=0.2)
    assert out["true_peak_dbtp"] <= log["master"]["ceiling_dbtp"] + 0.01
    assert all(c["ok"] for c in log["delivery_check"][:3])
    assert log["key"]["key"] in ("A minor", "C major")
    assert log["vocal"]["autotune"]["avg_correction_cents"] > 5
    assert "adlib_stem" in log["files"]
    for f in log["files"].values():
        assert (tmp_path / "out" / f).exists()
    m16, sr16 = audio_io.load(tmp_path / "out" / log["files"]["master_16bit_cd"])
    assert sr16 == 44100 and dynamics.true_peak_db(m16) <= log["master"]["ceiling_dbtp"] + 0.05
    assert json.loads((tmp_path / "out" / "song_report.json").read_text())["preset"] == "pop"


def test_runs_without_numba_pedalboard_or_soundfile(demo_files, tmp_path):
    """What Termux/Android has: numpy + scipy + pyloudnorm only."""
    d = demo_files
    script = (
        "import sys\n"
        "for m in ('numba', 'llvmlite', 'pedalboard', 'soundfile'): sys.modules[m] = None\n"
        "from studiomix.dsp import _jit; assert not _jit.HAS_NUMBA\n"
        "from studiomix.cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", script, str(d / "v.wav"), str(d / "b.wav"), "-a", str(d / "a.wav"),
         "-p", "trap", "-o", str(tmp_path), "-n", "t", "-q", "--no-stems"],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    log = json.loads((tmp_path / "t_report.json").read_text())
    assert log["output"]["integrated_lufs"] == pytest.approx(-8.5, abs=0.2)
    assert log["output"]["true_peak_dbtp"] <= -2.0 + 0.01


@pytest.mark.parametrize("case", ["high", "breathy", "bass"])
def test_tuner_benchmark_regression(case):
    """Guards the measured accuracy of the tracker + note decisions + PSOLA (tools/bench_tune.py)."""
    import bench_tune

    spec = {c[0].split()[0]: c for c in bench_tune.CASES}[case]
    r = bench_tune.run_case(*spec)
    assert r["track_gross_%"] < 2.0
    assert r["notes_ok_%"] >= 90.0
    assert r["note_err_c"] < 3.0


def test_tracker_is_time_aligned_on_fast_slides():
    """Pitch estimates must describe the audio at their own timestamps (no ~5 ms lag)."""
    t = np.arange(int(SR * 1.5)) / SR
    midi_true = 50 + 24 * np.clip((t - 0.5) / 0.5, 0, 1)  # two-octave slide in 0.5 s
    x = sung(pitch.midi_to_hz(midi_true))
    trk = pitch.track(x, SR)
    i = (trk["times"] > 0.55) & (trk["times"] < 0.95) & trk["voiced"]
    lags = np.arange(-10, 10.5, 0.5) / 1000
    err = [np.median(np.abs(trk["midi"][i] - np.interp(trk["times"][i] + lag, t, midi_true))) for lag in lags]
    assert abs(lags[int(np.argmin(err))]) <= 0.0015


def test_short_onset_notes_are_merged():
    seq = np.array([62.0] * 5 + [60.0] * 40 + [64.0] * 30)  # 25 ms blip at the onset
    m = np.concatenate([np.full(5, 61.2), np.full(40, 60.2), np.full(30, 64.1)])
    out = pitch._merge_short_notes(seq, m, min_frames=12)
    assert (out[:45] == 60.0).all() and (out[45:] == 64.0).all()


# ------------------------------------------------------------------ new features


@pytest.fixture(scope="module")
def tuned_two_notes():
    t = np.arange(SR * 3) / SR
    x = sung(np.where(t < 1.5, pitch.midi_to_hz(57.2), pitch.midi_to_hz(60.15)))  # A3, C4 (a bit sharp)
    cap = {}
    y, _ = pitch.autotune(x[None], SR, 0, "major", 0, 0, 1.0, capture=cap)
    return y, cap


@pytest.mark.parametrize("interval,expect_a3,expect_c4", [
    ("3up", 60, 64), ("3down", 53, 57), ("5up", 64, 67), ("5down", 50, 53), ("8up", 69, 72), ("8down", 45, 48),
])
def test_harmony_intervals_follow_the_key(tuned_two_notes, interval, expect_a3, expect_c4):
    y, cap = tuned_two_notes
    h = pitch.harmony(y, SR, cap, interval)
    # measure each half within +-5 semitones of the expected note (verified against Praat too;
    # our own tracker can octave-jump on heavily overlapped PSOLA grains, which is unrelated)
    for sl, exp in ((slice(0, SR * 3 // 2), expect_a3), (slice(SR * 3 // 2, SR * 3), expect_c4)):
        f = float(pitch.midi_to_hz(exp))
        tr = pitch.track(h[0, sl], SR, fmin=f / 1.33, fmax=f * 1.33, adapt=False)
        assert np.nanmedian(tr["midi"][60:230]) == pytest.approx(exp, abs=0.1)


def test_double_is_late_and_close_in_pitch(tuned_two_notes):
    y, cap = tuned_two_notes
    d = pitch.double(y, SR, cap, seed=3)
    tr = pitch.track(d[0], SR)
    assert np.nanmedian(tr["midi"][60:250]) == pytest.approx(57.0, abs=0.12)
    # onset of the double comes 5-22 ms after the lead's
    on = lambda z: np.argmax(np.abs(z) > 0.05 * np.max(np.abs(z)))  # noqa: E731
    lag_ms = (on(d[0]) - on(y[0])) / SR * 1000
    assert 4.0 <= lag_ms <= 23.0


def test_flex_tune_leaves_intentional_bends():
    t = np.arange(SR * 3) / SR
    x = sung(np.where(t < 1.5, pitch.midi_to_hz(57.25), pitch.midi_to_hz(57.8)))
    y, _ = pitch.autotune(x[None], SR, 0, "major", 0, 0, 1.0, flex_cents=35)
    tr = pitch.track(y[0], SR)
    assert np.nanmedian(tr["midi"][60:250]) == pytest.approx(57.0, abs=0.05)   # 25c off: corrected
    assert np.nanmedian(tr["midi"][360:560]) == pytest.approx(57.8, abs=0.05)  # 80c bend: kept


def test_key_change_detection():
    a = make_demo.beat(44100, 30.0)
    b = audio_io.resample(make_demo.beat(44100, 36.0), int(round(44100 * 2 ** (3 / 12))), 44100)[:, : 44100 * 30]
    secs = pitch.detect_key_sections(np.concatenate([a, b], axis=1), 44100)
    assert [(s[2], s[3]) for s in secs] == [(9, "minor"), (0, "minor")]
    assert abs(secs[0][1] - 30.0) < 3.0
    assert len(pitch.detect_key_sections(a, 44100)) == 1


def test_time_ranges_and_section_mask():
    from studiomix.cli import parse_time_ranges

    assert parse_time_ranges("0:45-1:15, 130-160") == [(45.0, 75.0), (130.0, 160.0)]
    with pytest.raises(ValueError):
        parse_time_ranges("1:00-0:30")
    m = chains.section_mask(SR * 10, SR, [(2.0, 4.0)])
    assert m[int(SR * 1)] == 0 and m[int(SR * 3)] == 1 and m[int(SR * 6)] == 0


def test_end_to_end_with_stack(demo_files, tmp_path):
    from studiomix.engine import run

    d = demo_files
    p = get_preset("trap", doubles=True, harmonies="3up")
    log = run(d / "v.wav", d / "b.wav", tmp_path, p, name="s", verbose=False, stack_at=[(3.0, 9.0)])
    assert log["stack"]["stack_voices"] == ["double", "double", "harmony 3up"]
    st, sr = audio_io.load(tmp_path / log["files"]["stack_stem"])
    rms = lambda a, b: np.sqrt(np.mean(st[:, int(a * sr):int(b * sr)] ** 2))  # noqa: E731
    assert rms(0.0, 2.5) < 1e-6 < rms(4.0, 8.0)
    assert log["output"]["true_peak_dbtp"] <= log["master"]["ceiling_dbtp"] + 0.01

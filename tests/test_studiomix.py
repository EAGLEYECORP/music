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


def test_web_app_end_to_end(demo_files, tmp_path):
    """POST a song to `studiomix serve`, poll the job, download the master."""
    import threading
    import time as _t
    import urllib.error
    import urllib.request
    import uuid as _uuid
    from http.server import ThreadingHTTPServer

    from studiomix import web

    web.Handler.jobs_dir = tmp_path
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert b"studio" in urllib.request.urlopen(base + "/").read()
        boundary = _uuid.uuid4().hex
        parts = []
        for name, path in (("lead", demo_files / "v.wav"), ("beat", demo_files / "b.wav")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
                         f"Content-Type: audio/wav\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
        for k, v in (("preset", "pop"), ("tune", "hard"), ("harmony", "3up")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(base + "/api/jobs", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        job_id = json.loads(urllib.request.urlopen(req).read())["id"]
        for _ in range(300):
            j = json.loads(urllib.request.urlopen(f"{base}/api/jobs/{job_id}").read())
            if j["status"] in ("done", "error"):
                break
            _t.sleep(0.5)
        assert j["status"] == "done", j.get("error")
        assert j["result"]["output"]["integrated_lufs"] == pytest.approx(-11.0, abs=0.2)
        assert "stack_stem" in j["result"]["files"]
        master = urllib.request.urlopen(f"{base}/jobs/{job_id}/{j['result']['files']['master_24bit']}").read()
        assert master[:4] == b"RIFF" and len(master) > 100_000
        # nothing outside the job's own outputs is served
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/jobs/{job_id}/..%2F..%2Fetc%2Fpasswd")
    finally:
        httpd.shutdown()


# ------------------------------------------------------------------ metering compliance


def _ebu_sine(db, secs, sr=48000, f=1000.0):
    t = np.arange(int(sr * secs)) / sr
    return np.sin(2 * np.pi * f * t) * 10 ** (db / 20)


@pytest.mark.parametrize("segments,expected", [
    ([(-23, 20)], -23.0),                                           # EBU Tech 3341 case 1
    ([(-33, 20)], -33.0),                                           # case 2
    ([(-36, 10), (-23, 60), (-36, 10)], -23.0),                     # case 3: relative gate
    ([(-72, 10), (-36, 10), (-23, 60), (-36, 10), (-72, 10)], -23.0),  # case 4: both gates
    ([(-26, 20), (-20, 20.1), (-26, 20)], -23.0),                   # case 5
])
def test_ebu_3341_integrated_loudness(segments, expected):
    x = np.concatenate([_ebu_sine(db, s) for db, s in segments])
    assert analysis.integrated_lufs(np.vstack([x, x]), 48000) == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize("a,b,expected", [(-20, -30, 10.0), (-20, -15, 5.0), (-40, -20, 20.0)])
def test_ebu_3342_loudness_range(a, b, expected):
    x = np.concatenate([_ebu_sine(a, 20), _ebu_sine(b, 20)])
    assert analysis.loudness_range(np.vstack([x, x]), 48000) == pytest.approx(expected, abs=1.0)


@pytest.mark.parametrize("sr", [44100, 48000])
def test_true_peak_worst_case_tones(sr):
    """Tones whose samples always miss the peak, up to 20 kHz: error must stay within 0.05 dB."""
    n = np.arange(sr // 2)
    fade = np.minimum(1, np.minimum(n, len(n) - 1 - n) / 2000)
    for num, den in ((1, 4), (1, 3), (3, 8), (2, 5), (5, 12), (1, 8), (7, 16)):
        if num / den * sr > 20000:
            continue
        for ph in np.linspace(0, np.pi, 9):
            x = np.sin(2 * np.pi * num * n / den + ph) * 0.5 * fade
            err = dynamics.true_peak_db(np.vstack([x, x])) - 20 * np.log10(0.5)
            assert abs(err) < 0.05, (num, den, ph, err)


def test_loudness_lands_exactly_on_target():
    x = np.random.default_rng(0).standard_normal((2, 48000 * 12)) * 0.05
    for target, ceiling in ((-14.0, -1.0), (-9.0, -2.0)):
        y = chains.finalize_loudness(x, 48000, target, ceiling, 2.0, 80.0, {})
        got = analysis.integrated_lufs(y, 48000)
        assert target - 0.03 <= got <= target + 1e-6
        assert dynamics.true_peak_db(y) <= ceiling


# ------------------------------------------------------------------ finished-mix mastering + broadcast versions


@pytest.fixture(scope="module")
def rough_mix(demo_files, tmp_path_factory):
    """A deliberately bad bounce: clipped, DC offset, out-of-phase bass, 2.5 s of dead air."""
    from scipy import signal as _sig

    sr = 48000
    v, _ = audio_io.load(demo_files / "v.wav")
    b, sb = audio_io.load(demo_files / "b.wav")
    b = audio_io.resample(b, sb, sr)
    n = min(v.shape[1], b.shape[1])
    mix = b[:, :n] * 0.9 + np.vstack([v[0, :n], v[0, :n]]) * 0.6
    lo = _sig.sosfiltfilt(_sig.butter(4, 120, fs=sr, output="sos"), mix[1])
    mix[1] = mix[1] - 2 * lo
    mix = np.clip(mix / np.max(np.abs(mix)) * 2.5, -1, 1) + 0.004  # flat-topped, like a real clipped bounce
    mix = np.concatenate([np.zeros((2, int(2.5 * sr))), mix], axis=1)
    p = tmp_path_factory.mktemp("rough") / "rough.wav"
    audio_io.write_wav(p, mix, sr, 16)
    return p


def test_diagnosis_finds_planted_problems(rough_mix):
    x, sr = audio_io.load(rough_mix)
    d = chains.diagnose_mix(x, sr)
    text = " ".join(d["findings"])
    assert d["metrics"]["clipped_spots"] > 0
    assert "DC offset" in text and "out of phase" in text and "silence" in text


def test_master_finished_mix_with_broadcast_versions(rough_mix, tmp_path):
    from studiomix.engine import master_mix

    log = master_mix(rough_mix, tmp_path, get_preset("hiphop"), name="m", vocal_lift_db=2.0,
                     deliver_extra=["ebu-r128", "atsc-a85", "apple"], verbose=False)
    assert all(c["ok"] for c in log["delivery_check"] if not c["check"].startswith("Dynamics"))
    assert log["mix_repair"]["low_end_phase_fix"]
    for key, lufs, tp in (("version_ebu-r128", -23.0, -1.0), ("version_atsc-a85", -24.0, -2.0),
                          ("version_apple", -16.0, -1.0)):
        x, sr = audio_io.load(tmp_path / log["files"][key])
        assert analysis.integrated_lufs(x, sr) == pytest.approx(lufs, abs=0.1)
        assert dynamics.true_peak_db(x) <= tp
    # the master starts right away (dead air trimmed) and the bass survived the phase fix
    m, sr = audio_io.load(tmp_path / log["files"]["master_24bit"])
    assert np.argmax(np.max(np.abs(m), axis=0) > 1e-3) / sr < 0.5
    lo = np.mean(filters.lowpass(m, sr, 120.0, order=4), axis=0)
    assert 10 * np.log10(np.mean(lo ** 2)) > -30


def test_cli_master_and_deliver_validation(rough_mix, tmp_path, capsys):
    from studiomix.cli import main

    assert main(["master", str(rough_mix), "-o", str(tmp_path), "--deliver", "nope", "-q"]) == 1
    assert "unknown delivery version" in capsys.readouterr().err


# ------------------------------------------------------------------ reference profiles


def test_learn_profile_and_master_toward_it(rough_mix, tmp_path, monkeypatch):
    from studiomix import profiles
    from studiomix.engine import master_mix

    monkeypatch.setenv("STUDIOMIX_PROFILES", str(tmp_path / "profiles"))
    # a "reference": a bright, wide, -10 LUFS take on the demo beat
    ref = make_demo.beat(48000, 20.0)
    ref = filters.eq(ref, 48000, "highshelf", 4000.0, 6.0)
    mid, side = 0.5 * (ref[0] + ref[1]), 0.5 * (ref[0] - ref[1]) + 0.3 * filters.bandpass(ref[:1], 48000, 500, 8000)[0]
    ref = chains.finalize_loudness(np.vstack([mid + side, mid - side]), 48000, -10.0, -1.0, 0.0, 80.0, {})
    audio_io.write_wav(tmp_path / "ref.wav", ref, 48000)

    prof = profiles.learn([tmp_path / "ref.wav"], "My Sound")
    assert prof["name"] == "my-sound" and profiles.list_profiles() == ["my-sound"]
    assert prof["target_lufs"] == pytest.approx(-10.0, abs=0.1)
    assert "my-sound" in profiles.describe(profiles.load("my-sound"))

    def distance(path):
        a = profiles.analyse(path)
        c = np.array([np.nan if v is None else v for v in a["curve_db"]])
        g = np.array(prof["grid_hz"])
        band = (g >= 100) & (g <= 12000)
        return np.sqrt(np.nanmean((c - np.array(prof["curve_db"]))[band] ** 2))

    p = get_preset("hiphop")
    plain = master_mix(rough_mix, tmp_path / "a", p, name="a", verbose=False)
    matched = master_mix(rough_mix, tmp_path / "b", profiles.apply(prof, p), name="b", verbose=False, profile=prof)
    assert matched["output"]["integrated_lufs"] == pytest.approx(-10.0, abs=0.1)
    assert distance(tmp_path / "b" / matched["files"]["master_24bit"]) < \
        distance(tmp_path / "a" / plain["files"]["master_24bit"]) - 0.5
    assert matched["delivery_check"][0]["ok"]  # still within the true-peak ceiling


def test_profile_errors(tmp_path, monkeypatch):
    from studiomix import profiles

    monkeypatch.setenv("STUDIOMIX_PROFILES", str(tmp_path))
    with pytest.raises(KeyError):
        profiles.load("nope")
    with pytest.raises(ValueError):
        profiles._safe("!!!")


def test_song_body_skips_video_intro_and_outro():
    from studiomix import profiles

    sr = 48000
    rng = np.random.default_rng(0)
    intro = rng.standard_normal((2, sr * 20)) * 0.002          # quiet talking/ambience
    song = rng.standard_normal((2, sr * 60)) * 0.2
    outro = np.zeros((2, sr * 25))
    a, b = profiles.song_body(np.concatenate([intro, song, outro], axis=1), sr)
    assert abs(a / sr - 20) <= 3 and abs(b / sr - 80) <= 4


def test_width_matching_reaches_targets():
    from studiomix.profiles import side_mid_db

    sr = 48000
    rng = np.random.default_rng(1)
    common = rng.standard_normal(sr * 10)
    x = np.vstack([common + 0.3 * rng.standard_normal(sr * 10), common + 0.3 * rng.standard_normal(sr * 10)]) * 0.1
    targets = {"500-2000": -6.0, "2000-8000": -12.0, "8000-16000": -20.0}
    y = x.copy()
    chains.match_width(y, sr, targets)
    for k, t in targets.items():
        assert side_mid_db(y, sr, *map(float, k.split("-"))) == pytest.approx(t, abs=1.0)
    # the low end is never widened
    assert side_mid_db(y, sr, 20, 110) == pytest.approx(side_mid_db(x, sr, 20, 110), abs=0.1)


# ------------------------------------------------------------------ phone studio


def test_studio_session_builds_timeline(tmp_path, demo_files):
    from studiomix import studio

    def wav_bytes(x, sr=48000):
        p = tmp_path / "tmp.wav"
        audio_io.write_wav(p, x, sr, 32)
        return p.read_bytes()

    beat = (demo_files / "b.wav").read_bytes()
    studio.save_beat(tmp_path, "Night Drive", "beat.wav", beat)
    tone = np.sin(2 * np.pi * 220 * np.arange(48000) / 48000)[None, :] * 0.3
    t1 = studio.add_take(tmp_path, "night-drive", wav_bytes(tone), "lead", 1.0, 180.0)
    studio.add_take(tmp_path, "night-drive", wav_bytes(tone * 0.5), "lead", 4.0, 180.0)
    studio.add_take(tmp_path, "night-drive", wav_bytes(tone), "adlib", 2.5, 180.0)
    loud = studio.add_take(tmp_path, "night-drive", wav_bytes(np.clip(tone * 5, -1, 1)), "adlib", 6.0, 0)
    assert "clipped" in loud["warning"]
    studio.delete_take(tmp_path, "night-drive", loud["id"])
    info = studio.info(tmp_path, "night-drive")
    assert info["beat"]["name"] == "beat.wav" and len(info["takes"]) == 3

    tr = studio.build_tracks(tmp_path, "night-drive")
    lead, sr = audio_io.load(tr["lead"])
    env = lambda a, b: np.max(np.abs(lead[0, int(a * sr):int(b * sr)]))  # noqa: E731
    assert env(0.0, 0.95) < 1e-6 and env(1.05, 1.95) > 0.25 and env(2.05, 3.95) < 1e-6 and env(4.05, 4.95) > 0.1
    ad, _ = audio_io.load(tr["adlib"])
    assert np.max(np.abs(ad[0, : int(2.45 * sr)])) < 1e-6
    with pytest.raises(ValueError):
        studio.add_take(tmp_path, "night-drive", wav_bytes(tone), "drums", 0, 0)
    with pytest.raises(ValueError):
        studio.safe_session("!!!")
    assert t1["latency_ms"] == 180.0


def test_studio_requires_beat_and_lead(tmp_path):
    from studiomix import studio

    with pytest.raises(ValueError):
        studio.build_tracks(tmp_path, "empty-song")


# ------------------------------------------------------------------ noise reduction


def _noisy_take(snr_db, seed=0):
    t = np.arange(SR * 8) / SR
    f = 220 * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
    clean = sung(f) * ((t % 2.0) < 1.4)  # phrases with gaps
    act = (t % 2.0) < 1.4
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(len(t))
    nz = signal.lfilter([1], [1, -0.99], w) + 0.3 * w  # fan-like
    nz *= np.sqrt(np.mean(clean[act] ** 2) / np.mean(nz ** 2) / 10 ** (snr_db / 10))
    return clean, clean + nz, act


@pytest.mark.parametrize("snr", [12, 20])
def test_denoise_removes_noise_without_hurting_the_voice(snr):
    from studiomix.dsp.denoise import denoise

    clean, noisy, act = _noisy_take(snr)
    out, st = denoise(noisy[None], SR, act)
    out = out[0]
    gaps = ~act
    assert 10 * np.log10(np.mean(out[gaps] ** 2) / np.mean(noisy[gaps] ** 2)) < -6.0
    sdr = lambda y: 10 * np.log10(np.sum(clean[act] ** 2) / np.sum((y[act] - clean[act]) ** 2))  # noqa: E731
    assert sdr(out) >= sdr(noisy) - 0.3


def test_denoise_leaves_clean_takes_alone():
    from studiomix.dsp.denoise import denoise

    clean, noisy, act = _noisy_take(50)
    out, st = denoise(noisy[None], SR, act)
    assert st["applied"] is False and np.array_equal(out[0], noisy)


def test_studio_comping_trim_nudge_gain(tmp_path, demo_files):
    from studiomix import studio

    def wav_bytes(x, sr=48000):
        p = tmp_path / "tmp.wav"
        audio_io.write_wav(p, x, sr, 32)
        return p.read_bytes()

    studio.save_beat(tmp_path, "hook", "beat.wav", (demo_files / "b.wav").read_bytes())
    tone = np.sin(2 * np.pi * 220 * np.arange(48000 * 2) / 48000)[None, :] * 0.3
    # a loop recording: 3 passes of the same section; the last one is active by default
    passes = [studio.add_take(tmp_path, "hook", wav_bytes(tone * g), "lead", 2.0, 100, group="g1", active=(i == 2))
              for i, g in enumerate((0.2, 0.5, 1.0))]
    info = studio.info(tmp_path, "hook")
    assert [t["active"] for t in info["takes"]] == [False, False, True]
    # pick pass 1, trim 0.5 s off its start, nudge it 20 ms earlier, +6 dB
    info = studio.update_take(tmp_path, "hook", passes[0]["id"],
                              {"active": "1", "trim_start_s": "0.5", "nudge_ms": "-20", "gain_db": "6"})
    assert [t["active"] for t in info["takes"]] == [True, False, False]
    tr = studio.build_tracks(tmp_path, "hook")
    lead, sr = audio_io.load(tr["lead"])
    nz = np.flatnonzero(np.abs(lead[0]) > 1e-4)
    assert nz[0] / sr == pytest.approx(2.0 + 0.5 - 0.02, abs=0.012)  # trimmed + nudged start
    assert np.max(np.abs(lead)) == pytest.approx(0.06 * 2.0, rel=0.05)  # pass 1 (0.2 * 0.3) at +6 dB
    with pytest.raises(ValueError):
        studio.update_take(tmp_path, "hook", "0000000000", {"active": "1"})
    # setting every lead take aside leaves nothing to mix
    studio.update_take(tmp_path, "hook", passes[0]["id"], {"active": "0"})
    with pytest.raises(ValueError):
        studio.build_tracks(tmp_path, "hook")


def test_web_server_blocks_rebinding_and_csrf(tmp_path):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from studiomix import web

    web.Handler.jobs_dir = tmp_path
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        assert urllib.request.urlopen(base + "/").status == 200
        # DNS rebinding: a hostile domain pointed at 127.0.0.1 sends its own name as Host
        req = urllib.request.Request(base + "/api/studio/my-song", headers={"Host": f"evil.example:{port}"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403
        # CSRF: a POST from another site's page
        req = urllib.request.Request(base + "/api/studio/my-song/delete/0123456789", data=b"", method="POST",
                                     headers={"Origin": "https://evil.example"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403
        # the app's own page is fine
        req = urllib.request.Request(base + "/api/studio/my-song/delete/0123456789", data=b"", method="POST",
                                     headers={"Origin": base})
        assert urllib.request.urlopen(req).status == 200
    finally:
        httpd.shutdown()


def test_studio_rejects_bad_uploads_and_edits(tmp_path, demo_files):
    from studiomix import studio

    good = (demo_files / "b.wav").read_bytes()
    studio.save_beat(tmp_path, "s", "beat.wav", good)
    # a broken beat upload must not destroy the beat already there
    with pytest.raises(Exception):
        studio.save_beat(tmp_path, "s", "beat.mp3", b"not audio at all")
    d = studio.session_dir(tmp_path, "s")
    assert (d / "beat.wav").exists() and not list(d.glob("upload.*"))
    assert studio.info(tmp_path, "s")["beat"]["name"] == "beat.wav"
    # a broken take leaves no file behind
    with pytest.raises(Exception):
        studio.add_take(tmp_path, "s", b"RIFFjunk", "lead", 0, 0)
    assert not list((d / "takes").glob("*.wav"))
    # NaN edits are refused (they would corrupt the session file for the browser)
    p = tmp_path / "t.wav"
    sr = 44100
    audio_io.write_wav(p, np.full((1, sr), 0.1), sr, 32)
    # offsets that land on a rounding boundary must not overflow the timeline
    for off in (0.0, 1.0, 2.0):
        t = studio.add_take(tmp_path, "s", p.read_bytes(), "lead", off, 0)
    # a nudge that lands the last take half a sample past a sample boundary must not overflow
    studio.update_take(tmp_path, "s", t["id"], {"nudge_ms": 0.0113})
    with pytest.raises(ValueError):
        studio.update_take(tmp_path, "s", t["id"], {"gain_db": "nan"})
    tr = studio.build_tracks(tmp_path, "s")
    assert audio_io.load(tr["lead"])[0].shape[-1] > 3 * 48000 - 10


def test_preset_overrides_out_of_range_are_refused():
    with pytest.raises(ValueError):
        get_preset("pop", ceiling_dbtp=1.0)
    with pytest.raises(ValueError):
        get_preset("pop", target_lufs=-50.0)
    assert get_preset("pop", target_lufs=-9.0, ceiling_dbtp=-1.0).target_lufs == -9.0

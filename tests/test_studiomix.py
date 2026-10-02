import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from studiomix import audio_io, chains  # noqa: E402
from studiomix.dsp import analysis, dynamics, filters  # noqa: E402
from studiomix.presets import PRESETS, get_preset  # noqa: E402

SR = 48000


def noise(seconds=5.0, level=0.1, ch=2, seed=0):
    return np.random.default_rng(seed).standard_normal((ch, int(SR * seconds))) * level


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
    x = np.vstack([np.sin(2 * np.pi * 997 * t)] * 2) * 10 ** (-23 / 20) * 1.0
    # BS.1770: a stereo 997 Hz sine at -23 dBFS peak per channel reads about -23 LUFS
    assert analysis.integrated_lufs(x, SR) == pytest.approx(-23.0, abs=0.3)


def test_dithered_16bit_roundtrip(tmp_path):
    x = noise(1.0, 0.2)
    p = tmp_path / "x.wav"
    audio_io.write_wav16_dithered(p, x, SR)
    y, sr = audio_io.load(p)
    assert sr == SR and sf.info(str(p)).subtype == "PCM_16"
    assert np.max(np.abs(y - x)) < 3 / 32768


def test_preset_overrides():
    p = get_preset("hiphop", target_lufs=-12.0, vocal_reverb=None)
    assert p.target_lufs == -12.0 and p.vocal_reverb == PRESETS["hiphop"].vocal_reverb
    with pytest.raises(KeyError):
        get_preset("polka")


def test_end_to_end(tmp_path):
    import make_demo
    from studiomix.engine import run

    sf.write(tmp_path / "v.wav", make_demo.vocal(48000, 12.0).T, 48000)
    sf.write(tmp_path / "b.wav", make_demo.beat(44100, 12.0).T, 44100)
    log = run(tmp_path / "v.wav", tmp_path / "b.wav", tmp_path / "out", get_preset("pop"), name="song",
              verbose=False)
    out = log["output"]
    assert out["integrated_lufs"] == pytest.approx(-11.0, abs=0.2)
    assert out["true_peak_dbtp"] <= log["master"]["ceiling_dbtp"] + 0.01
    assert all(c["ok"] for c in log["delivery_check"][:3])
    for f in log["files"].values():
        assert (tmp_path / "out" / f).exists()
    m16, sr16 = audio_io.load(tmp_path / "out" / log["files"]["master_16bit_cd"])
    assert sr16 == 44100 and dynamics.true_peak_db(m16) <= log["master"]["ceiling_dbtp"] + 0.05
    assert json.loads((tmp_path / "out" / "song_report.json").read_text())["preset"] == "pop"

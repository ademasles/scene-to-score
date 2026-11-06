from pathlib import Path

import numpy as np
import pytest

from scene_to_score.audio import load_audio, resample, write_wav


def _tone(rate: int, seconds: float, hz: float = 440.0) -> np.ndarray:
    t = np.arange(round(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_wav_round_trip_through_pyav(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    write_wav(path, _tone(16_000, 2.0), 16_000)
    samples, rate = load_audio(path)
    assert rate == 16_000
    assert samples.dtype == np.float32
    np.testing.assert_allclose(samples, _tone(16_000, 2.0), atol=1e-3)


def test_load_audio_cuts_a_window_and_refuses_overruns(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    write_wav(path, _tone(8_000, 3.0), 8_000)
    samples, _ = load_audio(path, start_s=1.0, duration_s=1.5)
    assert len(samples) == 12_000
    with pytest.raises(ValueError, match="outside"):
        load_audio(path, start_s=2.0, duration_s=1.5)


def test_write_wav_scales_down_only_when_clipping(tmp_path: Path) -> None:
    path = tmp_path / "loud.wav"
    write_wav(path, 3 * _tone(8_000, 0.5), 8_000)
    samples, _ = load_audio(path)
    assert np.abs(samples).max() == pytest.approx(1.0, abs=1e-3)


def test_resample_changes_length_and_keeps_pitch() -> None:
    out = resample(_tone(44_100, 1.0), 44_100, 48_000)
    assert len(out) == 48_000
    peak_hz = np.argmax(np.abs(np.fft.rfft(out))) * 48_000 / len(out)
    assert peak_hz == pytest.approx(440.0, abs=2.0)
    assert resample(out, 48_000, 48_000) is not out

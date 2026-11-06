"""Audio I/O shared by the music targets, the generation and the demo."""

import wave
from pathlib import Path

import av
import numpy as np
import soxr


def load_audio(
    path: Path, start_s: float = 0.0, duration_s: float | None = None
) -> tuple[np.ndarray, int]:
    """Decode ``path`` to mono float32 at its own sampling rate, then cut a window.

    Returns the samples and the sampling rate. A window running past the end is an error,
    not silently shortened.
    """
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        rate = stream.rate
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        chunks = [
            out.to_ndarray()[0]
            for frame in container.decode(stream)
            for out in resampler.resample(frame)
        ]
        chunks += [out.to_ndarray()[0] for out in resampler.resample(None)]
    samples = np.concatenate(chunks).astype(np.float32)
    start = round(start_s * rate)
    end = len(samples) if duration_s is None else start + round(duration_s * rate)
    if start < 0 or end > len(samples):
        raise ValueError(
            f"{path}: window {start_s}+{duration_s} s outside {len(samples) / rate:.2f} s"
        )
    return samples[start:end], rate


def resample(samples: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    if rate_in == rate_out:
        return samples.astype(np.float32)
    return soxr.resample(samples, rate_in, rate_out, quality="HQ").astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    """Write mono 16-bit PCM, peak-normalised only if the signal would clip."""
    peak = float(np.abs(samples).max(initial=0.0))
    scaled = samples / peak if peak > 1.0 else samples
    pcm = np.round(np.clip(scaled, -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())

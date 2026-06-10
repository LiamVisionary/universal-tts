"""Regression tests for KittenTTS audio conversion (the int16 truncation bug).

KittenTTS returns float32 NumPy arrays with peak around 0.5-1.0. The sidecar
and provider must convert these to PCM int16 with proper scaling (clip to
[-1, 1] then multiply by 32767). Casting float32 directly to int16 (which is
what the previous code did) truncates every value to 0 -> silent output.
"""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from universal_tts.providers.kitten import (
    _samples_to_pcm16,
    _samples_to_wav,
)


def _fake_kitten_audio(samples: int = 48000, peak: float = 0.6) -> np.ndarray:
    """Mimic KittenTTS output: float32, 1D, range roughly [-peak, +peak]."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(samples).astype(np.float32) * peak)


def test_pcm16_scaling_does_not_truncate_to_silence():
    audio = _fake_kitten_audio(peak=0.6)
    pcm = _samples_to_pcm16(audio)
    assert pcm.dtype == np.int16
    assert pcm.shape == (audio.size,)
    # The old broken behavior: every sample was 0. Make sure we get real values.
    assert int((pcm != 0).sum()) > 0, "PCM is all zeros — float32→int16 truncation bug regressed"
    # Peak should land near int16_max * 0.6 ≈ 19660
    assert int(np.max(np.abs(pcm))) > 10000
    assert int(np.max(np.abs(pcm))) < 32768


def test_pcm16_peak_clipped_to_int16_range():
    # If the model returns a sample > 1.0, the int16 conversion must clip, not wrap
    audio = np.array([1.5, -1.5, 2.0, -2.0, 0.0, 0.5], dtype=np.float32)
    pcm = _samples_to_pcm16(audio)
    # Clipped to int16 range, never wrapping around
    assert int(pcm[0]) == 32767   # clipped
    assert int(pcm[1]) in (-32768, -32767)  # clipped (np.round uses banker's rounding on .5)
    assert int(pcm[2]) == 32767
    assert int(pcm[3]) in (-32768, -32767)
    assert int(pcm[4]) == 0
    assert int(pcm[5]) > 0


def test_pcm16_handles_2d_input():
    audio = _fake_kitten_audio(48000).reshape(-1, 1)
    pcm = _samples_to_pcm16(audio)
    assert pcm.ndim == 1
    assert pcm.size == 48000
    assert int((pcm != 0).sum()) > 0


def test_wav_roundtrip_is_decodable_and_loud():
    audio = _fake_kitten_audio(samples=24000 * 2, peak=0.5)
    wav_bytes = _samples_to_wav(audio)
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 48000
        decoded = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert int((decoded != 0).sum()) > 0
    assert int(np.max(np.abs(decoded))) > 5000  # real audio, not silence

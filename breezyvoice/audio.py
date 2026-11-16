"""Measurements on a waveform: where the sound is and where the silence is."""
from __future__ import annotations

import numpy as np

_FRAME_SECONDS = 0.02
_SILENCE_DB = -40.0


def _silent_frames(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """One flag per 20 ms frame: is it 40 dB or more below the loudest frame?"""
    frame = int(_FRAME_SECONDS * sample_rate)
    count = len(audio) // frame
    if count == 0:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt(np.mean(audio[:count * frame].reshape(count, frame) ** 2, axis=1))
    return 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12) < _SILENCE_DB


def pauses(audio: np.ndarray, sample_rate: int, min_seconds: float = 0.25) -> list[tuple[float, float]]:
    """Silent stretches inside the speech, as (start, end) in seconds.

    A frame is silent when it is 40 dB below the loudest frame. Silence before the first and after
    the last sound is not a pause.
    """
    silent = _silent_frames(audio, sample_rate)
    voiced = np.flatnonzero(~silent)
    if voiced.size == 0:
        return []
    found, start = [], None
    for i in range(voiced[0], voiced[-1] + 1):
        if silent[i] and start is None:
            start = i
        elif not silent[i] and start is not None:
            if (i - start) * _FRAME_SECONDS >= min_seconds:
                found.append((start * _FRAME_SECONDS, i * _FRAME_SECONDS))
            start = None
    return found


def speech_pieces(audio: np.ndarray, sample_rate: int, min_gap_seconds: float) -> list[tuple[float, float]]:
    """Stretches of sound separated by at least ``min_gap_seconds`` of silence, as (start, end)."""
    silent = _silent_frames(audio, sample_rate)
    voiced = np.flatnonzero(~silent)
    if voiced.size == 0:
        return []
    cuts = pauses(audio, sample_rate, min_gap_seconds)
    edges = [voiced[0] * _FRAME_SECONDS] + [t for cut in cuts for t in cut] + [(voiced[-1] + 1) * _FRAME_SECONDS]
    return list(zip(edges[0::2], edges[1::2]))

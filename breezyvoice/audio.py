"""Measurements on a synthesized waveform that callers act on."""
from __future__ import annotations

import numpy as np

_FRAME_SECONDS = 0.02
_SILENCE_DB = -40.0


def pauses(audio: np.ndarray, sample_rate: int, min_seconds: float = 0.25) -> list[tuple[float, float]]:
    """Silent stretches inside the speech, as (start, end) in seconds.

    A frame is silent when it is 40 dB below the loudest frame. Silence before the first and after
    the last sound is not a pause.
    """
    frame = int(_FRAME_SECONDS * sample_rate)
    count = len(audio) // frame
    if count == 0:
        return []
    rms = np.sqrt(np.mean(audio[:count * frame].reshape(count, frame) ** 2, axis=1))
    silent = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12) < _SILENCE_DB
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

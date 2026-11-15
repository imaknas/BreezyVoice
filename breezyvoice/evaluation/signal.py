"""Cheap checks on the waveform itself, and a distance between two takes of the same sentence."""
from __future__ import annotations

import numpy as np
import torch

from cosyvoice.matcha.audio import mel_spectrogram

_CLIP_LEVEL = 0.999
_TAIL_SECONDS = 0.1


def clipped_fraction(audio: np.ndarray) -> float:
    return float(np.mean(np.abs(audio) >= _CLIP_LEVEL))


def tail_level_db(audio: np.ndarray, sample_rate: int) -> float:
    """Level of the last 100 ms relative to the whole clip, in dB.

    Speech that ends properly decays; a clip cut off mid-syllable ends near its average level.
    """
    tail = audio[-int(_TAIL_SECONDS * sample_rate):]
    whole = float(np.sqrt(np.mean(audio**2))) + 1e-9
    return float(20 * np.log10((float(np.sqrt(np.mean(tail**2))) + 1e-9) / whole))


def mel_distance(a: np.ndarray, b: np.ndarray, sample_rate: int) -> float | None:
    """Mean absolute difference of the log-mel spectrograms, or None when the lengths differ.

    Two takes from the same speech tokens have the same length, and this measures how far the
    acoustic stages moved them apart; different lengths mean the tokens themselves differ.
    """
    if a.shape != b.shape:
        return None

    def mel(x: np.ndarray) -> torch.Tensor:
        return mel_spectrogram(torch.from_numpy(x).unsqueeze(0), n_fft=1024, num_mels=80,
                               sampling_rate=sample_rate, hop_size=256, win_size=1024, fmin=0, fmax=8000)

    return float((mel(a) - mel(b)).abs().mean())


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

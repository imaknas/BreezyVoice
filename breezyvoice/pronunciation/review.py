"""One pass over a finished take: what a listener would object to, and the part of it that can be mended."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from breezyvoice.audio import pauses as silences
from breezyvoice.pronunciation.align import stray_units, syllable_times
from breezyvoice.pronunciation.check import score
from breezyvoice.pronunciation.pauses import Limits, Pause, place, tidy, unnatural
from breezyvoice.pronunciation.readings import Reading, Unsupported
from breezyvoice.pronunciation.units import flatten

# The intended reading must fit this much worse than another reading of the character before the
# take is reported. On the takes a listener judged, the two with a wrong reading scored -9.8 and
# -6.7 and the clean ones above -1, apart from one he was unsure of at -6.4.
READING_MARGIN = -5.0
MIN_SILENCE = 0.10


@dataclass(frozen=True)
class WrongReading:
    character: str
    position: int  # among the spoken characters
    intended: str
    heard: str  # the reading that fits the audio better
    margin: float


@dataclass(frozen=True)
class Review:
    characters: tuple[str, ...]
    wrong_readings: tuple[WrongReading, ...]
    extra_before: tuple[str, ...]  # sound units heard before the sentence begins
    extra_after: tuple[str, ...]  # ... and after it ends
    pauses: tuple[Pause, ...]
    unnatural_pauses: tuple[Pause, ...]

    @property
    def needs_retake(self) -> bool:
        """Something only another take can fix. Pauses are not counted: ``Reviewer.tidy`` mends them."""
        return bool(self.wrong_readings or self.extra_before or self.extra_after)


class Reviewer:
    """``recognizer`` gives frame posteriors over sound units for 16 kHz audio; ``read`` gives the
    intended readings of a text; ``to_16k`` resamples (audio, rate) for the recogniser."""

    def __init__(self, recognizer, read: Callable[[str], Reading],
                 to_16k: Callable[[np.ndarray, int], torch.Tensor], limits: Limits = Limits(),
                 reading_margin: float = READING_MARGIN):
        self._recognizer = recognizer
        self._read = read
        self._to_16k = to_16k
        self._limits = limits
        self._reading_margin = reading_margin

    def review(self, audio: np.ndarray, sample_rate: int, text: str) -> Review:
        """Raises ``Unsupported`` for text the recogniser has no units for."""
        reading = self._read(text)
        index = self._recognizer.unit_index
        missing = [u for u in flatten(reading.readings) if u not in index]
        if missing:
            raise Unsupported(f"the recogniser has no unit for {missing[0]}")
        log_probs = self._recognizer.log_probs(self._to_16k(audio, sample_rate))
        wrong = tuple(
            WrongReading(reading.characters[s.index], s.index, s.reading, s.rival, s.margin)
            for s in score(log_probs, reading.readings, reading.others, index, tones=False)
            if s.rival is not None and s.margin < self._reading_margin)
        times = syllable_times(log_probs, reading.readings, index)
        before, after = stray_units(log_probs, times, self._recognizer.vocabulary)
        placed = tuple(place(silences(audio, sample_rate, MIN_SILENCE), times, reading.positions, reading.spoken))
        return Review(reading.characters, wrong, tuple(before), tuple(after), placed,
                      tuple(unnatural(placed, self._limits)))

    def tidy(self, audio: np.ndarray, sample_rate: int, review: Review) -> np.ndarray:
        """The take with its unnatural pauses shortened; the speech itself is untouched."""
        return tidy(audio, sample_rate, review.pauses, self._limits)

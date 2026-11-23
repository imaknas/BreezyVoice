"""One pass over a finished take: what a listener would object to, and the part of it that can be mended."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from breezyvoice.audio import pauses as silences, silence_before
from breezyvoice.pronunciation.align import Inserted, Repeat, clarity, inserted, repeated, stray_units, syllable_times
from breezyvoice.pronunciation.check import score
from breezyvoice.pronunciation.pauses import Limits, Missing, Pause, missing, place, tidy, unnatural
from breezyvoice.pronunciation.readings import Reading, Unsupported
from breezyvoice.pronunciation.units import flatten

# The intended reading must fit this much worse than another reading of the character before the
# take is reported. On the takes a listener judged, the two with a wrong reading scored -9.8 and
# -6.7 and the clean ones above -1, apart from one they were unsure of at -6.4.
READING_MARGIN = -5.0
MIN_SILENCE = 0.10
# A take that starts with this much silence has gone wrong: the four a listener pointed out began
# after 1.8 to 9.2 s (one of them left out its first words), and none the listener let pass waited this long.
MAX_SILENCE_BEFORE = 1.5
# A syllable heard this faintly is reported but does not ask for another take. Of fifteen takes
# with one, a listener wanted eight redone and found nothing wrong with four.
UNCLEAR = -10.0


@dataclass(frozen=True)
class WrongReading:
    character: str
    position: int  # among the spoken characters
    intended: str
    heard: str  # the reading that fits the audio better
    margin: float


@dataclass(frozen=True)
class Unclear:
    character: str
    position: int  # among the spoken characters
    score: float  # log-probability of its least clear sound


@dataclass(frozen=True)
class Review:
    characters: tuple[str, ...]
    wrong_readings: tuple[WrongReading, ...]
    extra_before: tuple[str, ...]  # sound units heard before the sentence begins
    extra_after: tuple[str, ...]  # ... and after it ends
    pauses: tuple[Pause, ...]
    unnatural_pauses: tuple[Pause, ...]
    missing_pauses: tuple[Missing, ...] = ()  # strong marks spoken without a pause
    repeated: tuple[Repeat, ...] = ()  # syllables said twice in a row where the text has them once
    inserted: tuple[Inserted, ...] = ()  # other whole syllables heard in mid-sentence that the text lacks
    silence_before: float = 0.0  # seconds before the first sound
    unclear: tuple[Unclear, ...] = ()  # syllables heard faintly: slurred, cut short or left out

    @property
    def needs_retake(self) -> bool:
        """Something only another take can fix. Pauses are not counted: ``Reviewer.tidy`` mends them.
        Unclear syllables are not counted either: about half of them a listener lets pass."""
        return bool(self.wrong_readings or self.extra_before or self.extra_after or self.repeated
                    or self.inserted or self.long_silence_before)

    @property
    def long_silence_before(self) -> bool:
        return self.silence_before > MAX_SILENCE_BEFORE


def report(review: Review) -> dict:
    """The review as plain data: what needs another take, and where the pauses were."""
    return {
        "needs_retake": review.needs_retake,
        "wrong_readings": [{"character": w.character, "position": w.position, "intended": w.intended,
                            "heard": w.heard, "margin": round(w.margin, 2)} for w in review.wrong_readings],
        "extra_before": list(review.extra_before),
        "extra_after": list(review.extra_after),
        "repeated": [{"characters": "".join(review.characters[r.first:r.first + r.count]), "position": r.first,
                      "at": round(r.at, 2)} for r in review.repeated],
        "inserted": [{"units": "".join(i.units), "after": review.characters[i.after], "position": i.after,
                      "at": round(i.at, 2)} for i in review.inserted],
        "silence_before": round(review.silence_before, 2),
        "long_silence_before": review.long_silence_before,
        "unclear": [{"character": u.character, "position": u.position, "score": round(u.score, 1)}
                    for u in review.unclear],
        "unnatural_pauses": [{"after": review.characters[p.after], "position": p.after, "seconds": round(p.length, 2),
                              "at_punctuation": p.at_punctuation} for p in review.unnatural_pauses],
        "missing_pauses": [{"after": review.characters[m.after], "position": m.after, "seconds": round(m.seconds, 2),
                            "can_add": m.at is not None} for m in review.missing_pauses],
    }


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
        unknown = [u for u in flatten(reading.readings) if u not in index]
        if unknown:
            raise Unsupported(f"the recogniser has no unit for {unknown[0]}")
        log_probs = self._recognizer.log_probs(self._to_16k(audio, sample_rate))
        wrong = tuple(
            WrongReading(reading.characters[s.index], s.index, s.reading, s.rival, s.margin)
            for s in score(log_probs, reading.readings, reading.others, index, tones=False)
            if s.rival is not None and s.margin < self._reading_margin)
        times = syllable_times(log_probs, reading.readings, index)
        before, after = stray_units(log_probs, times, self._recognizer.vocabulary)
        placed = tuple(place(silences(audio, sample_rate, MIN_SILENCE), times, reading.positions, reading.spoken))
        lacking = missing(audio, sample_rate, placed, times, reading.positions, reading.spoken, self._limits)
        twice = tuple(repeated(log_probs, reading.readings, index))
        added = tuple(i for i in inserted(log_probs, reading.readings, index) if all(i.at != r.at for r in twice))
        faint = tuple(Unclear(reading.characters[k], k, s)
                      for k, s in enumerate(clarity(log_probs, reading.readings, index)) if s < UNCLEAR)
        return Review(reading.characters, wrong, tuple(before), tuple(after), placed,
                      tuple(unnatural(placed, self._limits)), tuple(lacking), twice, added,
                      silence_before(audio, sample_rate), faint)

    def tidy(self, audio: np.ndarray, sample_rate: int, review: Review) -> np.ndarray:
        """The take with its unnatural pauses shortened and its missing ones added; the speech itself
        is untouched."""
        return tidy(audio, sample_rate, review.pauses, self._limits, add=review.missing_pauses)

"""Pauses in their place: which two characters a silence falls between, and whether the text calls for it.

Counting silences against punctuation marks is not enough: a pause in the middle of a phrase and a
comma spoken without one cancel out, and a pause at a comma can itself be too long. Each silence is
therefore placed in the text, using the times at which the syllables are spoken.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Pause:
    start: float  # seconds
    end: float
    after: int  # index of the syllable it follows
    at_punctuation: bool  # the text has punctuation between that syllable and the next

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Limits:
    # Calibrated on fifty takes a listener marked: with these values the rule agreed with him on
    # eleven of the fifteen takes he found unnatural and on thirty-two of the thirty-five he did not.
    unmarked: float = 0.14  # a silence this long where the text has no punctuation is heard as a break
    at_punctuation: float = 0.50  # longer than this at a comma is heard as dragging


def place(silences: Sequence[tuple[float, float]], syllables: Sequence[tuple[float, float]],
          positions: Sequence[int], spoken: str) -> list[Pause]:
    """Assign each silence to the boundary between two syllables.

    ``syllables`` are the (start, end) times of the spoken characters and ``positions`` their indices
    in ``spoken``. Silences before the first or after the last syllable are not pauses. Silences that
    fall on the same boundary are merged: a breath in the middle of a pause splits it in two.
    """
    centres = [(a + b) / 2 for a, b in syllables]
    merged: dict[int, list[float]] = {}
    for start, end in silences:
        middle = (start + end) / 2
        before = [i for i, c in enumerate(centres) if c < middle]
        if not before or before[-1] + 1 >= len(syllables):
            continue
        boundary = before[-1]
        if boundary in merged:
            merged[boundary] = [min(merged[boundary][0], start), max(merged[boundary][1], end)]
        else:
            merged[boundary] = [start, end]
    pauses = []
    for boundary, (start, end) in sorted(merged.items()):
        between = spoken[positions[boundary] + 1: positions[boundary + 1]]
        marked = any(not ch.isalnum() and not ch.isspace() for ch in between)
        pauses.append(Pause(start, end, boundary, marked))
    return pauses


def unnatural(pauses: Sequence[Pause], limits: Limits = Limits()) -> list[Pause]:
    """The pauses a listener would notice, in order."""
    return [p for p in pauses
            if (p.at_punctuation and p.length > limits.at_punctuation)
            or (not p.at_punctuation and p.length >= limits.unmarked)]


def tidy(audio: np.ndarray, sample_rate: int, pauses: Sequence[Pause], limits: Limits = Limits(),
         unmarked_to: float = 0.04, punctuation_to: float = 0.32, fade: float = 0.008) -> np.ndarray:
    """Shorten the pauses ``unnatural`` would report: those in mid-phrase to almost nothing, those that
    drag at punctuation to an ordinary length. The cut is taken from the middle of the silence and the
    two sides are crossfaded, so no speech is touched."""
    cuts = []
    for pause in unnatural(pauses, limits):
        keep = punctuation_to if pause.at_punctuation else unmarked_to
        if pause.length <= keep + 2 * fade:
            continue
        cuts.append((pause.start + keep / 2, pause.end - keep / 2))
    out, position = [], 0
    n = int(fade * sample_rate)
    ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
    for start, end in sorted(cuts):
        a, b = int(start * sample_rate), int(end * sample_rate)
        head = audio[position:a].astype(np.float32, copy=True)
        head[-n:] = head[-n:] * ramp + audio[b:b + n] * (1 - ramp)
        out.append(head)
        position = b + n
    out.append(audio[position:])
    return np.concatenate(out)

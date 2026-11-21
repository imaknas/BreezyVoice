"""Pauses in their place: which two characters a silence falls between, and whether the text calls for it.

Counting silences against punctuation marks is not enough: a pause in the middle of a phrase and a
comma spoken without one cancel out, and a pause at a comma can itself be too long. Each silence is
therefore placed in the text, using the times at which the syllables are spoken.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Marks that announce or separate, after which a listener expects a beat even in quick speech. Dashes
# are left out: between two names or numbers they join rather than separate, and the two uses cannot be
# told apart from the text alone.
STRONG_MARKS = frozenset("：:；;…⋯")
# Without an existing pause, silence is only added where the sound dips at least this far below the
# louder of the two syllables. With no dip there, the syllables run into each other, or the alignment
# is off, and a cut would land inside a word.
MIN_DIP_DB = 10.0


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
class Missing:
    """A place where the text calls for a pause and the take has none, or too short a one."""
    after: int  # index of the syllable it should follow
    at: float | None  # where in the audio to add silence, seconds; None when no safe place was found
    seconds: float  # how much is lacking


@dataclass(frozen=True)
class Limits:
    # Calibrated on fifty takes a listener marked: with these values the rule agreed with him on
    # eleven of the fifteen takes he found unnatural and on thirty-two of the thirty-five he did not.
    unmarked: float = 0.14  # a silence this long where the text has no punctuation is heard as a break
    at_punctuation: float = 0.50  # longer than this at a comma is heard as dragging
    after_strong_mark: float = 0.22  # a colon or semicolon run straight through is heard as rushed


def _between(spoken: str, positions: Sequence[int], boundary: int) -> str:
    return spoken[positions[boundary] + 1: positions[boundary + 1]]


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
        marked = any(not ch.isalnum() and not ch.isspace() for ch in _between(spoken, positions, boundary))
        pauses.append(Pause(start, end, boundary, marked))
    return pauses


def unnatural(pauses: Sequence[Pause], limits: Limits = Limits()) -> list[Pause]:
    """The pauses a listener would notice, in order."""
    return [p for p in pauses
            if (p.at_punctuation and p.length > limits.at_punctuation)
            or (not p.at_punctuation and p.length >= limits.unmarked)]


def missing(audio: np.ndarray, sample_rate: int, pauses: Sequence[Pause], syllables: Sequence[tuple[float, float]],
            positions: Sequence[int], spoken: str, limits: Limits = Limits()) -> list[Missing]:
    """Strong marks spoken with no pause, or one shorter than ``limits.after_strong_mark``.

    The silence is to be added inside the existing pause if there is one, and otherwise at the
    quietest moment between the two syllables, provided the sound really dips there.
    """
    existing = {p.after: p for p in pauses}
    out = []
    for boundary in range(len(syllables) - 1):
        if not STRONG_MARKS & set(_between(spoken, positions, boundary)):
            continue
        pause = existing.get(boundary)
        have = pause.length if pause else 0.0
        if have >= limits.after_strong_mark:
            continue
        if pause:
            at = (pause.start + pause.end) / 2
        else:
            a = int(sum(syllables[boundary]) / 2 * sample_rate)
            b = int(sum(syllables[boundary + 1]) / 2 * sample_rate)
            frame = max(1, int(0.02 * sample_rate))
            levels = [(float(np.mean(audio[s:s + frame] ** 2)), s) for s in range(a, max(a + 1, b - frame), frame)]
            quietest, start = min(levels)
            loudest = max(level for level, _ in levels)
            dip = 10 * np.log10((loudest + 1e-12) / (quietest + 1e-12))
            at = (start + frame / 2) / sample_rate if dip >= MIN_DIP_DB else None
        out.append(Missing(boundary, at, limits.after_strong_mark - have))
    return out


def tidy(audio: np.ndarray, sample_rate: int, pauses: Sequence[Pause], limits: Limits = Limits(),
         add: Sequence[Missing] = (), unmarked_to: float = 0.04, punctuation_to: float = 0.32,
         fade: float = 0.008) -> np.ndarray:
    """Shorten the pauses ``unnatural`` would report and add the ones in ``add``.

    Pauses in mid-phrase go down to almost nothing, those that drag at punctuation to an ordinary
    length; the cut is taken from the middle of the silence and the two sides are crossfaded, so no
    speech is touched. An added pause fades the sound out and back in around the silence.
    """
    n = int(fade * sample_rate)
    edits: list[tuple[float, str, float]] = []  # (time, kind, cut end or seconds to add)
    for pause in unnatural(pauses, limits):
        keep = punctuation_to if pause.at_punctuation else unmarked_to
        if pause.length > keep + 2 * fade:
            edits.append((pause.start + keep / 2, "cut", pause.end - keep / 2))
    edits += [(m.at, "add", m.seconds) for m in add if m.at is not None]
    out, position = [], 0
    ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
    for time, kind, value in sorted(edits):
        a = int(time * sample_rate)
        if a - position < n:
            continue  # too close to the previous edit to fade
        head = audio[position:a].astype(np.float32, copy=True)
        if kind == "cut":
            b = int(value * sample_rate)
            head[-n:] = head[-n:] * ramp + audio[b:b + n] * (1 - ramp)
            out.append(head)
            position = b + n
        else:
            head[-n:] *= ramp
            tail = audio[a:a + n].astype(np.float32, copy=True) * ramp[::-1]
            out += [head, np.zeros(int(value * sample_rate), dtype=np.float32), tail]
            position = a + n
    out.append(audio[position:])
    return np.concatenate(out)

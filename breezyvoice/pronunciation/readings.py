"""The intended reading of every character of a text, and the other readings it has."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from breezyvoice.pronunciation.units import is_reading


class Unsupported(ValueError):
    """The text holds something the recogniser has no units for (Latin words, for instance)."""


@dataclass(frozen=True)
class Reading:
    characters: tuple[str, ...]  # the characters that are spoken, in order
    readings: tuple[str, ...]  # one bopomofo reading with tone per character
    others: tuple[tuple[str, ...], ...]  # per character, its other dictionary readings
    spoken: str = ""  # the text as it is spoken (numbers written out), punctuation included
    positions: tuple[int, ...] = ()  # where each character sits in ``spoken``


def load_polyphones(path: Path) -> dict[str, list[str]]:
    """Character -> all its readings, from g2pW's ``POLYPHONIC_CHARS.txt`` (tab-separated pairs)."""
    table: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        character, _, reading = line.partition("\t")
        if character and is_reading(reading):
            table.setdefault(character, []).append(reading)
    return table


def read(text: str, normalise: Callable[[str], str], predict: Callable[[str], Sequence[str | None]],
         polyphones: Mapping[str, Sequence[str]]) -> Reading:
    """``normalise`` writes numbers and symbols out as the synthesizer does; ``predict`` returns one
    entry per character of its input, a reading or None for what is not spoken."""
    spoken = normalise(text)
    predicted = predict(spoken)
    if len(predicted) != len(spoken):
        raise Unsupported(f"reading predictor returned {len(predicted)} entries for {len(spoken)} characters")
    characters, readings, positions = [], [], []
    for position, (character, reading) in enumerate(zip(spoken, predicted)):
        if reading is None:
            if character.isalnum():
                raise Unsupported(f"no reading for {character!r} in {text!r}")
            continue
        if not is_reading(reading):
            raise Unsupported(f"unexpected reading {reading!r} for {character!r}")
        characters.append(character)
        readings.append(reading)
        positions.append(position)
    if not readings:
        raise Unsupported(f"nothing to read in {text!r}")
    others = tuple(tuple(r for r in polyphones.get(c, ()) if r != reading) for c, reading in zip(characters, readings))
    return Reading(tuple(characters), tuple(readings), others, spoken, tuple(positions))

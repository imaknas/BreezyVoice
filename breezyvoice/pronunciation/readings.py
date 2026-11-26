"""The intended reading of every character of a text, and the other readings it has."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from breezyvoice.pronunciation.units import is_reading


# Readings of one character that are all in ordinary use in Taiwan: hearing one where another was
# predicted is not a mistake.
FREE_VARIANTS = {
    "液": ("ㄧㄝ4", "ㄧ4"), "那": ("ㄋㄚ4", "ㄋㄟ4"), "這": ("ㄓㄜ4", "ㄓㄟ4"), "誰": ("ㄕㄟ2", "ㄕㄨㄟ2"),
    "熟": ("ㄕㄡ2", "ㄕㄨ2"), "暴": ("ㄅㄠ4", "ㄆㄨ4"),
    "哪": ("ㄋㄚ3", "ㄋㄟ3"), "角": ("ㄐㄧㄠ3", "ㄐㄩㄝ2"),
}
# Characters with one reading a listener in Taiwan takes as right and another, standard elsewhere,
# that they hear as wrong: (the reading a take is held to, the readings refused, the margin below
# which the take is reported). The two readings of 血 differ by one glide and score close together,
# far inside the general margin. Of twenty takes a listener went through, the twelve scoring 3.2 or
# less were heard as ㄒㄩㄝ (ten) or could not be told (two), and the eight from 3.7 up were not.
SETTLED = {"血": ("ㄒㄧㄝ3", ("ㄒㄩㄝ4",), 3.5)}
# Initials and finals most speakers in Taiwan do not keep apart.
_MERGERS = str.maketrans({"ㄓ": "ㄗ", "ㄔ": "ㄘ", "ㄕ": "ㄙ"})


def _merged(reading: str) -> str:
    body = reading[:-1].translate(_MERGERS)
    if body.endswith("ㄥ") and not body.endswith(("ㄨㄥ", "ㄩㄥ")):
        body = body[:-1] + "ㄣ"
    return body


def _same_to_a_listener(character: str, a: str, b: str) -> bool:
    """Would hearing ``b`` for ``a`` pass as correct: a free variant, or the same sounds after the mergers?"""
    variants = FREE_VARIANTS.get(character, ())
    return (a in variants and b in variants) or _merged(a) == _merged(b)


_WRITTEN = re.compile(r"\[:([^\]]*)\]")


def _written_readings(text: str) -> tuple[str, dict[int, str]]:
    """``text`` without its ``[:注音]`` annotations, and the reading each one gives the character before it."""
    plain, written, length, last = [], {}, 0, 0
    for match in _WRITTEN.finditer(text):
        plain.append(text[last:match.start()])
        length += len(plain[-1])
        last = match.end()
        if length:
            written[length - 1] = match.group(1)
    plain.append(text[last:])
    return "".join(plain), written


class Unsupported(ValueError):
    """The text holds something the recogniser has no units for (Latin words, for instance)."""


@dataclass(frozen=True)
class Reading:
    characters: tuple[str, ...]  # the characters that are spoken, in order
    readings: tuple[str, ...]  # one bopomofo reading with tone per character
    others: tuple[tuple[str, ...], ...]  # per character, its other dictionary readings
    spoken: str = ""  # the text as it is spoken (numbers written out), punctuation included
    positions: tuple[int, ...] = ()  # where each character sits in ``spoken``
    margins: tuple[float | None, ...] = ()  # per character, its own margin for a wrong reading, if it has one


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
    entry per character of its input, a reading or None for what is not spoken. A reading written
    after a character (``好[:ㄏㄠ3]``) replaces the predicted one, as it does for the synthesizer; a
    character in ``SETTLED`` is held to its settled reading unless one is written."""
    spoken, written = _written_readings(normalise(text))
    predicted = list(predict(spoken))
    if len(predicted) != len(spoken):
        raise Unsupported(f"reading predictor returned {len(predicted)} entries for {len(spoken)} characters")
    for position, character in enumerate(spoken):
        if character in SETTLED and predicted[position] is not None:
            predicted[position] = SETTLED[character][0]
    for position, reading in written.items():  # a reading written in the text is the one intended
        predicted[position] = reading
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
    # tone apart, a rival must be something a listener would hear as different
    others = tuple(tuple(r for r in polyphones.get(c, ())
                         if r != reading and (r[:-1] == reading[:-1] or not _same_to_a_listener(c, reading, r)))
                   for c, reading in zip(characters, readings))
    held = [c in SETTLED and reading == SETTLED[c][0] for c, reading in zip(characters, readings)]
    others = tuple(o + tuple(r for r in SETTLED[c][1] if r not in o) if h else o
                   for c, o, h in zip(characters, others, held))
    margins = tuple(SETTLED[c][2] if h else None for c, h in zip(characters, held))
    return Reading(tuple(characters), tuple(readings), others, spoken, tuple(positions), margins)

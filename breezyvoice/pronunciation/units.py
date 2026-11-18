"""Sound units of a bopomofo reading, and the readings a syllable could be confused with."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

INITIALS = frozenset("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ")
TONES = "12345"
_NO_FINAL = "-"  # ㄓ ㄔ ㄕ ㄖ ㄗ ㄘ ㄙ can stand alone


def is_reading(text: str) -> bool:
    """A bopomofo syllable followed by a tone digit, as the reading predictor writes it (ㄊㄧㄠ2)."""
    return len(text) >= 2 and text[-1] in TONES and all("ㄅ" <= ch <= "ㄯ" for ch in text[:-1])


def units(reading: str) -> list[str]:
    """Initial and tonal final: ㄊㄧㄠ2 -> [ㄊ, ㄧㄠ2]; ㄕ4 -> [ㄕ, -4]; ㄧ1 -> [ㄧ1].

    The tone goes with the final because that is where pitch is carried, and a final in two tones is
    then two units the recogniser has to tell apart.
    """
    if not is_reading(reading):
        raise ValueError(f"not a bopomofo reading with a tone: {reading!r}")
    body, tone = reading[:-1], reading[-1]
    if body[0] in INITIALS:
        return [body[0], (body[1:] or _NO_FINAL) + tone]
    return [body + tone]


def flatten(readings: Iterable[str]) -> list[str]:
    return [unit for reading in readings for unit in units(reading)]


def alternatives(reading: str, other_readings: Sequence[str], known_units: Mapping[str, int] | Iterable[str]) -> list[str]:
    """Readings ``reading`` could have been said as instead, most informative first.

    The same syllable in the other tones, then the character's other dictionary readings. Readings
    the recogniser has no unit for are left out: it cannot say anything about them.
    """
    known = set(known_units)
    body = reading[:-1]
    candidates = [body + tone for tone in TONES if tone != reading[-1]]
    candidates += [r for r in other_readings if r != reading and r not in candidates]
    return [r for r in candidates if is_reading(r) and all(u in known for u in units(r))]

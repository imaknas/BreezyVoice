"""Text → the annotated string BreezyVoice was trained on.

Steps: normalise the parts outside manual ``[...]`` annotations, clean punctuation the way upstream
did, then add ``[:注音]`` after characters that are rare or whose predicted reading is not their
most common one (upstream's ``get_bopomofo_rare``).
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

from breezyvoice.text.normalize import normalize_zh

# g2p(text) -> one reading per character (None where it has none), e.g. g2pw.G2PWConverter.
G2P = Callable[[str], Sequence[str | None]]

_CHINESE = re.compile(r"[一-鿿]")
_BRACKET = re.compile(r"(\[[^\]]*\])")


def contains_chinese(text: str) -> bool:
    return bool(_CHINESE.search(text))


def _replace_blank(text: str) -> str:
    """Drop spaces except between two ASCII words (upstream frontend_utils.replace_blank)."""
    out = []
    for i, c in enumerate(text):
        if c == " ":
            prev = text[i - 1] if i > 0 else " "
            nxt = text[i + 1] if i + 1 < len(text) else " "
            if prev.isascii() and prev != " " and nxt.isascii() and nxt != " ":
                out.append(c)
        else:
            out.append(c)
    return "".join(out)


def _clean_zh(text: str) -> str:
    text = text.replace("\n", "")
    text = _replace_blank(text)
    text = text.replace(".", "、")
    text = text.replace(" - ", "，")
    for ch in ("（", "）", "【", "】", "`"):
        text = text.replace(ch, "")
    text = text.replace("——", " ")
    return text


def _normalize_segment(text: str, spell_english_number: Callable[[str], str]) -> str:
    if contains_chinese(text):
        return _clean_zh(normalize_zh(text))
    return re.sub(r"\d+", lambda m: spell_english_number(m.group(0)), text)


def normalize(text: str, spell_english_number: Callable[[str], str]) -> str:
    """Normalise ``text``; ``[...]`` annotations pass through unchanged."""
    text = text.strip()
    parts = _BRACKET.split(text)
    out = [p if _BRACKET.fullmatch(p) else _normalize_segment(p, spell_english_number) for p in parts]
    joined = "".join(out).strip()
    if contains_chinese(joined):
        joined = re.sub(r"[，,]+$", "。", joined)
    return joined


class Annotator:
    """Adds 注音 to rare or non-default readings, given word statistics and a g2p model."""

    def __init__(self, g2p: G2P, char_frequency: dict[str, int], char_readings: dict[str, list[str]],
                 always_annotate: set[str]):
        self._g2p = g2p
        self._freq = char_frequency
        self._readings = char_readings
        self._always = always_annotate

    def __call__(self, text: str) -> str:
        readings = self._g2p(text)
        out = []
        for i, (ch, reading) in enumerate(zip(text, readings)):
            next_ch = text[i + 1] if i + 1 < len(text) else None
            if next_ch == "[" or reading is None:
                out.append(ch)
                continue
            freq = self._freq.get(ch, 0)
            known = self._readings.get(ch, [])
            if freq < 500:
                out.append(f"{ch}[:{reading}]")
            elif len(known) >= 2 and reading != known[0] and (freq < 10000 or ch in self._always):
                out.append(f"{ch}[:{reading}]")
            else:
                out.append(ch)
        return "".join(out)


class TextFrontend:
    def __init__(self, annotator: Callable[[str], str], spell_english_number: Callable[[str], str]):
        self._annotate = annotator
        self._spell = spell_english_number

    def __call__(self, text: str) -> str:
        return self._annotate(normalize(text, self._spell))

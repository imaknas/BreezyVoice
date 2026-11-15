"""Character error rate between what was asked for and what a recogniser heard."""
from __future__ import annotations

import difflib
from typing import Callable, Sequence


def comparable(text: str, unify_script: Callable[[str], str]) -> str:
    """Reduce a text to what a recogniser can be held to: letters and digits, one script, lower case.

    Recognisers differ in punctuation and in Traditional or Simplified output; neither is an error
    of the synthesizer.
    """
    return "".join(ch for ch in unify_script(text) if ch.isalnum()).lower()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insertions, deletions, substitutions)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str, unify_script: Callable[[str], str]) -> float:
    """Edit distance over the reference length, after ``comparable``. Can exceed 1."""
    ref = comparable(reference, unify_script)
    if not ref:
        raise ValueError("reference has no comparable characters")
    return edit_distance(ref, comparable(hypothesis, unify_script)) / len(ref)


def syllable_match(reference: str, hypothesis: str, to_syllables: Callable[[str], Sequence[str]]) -> float:
    """Similarity (0-1) of the two texts as syllable sequences.

    Homophones and script differences do not count against the synthesizer here, which makes this
    the measure to gate on when deciding whether a take has to be redone.
    """
    return difflib.SequenceMatcher(None, to_syllables(reference), to_syllables(hypothesis), autojunk=False).ratio()

"""Character error rate between what was asked for and what a recogniser heard."""
from __future__ import annotations

import difflib
from typing import Callable, Collection, Sequence


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


def syllable_match(reference: str, hypothesis: str, to_syllables: Callable[[str], Sequence[str]],
                   variants: Collection[tuple[str, str]] = frozenset()) -> float:
    """Similarity (0-1) of the two texts as syllable sequences.

    Homophones and script differences do not count against the synthesizer here, which makes this
    the measure to gate on when deciding whether a take has to be redone. ``variants`` lists
    (expected, heard) syllable pairs that are also accepted: readings that are correct for the
    speaker's variety of the language although the syllable converter expects another.
    """
    ref, hyp = list(to_syllables(reference)), list(to_syllables(hypothesis))
    if not ref and not hyp:
        return 1.0
    matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    same = sum(block.size for block in matcher.get_matching_blocks())
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "replace" and a1 - a0 == b1 - b0:
            same += sum(pair in variants for pair in zip(ref[a0:a1], hyp[b0:b1]))
    return 2 * same / (len(ref) + len(hyp))


_BREAKS = set("，、；：,;:—…。！？!?")


def marked_breaks(text: str) -> int:
    """How many places inside the text call for a pause: runs of punctuation, not counting the end."""
    count, in_run = 0, False
    for ch in text.strip().rstrip("".join(_BREAKS)):
        if ch in _BREAKS and not in_run:
            count += 1
        in_run = ch in _BREAKS
    return count

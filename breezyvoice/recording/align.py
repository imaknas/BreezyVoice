"""Group recognised pieces of a recording into takes of script lines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from breezyvoice.evaluation.text import comparable, edit_distance

# A group of pieces counts as a take of a line when it differs from it by at most this share of the
# line's characters. Recognisers mishear a few characters even of clean speech.
MAX_ERROR_RATE = 0.3
# A reader may stop for breath a few times inside a long line.
MAX_PIECES_PER_TAKE = 5


@dataclass(frozen=True)
class Take:
    line: str  # script line id
    first: int  # index of its first piece
    last: int  # index of its last piece, inclusive
    error_rate: float  # character error rate against the line, by the recogniser used


def assign(heard: Sequence[str], lines: Mapping[str, str], unify_script: Callable[[str], str]) -> list[Take]:
    """Choose non-overlapping groups of consecutive pieces that best match script lines.

    ``heard`` is the recognised text of each piece, in recording order. Every group is scored by how
    many characters of its line it gets right; the grouping with the highest total wins, so a pause
    inside a sentence does not split it and a piece that belongs to no line (a cough, an aside, an
    abandoned start) is left out. Pieces are not required to follow the script order: a reader may
    go back to a line.
    """
    texts = [comparable(h, unify_script) for h in heard]
    targets = {k: comparable(v, unify_script) for k, v in lines.items()}

    def best_line(i: int, j: int) -> tuple[float, str, float] | None:
        said = "".join(texts[i:j + 1])
        if not said:
            return None
        found = None
        for key, target in targets.items():
            if not target or abs(len(said) - len(target)) > MAX_ERROR_RATE * len(target):
                continue
            rate = edit_distance(target, said) / len(target)
            score = len(target) * (1 - rate)
            if rate <= MAX_ERROR_RATE and (found is None or score > found[0]):
                found = (score, key, rate)
        return found

    n = len(texts)
    best = [0.0] * (n + 1)  # best[i]: highest total score of the first i pieces
    choice: list[Take | None] = [None] * (n + 1)
    for end in range(1, n + 1):
        best[end], choice[end] = best[end - 1], None
        for start in range(max(0, end - MAX_PIECES_PER_TAKE), end):
            match = best_line(start, end - 1)
            if match and best[start] + match[0] > best[end]:
                best[end] = best[start] + match[0]
                choice[end] = Take(match[1], start, end - 1, match[2])
    takes, end = [], n
    while end > 0:
        take = choice[end]
        if take is None:
            end -= 1
        else:
            takes.append(take)
            end = take.first
    return takes[::-1]


def preferred(takes: Sequence[Take]) -> dict[str, Take]:
    """One take per line: the last clean one, since a reader repeats a line to improve on it;
    failing that, the one with the fewest errors."""
    by_line: dict[str, list[Take]] = {}
    for take in takes:
        by_line.setdefault(take.line, []).append(take)
    chosen = {}
    for line, mine in by_line.items():
        clean = [t for t in mine if t.error_rate <= 0.1]
        chosen[line] = clean[-1] if clean else min(mine, key=lambda t: t.error_rate)
    return chosen

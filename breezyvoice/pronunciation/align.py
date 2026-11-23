"""When each syllable of the intended text is spoken, from the recogniser's frame posteriors."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from breezyvoice.pronunciation.units import INITIALS, units

FRAME_SECONDS = 0.02  # the speech encoder produces one frame per 20 ms


def unit_frames(log_probs: torch.Tensor, sequence: Sequence[int], blank: int = 0) -> list[tuple[int, int]]:
    """The most likely frames (first, last) of each unit of ``sequence``, in order.

    The standard best path through the CTC lattice: blanks may sit between units, a unit may span
    several frames, and every unit is given at least one.
    """
    frames = log_probs.size(0)
    states = [blank]
    for unit in sequence:
        states += [unit, blank]
    count = len(states)
    if frames * 1 < len(sequence):
        raise ValueError(f"{len(sequence)} units do not fit into {frames} frames")
    minus_inf = float("-inf")
    emit = log_probs[:, states]  # (frames, states)
    score = torch.full((frames, count), minus_inf)
    back = torch.zeros((frames, count), dtype=torch.long)
    score[0, 0] = emit[0, 0]
    if count > 1:
        score[0, 1] = emit[0, 1]
    # a unit may be entered by skipping the blank before it unless it repeats the previous unit
    can_skip = torch.tensor([s >= 2 and states[s] != blank and states[s] != states[s - 2] for s in range(count)])
    for t in range(1, frames):
        stay = score[t - 1]
        step = torch.cat([torch.tensor([minus_inf]), score[t - 1, :-1]])
        skip = torch.cat([torch.tensor([minus_inf, minus_inf]), score[t - 1, :-2]])
        skip = torch.where(can_skip, skip, torch.full_like(skip, minus_inf))
        best, choice = torch.stack([stay, step, skip]).max(dim=0)
        score[t] = best + emit[t]
        back[t] = choice
    state = count - 1 if count == 1 or score[-1, count - 1] >= score[-1, count - 2] else count - 2
    path = [state]
    for t in range(frames - 1, 0, -1):
        state -= int(back[t, state])
        path.append(state)
    path.reverse()
    spans: list[list[int]] = [[-1, -1] for _ in sequence]
    for t, s in enumerate(path):
        if s % 2 == 1:
            k = s // 2
            if spans[k][0] < 0:
                spans[k][0] = t
            spans[k][1] = t
    if any(a < 0 for a, _ in spans):
        raise ValueError("alignment did not reach every unit")
    return [(a, b) for a, b in spans]


def syllable_times(log_probs: torch.Tensor, readings: Sequence[str], unit_index: Mapping[str, int]) -> list[tuple[float, float]]:
    """(start, end) in seconds of each syllable: from its first unit's first frame to its last unit's last."""
    per_syllable = [units(r) for r in readings]
    flat = [unit_index[u] for group in per_syllable for u in group]
    spans = unit_frames(log_probs, flat)
    times, position = [], 0
    for group in per_syllable:
        first, last = spans[position][0], spans[position + len(group) - 1][1]
        times.append((first * FRAME_SECONDS, (last + 1) * FRAME_SECONDS))
        position += len(group)
    return times


def heard_units(log_probs: torch.Tensor, blank: int = 0) -> list[tuple[int, int, float]]:
    """(unit, first frame, probability there) of each unit on the frame-by-frame best path."""
    out, previous = [], blank
    if not log_probs.numel():
        return out
    best = log_probs.max(dim=-1)
    for t, (i, value) in enumerate(zip(best.indices.tolist(), best.values.tolist())):
        if i != previous and i != blank:
            out.append((i, t, math.exp(value)))
        previous = i
    return out


def stray_units(log_probs: torch.Tensor, syllables: Sequence[tuple[float, float]], vocabulary: Sequence[str],
                blank: int = 0) -> tuple[list[str], list[str]]:
    """Units the recogniser hears before the first and after the last syllable of the text.

    A take sometimes starts or ends with a syllable or two that the text does not contain. They sit
    outside the aligned sentence, where anything recognised is by definition not part of it.
    """
    def heard(frames: torch.Tensor) -> list[str]:
        return [vocabulary[i] for i, _, _ in heard_units(frames, blank)]

    first = max(0, int(round(syllables[0][0] / FRAME_SECONDS)) - 1)
    last = int(round(syllables[-1][1] / FRAME_SECONDS)) + 1
    return heard(log_probs[:first]), heard(log_probs[last:])


@dataclass(frozen=True)
class Repeat:
    first: int  # the first syllable said twice, among the syllables of the text
    count: int  # how many syllables in a row are said twice
    at: float  # seconds, where the extra copy starts


def _inserted(text: Sequence[str], heard: Sequence[str]) -> list[tuple[int, list[int]]]:
    """Runs of ``heard`` that a shortest edit path leaves unmatched: (units of ``text`` before the
    run, positions of the run in ``heard``)."""
    n, m = len(text), len(heard)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + (text[i - 1] != heard[j - 1]), cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    before: dict[int, int] = {}
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (text[i - 1] != heard[j - 1]):
            i, j = i - 1, j - 1
        elif j and cost[i][j] == cost[i][j - 1] + 1:
            before[j - 1] = i
            j -= 1
        else:
            i -= 1
    runs: list[tuple[int, list[int]]] = []
    for j in sorted(before):
        if runs and runs[-1][1][-1] == j - 1 and runs[-1][0] == before[j]:
            runs[-1][1].append(j)
        else:
            runs.append((before[j], [j]))
    return runs


@dataclass(frozen=True)
class Inserted:
    after: int  # the syllable of the text it is heard after
    units: tuple[str, ...]
    at: float  # seconds, where it starts


def inserted(log_probs: torch.Tensor, readings: Sequence[str], unit_index: Mapping[str, int],
             min_probability: float = 0.8) -> list[Inserted]:
    """Whole syllables heard inside the sentence that its text does not have (回神來才發現 for 回神才發現).

    Like ``repeated`` this reads the best path frame by frame, and it includes what ``repeated``
    finds. Only runs made of an initial and a final count, each unit heard with at least
    ``min_probability``: a lone final between two syllables is nearly always a vowel carried over
    from its neighbour, and a listener hears nothing added there. Sounds before the first and after
    the last syllable are ``stray_units``.
    """
    names = {i: unit for unit, i in unit_index.items()}
    groups = [units(r) for r in readings]
    flat = [u for g in groups for u in g]
    starts, position = [], 0
    for group in groups:
        starts.append(position)
        position += len(group)
    heard = heard_units(log_probs)
    said = [names[i] for i, _, _ in heard]
    found = []
    for before, run in _inserted(flat, said):
        extra = [said[j] for j in run]
        whole = len(extra) % 2 == 0 and all((u in INITIALS) == (k % 2 == 0) for k, u in enumerate(extra))
        if before in (0, len(flat)) or not whole or min(heard[j][2] for j in run) < min_probability:
            continue
        after = sum(start < before for start in starts) - 1
        found.append(Inserted(after, tuple(extra), heard[run[0]][1] * FRAME_SECONDS))
    return found


def clarity(log_probs: torch.Tensor, readings: Sequence[str], unit_index: Mapping[str, int]) -> list[float]:
    """How plainly each syllable is heard: the log-probability of its least clear unit, taken at the
    frame of the unit's span where the recogniser hears it best. Near 0 for a clearly spoken
    syllable; far below when the sound is slurred, cut short or not there."""
    groups = [units(r) for r in readings]
    flat = [unit_index[u] for g in groups for u in g]
    best = [float(log_probs[a:b + 1, u].max()) for u, (a, b) in zip(flat, unit_frames(log_probs, flat))]
    scores, position = [], 0
    for group in groups:
        scores.append(min(best[position:position + len(group)]))
        position += len(group)
    return scores


def _same_syllables(groups: Sequence[Sequence[str]], said: Sequence[str]) -> bool:
    """Whether ``said`` is these syllables again. The tone of a syllable with an initial may differ,
    as it does when a word is stumbled over; a bare final must match exactly, because one long vowel
    is easily heard as the same vowel in two tones."""
    position = 0
    for group in groups:
        now = said[position:position + len(group)]
        position += len(group)
        if len(group) == 1:
            if list(now) != list(group):
                return False
        elif len(now) != 2 or now[0] != group[0] or now[1][:-1] != group[1][:-1]:
            return False
    return position == len(said)


def repeated(log_probs: torch.Tensor, readings: Sequence[str], unit_index: Mapping[str, int],
             min_probability: float = 0.5, min_apart: float = 0.1) -> list[Repeat]:
    """Syllables of the text that are heard a second time right beside themselves (就就看 for 就看).

    The forced alignment cannot show this: it gives the syllable both copies' frames. The best path
    frame by frame can, since it carries the units twice. Only an extra run that spells out whole
    neighbouring syllables counts, each unit heard with at least ``min_probability``; other extra
    units are too often the recogniser wavering to be reported. The two copies must also start at
    least ``min_apart`` seconds from each other: a held vowel is sometimes heard as the same unit
    twice a few frames apart, which is too close together to be two syllables.
    """
    names = {i: unit for unit, i in unit_index.items()}
    groups = [units(r) for r in readings]
    starts, position = [], 0
    for group in groups:
        starts.append(position)
        position += len(group)
    heard = heard_units(log_probs)
    said = [names[i] for i, _, _ in heard]
    found = []
    for before, run in _inserted([u for g in groups for u in g], said):
        if min(heard[j][2] for j in run) < min_probability:
            continue
        extra = [said[j] for j in run]
        for first in range(len(groups)):
            count, length = 0, 0
            while first + count < len(groups) and length < len(extra):
                length += len(groups[first + count])
                count += 1
            if length != len(extra) or not _same_syllables(groups[first:first + count], extra):
                continue
            if starts[first] == before:  # the copy that was kept follows the extra one
                other = run[-1] + 1
            elif starts[first] + length == before:  # ... or comes before it
                other = run[0] - length
            else:
                continue
            if not 0 <= other < len(heard):
                continue
            if abs(heard[other][1] - heard[run[0]][1]) * FRAME_SECONDS >= min_apart:
                found.append(Repeat(first, count, heard[run[0]][1] * FRAME_SECONDS))
                break
    return found

"""When each syllable of the intended text is spoken, from the recogniser's frame posteriors."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch

from breezyvoice.pronunciation.units import units

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

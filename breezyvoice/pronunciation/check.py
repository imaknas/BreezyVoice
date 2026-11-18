"""Score a take against its intended readings, one syllable at a time."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from breezyvoice.pronunciation.units import TONES, alternatives, flatten, is_reading, units


@dataclass(frozen=True)
class Syllable:
    index: int  # position among the syllables of the text
    reading: str  # the intended reading
    rival: str | None  # the alternative that fits the audio best, or None when there was none to try
    margin: float  # how much better the intended reading fits than that rival; negative: the rival fits better


def _costs(log_probs: torch.Tensor, sequences: Sequence[Sequence[int]]) -> torch.Tensor:
    """Negative log-likelihood of each unit sequence given the frame posteriors (time, units)."""
    frames = log_probs.size(0)
    batch = log_probs.unsqueeze(1).expand(frames, len(sequences), log_probs.size(1))
    lengths = torch.tensor([len(s) for s in sequences])
    targets = torch.nn.utils.rnn.pad_sequence([torch.tensor(s) for s in sequences], batch_first=True)
    return torch.nn.functional.ctc_loss(batch, targets, torch.full((len(sequences),), frames), lengths,
                                        blank=0, reduction="none", zero_infinity=True)


def _in_any_tone(reading: str, unit_index: Mapping[str, int]) -> list[str]:
    """The syllable of ``reading`` in every tone the recogniser has units for."""
    candidates = [reading[:-1] + tone for tone in TONES]
    return [r for r in candidates if is_reading(r) and all(u in unit_index for u in units(r))]


def score(log_probs: torch.Tensor, readings: Sequence[str], other_readings: Sequence[Sequence[str]],
          unit_index: Mapping[str, int], tones: bool = True) -> list[Syllable]:
    """Compare the whole sentence as intended with the sentence changed in one syllable at a time.

    ``other_readings[i]`` lists the other dictionary readings of the character behind syllable ``i``.
    Changing one syllable and keeping the rest makes the comparison about that syllable alone, and
    rivals restricted to plausible confusions keep accent from being read as error: a merger of two
    initials is not among the rivals, so it cannot lose to one.

    With ``tones`` the rivals are the same syllable in the other tones and the character's other
    readings. Without, the question is only whether another reading's sounds were said: each side is
    taken in whichever tone fits best, so a slip of tone alone does not count against the intended
    reading, nor for a rival that happens to carry the tone that was spoken.
    """
    intended = list(readings)
    variants: list[list[str]] = [intended]
    owners: list[tuple[int, str, str]] = []  # (syllable, "intended" or the rival's dictionary reading, variant reading)

    def add(i: int, label: str, reading: str) -> None:
        variants.append(intended[:i] + [reading] + intended[i + 1:])
        owners.append((i, label, reading))

    for i, reading in enumerate(readings):
        if tones:
            for rival in alternatives(reading, other_readings[i], unit_index):
                add(i, rival, rival)
            continue
        rivals = alternatives(reading, other_readings[i], unit_index, tones=False)
        if not rivals:
            continue
        for same in _in_any_tone(reading, unit_index):
            if same != reading:
                add(i, "intended", same)
        for rival in rivals:
            for variant in _in_any_tone(rival, unit_index):
                add(i, rival, variant)

    costs = _costs(log_probs, [[unit_index[u] for u in flatten(v)] for v in variants]).tolist()
    base = costs[0]
    intended_cost = {i: base for i in range(len(readings))}
    rival_cost: dict[int, tuple[float, str]] = {}
    for (i, label, _), cost in zip(owners, costs[1:]):
        if label == "intended":
            intended_cost[i] = min(intended_cost[i], cost)
        elif i not in rival_cost or cost < rival_cost[i][0]:
            rival_cost[i] = (cost, label)
    out = []
    for i, reading in enumerate(readings):
        if i in rival_cost:
            out.append(Syllable(i, reading, rival_cost[i][1], rival_cost[i][0] - intended_cost[i]))
        else:
            out.append(Syllable(i, reading, None, float("inf")))
    return out

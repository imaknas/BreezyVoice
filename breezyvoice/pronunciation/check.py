"""Score a take against its intended readings, one syllable at a time."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from breezyvoice.pronunciation.units import alternatives, flatten


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


def score(log_probs: torch.Tensor, readings: Sequence[str], other_readings: Sequence[Sequence[str]],
          unit_index: Mapping[str, int]) -> list[Syllable]:
    """Compare the whole sentence as intended with the sentence changed in one syllable at a time.

    ``other_readings[i]`` lists the other dictionary readings of the character behind syllable ``i``.
    Changing one syllable and keeping the rest makes the comparison about that syllable alone, and
    a rival restricted to plausible confusions keeps accent from being read as error: a merger of
    two initials is not among the rivals, so it cannot lose to one.
    """
    intended = [unit_index[u] for u in flatten(readings)]
    variants, owners = [intended], []
    for i, reading in enumerate(readings):
        for rival in alternatives(reading, other_readings[i], unit_index):
            changed = list(readings[:i]) + [rival] + list(readings[i + 1:])
            variants.append([unit_index[u] for u in flatten(changed)])
            owners.append((i, rival))
    costs = _costs(log_probs, variants)
    base = float(costs[0])
    best: dict[int, tuple[float, str]] = {}
    for (i, rival), cost in zip(owners, costs[1:].tolist()):
        if i not in best or cost - base < best[i][0]:
            best[i] = (cost - base, rival)
    return [Syllable(i, reading, best[i][1] if i in best else None, best[i][0] if i in best else float("inf"))
            for i, reading in enumerate(readings)]

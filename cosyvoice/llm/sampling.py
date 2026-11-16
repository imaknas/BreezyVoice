"""How the language model picks the next speech token.

A sampler takes the log-probabilities of one decoding step and the tokens decoded so far and
returns a token id. All of them draw from the CPU generator, so a seed gives the same take on every
device.
"""
from typing import Callable, Sequence

import torch

Sampler = Callable[[torch.Tensor, Sequence[int]], int]


def _draw(prob: torch.Tensor) -> int:
    return int(prob.cpu().multinomial(1, replacement=True))


def top_k(k: int = 25) -> Sampler:
    """Sample among the ``k`` most likely tokens, in proportion to their probability."""

    def sample(log_probs: torch.Tensor, decoded: Sequence[int]) -> int:
        prob, indices = log_probs.softmax(dim=-1).topk(k)
        return int(indices[_draw(prob)])

    return sample


def repetition_aware(top_p: float = 0.8, k: int = 25, window: int = 10, max_share: float = 0.1) -> Sampler:
    """Nucleus sampling that backs off when it starts to repeat itself.

    The token is drawn from the most likely tokens that together reach ``top_p`` of the probability
    (at most ``k`` of them), which keeps unlikely tokens out; a token-by-token decoder that samples
    one rarely recovers its place. If the drawn token already makes up ``max_share`` of the last
    ``window`` tokens, it is excluded and the step is drawn again from everything else, which breaks
    the loops such a decoder otherwise falls into (VALL-E 2, arXiv:2406.05370; the defaults are
    CosyVoice's).
    """

    def sample(log_probs: torch.Tensor, decoded: Sequence[int]) -> int:
        prob = log_probs.softmax(dim=-1)
        ordered, indices = prob.sort(descending=True, stable=True)
        before = ordered.cumsum(dim=-1) - ordered  # probability mass ahead of each token
        keep = int(min(k, max(1, int((before < top_p).sum()))))
        token = int(indices[_draw(ordered[:keep])])
        recent = decoded[-window:]
        if sum(t == token for t in recent) >= window * max_share:
            rest = prob.clone()
            rest[token] = 0
            token = _draw(rest)
        return token

    return sample

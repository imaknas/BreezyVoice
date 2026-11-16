"""Token samplers (no model)."""
import torch

from cosyvoice.llm.sampling import repetition_aware, top_k


def _log_probs(values):
    return torch.tensor(values, dtype=torch.float32).log_softmax(dim=-1)


def test_top_k_draws_as_the_original_decoder_did():
    g = torch.Generator().manual_seed(3)
    scores = torch.randn(4097, generator=g).log_softmax(dim=-1)
    torch.manual_seed(5)
    prob, indices = scores.softmax(dim=-1).topk(25)
    expected = [int(indices[prob.multinomial(1, replacement=True)]) for _ in range(20)]
    torch.manual_seed(5)
    sampler = top_k(25)
    assert [sampler(scores, []) for _ in range(20)] == expected


def test_top_k_stays_within_the_k_most_likely():
    scores = _log_probs([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    torch.manual_seed(0)
    assert {top_k(2)(scores, []) for _ in range(200)} == {0, 1}


def test_nucleus_keeps_only_tokens_needed_to_reach_top_p():
    # the first two tokens already pass 0.8, so the third is never needed
    scores = torch.tensor([0.5, 0.35, 0.1, 0.05]).log()
    torch.manual_seed(0)
    sampler = repetition_aware(top_p=0.8, k=25)
    assert {sampler(scores, []) for _ in range(300)} == {0, 1}


def test_nucleus_is_capped_at_k_tokens():
    scores = torch.full((100,), 0.01).log()
    torch.manual_seed(0)
    sampler = repetition_aware(top_p=0.99, k=3)
    assert {sampler(scores, []) for _ in range(300)} <= {0, 1, 2}


def test_a_token_that_already_fills_the_window_is_not_drawn_again():
    scores = torch.tensor([0.97, 0.01, 0.01, 0.01]).log()
    torch.manual_seed(0)
    sampler = repetition_aware(window=10, max_share=0.1)
    assert all(sampler(scores, [3, 2, 0]) != 0 for _ in range(100))
    assert all(sampler(scores, [3, 2, 1]) == 0 for _ in range(100))


def test_only_the_recent_window_counts():
    scores = torch.tensor([0.97, 0.01, 0.01, 0.01]).log()
    torch.manual_seed(0)
    sampler = repetition_aware(window=3, max_share=0.34)
    assert sampler(scores, [0, 1, 2, 3]) == 0  # the earlier 0 has left the window
    assert sampler(scores, [1, 2, 0, 0]) != 0

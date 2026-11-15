"""The in-place decode path must give exactly what the concatenating ``forward_chunk`` path gives."""
import pytest
import torch

from cosyvoice.transformer.encoder import TransformerEncoder


def _encoder(blocks: int = 2, size: int = 32, heads: int = 4) -> TransformerEncoder:
    torch.manual_seed(0)
    return TransformerEncoder(
        input_size=size, output_size=size, attention_heads=heads, linear_units=64, num_blocks=blocks,
        dropout_rate=0.1, positional_dropout_rate=0.1, attention_dropout_rate=0,
        input_layer="linear_legacy", pos_enc_layer_type="rel_pos_espnet",
        selfattention_layer_type="rel_selfattn", static_chunk_size=1,
    ).eval()


def _reference(enc: TransformerEncoder, prefix: torch.Tensor, steps: torch.Tensor) -> list[torch.Tensor]:
    """Outputs of the prefix and of each later position, growing the cache by concatenation."""
    empty = torch.zeros((0, 0, 0, 0), device=prefix.device)
    n = prefix.size(1)
    causal = torch.tril(torch.ones((1, n, n), device=prefix.device)).to(torch.bool)
    y, att, cnn = enc.forward_chunk(prefix, 0, -1, empty, empty, causal)
    outs = [y]
    one = torch.ones((1, 1, 1), dtype=torch.bool, device=prefix.device)
    for x in steps.unbind(1):
        y, att, cnn = enc.forward_chunk(x.view(1, 1, -1), 0, -1, att, cnn, one)
        outs.append(y)
    return outs


def _in_place(enc: TransformerEncoder, prefix: torch.Tensor, steps: torch.Tensor) -> list[torch.Tensor]:
    y, caches = enc.decode_prefill(prefix, capacity=prefix.size(1) + steps.size(1))
    outs = [y]
    for x in steps.unbind(1):
        outs.append(enc.decode_step(x.view(1, 1, -1), caches))
    return outs


@pytest.mark.parametrize("prefix_len", [1, 7, 40])
def test_decode_step_matches_forward_chunk(prefix_len):
    enc = _encoder()
    g = torch.Generator().manual_seed(1)
    prefix = torch.randn(1, prefix_len, 32, generator=g)
    steps = torch.randn(1, 25, 32, generator=g)
    with torch.inference_mode():
        expected = _reference(enc, prefix, steps)
        got = _in_place(enc, prefix, steps)
    assert len(got) == len(expected)
    # Rounding only: with heads this narrow the shorter position product takes a different BLAS
    # kernel. Exact equality is asserted on the released weights below.
    for i, (a, b) in enumerate(zip(expected, got)):
        assert torch.allclose(a, b, rtol=0, atol=1e-5), f"output {i} differs by {(a - b).abs().max().item():.3e}"


def test_decode_step_beyond_capacity_fails():
    enc = _encoder()
    with torch.inference_mode():
        _, caches = enc.decode_prefill(torch.randn(1, 4, 32), capacity=5)
        enc.decode_step(torch.randn(1, 1, 32), caches)
        with pytest.raises(ValueError, match="overflow"):
            enc.decode_step(torch.randn(1, 1, 32), caches)


def test_decode_longer_than_positional_table_fails():
    enc = _encoder()
    limit = enc.embed.pos_enc.pe.size(1) // 2 + 1
    with torch.inference_mode(), pytest.raises(ValueError, match="positional table"):
        enc.decode_prefill(torch.randn(1, 4, 32), capacity=limit + 1)


# --- with the released weights -------------------------------------------------------------------

def _devices() -> list[str]:
    return ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]


@pytest.fixture(scope="module")
def llm():
    from breezyvoice import assets
    from breezyvoice.model import build_llm

    try:
        weights = assets.resolve().weights_dir
    except assets.AssetsMissing as e:
        pytest.skip(str(e))
    net = build_llm()
    net.load_state_dict(torch.load(weights / "llm.pt", map_location="cpu", weights_only=True))
    return net.eval()


@pytest.mark.slow
@pytest.mark.parametrize("device", _devices())
def test_released_lm_logits_are_bit_identical(llm, device):
    """A long prefix, as a real reference clip gives, then enough steps to cross cache-growth sizes."""
    llm.to(device)
    enc = llm.llm
    g = torch.Generator().manual_seed(0)
    prefix = torch.randn(1, 900, 1024, generator=g).to(device)
    tokens = torch.randint(0, llm.speech_token_size, (120,), generator=g)
    steps = llm.speech_embedding.weight[tokens.to(device)].unsqueeze(0)
    with torch.inference_mode():
        expected = [llm.llm_decoder(y[:, -1]) for y in _reference(enc, prefix, steps)]
        got = [llm.llm_decoder(y[:, -1]) for y in _in_place(enc, prefix, steps)]
    for i, (a, b) in enumerate(zip(expected, got)):
        assert torch.equal(a, b), f"logits at position {i} differ by {(a - b).abs().max().item():.3e}"


def _sample(llm, device: str, seed: int) -> list[int]:
    llm.to(device)
    g = torch.Generator().manual_seed(seed)
    text = torch.randint(0, 50000, (1, 12), generator=g, dtype=torch.int32)
    prompt_text = torch.randint(0, 50000, (1, 20), generator=g, dtype=torch.int32)
    prompt_speech = torch.randint(0, llm.speech_token_size, (1, 300), generator=g, dtype=torch.int32)
    embedding = torch.randn(1, 192, generator=g)
    torch.manual_seed(seed)
    out = llm.inference(
        text=text.to(device), text_len=torch.tensor([text.size(1)], dtype=torch.int32, device=device),
        prompt_text=prompt_text.to(device),
        prompt_text_len=torch.tensor([prompt_text.size(1)], dtype=torch.int32, device=device),
        prompt_speech_token=prompt_speech.to(device),
        prompt_speech_token_len=torch.tensor([prompt_speech.size(1)], dtype=torch.int32, device=device),
        embedding=embedding.to(device), sampling=25, max_token_text_ratio=10, min_token_text_ratio=3,
    )
    return out.squeeze(0).tolist()


@pytest.mark.slow
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_seed_gives_the_same_tokens_on_cpu_and_mps(llm, seed):
    cpu = _sample(llm, "cpu", seed)
    assert len(cpu) >= 36
    assert _sample(llm, "mps", seed) == cpu

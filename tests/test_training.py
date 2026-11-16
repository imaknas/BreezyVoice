"""Fine-tuning logic on a toy language model (no released weights)."""
import pytest
import torch

from breezyvoice.training.data import Example, collate, extract, speaker_embedding, split
from breezyvoice.training.finetune import Settings, fine_tune, held_out_loss
from cosyvoice.llm.llm import TransformerLM
from cosyvoice.transformer.encoder import ConformerEncoder, TransformerEncoder

TEXT_VOCAB, SPEECH_VOCAB, DIM = 50, 16, 32


def _toy_lm() -> TransformerLM:
    torch.manual_seed(0)
    return TransformerLM(
        text_encoder_input_size=DIM, llm_input_size=DIM, llm_output_size=DIM, text_token_size=TEXT_VOCAB,
        speech_token_size=SPEECH_VOCAB, spk_embed_dim=192, lsm_weight=0, length_normalized_loss=True,
        text_encoder=ConformerEncoder(
            input_size=DIM, output_size=DIM, attention_heads=4, linear_units=64, num_blocks=1,
            dropout_rate=0.0, positional_dropout_rate=0.0, attention_dropout_rate=0, normalize_before=True,
            input_layer="linear", pos_enc_layer_type="rel_pos_espnet", selfattention_layer_type="rel_selfattn",
            use_cnn_module=False, macaron_style=False, use_dynamic_chunk=False, use_dynamic_left_chunk=False,
            static_chunk_size=1),
        llm=TransformerEncoder(
            input_size=DIM, output_size=DIM, attention_heads=4, linear_units=64, num_blocks=2,
            dropout_rate=0.0, positional_dropout_rate=0.0, attention_dropout_rate=0,
            input_layer="linear_legacy", pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn", static_chunk_size=1),
    )


def _examples(n_lines: int = 16, takes: int = 2) -> list[Example]:
    """Each line's speech tokens are a fixed function of its text, so there is something to learn."""
    g = torch.Generator().manual_seed(1)
    out = []
    for i in range(n_lines):
        text = torch.randint(1, TEXT_VOCAB, (6,), generator=g, dtype=torch.int32)
        speech = (text.repeat_interleave(3) % SPEECH_VOCAB).to(torch.int32)
        for _ in range(takes):
            out.append(Example(line=f"L{i:02d}", text_token=text, speech_token=speech,
                               embedding=torch.randn(192, generator=g)))
    return out


def test_extract_uses_the_given_front_ends():
    example = extract(
        "B01", "你好", torch.zeros(1, 16000),
        text_tokens=lambda text: torch.tensor([[len(text), 7]]),
        speech_token=lambda wav: torch.tensor([[1, 2, 3]]),
        embedding=lambda wav: torch.ones(1, 192),
    )
    assert example.line == "B01"
    assert example.text_token.tolist() == [2, 7] and example.speech_token.tolist() == [1, 2, 3]
    assert example.embedding.shape == (192,) and example.seconds == pytest.approx(3 / 50)


def test_split_holds_out_whole_lines():
    train, held = split(_examples(), held_out_every=4)
    assert {e.line for e in held} == {"L00", "L04", "L08", "L12"}
    assert not {e.line for e in train} & {e.line for e in held}
    assert len(train) + len(held) == 32


def test_collate_pads_and_shares_one_embedding():
    a = Example("a", torch.tensor([1, 2, 3], dtype=torch.int32), torch.tensor([4, 5], dtype=torch.int32), torch.zeros(192))
    b = Example("b", torch.tensor([6], dtype=torch.int32), torch.tensor([7, 8, 9, 1], dtype=torch.int32), torch.ones(192))
    batch = collate([a, b], speaker_embedding([a, b]))
    assert batch["text_token"].tolist() == [[1, 2, 3], [6, 0, 0]] and batch["text_token_len"].tolist() == [3, 1]
    assert batch["speech_token"].shape == (2, 4) and batch["speech_token_len"].tolist() == [2, 4]
    assert torch.equal(batch["embedding"], torch.full((2, 192), 0.5))


def test_fine_tune_lowers_held_out_loss_and_returns_the_best_weights():
    examples = _examples()
    train, held = split(examples)
    embedding = speaker_embedding(examples)
    llm = _toy_lm()
    device = torch.device("cpu")
    before, _ = held_out_loss(llm, held, embedding, device, 8)
    seen = []
    result = fine_tune(llm, train, held, embedding, device,
                       Settings(steps=60, batch_size=8, learning_rate=3e-3, evaluate_every=20), log=seen.append)
    assert [e["step"] for e in result.history] == [0, 20, 40, 60] and seen == result.history
    assert result.history[0]["held_out_loss"] == pytest.approx(before)
    assert result.best_loss < before * 0.8
    assert result.best_loss == min(e["held_out_loss"] for e in result.history)
    assert not llm.training

    fresh = _toy_lm()
    fresh.load_state_dict(result.best_state)
    assert held_out_loss(fresh, held, embedding, device, 8)[0] == pytest.approx(result.best_loss, rel=1e-4)


def test_fine_tune_needs_held_out_examples():
    with pytest.raises(ValueError):
        fine_tune(_toy_lm(), _examples(2), [], torch.zeros(192), torch.device("cpu"))

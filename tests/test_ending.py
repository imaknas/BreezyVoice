"""A sentence that stops during its last sound is carried on; one that has finished is left alone."""
from types import SimpleNamespace

import numpy as np
import torch

from breezyvoice.engine import CARRY_ON_LIMIT, CARRY_ON_TOKENS, BreezyVoice
from breezyvoice.ending import ends_loud

RATE = 22050


def _tone(seconds: float, level: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_a_take_that_stops_mid_sound_ends_loud():
    assert ends_loud(_tone(0.5), RATE)


def test_a_take_that_fades_out_does_not():
    faded = _tone(0.5) * np.linspace(1, 0, int(0.5 * RATE), dtype=np.float32) ** 4
    assert not ends_loud(faded, RATE)
    assert not ends_loud(np.concatenate([_tone(0.5), np.zeros(int(0.1 * RATE), dtype=np.float32)]), RATE)


def test_a_quiet_recording_is_judged_against_its_own_peak():
    assert ends_loud(_tone(0.5, level=0.01), RATE)
    assert not ends_loud(np.zeros(100, dtype=np.float32), RATE) and not ends_loud(np.zeros(0, dtype=np.float32), RATE)


class FakeLM:
    """Ends after five tokens; carries on with token 9 while there is room."""

    def __init__(self, room: int = 1000):
        self.room, self.asked = room, []

    def inference(self, **prompt):
        prompt["sampler"](torch.zeros(4).log_softmax(-1), [])
        return torch.tensor([[1, 2, 3, 4, 5]])

    def carry_on(self, tokens, count, **prompt):
        assert "max_token_text_ratio" not in prompt
        self.asked.append((tokens.shape[1], count))
        count = min(count, self.room - tokens.shape[1])
        return torch.cat([tokens, torch.full((1, max(count, 0)), 9)], dim=1)


def _engine(lm, unfinished) -> BreezyVoice:
    place = SimpleNamespace(llm="cpu", flow="cpu", hift="cpu")
    nets = SimpleNamespace(
        llm=lm, placement=place,
        flow=SimpleNamespace(inference=lambda token, **_: token.float()),
        hift=SimpleNamespace(inference=lambda mel: mel.repeat_interleave(10, dim=1)),  # ten samples a token
    )
    return BreezyVoice(nets, text_frontend=lambda s: s, encode=lambda s: [1] * len(s), speech_frontend=None,
                       sampler=lambda log_probs, decoded: 0, unfinished=unfinished)


def _speaker():
    audio = SimpleNamespace(speech_token=torch.zeros((1, 3), dtype=torch.int32), speech_feat=torch.zeros((1, 6, 80)),
                            embedding=torch.zeros((1, 192)))
    return SimpleNamespace(text_token=torch.zeros((1, 2), dtype=torch.int32), audio=audio)


def test_a_finished_sentence_is_left_as_the_model_ended_it():
    lm = FakeLM()
    result = _engine(lm, unfinished=lambda audio, rate: False).synthesize("你好。", _speaker(), seed=0)
    assert lm.asked == [] and len(result.audio) == 50 and result.decoding[0].carried_on == 0


def test_an_unfinished_sentence_is_carried_on_until_it_is_finished():
    lm = FakeLM()
    seen = []

    def unfinished(audio, rate):
        seen.append(len(audio))
        return len(audio) < 130  # finished once thirteen tokens are rendered

    result = _engine(lm, unfinished).synthesize("你好。", _speaker(), seed=0)
    assert lm.asked == [(5, CARRY_ON_TOKENS), (9, CARRY_ON_TOKENS)]
    assert seen == [50, 90, 130] and len(result.audio) == 130
    (decoding,) = result.decoding
    assert decoding.carried_on == 8 and decoding.samples == 130


def test_carrying_on_stops_at_the_limit():
    lm = FakeLM()
    result = _engine(lm, unfinished=lambda audio, rate: True).synthesize("你好。", _speaker(), seed=0)
    assert result.decoding[0].carried_on == CARRY_ON_LIMIT and len(lm.asked) == CARRY_ON_LIMIT // CARRY_ON_TOKENS


def test_carrying_on_stops_when_the_decoder_has_no_room():
    lm = FakeLM(room=7)
    result = _engine(lm, unfinished=lambda audio, rate: True).synthesize("你好。", _speaker(), seed=0)
    assert result.decoding[0].carried_on == 2 and len(result.audio) == 70


def test_without_a_judge_nothing_is_carried_on():
    lm = FakeLM()
    result = _engine(lm, unfinished=None).synthesize("你好。", _speaker(), seed=0)
    assert lm.asked == [] and result.decoding[0].carried_on == 0


def test_each_sentence_is_judged_on_its_own():
    lm = FakeLM()
    calls = iter([True, False, False])  # only the first sentence stopped too soon
    result = _engine(lm, unfinished=lambda audio, rate: next(calls)).synthesize("你好。再見。", _speaker(), seed=0)
    assert [d.carried_on for d in result.decoding] == [CARRY_ON_TOKENS, 0]
    assert len(result.audio) == 90 + 50

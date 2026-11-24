"""Pronunciation checking logic with a toy recogniser (no pre-trained encoder, no audio files)."""
from types import SimpleNamespace

import pytest
import torch

from breezyvoice.pronunciation.check import score
from breezyvoice.pronunciation.model import BLANK, Recognizer
from breezyvoice.pronunciation.readings import Unsupported, load_polyphones, read
from breezyvoice.pronunciation.train import Settings, Utterance, edit_distance, error_rates, greedy, split, train, vocabulary
from breezyvoice.pronunciation.units import alternatives, flatten, is_reading, units


def test_units_split_initial_and_tonal_final():
    assert units("ㄊㄧㄠ2") == ["ㄊ", "ㄧㄠ2"]
    assert units("ㄕ4") == ["ㄕ", "-4"]
    assert units("ㄧ1") == ["ㄧ1"]
    assert units("ㄦ2") == ["ㄦ2"]
    assert flatten(["ㄏㄠ3", "ㄧ1"]) == ["ㄏ", "ㄠ3", "ㄧ1"]


@pytest.mark.parametrize("bad", ["", "ㄊㄧㄠ", "tiao2", "ㄊㄧㄠ6", "2"])
def test_units_reject_what_is_not_a_reading(bad):
    assert not is_reading(bad)
    with pytest.raises(ValueError):
        units(bad)


def test_alternatives_are_other_tones_then_other_readings_that_have_units():
    known = {"ㄊ", "ㄉ", "ㄧㄠ1", "ㄧㄠ2", "ㄧㄠ4", "ㄓ", "ㄡ1"}
    assert alternatives("ㄊㄧㄠ2", ["ㄉㄧㄠ4", "ㄓㄡ1", "ㄊㄧㄠ4"], known) == ["ㄊㄧㄠ1", "ㄊㄧㄠ4", "ㄉㄧㄠ4", "ㄓㄡ1"]
    assert alternatives("ㄊㄧㄠ2", ["ㄉㄧㄠ3"], known) == ["ㄊㄧㄠ1", "ㄊㄧㄠ4"]  # no unit for ㄧㄠ3
    assert alternatives("ㄊㄧㄠ2", ["ㄉㄧㄠ4", "ㄊㄧㄠ4"], known, tones=False) == ["ㄉㄧㄠ4"]


def test_load_polyphones(tmp_path):
    path = tmp_path / "POLYPHONIC_CHARS.txt"
    path.write_text("調\tㄊㄧㄠ2\n調\tㄉㄧㄠ4\n和\tㄏㄜ2\n壞行\n", encoding="utf-8")
    assert load_polyphones(path) == {"調": ["ㄊㄧㄠ2", "ㄉㄧㄠ4"], "和": ["ㄏㄜ2"]}


def _predict(text):
    table = {"調": "ㄊㄧㄠ2", "大": "ㄉㄚ4", "一": "ㄧ1"}
    return [table.get(ch) for ch in text]


def test_read_skips_punctuation_and_lists_other_readings():
    reading = read("調大，一。", lambda t: t, _predict, {"調": ["ㄊㄧㄠ2", "ㄉㄧㄠ4"]})
    assert reading.characters == ("調", "大", "一")
    assert reading.readings == ("ㄊㄧㄠ2", "ㄉㄚ4", "ㄧ1")
    assert reading.others == (("ㄉㄧㄠ4",), (), ())


def test_read_leaves_out_rivals_a_listener_would_not_hear_as_different():
    predict = lambda text: [{"液": "ㄧㄝ4", "暫": "ㄓㄢ4", "行": "ㄒㄧㄥ2"}.get(ch) for ch in text]  # noqa: E731
    table = {"液": ["ㄧㄝ4", "ㄧ4"], "暫": ["ㄓㄢ4", "ㄗㄢ4", "ㄓㄢ3"], "行": ["ㄒㄧㄥ2", "ㄒㄧㄣ2", "ㄏㄤ2"]}
    reading = read("液暫行", lambda t: t, predict, table)
    assert reading.others == ((), ("ㄓㄢ3",), ("ㄏㄤ2",))  # free variant; retroflex and -ng mergers


def test_read_applies_normalisation_first():
    reading = read("1", lambda t: t.replace("1", "一"), _predict, {})
    assert reading.characters == ("一",)


def test_read_takes_a_reading_written_in_the_text_as_the_one_intended():
    reading = read("調[:ㄉㄧㄠ4]大，好[:ㄏㄠ3]一。", lambda t: t, _predict, {"調": ["ㄊㄧㄠ2", "ㄉㄧㄠ4"]})
    assert reading.characters == ("調", "大", "好", "一")
    assert reading.readings == ("ㄉㄧㄠ4", "ㄉㄚ4", "ㄏㄠ3", "ㄧ1")
    assert reading.others == (("ㄊㄧㄠ2",), (), (), ())
    assert (reading.spoken, reading.positions) == ("調大，好一。", (0, 1, 3, 4))


def test_read_refuses_a_written_reading_that_is_not_one():
    with pytest.raises(Unsupported):
        read("調[:tiao2]大", lambda t: t, _predict, {})


@pytest.mark.parametrize("text", ["調OK", "。，", "調好"])
def test_read_refuses_what_it_cannot_read(text):
    with pytest.raises(Unsupported):
        read(text, lambda t: t, _predict, {})


def _posteriors(vocab, sequence, frames_per_unit=3, confidence=0.9):
    """Frame log-probabilities that clearly say ``sequence`` (with a blank frame between units)."""
    index = {u: i for i, u in enumerate(vocab)}
    rows = []
    for unit in sequence:
        for target in [unit] * frames_per_unit + [BLANK]:
            row = torch.full((len(vocab),), (1 - confidence) / (len(vocab) - 1))
            row[index[target]] = confidence
            rows.append(row)
    return torch.stack(rows).log()


VOCAB = [BLANK, "ㄊ", "ㄉ", "ㄧㄠ2", "ㄧㄠ4", "ㄚ4"]
INDEX = {u: i for i, u in enumerate(VOCAB)}


def test_score_prefers_the_reading_that_was_spoken():
    readings, others = ["ㄊㄧㄠ2", "ㄉㄚ4"], [["ㄉㄧㄠ4"], []]
    right = score(_posteriors(VOCAB, flatten(readings)), readings, others, INDEX)
    assert right[0].margin > 3 and right[0].index == 0 and right[0].reading == "ㄊㄧㄠ2"
    assert right[1].rival is None and right[1].margin == float("inf")  # ㄚ in other tones has no unit

    wrong = score(_posteriors(VOCAB, flatten(["ㄉㄧㄠ4", "ㄉㄚ4"])), readings, others, INDEX)
    assert wrong[0].margin < -3 and wrong[0].rival == "ㄉㄧㄠ4"


def test_score_without_tones_ignores_a_tone_slip_but_not_another_reading():
    readings = ["ㄊㄧㄠ2"]
    tone_slip = score(_posteriors(VOCAB, ["ㄊ", "ㄧㄠ4"]), readings, [["ㄉㄧㄠ4"]], INDEX, tones=False)
    assert tone_slip[0].rival == "ㄉㄧㄠ4" and tone_slip[0].margin > 0
    other = score(_posteriors(VOCAB, ["ㄉ", "ㄧㄠ4"]), readings, [["ㄉㄧㄠ4"]], INDEX, tones=False)
    assert other[0].margin < -3


def test_score_names_the_tone_when_only_the_tone_is_off():
    readings = ["ㄊㄧㄠ2"]
    result = score(_posteriors(VOCAB, ["ㄊ", "ㄧㄠ4"]), readings, [["ㄉㄧㄠ4"]], INDEX)
    assert result[0].rival == "ㄊㄧㄠ4" and result[0].margin < 0


def test_edit_distance_and_greedy():
    assert edit_distance("abc", "abd") == 1 and edit_distance([], [1, 2]) == 2
    assert greedy(_posteriors(VOCAB, ["ㄊ", "ㄧㄠ2", "ㄊ"]), VOCAB) == ["ㄊ", "ㄧㄠ2", "ㄊ"]


def test_split_keeps_a_text_on_one_side():
    utterances = [Utterance(key=f"t{i % 20}", wav=f"{i}.wav", readings=("ㄧ1",)) for i in range(60)]
    training, held = split(utterances, held_out_share=0.1)
    assert {u.key for u in held} and not {u.key for u in training} & {u.key for u in held}
    assert len(training) + len(held) == 60 and len({u.key for u in held}) == 2


class _ToyEncoder(torch.nn.Module):
    """Stands in for the speech encoder: one frame per 320 samples, described by its low spectrum."""

    def __init__(self, hidden=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, to_dict=lambda: {})
        self.project = torch.nn.Linear(16, hidden)

    def forward(self, audio, attention_mask=None):
        frames = audio[:, :audio.size(1) // 320 * 320].reshape(audio.size(0), -1, 320)
        spectrum = torch.fft.rfft(frames).abs()[..., :16]
        return SimpleNamespace(last_hidden_state=torch.tanh(self.project(torch.log1p(spectrum))))

    def _get_feat_extract_output_lengths(self, lengths):
        return lengths // 320


def _tone(frequency, seconds=0.02):  # one frame, so the unit has exactly one place to be emitted
    t = torch.arange(int(16000 * seconds)) / 16000
    return torch.sin(2 * torch.pi * frequency * t)


def test_training_teaches_a_toy_recogniser_to_tell_two_sounds_apart():
    # two "syllables" that differ only in pitch, standing in for two tones
    sounds = {"ㄧ1": _tone(200), "ㄧ4": _tone(500)}
    # a silent gap between them: the toy encoder sees one frame at a time, and CTC needs a blank to
    # separate two equal units
    gap = torch.zeros(1600)
    audio = {f"{a}{b}.wav": torch.cat([sounds[a], gap, sounds[b]]) for a in sounds for b in sounds}
    utterances = [Utterance(key=name, wav=name, readings=(name[:2], name[2:4])) for name in audio]
    torch.manual_seed(0)
    recognizer = Recognizer(_ToyEncoder(), vocabulary(utterances))
    assert recognizer.vocabulary == [BLANK, "ㄧ1", "ㄧ4"]
    load = lambda name: audio[name]  # noqa: E731
    seen = []
    result = train(recognizer, utterances, utterances[:2], load, torch.device("cpu"),
                   Settings(steps=600, batch_size=4, encoder_learning_rate=1e-2, head_learning_rate=1e-2, evaluate_every=200),
                   log=seen.append)
    assert [e["step"] for e in result.history] == [200, 400, 600] and seen == result.history
    recognizer.load_state_dict(result.best_state)
    assert error_rates(recognizer, utterances, load, torch.device("cpu"), 4) == (0.0, 0.0)
    scored = score(recognizer.log_probs(audio["ㄧ1ㄧ4.wav"]), ["ㄧ1", "ㄧ4"], [[], []], recognizer.unit_index)
    assert all(s.margin > 0 for s in scored)
    mistaken = score(recognizer.log_probs(audio["ㄧ4ㄧ4.wav"]), ["ㄧ1", "ㄧ4"], [[], []], recognizer.unit_index)
    assert mistaken[0].margin < 0 and mistaken[0].rival == "ㄧ4" and mistaken[1].margin > 0

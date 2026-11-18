"""Aligning syllables to frames and placing pauses in the text (no models)."""
import numpy as np
import pytest
import torch

from breezyvoice.pronunciation.align import FRAME_SECONDS, stray_units, syllable_times, unit_frames
from breezyvoice.pronunciation.pauses import Limits, Pause, place, tidy, unnatural

VOCAB = ["<blank>", "ㄊ", "ㄧㄠ2", "ㄉ", "ㄚ4", "ㄧ1"]
INDEX = {u: i for i, u in enumerate(VOCAB)}


def _frames(pattern):
    """Frame posteriors that say the given unit (or blank, written _) in each frame."""
    rows = []
    for unit in pattern:
        row = torch.full((len(VOCAB),), 0.01)
        row[INDEX["<blank>" if unit == "_" else unit]] = 0.95
        rows.append(row)
    return torch.stack(rows).log()


def test_unit_frames_follow_the_posteriors():
    lp = _frames(["_", "ㄊ", "ㄊ", "ㄧㄠ2", "ㄧㄠ2", "ㄧㄠ2", "_", "_", "ㄉ", "ㄚ4", "_"])
    assert unit_frames(lp, [INDEX[u] for u in ("ㄊ", "ㄧㄠ2", "ㄉ", "ㄚ4")]) == [(1, 2), (3, 5), (8, 8), (9, 9)]


def test_repeated_units_are_kept_apart_by_a_blank():
    lp = _frames(["ㄧ1", "ㄧ1", "_", "ㄧ1", "_"])
    assert unit_frames(lp, [INDEX["ㄧ1"], INDEX["ㄧ1"]]) == [(0, 1), (3, 3)]


def test_unit_frames_needs_room():
    with pytest.raises(ValueError):
        unit_frames(_frames(["ㄊ"]), [INDEX["ㄊ"], INDEX["ㄧㄠ2"]])


def test_syllable_times_span_initial_and_final():
    lp = _frames(["_", "ㄊ", "ㄧㄠ2", "ㄧㄠ2", "_", "_", "ㄉ", "ㄚ4", "_"])
    times = syllable_times(lp, ["ㄊㄧㄠ2", "ㄉㄚ4"], INDEX)
    assert times == [pytest.approx((1 * FRAME_SECONDS, 4 * FRAME_SECONDS)), pytest.approx((6 * FRAME_SECONDS, 8 * FRAME_SECONDS))]


# 「你好，再見了」: five spoken characters at positions 0 1 3 4 5 of the text
SPOKEN, POSITIONS = "你好，再見了。", [0, 1, 3, 4, 5]
SYLLABLES = [(0.0, 0.2), (0.2, 0.4), (0.9, 1.1), (1.1, 1.3), (1.6, 1.8)]


def test_place_tells_a_comma_pause_from_one_inside_a_phrase():
    pauses = place([(0.42, 0.88), (1.32, 1.58)], SYLLABLES, POSITIONS, SPOKEN)
    assert [(p.after, p.at_punctuation) for p in pauses] == [(1, True), (3, False)]
    assert pauses[0].length == pytest.approx(0.46)


def test_place_ignores_silence_outside_the_sentence_and_merges_a_split_pause():
    pauses = place([(-0.3, -0.1), (0.42, 0.6), (0.66, 0.88), (1.85, 2.2)], SYLLABLES, POSITIONS, SPOKEN)
    assert len(pauses) == 1
    assert (pauses[0].start, pauses[0].end, pauses[0].after) == (0.42, 0.88, 1)


def test_unnatural_uses_one_limit_in_mid_phrase_and_another_at_punctuation():
    pauses = [Pause(0.0, 0.3, 0, True), Pause(1.0, 1.6, 1, True), Pause(2.0, 2.1, 2, False), Pause(3.0, 3.2, 3, False)]
    assert unnatural(pauses, Limits(unmarked=0.14, at_punctuation=0.5)) == [pauses[1], pauses[3]]


def test_tidy_shortens_only_the_unnatural_pauses_and_keeps_the_speech():
    rate = 16000
    rng = np.random.default_rng(0)
    speech = lambda seconds: (0.2 * rng.standard_normal(int(rate * seconds))).astype(np.float32)  # noqa: E731
    quiet = lambda seconds: np.zeros(int(rate * seconds), dtype=np.float32)  # noqa: E731
    parts = [speech(0.5), quiet(0.3), speech(0.5), quiet(0.2), speech(0.5), quiet(0.7), speech(0.5)]
    audio = np.concatenate(parts)
    pauses = [Pause(0.5, 0.8, 0, True), Pause(1.3, 1.5, 1, False), Pause(2.0, 2.7, 2, True)]
    out = tidy(audio, rate, pauses, Limits(), unmarked_to=0.04, punctuation_to=0.32, fade=0.008)
    removed = (0.2 - 0.04) + (0.7 - 0.32)
    assert len(out) / rate == pytest.approx(len(audio) / rate - removed, abs=0.02)
    assert np.array_equal(out[:int(0.5 * rate)], audio[:int(0.5 * rate)])  # speech before the first cut is untouched
    assert np.array_equal(out[-int(0.4 * rate):], audio[-int(0.4 * rate):])
    assert tidy(audio, rate, [pauses[0]]).shape == audio.shape  # an ordinary comma pause is left alone


def test_stray_units_are_what_is_heard_outside_the_sentence():
    lp = _frames(["ㄧ1", "_", "_", "ㄊ", "ㄧㄠ2", "_", "ㄉ", "ㄚ4", "_", "_", "ㄉ", "ㄚ4", "_"])
    readings = ["ㄊㄧㄠ2", "ㄉㄚ4"]
    assert syllable_times(lp, readings, INDEX)[0][0] == pytest.approx(3 * FRAME_SECONDS)
    # the alignment may take either ㄉㄚ4 as the sentence's; whichever it leaves out is stray
    before, after = stray_units(lp, [(3 * FRAME_SECONDS, 5 * FRAME_SECONDS), (6 * FRAME_SECONDS, 8 * FRAME_SECONDS)], VOCAB)
    assert before == ["ㄧ1"] and after == ["ㄉ", "ㄚ4"]
    clean = _frames(["_", "ㄊ", "ㄧㄠ2", "_", "ㄉ", "ㄚ4", "_"])
    assert stray_units(clean, syllable_times(clean, readings, INDEX), VOCAB) == ([], [])

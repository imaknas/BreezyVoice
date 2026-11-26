"""Aligning syllables to frames and placing pauses in the text (no models)."""
import numpy as np
import pytest
import torch

from breezyvoice.audio import silence_before
from breezyvoice.pronunciation.align import (FRAME_SECONDS, Inserted, clarity, inserted, repeated, stray_units,
                                              syllable_times, unit_frames)
from breezyvoice.pronunciation.pauses import Limits, Missing, Pause, missing, place, tidy, unnatural

VOCAB = ["<blank>", "ㄊ", "ㄧㄠ2", "ㄉ", "ㄚ4", "ㄧ1", "ㄧㄠ4", "ㄧ4"]
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


# 「試試：在哪」: four spoken characters at positions 0 1 3 4
COLON, COLON_POS = "試試：在哪", [0, 1, 3, 4]
COLON_SYL = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]


def _speech_with_dip(rate=16000):
    """0.8 s of noise with a quieter stretch from 0.38 to 0.42 s."""
    rng = np.random.default_rng(1)
    audio = (0.2 * rng.standard_normal(int(rate * 0.8))).astype(np.float32)
    audio[int(0.38 * rate):int(0.42 * rate)] *= 0.05
    return audio, rate


def test_missing_finds_a_colon_spoken_straight_through_and_picks_the_quietest_moment():
    audio, rate = _speech_with_dip()
    (lack,) = missing(audio, rate, [], COLON_SYL, COLON_POS, COLON, Limits(after_strong_mark=0.22))
    assert lack.after == 1 and lack.seconds == pytest.approx(0.22)
    assert 0.37 <= lack.at <= 0.43


def test_missing_tops_up_a_short_pause_and_accepts_a_long_enough_one():
    audio, rate = _speech_with_dip()
    short = [Pause(0.36, 0.44, 1, True)]
    (lack,) = missing(audio, rate, short, COLON_SYL, COLON_POS, COLON, Limits(after_strong_mark=0.22))
    assert lack.at == pytest.approx(0.40) and lack.seconds == pytest.approx(0.14)
    assert missing(audio, rate, [Pause(0.3, 0.55, 1, True)], COLON_SYL, COLON_POS, COLON) == []
    # a comma is not a strong mark
    assert missing(audio, rate, [], SYLLABLES, POSITIONS, SPOKEN) == []


def test_tidy_adds_the_missing_pause_as_silence():
    audio, rate = _speech_with_dip()
    out = tidy(audio, rate, [], add=[Missing(1, 0.40, 0.22)])
    assert len(out) / rate == pytest.approx(0.8 + 0.22, abs=0.02)
    added = out[int(0.42 * rate):int(0.60 * rate)]
    assert np.all(added == 0)
    assert np.array_equal(out[:int(0.3 * rate)], audio[:int(0.3 * rate)])
    assert np.array_equal(out[-int(0.3 * rate):], audio[-int(0.3 * rate):])


def test_reviewer_puts_the_pieces_together():
    """A stand-in recogniser whose posteriors say 調大 with a stray ㄧ1 at the end, 20 ms per frame."""
    from types import SimpleNamespace

    from breezyvoice.pronunciation.readings import Reading, Unsupported
    from breezyvoice.pronunciation.review import Reviewer

    pattern = ["_"] * 5 + ["ㄊ", "ㄧㄠ2"] + ["_"] * 20 + ["ㄉ", "ㄚ4"] + ["_"] * 5 + ["ㄧ1"] + ["_"] * 3
    recognizer = SimpleNamespace(unit_index=INDEX, vocabulary=VOCAB, log_probs=lambda audio: _frames(pattern))
    reading = Reading(("調", "大"), ("ㄊㄧㄠ2", "ㄉㄚ4"), ((), ()), "調大。", (0, 1))
    rate = 16000
    rng = np.random.default_rng(0)
    audio = (0.2 * rng.standard_normal(len(pattern) * 320)).astype(np.float32)
    audio[8 * 320:26 * 320] = 0  # silence between the two syllables
    reviewer = Reviewer(recognizer, lambda text: reading, lambda a, sr: a)
    review = reviewer.review(audio, rate, "調大。")
    assert review.extra_after == ("ㄧ1",) and review.extra_before == () and review.needs_retake
    assert [(p.after, p.at_punctuation) for p in review.unnatural_pauses] == [(0, False)]
    assert len(reviewer.tidy(audio, rate, review)) < len(audio)

    unknown = Reading(("好",), ("ㄏㄠ3",), ((),), "好", (0,))
    with pytest.raises(Unsupported):
        Reviewer(recognizer, lambda text: unknown, lambda a, sr: a).review(audio, rate, "好")


def test_reviewer_uses_the_margin_a_character_has_of_its_own():
    """調 heard a little more like ㄊㄧㄠ2 than ㄉㄧㄠ4: inside the general margin, outside a stricter one."""
    from types import SimpleNamespace

    from breezyvoice.pronunciation.readings import Reading
    from breezyvoice.pronunciation.review import Reviewer

    def frame(**probabilities):
        row = torch.full((len(VOCAB),), 0.01)
        for unit, value in probabilities.items():
            row[INDEX["<blank>" if unit == "_" else unit]] = value
        return row

    blank = frame(_=0.95)
    frames = torch.stack([blank, frame(ㄊ=0.5, ㄉ=0.3), frame(ㄧㄠ2=0.5, ㄧㄠ4=0.3), blank]).log()
    recognizer = SimpleNamespace(unit_index=INDEX, vocabulary=VOCAB, log_probs=lambda audio: frames)
    audio = np.full(4 * 320, 0.1, dtype=np.float32)

    def wrong(margins):
        reading = Reading(("調",), ("ㄊㄧㄠ2",), (("ㄉㄧㄠ4",),), "調", (0,), margins)
        return Reviewer(recognizer, lambda text: reading, lambda a, sr: a).review(audio, 16000, "調").wrong_readings

    assert wrong(()) == () and wrong((None,)) == ()
    (found,) = wrong((3.5,))
    assert found.heard == "ㄉㄧㄠ4" and 0 < found.margin < 3.5


def test_what_a_repeat_reports_is_not_listed_again_as_inserted():
    from breezyvoice.pronunciation.align import Repeat
    from breezyvoice.pronunciation.review import _part_of

    repeat = Repeat(first=3, count=2, at=1.0)  # syllables 3 and 4 said again from 1.0 s
    assert _part_of(Inserted(2, ("ㄉ", "ㄚ4"), 1.0), repeat)  # the copy, found after syllable 2
    assert _part_of(Inserted(4, ("ㄉ", "ㄚ4"), 1.3), repeat)  # a later part of it
    assert not _part_of(Inserted(2, ("ㄉ", "ㄚ4"), 0.6), repeat)  # before the copy starts
    assert not _part_of(Inserted(7, ("ㄉ", "ㄚ4"), 2.0), repeat)  # elsewhere in the sentence


def test_missing_does_not_place_a_pause_where_the_sound_never_dips():
    rate = 16000
    rng = np.random.default_rng(2)
    steady = (0.2 * rng.standard_normal(int(rate * 0.8))).astype(np.float32)
    (lack,) = missing(steady, rate, [], COLON_SYL, COLON_POS, COLON)
    assert lack.at is None and lack.seconds == pytest.approx(0.22)
    assert tidy(steady, rate, [], add=[lack]).shape == steady.shape  # nothing is cut into


def test_dashes_join_and_other_ellipsis_characters_count():
    audio, rate = _speech_with_dip()
    assert missing(audio, rate, [], COLON_SYL, [0, 1, 3, 4], "台北—高雄") == []
    assert len(missing(audio, rate, [], COLON_SYL, [0, 1, 4, 5], "等等⋯⋯你說")) == 1


def test_repeated_finds_a_syllable_said_twice_in_mid_sentence():
    readings = ["ㄊㄧㄠ2", "ㄉㄚ4"]
    twice = _frames(["_", "ㄊ", "ㄧㄠ2", "_", "_", "_", "ㄊ", "ㄧㄠ2", "_", "ㄉ", "ㄚ4", "_"])
    (repeat,) = repeated(twice, readings, INDEX)
    assert (repeat.first, repeat.count) == (0, 1) and repeat.at in (pytest.approx(FRAME_SECONDS), pytest.approx(6 * FRAME_SECONDS))
    assert repeated(_frames(["_", "ㄊ", "ㄧㄠ2", "_", "ㄉ", "ㄚ4", "_"]), readings, INDEX) == []
    # the text itself may say a syllable twice
    assert repeated(twice, ["ㄊㄧㄠ2", "ㄊㄧㄠ2", "ㄉㄚ4"], INDEX) == []


def test_repeated_covers_a_whole_word_and_a_changed_tone_but_not_other_extra_sounds():
    readings = ["ㄊㄧㄠ2", "ㄉㄚ4"]
    word = _frames(["ㄊ", "ㄧㄠ2", "ㄉ", "ㄚ4", "_", "_", "ㄊ", "ㄧㄠ2", "ㄉ", "ㄚ4"])
    assert [(r.first, r.count) for r in repeated(word, readings, INDEX)] == [(0, 2)]
    other_tone = _frames(["ㄊ", "ㄧㄠ4", "_", "_", "_", "ㄊ", "ㄧㄠ2", "ㄉ", "ㄚ4"])
    assert [(r.first, r.count) for r in repeated(other_tone, readings, INDEX)] == [(0, 1)]
    # an extra sound that is not a neighbouring syllable, and half a syllable, are left alone
    assert repeated(_frames(["ㄊ", "ㄧㄠ2", "_", "ㄧ1", "_", "ㄉ", "ㄚ4"]), readings, INDEX) == []
    assert repeated(_frames(["ㄊ", "ㄧㄠ2", "_", "ㄉ", "_", "ㄉ", "ㄚ4"]), readings, INDEX) == []
    # a bare final heard in two tones is one long vowel, not a repeat
    assert repeated(_frames(["ㄧ4", "ㄧ1", "_", "ㄉ", "ㄚ4"]), ["ㄧ1", "ㄉㄚ4"], INDEX) == []


def test_repeated_needs_the_two_copies_far_enough_apart_to_be_two_syllables():
    held = _frames(["ㄧ1", "_", "ㄧ1", "_", "ㄉ", "ㄚ4"])  # 40 ms apart: one held vowel
    assert repeated(held, ["ㄧ1", "ㄉㄚ4"], INDEX) == []
    twice = _frames(["ㄧ1", "_", "_", "_", "_", "_", "ㄧ1", "_", "ㄉ", "ㄚ4"])  # 120 ms apart
    assert [(r.first, r.count) for r in repeated(twice, ["ㄧ1", "ㄉㄚ4"], INDEX)] == [(0, 1)]


def test_repeated_ignores_units_the_recogniser_is_unsure_of():
    lp = _frames(["_", "ㄊ", "ㄧㄠ2", "_", "_", "_", "ㄊ", "ㄧㄠ2", "_", "ㄉ", "ㄚ4", "_"])
    assert repeated(lp, ["ㄊㄧㄠ2", "ㄉㄚ4"], INDEX) != []
    assert repeated(lp, ["ㄊㄧㄠ2", "ㄉㄚ4"], INDEX, min_probability=1.01) == []


def test_inserted_finds_a_whole_syllable_the_text_lacks_in_mid_sentence():
    readings = ["ㄊㄧㄠ2", "ㄧ1", "ㄉㄚ4"]
    lp = _frames(["_", "ㄊ", "ㄧㄠ2", "_", "ㄧ1", "_", "ㄉ", "ㄧㄠ4", "_", "ㄉ", "ㄚ4", "_"])
    assert inserted(lp, readings, INDEX) == [Inserted(1, ("ㄉ", "ㄧㄠ4"), pytest.approx(6 * FRAME_SECONDS))]
    # a lone final or a lone initial between two syllables is not a syllable
    assert inserted(_frames(["ㄊ", "ㄧㄠ2", "_", "ㄧ4", "_", "ㄧ1", "_", "ㄉ", "ㄚ4"]), readings, INDEX) == []
    assert inserted(_frames(["ㄊ", "ㄧㄠ2", "_", "ㄧ1", "_", "ㄊ", "_", "ㄉ", "ㄚ4"]), readings, INDEX) == []
    # before the first and after the last syllable is not mid-sentence
    assert inserted(_frames(["ㄉ", "ㄧㄠ4", "_", "ㄊ", "ㄧㄠ2", "_", "ㄧ1", "_", "ㄉ", "ㄚ4", "_", "ㄊ", "ㄚ4"]), readings, INDEX) == []
    assert inserted(lp, readings, INDEX, min_probability=0.99) == []


def test_inserted_reports_whole_syllables_beside_a_unit_that_is_not_one():
    readings = ["ㄊㄧㄠ2", "ㄉㄚ4"]
    # ㄉㄧㄠ4 and a stray ㄧ4 between the two syllables: the stray final does not excuse the syllable
    lp = _frames(["ㄊ", "ㄧㄠ2", "_", "ㄉ", "ㄧㄠ4", "_", "ㄧ4", "_", "ㄉ", "ㄚ4"])
    assert inserted(lp, readings, INDEX) == [Inserted(0, ("ㄉ", "ㄧㄠ4"), pytest.approx(3 * FRAME_SECONDS))]
    # two extra syllables in a row are one finding; apart, two
    lp = _frames(["ㄊ", "ㄧㄠ2", "_", "ㄉ", "ㄧㄠ4", "ㄊ", "ㄧㄠ4", "_", "ㄉ", "ㄚ4"])
    assert [i.units for i in inserted(lp, readings, INDEX)] == [("ㄉ", "ㄧㄠ4", "ㄊ", "ㄧㄠ4")]
    lp = _frames(["ㄊ", "ㄧㄠ2", "_", "ㄉ", "ㄧㄠ4", "_", "ㄧ4", "_", "ㄊ", "ㄧㄠ4", "_", "ㄉ", "ㄚ4"])
    assert [i.units for i in inserted(lp, readings, INDEX)] == [("ㄉ", "ㄧㄠ4"), ("ㄊ", "ㄧㄠ4")]
    # a faint unit inside a pair still disqualifies that pair alone
    faint = _frames(["ㄊ", "ㄧㄠ2", "_", "ㄉ", "ㄧㄠ4", "_", "ㄊ", "ㄧㄠ4", "_", "ㄉ", "ㄚ4"])
    faint[4] = torch.full((len(VOCAB),), 0.1).log()
    faint[4, INDEX["ㄧㄠ4"]] = torch.tensor(0.3).log()
    assert [i.units for i in inserted(faint, readings, INDEX)] == [("ㄊ", "ㄧㄠ4")]


def test_clarity_is_low_for_a_syllable_that_is_not_heard():
    lp = _frames(["_", "ㄊ", "ㄧㄠ2", "_", "_", "_", "ㄉ", "ㄚ4", "_"])
    scores = clarity(lp, ["ㄊㄧㄠ2", "ㄧ1", "ㄉㄚ4"], INDEX)
    assert scores[0] > -0.1 and scores[2] > -0.1
    assert scores[1] < -3


def test_silence_before_is_the_time_to_the_first_sound():
    rate = 16000
    audio = np.concatenate([np.zeros(rate * 2, dtype=np.float32), 0.5 * np.ones(rate, dtype=np.float32)])
    assert silence_before(audio, rate) == pytest.approx(2.0)
    assert silence_before(np.zeros(rate, dtype=np.float32), rate) == pytest.approx(1.0)

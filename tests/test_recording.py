"""Cutting a recorded reading into takes, with stand-in recognition (no models)."""
import numpy as np
import pytest

from breezyvoice.audio import speech_pieces
from breezyvoice.recording.align import Take, assign, preferred
from breezyvoice.recording.script import load_script, section

RATE = 16000
LINES = {"B01": "你有沒有想過為什麼早上總是起不來？", "B02": "其實問題不在鬧鐘，而在你前一晚的習慣。",
         "B03": "睡前滑手機會讓大腦以為現在還是白天。"}


def same(text: str) -> str:
    return text


def test_load_script_and_section(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("# note\nA01 第一段。\n\nB01 第一句。\nB02 第二句。\nBB01 另一組。\n", encoding="utf-8")
    script = load_script(path)
    assert list(script) == ["A01", "B01", "B02", "BB01"]
    assert list(section(script, "B")) == ["B01", "B02"]


def test_load_script_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("A01 一。\nA01 二。\n", encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        load_script(path)


def test_speech_pieces_split_at_long_silences_only():
    rng = np.random.default_rng(0)
    sound = lambda s: 0.2 * rng.standard_normal(int(RATE * s))  # noqa: E731
    quiet = lambda s: np.zeros(int(RATE * s))  # noqa: E731
    audio = np.concatenate([quiet(0.5), sound(1.0), quiet(0.2), sound(1.0), quiet(1.0), sound(0.5), quiet(0.5)])
    pieces = speech_pieces(audio.astype(np.float32), RATE, 0.35)
    assert len(pieces) == 2
    assert pieces[0] == pytest.approx((0.5, 2.7), abs=0.05) and pieces[1] == pytest.approx((3.7, 4.2), abs=0.05)
    assert speech_pieces(np.zeros(RATE, dtype=np.float32), RATE, 0.35) == []


def test_assign_one_piece_per_line_with_repeats():
    heard = [LINES["B01"], LINES["B01"], LINES["B02"], LINES["B03"], LINES["B03"]]
    takes = assign(heard, LINES, same)
    assert [(t.line, t.first, t.last) for t in takes] == [("B01", 0, 0), ("B01", 1, 1), ("B02", 2, 2),
                                                           ("B03", 3, 3), ("B03", 4, 4)]
    assert all(t.error_rate == 0 for t in takes)


def test_assign_rejoins_a_sentence_split_by_a_pause():
    heard = ["其實問題不在鬧鐘", "而在你前一晚的習慣", LINES["B03"]]
    takes = assign(heard, LINES, same)
    assert [(t.line, t.first, t.last) for t in takes] == [("B02", 0, 1), ("B03", 2, 2)]


def test_assign_leaves_out_pieces_that_match_no_line():
    heard = [LINES["B01"], "不好意思重來一次", "其實問題", LINES["B02"]]
    takes = assign(heard, LINES, same)
    assert [(t.line, t.first, t.last) for t in takes] == [("B01", 0, 0), ("B02", 3, 3)]


def test_assign_tolerates_recognition_errors_and_reports_them():
    heard = ["你有沒有想過為什麼早上總是起不賴"]
    (take,) = assign(heard, LINES, same)
    assert take.line == "B01" and take.error_rate == pytest.approx(1 / 16)


def test_assign_does_not_glue_two_lines_into_one_take():
    heard = [LINES["B01"] + LINES["B02"]]  # no silence between two sentences: one piece
    assert assign(heard, LINES, same) == []


def test_preferred_takes_the_last_clean_take():
    takes = [Take("B01", 0, 0, 0.0), Take("B01", 1, 1, 0.25), Take("B01", 2, 2, 0.05),
             Take("B02", 3, 3, 0.2), Take("B02", 4, 4, 0.15)]
    chosen = preferred(takes)
    assert chosen["B01"] is takes[2]  # the last one within tolerance
    assert chosen["B02"] is takes[4]  # none is clean: the one with the fewest errors

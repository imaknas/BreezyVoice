"""Splitting text into the pieces that are synthesized one at a time (no model)."""
from breezyvoice.engine import MAX_SENTENCE_CHARS, split_long


def test_short_sentence_is_untouched():
    text = "其實問題不在鬧鐘，而在你前一晚的習慣。"
    assert split_long(text) == [text]


def test_long_sentence_breaks_at_commas_within_the_limit():
    clauses = ["每天早上起床之後", "先喝一杯溫開水", "接著打開窗戶讓新鮮的空氣流進來", "然後花幾分鐘伸展一下肩膀和脖子",
               "再坐下來把今天最重要的三件事情寫在紙上", "寫完之後不要急著看手機", "先從第一件事情開始動手"]
    text = "，".join(clauses) + "。"
    pieces = split_long(text)
    assert "".join(pieces) == text
    assert len(pieces) > 1
    assert all(sum(ch.isalnum() for ch in p) <= MAX_SENTENCE_CHARS for p in pieces)
    assert all(p.endswith(("，", "。")) for p in pieces)


def test_annotations_do_not_count_as_spoken_characters():
    text = "好[:ㄏㄠ3]" * 40 + "。"
    assert split_long(text) == [text]


def test_a_long_clause_without_commas_is_left_whole():
    text = "字" * 100 + "。"
    assert split_long(text) == [text]


def test_limit_is_respected_greedily():
    assert split_long("一二三，四五六，七八九。", limit=6) == ["一二三，四五六，", "七八九。"]

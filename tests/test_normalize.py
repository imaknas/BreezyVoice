import re

import pytest

from breezyvoice.text.frontend import normalize
from breezyvoice.text.normalize import cardinal, normalize_zh

CASES = [
    # 兩 vs 二
    ("有2個人", "有兩個人"),
    ("12個人", "十二個人"),
    ("第2名", "第二名"),
    ("2月2號", "二月二號"),
    ("2樓", "二樓"),
    ("2年級", "二年級"),
    ("2年", "兩年"),
    ("約2成", "約兩成"),
    ("2小時", "兩小時"),
    ("2.5倍", "二點五倍"),
    ("20000元", "兩萬元"),
    ("200人", "兩百人"),
    ("２個", "兩個"),
    # cardinals
    ("10050", "一萬零五十"),
    ("100000", "十萬"),
    ("110", "一百一十"),
    ("105", "一百零五"),
    ("1010000", "一百零一萬"),
    ("1,234,567", "一百二十三萬四千五百六十七"),
    ("3.14", "三點一四"),
    ("-5度", "負五度"),
    # percentages
    ("50%的人", "百分之五十的人"),
    ("3.5％", "百分之三點五"),
    ("1-3%", "百分之一到三"),
    # years, dates, times
    ("2025年", "二零二五年"),
    ("2000年", "兩千年"),
    ("1990年代", "一九九零年代"),
    ("民國114年", "民國一百一十四年"),
    ("2025/3/5開幕", "二零二五年三月五日開幕"),
    ("2025-03-05", "二零二五年三月五日"),
    ("10:30開會", "十點三十分開會"),
    ("8:05", "八點零五分"),
    ("2:00", "兩點"),
    # ranges, fractions, ratios, codes, units
    ("3-5天", "三到五天"),
    ("3~5天", "三到五天"),
    ("1/3的人", "三分之一的人"),
    ("比數1:2", "比數一比二"),
    ("0912345678", "零九一二三四五六七八"),
    ("30°C", "三十度"),
    ("5km", "五公里"),
    ("NT$300", "三百元"),
    ("3x", "三倍"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_normalize_zh(text, expected):
    assert normalize_zh(text) == expected


@pytest.mark.parametrize("n,expected", [(0, "零"), (10, "十"), (12, "十二"), (20, "二十"), (2, "二"),
                                        (1001, "一千零一"), (2200, "兩千兩百"), (100000001, "一億零一")])
def test_cardinal(n, expected):
    assert cardinal(n) == expected


def _spell(digits: str) -> str:
    return f"<{digits}>"


def test_annotations_untouched_and_preceding_char_kept():
    # Upstream dropped the character before a manual annotation (今天天氣真[:ㄏㄠ3]).
    assert normalize("今天天氣真好[:ㄏㄠ3]的一天。", _spell) == "今天天氣真好[:ㄏㄠ3]的一天。"
    assert normalize("有2[:ㄦ4]個", _spell) == "有二[:ㄦ4]個"


def test_no_ascii_digits_left_in_chinese_text():
    text = "2024年有3.5%的人，在10:30到12:00之間買了2杯、共NT$1,200，約1/4。"
    assert not re.search(r"\d", normalize(text, _spell))


def test_decimal_point_survives_period_rewrite():
    # "." becomes "、" for Chinese text; decimals must be read out before that.
    assert normalize("成長了3.5倍", _spell) == "成長了三點五倍"


def test_spacing_and_trailing_comma():
    assert normalize("歡迎使用 BreezyVoice 模型，", _spell) == "歡迎使用BreezyVoice模型。"


def test_english_only_text_uses_english_numbers():
    assert normalize("I have 3 apples", _spell) == "I have <3> apples"


@pytest.mark.parametrize("text, expected", [
    ("在戶外待一會兒。", "在戶外待一會。"),
    ("有點兒冷，等會兒再說。", "有點冷，等會再說。"),
    ("這兒比那兒暖。", "這裡比那裡暖。"),
    ("你要去哪兒？", "你要去哪裡？"),
])
def test_an_unspoken_er_is_left_out(text, expected):
    assert normalize(text, _spell) == expected


@pytest.mark.parametrize("text", ["兒子和女兒都來了。", "嬰兒與幼兒，還有兒童。", "那兒子不聽話。", "花兒開了。"])
def test_other_er_stay(text):
    assert normalize(text, _spell) == text


def test_an_er_with_a_reading_written_for_it_is_kept():
    assert normalize("等一會兒[:ㄦ2]再去那兒。", _spell) == "等一會兒[:ㄦ2]再去那裡。"

"""Text normalisation for Taiwan Mandarin: write numbers and symbols out as they are read aloud.

Pure functions, no model or I/O. Covered:

- integers (1,234 / 1234 / 12000 → 一千兩百三十四 / 一萬兩千), with 兩 where Taiwan Mandarin says it
  (兩百, 兩千, 兩萬, 兩億, and a bare 2 before a measure word: 2個 → 兩個, 2小時 → 兩小時),
  but 二 in 十二, 第2, 2月, 2號, 2樓 and other ordinals or dates;
- decimals (3.14 → 三點一四), negatives (-5 → 負五);
- percentages (50% → 百分之五十, 3.5％ → 百分之三點五, 1-3% → 百分之一到三);
- years read digit by digit (2025年 → 二零二五年, 1990年代 → 一九九零年代; but 2000年 → 兩千年), dates
  (2025/3/5, 2025-03-05, 2025年3月5日), times (10:30 → 十點三十分, 8:05 → 八點零五分);
- ranges (3-5、3~5、3～5 → 三到五), fractions (1/3 → 三分之一; write dates as 3月5日), ratios (1:2 → 一比二);
- digit strings with a leading zero or longer than 12 digits (phone numbers, codes) read digit by digit;
- a few units and symbols (km, kg, cm, °C, NT$, $, +, ×, x as in 3x, ² ³).

Text inside [...] (manual 注音 annotations) is left untouched by the caller.
"""
from __future__ import annotations

import re
import unicodedata

DIGITS = "零一二三四五六七八九"

# Measure words after which a bare 2 is read 兩.
_LIANG_CLASSIFIERS = (
    "個 位 隻 次 種 件 條 張 本 天 年 週 周 歲 倍 人 名 句 組 杯 碗 塊 元 份 點 分鐘 秒 小時 公里 公斤 公分 "
    "顆 根 片 支 枝 台 臺 部 家 間 層 輛 雙 對 段 頁 篇 首 招 步 口 下 遍 趟 場 堂 節 門 項 套 款 籃 匹 頭 "
    "條 道 筆 張 封 粒 滴 包 盒 瓶 箱 袋 斤 兩 碼 億 萬 千 百 歲 週年 星期 個月 禮拜 成 圈 輪 集 屆 季 代 餐 晚 夜"
).split()
_LIANG_RE = "|".join(sorted(set(_LIANG_CLASSIFIERS), key=len, reverse=True))

_UNITS = [
    (r"°C|℃", "度"),
    (r"°F|℉", "華氏度"),
    (r"km/h", "公里每小時"),
    (r"km", "公里"),
    (r"kg", "公斤"),
    (r"cm", "公分"),
    (r"mm", "毫米"),
    (r"ml|mL", "毫升"),
    (r"(?<![A-Za-z])m(?![A-Za-z])", "公尺"),
    (r"(?<![A-Za-z])g(?![A-Za-z])", "公克"),
]


def digits_reading(s: str) -> str:
    """Read a digit string one digit at a time: 2025 → 二零二五."""
    return "".join(DIGITS[int(c)] for c in s)


def _section(n: int) -> str:
    """Read 0 < n < 10000 (no leading 一 on 十 is added here; see cardinal)."""
    units = ["千", "百", "十", ""]
    s = f"{n:04d}"
    out = []
    zero_pending = False
    started = False
    for i, ch in enumerate(s):
        d = int(ch)
        if d == 0:
            if started:
                zero_pending = True
            continue
        if zero_pending:
            out.append("零")
            zero_pending = False
        digit = DIGITS[d]
        if d == 2 and units[i] in ("千", "百"):
            digit = "兩"
        out.append(digit + units[i])
        started = True
    return "".join(out)


def cardinal(n: int) -> str:
    """Read a non-negative integer: 12 → 十二, 105 → 一百零五, 20000 → 兩萬, 110 → 一百一十."""
    if n == 0:
        return "零"
    big_units = ["", "萬", "億", "兆"]
    parts = []
    i = 0
    while n > 0:
        parts.append((n % 10000, big_units[i]))
        n //= 10000
        i += 1
    out = []
    zero_pending = False
    for idx, (sec, unit) in enumerate(reversed(parts)):
        if sec == 0:
            zero_pending = True
            continue
        text = _section(sec)
        if out and (zero_pending or sec < 1000):
            text = "零" + text
        if unit and text == "二":
            text = "兩"
        out.append(text + unit)
        zero_pending = False
    result = "".join(out)
    # 一十 at the very start is read 十 (十二, 十萬); elsewhere it stays (一百一十).
    if result.startswith("一十"):
        result = result[1:]
    return result


def number(s: str) -> str:
    """Read a numeral string that may contain commas and one decimal point."""
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        return cardinal(int(whole or "0")) + "點" + digits_reading(frac)
    if len(s) > 1 and s.startswith("0") or len(s) > 12:
        return digits_reading(s)
    return cardinal(int(s))


_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"


def _to_halfwidth(text: str) -> str:
    out = []
    for ch in text:
        if ch in "０１２３４５６７８９％．：／～－＋，":
            ch = unicodedata.normalize("NFKC", ch)
            if ch == ",":
                ch = "，"  # keep the full-width comma as Chinese punctuation
        out.append(ch)
    return "".join(out)


def _year(m: re.Match) -> str:
    y = m.group(1)
    # 2000年 is 兩千年, 1000年 一千年; other years digit by digit (二零二五年).
    return cardinal(int(y)) if int(y) % 1000 == 0 else digits_reading(y)


def _date_ymd(m: re.Match) -> str:
    y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
    return f"{digits_reading(y)}年{cardinal(mo)}月{cardinal(d)}日"


def _time(m: re.Match) -> str:
    h, mi = int(m.group(1)), m.group(2)
    hour = "兩" if h == 2 else cardinal(h)
    if int(mi) == 0:
        return f"{hour}點"
    minutes = cardinal(int(mi)) if int(mi) >= 10 else "零" + cardinal(int(mi))
    return f"{hour}點{minutes}分"


def _bare_number(m: re.Match) -> str:
    s = m.group(0)
    return number(s)


def normalize_zh(text: str) -> str:
    """Write out numbers and symbols in Chinese text so that no ASCII digit remains."""
    t = _to_halfwidth(text)

    t = t.replace("²", "平方").replace("³", "立方")
    t = re.sub(r"NT\$\s*(" + _NUM + ")", lambda m: number(m.group(1)) + "元", t)
    t = re.sub(r"(?:US)?\$\s*(" + _NUM + ")", lambda m: number(m.group(1)) + "美元", t)

    # Dates and times before anything else touches their digits.
    t = re.sub(r"(?<!\d)(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?!\d)", _date_ymd, t)
    t = re.sub(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])", lambda m: _time(m) if int(m.group(1)) <= 24 and int(m.group(2)) < 60 else m.group(0), t)
    t = re.sub(r"(?<!\d)(\d{4})(?=\s*年(?!級))", _year, t)
    t = re.sub(r"(?<![\d:])(\d+):(\d+)(?![\d:])", lambda m: f"{cardinal(int(m.group(1)))}比{cardinal(int(m.group(2)))}", t)

    # Percentages, including ranges such as 1-3%.
    t = re.sub(r"(" + _NUM + r")\s*[-~～到至]\s*(" + _NUM + r")\s*%",
               lambda m: f"百分之{number(m.group(1))}到{number(m.group(2))}", t)
    t = re.sub(r"(" + _NUM + r")\s*%", lambda m: "百分之" + number(m.group(1)), t)

    # Ranges and fractions.
    t = re.sub(r"(?<![\d/])(" + _NUM + r")\s*[~～]\s*(" + _NUM + r")", lambda m: f"{number(m.group(1))}到{number(m.group(2))}", t)
    t = re.sub(r"(?<=[^\dA-Za-z\-])(" + _NUM + r")\s*-\s*(" + _NUM + r")(?![\d\-])",
               lambda m: f"{number(m.group(1))}到{number(m.group(2))}", t)
    t = re.sub(r"^(" + _NUM + r")\s*-\s*(" + _NUM + r")(?![\d\-])", lambda m: f"{number(m.group(1))}到{number(m.group(2))}", t)
    t = re.sub(r"(?<![\d/])(\d+)/(\d+)(?![\d/])", lambda m: f"{cardinal(int(m.group(2)))}分之{cardinal(int(m.group(1)))}", t)

    # Signs and multipliers.
    t = re.sub(r"(?<![\dA-Za-z])-(?=\d)", "負", t)
    t = re.sub(r"(?<![\dA-Za-z])\+(?=\d)", "正", t)
    t = re.sub(r"(\d)\s*[x×]\s*(?=\d)", r"\1乘以", t)
    t = re.sub(r"(" + _NUM + r")\s*[x×](?![A-Za-z])", lambda m: number(m.group(1)) + "倍", t)

    # Units after a number.
    for pattern, word in _UNITS:
        t = re.sub(r"(" + _NUM + r")\s*(?:" + pattern + r")", lambda m, w=word: number(m.group(1)) + w, t)

    # A bare 2 before a measure word is 兩, except after 第 / in ordinals.
    t = re.sub(r"(?<![\d.第])2(?![.\d])(?=\s*(?:" + _LIANG_RE + r"))(?!\s*年級)", "兩", t)

    t = re.sub(_NUM, _bare_number, t)
    return t

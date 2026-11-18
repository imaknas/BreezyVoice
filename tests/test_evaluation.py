"""The evaluation logic, with stand-in recognisers and encoder (no models, no optional packages)."""
import numpy as np
import pytest

from breezyvoice.audio import pauses
from breezyvoice.evaluation import signal
from breezyvoice.evaluation.compare import compare
from breezyvoice.evaluation.run import evaluate, load_sentences, measure_reference, summarize
from breezyvoice.evaluation.text import character_error_rate, comparable, edit_distance, marked_breaks, syllable_match

RATE = 22050


def same(text: str) -> str:
    return text


def test_comparable_drops_punctuation_and_case():
    assert comparable("你好，World！ 3 次。", same) == "你好world3次"


def test_comparable_unifies_script():
    assert comparable("軟體", lambda t: t.replace("軟體", "软体")) == "软体"


@pytest.mark.parametrize("a, b, expected", [
    ("", "", 0), ("abc", "abc", 0), ("abc", "", 3), ("", "abc", 3),
    ("今天天氣真好", "今天天真好", 1), ("今天天氣真好", "今天天器真好", 1), ("kitten", "sitting", 3),
])
def test_edit_distance(a, b, expected):
    assert edit_distance(a, b) == expected


def test_character_error_rate_ignores_punctuation_and_can_exceed_one():
    assert character_error_rate("今天，天氣真好。", "今天天氣真好", same) == 0
    assert character_error_rate("今天天氣真好", "今天天器真", same) == pytest.approx(2 / 6)
    assert character_error_rate("好", "好的好的", same) == 3


def test_character_error_rate_needs_a_reference():
    with pytest.raises(ValueError):
        character_error_rate("，。", "好", same)


def test_syllable_match_counts_homophones_as_equal():
    sounds = {"刺": "ci", "喫": "chi", "吃": "chi", "痛": "tong"}
    to_syllables = lambda text: [sounds[ch] for ch in text]  # noqa: E731
    assert syllable_match("吃痛", "喫痛", to_syllables) == 1
    assert syllable_match("刺痛", "喫痛", to_syllables) == pytest.approx(0.5)


def test_syllable_match_accepts_listed_variants_only_one_for_one():
    to_syllables = lambda text: text.split()  # noqa: E731
    assert syllable_match("wo he ni", "wo han ni", to_syllables) == pytest.approx(2 / 3)
    assert syllable_match("wo he ni", "wo han ni", to_syllables, {("he", "han")}) == 1
    assert syllable_match("wo han ni", "wo he ni", to_syllables, {("he", "han")}) == pytest.approx(2 / 3)
    assert syllable_match("", "", to_syllables) == 1


def test_taiwan_syllables():
    pytest.importorskip("pypinyin")
    pytest.importorskip("opencc")
    from breezyvoice.evaluation.models import TAIWAN_READINGS, taiwan_syllables

    syllables = taiwan_syllables()
    assert syllables("軟體 1968，Hello！") == ["ruan", "ti", "yi", "jiu", "liu", "ba", "hello"]
    # the same word in either script reads the same
    for traditional, simplified in [("嘴乾", "嘴干"), ("音樂", "音乐"), ("沉沒", "沉没"), ("調節", "调节"), ("傳遞", "传递")]:
        assert syllables(traditional) == syllables(simplified), traditional
    # distinctions not made in Taiwan are merged
    assert syllables("是四十") == ["si", "si", "si"] and syllables("新星") == ["xin", "xin"]
    # a Taiwan reading heard as another character is accepted
    assert syllable_match("我和你", "我汗你", syllables, TAIWAN_READINGS) == 1
    assert syllable_match("液體", "一体", syllables, TAIWAN_READINGS) == 1
    assert syllable_match("我和你", "我看你", syllables, TAIWAN_READINGS) < 1


def test_load_sentences_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("# note\n\n第一句。\n  第二句。  \n", encoding="utf-8")
    assert load_sentences(path) == ["第一句。", "第二句。"]


def test_bundled_sentences_have_no_digits_or_latin_letters():
    from breezyvoice.evaluation.cli import SENTENCES

    sentences = load_sentences(SENTENCES)
    assert len(sentences) >= 30
    assert not [s for s in sentences if any(ch.isascii() and ch.isalnum() for ch in s)]
    assert max(len(comparable(s, same)) for s in sentences) > 120  # the comma-only long sentence


def test_accent_sentences_load():
    from breezyvoice.evaluation.cli import SENTENCES

    assert len(load_sentences(SENTENCES.with_name("accent_sentences.txt"))) >= 10


def test_code_switch_sentences_mix_both_scripts():
    from breezyvoice.evaluation.cli import SENTENCES

    sentences = load_sentences(SENTENCES.with_name("code_switch_sentences.txt"))
    assert len(sentences) >= 10
    for s in sentences:
        assert any(ch.isascii() and ch.isalpha() for ch in s) and any(ord(ch) > 0x3000 for ch in s), s


def test_signal_checks():
    t = np.arange(RATE, dtype=np.float32) / RATE
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    faded = tone * np.linspace(1, 0, RATE, dtype=np.float32) ** 4
    assert signal.clipped_fraction(tone) == 0
    assert signal.clipped_fraction(np.clip(tone * 3, -1, 1)) > 0.5
    assert signal.tail_level_db(tone, RATE) == pytest.approx(0, abs=0.5)
    assert signal.tail_level_db(faded, RATE) < -20


def _speech(*parts):
    """Alternating sound and silence, by duration in seconds."""
    rng = np.random.default_rng(0)
    out = [(0.2 * rng.standard_normal(int(RATE * d)) if i % 2 == 0 else np.zeros(int(RATE * d)))
           for i, d in enumerate(parts)]
    return np.concatenate(out).astype(np.float32)


def test_pauses_finds_long_gaps_inside_speech_only():
    audio = np.concatenate([np.zeros(RATE // 2, dtype=np.float32), _speech(0.5, 0.4, 0.5, 0.1, 0.5),
                            np.zeros(RATE // 2, dtype=np.float32)])
    found = pauses(audio, RATE)
    assert len(found) == 1
    assert found[0][0] == pytest.approx(1.0, abs=0.04) and found[0][1] == pytest.approx(1.4, abs=0.04)
    assert pauses(np.zeros(RATE, dtype=np.float32), RATE) == []


@pytest.mark.parametrize("text, expected", [
    ("下次名字卡住時試試。", 0), ("其實問題不在鬧鐘，而在你前一晚的習慣。", 1),
    ("好了……我們走吧！", 1), ("第一，第二；第三。", 2), ("你覺得呢？", 0),
])
def test_marked_breaks(text, expected):
    assert marked_breaks(text) == expected


def test_mel_distance():
    rng = np.random.default_rng(0)
    a = (0.1 * rng.standard_normal(RATE)).astype(np.float32)
    assert signal.mel_distance(a, a, RATE) == 0
    assert signal.mel_distance(a, a * 0.5, RATE) > 0.1
    assert signal.mel_distance(a, a[:-10], RATE) is None


def _fakes():
    """A synthesizer whose audio encodes the seed, a recogniser that drops a character on odd seeds."""
    spoken = {}

    def synthesize(text, seed):
        audio = np.full(RATE * 2, 0.01 * (seed + 1), dtype=np.float32)
        spoken[audio.tobytes()[:8]] = (text, seed)
        return audio, RATE

    def transcribe(audio, rate):
        text, seed = spoken[audio.tobytes()[:8]]
        return text[1:] if seed % 2 else text

    def embed(audio, rate):
        v = np.array([1.0, float(audio[0]) * 10], dtype=np.float32)
        return v / np.linalg.norm(v)

    return synthesize, transcribe, embed


def test_evaluate_records_each_take_and_summary_counts_bad_ones():
    synthesize, transcribe, embed = _fakes()
    saved = {}
    ticks = iter(range(100))
    records = evaluate(
        synthesize, ["今天天氣真好。", "我們去散步吧。"], [0, 1], {"fake": transcribe}, embed,
        reference_embedding=np.array([1.0, 0.0], dtype=np.float32), unify_script=same, to_syllables=list, syllable_variants=frozenset(),
        save_wav=lambda name, audio, rate: saved.__setitem__(name, audio),
        clock=lambda: float(next(ticks)), log=lambda _: None,
    )
    assert [(r["sentence"], r["seed"]) for r in records] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert set(saved) == {r["wav"] for r in records} and len(saved) == 4
    assert records[0]["cer"] == {"fake": 0} and records[1]["cer"]["fake"] == pytest.approx(1 / 6)
    assert records[0]["chars"] == 6 and records[0]["seconds"] == 2 and records[0]["wall_seconds"] == 1
    assert records[0]["similarity"] > records[1]["similarity"]
    assert records[0]["pauses"] == [] and records[0]["unmarked_pauses"] == 0

    summary = summarize(records)
    assert summary["takes"] == 4
    assert summary["takes_with_unmarked_pauses"] == 0
    assert summary["real_time_factor"] == pytest.approx(0.5)
    assert summary["bad_takes"] == {"fake": 2}
    assert summary["bad_takes_all_recognisers"] == 2
    assert records[0]["syllable_match"] == {"fake": 1} and records[1]["syllable_match"]["fake"] < 0.97
    assert summary["retakes"] == {"fake": 2} and summary["retakes_all_recognisers"] == 2
    assert summary["cer_mean"]["fake"] == pytest.approx(1 / 12)
    assert summary["similarity_min"] == pytest.approx(records[1]["similarity"])


def test_measure_reference():
    audio = np.concatenate([np.full(RATE, 0.1, dtype=np.float32), np.full(RATE, 0.2, dtype=np.float32)])
    embed = lambda a, r: np.array([1.0, 0.0], dtype=np.float32)  # noqa: E731
    out = measure_reference(audio, RATE, "你好嗎", {"fake": lambda a, r: "你好"}, embed, same)
    assert out["seconds"] == 2
    assert out["cer"]["fake"] == pytest.approx(1 / 3)
    assert out["split_half_similarity"] == pytest.approx(1)


def _record(sentence, seed, cer, similarity):
    return {"sentence": sentence, "seed": seed, "cer": {"fake": cer}, "similarity": similarity,
            "wav": f"{sentence}_{seed}"}


def test_compare_pairs_by_sentence_and_seed():
    base = [_record(0, 0, 0.0, 0.9), _record(0, 1, 0.2, 0.8), _record(1, 0, 0.0, 0.9)]
    candidate = [_record(0, 1, 0.0, 0.7), _record(0, 0, 0.1, 0.9), _record(2, 0, 0.0, 0.9)]
    waves = {"0_0": np.zeros(10, dtype=np.float32), "0_1": np.zeros(12, dtype=np.float32)}

    def load_pair(b, c):
        # the candidate's take of (0, 1) has a different length: its tokens changed
        return waves[b["wav"]], waves[c["wav"]] if c["seed"] == 0 else np.zeros(20, dtype=np.float32), RATE

    report = compare(base, candidate, load_pair,
                     lambda a, b, rate: None if a.shape != b.shape else 0.25)
    assert report["pairs"] == 2 and report["unpaired"] == 1
    assert report["same_length"] == 1
    assert report["mel_distance_mean"] == 0.25 and report["mel_distance_max"] == 0.25
    assert report["cer_worse"] == {"fake": 1} and report["cer_better"] == {"fake": 1}
    assert report["cer_delta_mean"]["fake"] == pytest.approx(-0.05)
    assert report["similarity_delta_min"] == pytest.approx(-0.1)


def test_compare_needs_common_takes():
    with pytest.raises(ValueError):
        compare([_record(0, 0, 0, 1)], [_record(1, 0, 0, 1)], lambda b, c: None, lambda a, b, r: None)


def test_fetch_checks_the_checksum(tmp_path, monkeypatch):
    from breezyvoice.evaluation import models

    monkeypatch.setenv("BREEZYVOICE_HOME", str(tmp_path))

    def bogus(url, dest):
        dest.write_bytes(b"not the model")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        models.fetch(source=bogus, log=lambda _: None)
    assert not list((tmp_path / "evaluation").glob("*.onnx"))

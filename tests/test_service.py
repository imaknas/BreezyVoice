"""Request handling with a stand-in engine (no model)."""
import numpy as np
import pytest

from breezyvoice.engine import Decoding, Synthesis
from breezyvoice.service import handle

RATE = 22050


class FakeEngine:
    """Speaks noise with one 0.16 s gap in the middle."""

    def speaker(self, wav, transcript):
        return object()

    def synthesize(self, text, speaker, seed):
        rng = np.random.default_rng(seed)
        sound = lambda seconds: 0.2 * rng.standard_normal(int(RATE * seconds))  # noqa: E731
        audio = np.concatenate([sound(0.5), np.zeros(int(RATE * 0.16)), sound(0.5)]).astype(np.float32)
        return Synthesis(audio=audio, sample_rate=RATE, text=text,
                         decoding=(Decoding((2.5, 0.25), (-1.5, -0.125), len(audio)),))


@pytest.fixture
def request_json(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"")
    return {"text": "你好。", "ref_wav": str(ref), "ref_text": "參考。", "out": str(tmp_path / "out.wav")}


def test_default_threshold_ignores_a_short_gap(request_json):
    result = handle(FakeEngine(), request_json)
    assert result["ok"], result
    assert result["pauses"] == [] and result["min_pause_s"] == 0.25


def test_requested_threshold_is_applied_and_echoed(request_json):
    result = handle(FakeEngine(), {**request_json, "min_pause_s": 0.12})
    assert result["ok"], result
    assert result["min_pause_s"] == 0.12
    ((start, end),) = result["pauses"]
    assert start == pytest.approx(0.5, abs=0.03) and end == pytest.approx(0.66, abs=0.03)


@pytest.mark.parametrize("value", [0, -1, 0.01, 11, "0.12", True, None])
def test_bad_threshold_is_a_bad_request(request_json, value):
    result = handle(FakeEngine(), {**request_json, "min_pause_s": value})
    assert not result["ok"] and result["kind"] == "bad_request" and "min_pause_s" in result["error"]


# --- with a reviewer ------------------------------------------------------------------------------

from breezyvoice.pronunciation.pauses import Pause, tidy  # noqa: E402
from breezyvoice.pronunciation.readings import Unsupported  # noqa: E402
from breezyvoice.pronunciation.review import Review, WrongReading  # noqa: E402


class FakeReviewer:
    """Finds the 0.16 s gap of FakeEngine's audio in mid-phrase and, on request, a wrong reading."""

    def __init__(self, wrong=False, unsupported=False):
        self.wrong, self.unsupported = wrong, unsupported

    def review(self, audio, sample_rate, text):
        if self.unsupported:
            raise Unsupported("no reading for 'O'")
        gap = Pause(0.5, 0.66, 0, False)
        wrong = (WrongReading("好", 1, "ㄏㄠ3", "ㄏㄠ4", -8.0),) if self.wrong else ()
        return Review(("你", "好"), wrong, (), (), (gap,), (gap,), times=((0.0, 0.5), (0.66, 1.16)))

    def tidy(self, audio, sample_rate, review):
        return tidy(audio, sample_rate, review.pauses)


def test_without_a_reviewer_there_is_no_review(request_json):
    assert "review" not in handle(FakeEngine(), request_json)


def test_reviewer_tidies_pauses_and_reports_them(request_json):
    result = handle(FakeEngine(), request_json, FakeReviewer())
    assert result["ok"], result
    review = result["review"]
    assert review["tidied"] and not review["needs_retake"]
    assert review["unnatural_pauses"] == [{"after": "你", "position": 0, "seconds": 0.16, "at_punctuation": False}]
    assert result["duration_s"] == pytest.approx(1.16 - 0.12, abs=0.02)  # the 0.16 s gap is cut to 0.04 s


def test_tidying_can_be_turned_off_per_request(request_json):
    result = handle(FakeEngine(), {**request_json, "tidy_pauses": False}, FakeReviewer())
    assert not result["review"]["tidied"] and result["duration_s"] == pytest.approx(1.16, abs=0.01)
    bad = handle(FakeEngine(), {**request_json, "tidy_pauses": "no"}, FakeReviewer())
    assert not bad["ok"] and bad["kind"] == "bad_request"


def test_each_take_is_recorded_as_it_was_before_tidying(request_json):
    kept = []
    result = handle(FakeEngine(), {**request_json, "seed": 4}, FakeReviewer(), kept.append)
    assert result["review"]["tidied"] and "record_error" not in result
    (take,) = kept
    assert (take["text"], take["seed"], take["out"]) == ("你好。", 4, request_json["out"])
    assert take["decoding"] == [{"samples": int(RATE * 0.5) * 2 + int(RATE * 0.16),
                                 "entropy": [2.5, 0.25], "log_prob": [-1.5, -0.125]}]
    assert take["syllables"] == [{"character": "你", "start": 0.0, "end": 0.5},
                                 {"character": "好", "start": 0.66, "end": 1.16}]
    assert take["review"] == result["review"]


def test_a_take_is_recorded_without_a_reviewer_and_when_it_cannot_be_reviewed(request_json):
    kept = []
    handle(FakeEngine(), request_json, None, kept.append)
    handle(FakeEngine(), request_json, FakeReviewer(unsupported=True), kept.append)
    assert [("syllables" in t, t.get("review")) for t in kept] == [(False, None), (False, {"unsupported": "no reading for 'O'"})]
    assert all(t["decoding"][0]["entropy"] == [2.5, 0.25] for t in kept)


@pytest.mark.parametrize("error", [OSError("no space left"), TypeError("not JSON serializable")])
def test_a_take_that_cannot_be_recorded_is_still_delivered(request_json, error):
    def failing(take):
        raise error

    result = handle(FakeEngine(), request_json, None, failing)
    assert result["ok"] and result["record_error"] == f"{type(error).__name__}: {error}"


def test_a_wrong_reading_asks_for_a_retake(request_json):
    review = handle(FakeEngine(), request_json, FakeReviewer(wrong=True))["review"]
    assert review["needs_retake"]
    assert review["wrong_readings"] == [{"character": "好", "position": 1, "intended": "ㄏㄠ3", "heard": "ㄏㄠ4", "margin": -8.0}]


def test_text_the_reviewer_cannot_read_is_still_synthesized(request_json):
    result = handle(FakeEngine(), request_json, FakeReviewer(unsupported=True))
    assert result["ok"] and result["review"] == {"unsupported": "no reading for 'O'"}


def test_inserted_syllables_a_long_silence_and_unclear_syllables_ask_for_another_take():
    from breezyvoice.pronunciation.align import Inserted
    from breezyvoice.pronunciation.review import Unclear, report

    def review(**found):
        return Review(("你", "好"), (), (), (), (), (), **found)

    assert not review(silence_before=1.2).needs_retake
    assert review(silence_before=1.2, unclear=(Unclear("好", 1, -14.0),)).needs_retake
    assert review(silence_before=2.0).needs_retake
    added = report(review(inserted=(Inserted(0, ("ㄌ", "ㄞ2"), 0.5),)))
    assert added["needs_retake"] and added["inserted"] == [{"units": "ㄌㄞ2", "after": "你", "position": 0, "at": 0.5}]
    assert report(review(unclear=(Unclear("好", 1, -14.02),)))["unclear"] == [{"character": "好", "position": 1, "score": -14.0}]

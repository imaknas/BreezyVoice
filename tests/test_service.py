"""Request handling with a stand-in engine (no model)."""
import numpy as np
import pytest

from breezyvoice.engine import Synthesis
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
        return Synthesis(audio=audio, sample_rate=RATE, text=text)


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

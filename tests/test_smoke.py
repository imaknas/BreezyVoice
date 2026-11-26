"""Synthesize one sentence end to end. Needs `breezyvoice setup` to have run once (skips otherwise)."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from breezyvoice import assets

ROOT = Path(__file__).resolve().parents[1]
REF_WAV = ROOT / "data" / "example.wav"
REF_TEXT = ("在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。"
            "只有擁有解密方法的對象，經由解密過程，才能將密文還原為正常可讀的內容。")
TEXT = "今天天氣真好，我們一起去公園散步吧。"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def engine():
    try:
        assets.resolve()
    except assets.AssetsMissing as e:
        pytest.skip(str(e))
    from breezyvoice.loader import load

    return load("auto")


def _check_speech(audio: np.ndarray, sample_rate: int, n_chars: int) -> None:
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() <= 1.0
    rms = float(np.sqrt(np.mean(audio**2)))
    assert rms > 0.005, f"near-silent output (rms={rms})"
    seconds = len(audio) / sample_rate
    # Mandarin narration runs roughly 3-7 characters per second.
    assert n_chars / 8 < seconds < n_chars / 2, f"{seconds:.2f}s for {n_chars} characters"


def test_synthesize_one_sentence(engine, tmp_path):
    from breezyvoice.service import handle

    out = tmp_path / "out.wav"
    result = handle(engine, {"text": TEXT, "ref_wav": str(REF_WAV), "ref_text": REF_TEXT, "seed": 7, "out": str(out)})
    assert result["ok"], result
    assert all(0 < a < b < result["duration_s"] and b - a >= 0.25 for a, b in result["pauses"])
    audio, sr = soundfile.read(out, dtype="float32")
    assert sr == engine.sample_rate == 22050
    _check_speech(audio, sr, n_chars=16)
    json.dumps(result)


def test_same_seed_same_audio(engine):
    speaker = engine.speaker(REF_WAV, REF_TEXT)
    first = engine.synthesize("今天天氣真好。", speaker, seed=11)
    a, b = first.audio, engine.synthesize("今天天氣真好。", speaker, seed=11).audio
    (decoding,) = first.decoding
    assert decoding.samples == len(a) and len(decoding.entropy) == len(decoding.log_prob)
    assert 45 < len(decoding.entropy) / first.duration_s < 55  # fifty speech tokens to the second
    assert all(e >= 0 for e in decoding.entropy) and all(p <= 0 for p in decoding.log_prob)
    assert a.shape == b.shape  # the same speech tokens
    # On CPU the two takes are bit-identical. The flow decoder and vocoder on MPS are not exactly
    # repeatable: samples differ by up to about 2e-4 from run to run, a few steps of a 16-bit file.
    assert np.allclose(a, b, atol=1e-3)


def test_a_sentence_carried_on_keeps_what_was_said_and_never_ends_again(engine, monkeypatch):
    speaker = engine.speaker(REF_WAV, REF_TEXT)
    monkeypatch.setattr(engine, "_unfinished", None)
    as_ended = engine.synthesize("今天天氣真好。", speaker, seed=11)
    monkeypatch.setattr(engine, "_unfinished", lambda audio, rate: True)
    carried = engine.synthesize("今天天氣真好。", speaker, seed=11)
    (before,), (after,) = as_ended.decoding, carried.decoding
    assert before.carried_on == 0 and after.carried_on == 12
    assert len(after.entropy) == len(before.entropy) + 12 == len(after.log_prob)
    # The same seed says the same thing up to where the model first stopped.
    assert after.log_prob[:len(before.log_prob)] == pytest.approx(before.log_prob, abs=1e-4)
    assert 0.2 < carried.duration_s - as_ended.duration_s < 0.28  # twelve tokens at fifty to the second
    assert all(np.isfinite(after.log_prob))  # the end of speech, left out, is never the token drawn


def test_carrying_one_sentence_on_leaves_the_next_one_as_it_was(engine, monkeypatch):
    speaker = engine.speaker(REF_WAV, REF_TEXT)
    text = "今天天氣真好。我們去公園散步吧。"
    monkeypatch.setattr(engine, "_unfinished", None)
    as_ended = engine.synthesize(text, speaker, seed=5).decoding
    calls = iter([True, False, False])  # only the first sentence is taken to have stopped short
    monkeypatch.setattr(engine, "_unfinished", lambda audio, rate: next(calls))
    carried = engine.synthesize(text, speaker, seed=5).decoding
    assert [d.carried_on for d in carried] == [4, 0]
    assert carried[1].log_prob == pytest.approx(as_ended[1].log_prob, abs=1e-4)
    assert carried[1].samples == as_ended[1].samples


def test_bad_request_is_reported_not_raised(engine, tmp_path):
    from breezyvoice.service import handle

    result = handle(engine, {"text": TEXT, "ref_wav": str(tmp_path / "missing.wav"), "ref_text": REF_TEXT,
                             "out": str(tmp_path / "x.wav")})
    assert result == {"ok": False, "error": f"ref_wav not found: {tmp_path / 'missing.wav'}", "kind": "bad_request"}

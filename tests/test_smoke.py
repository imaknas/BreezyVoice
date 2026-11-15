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
    audio, sr = soundfile.read(out, dtype="float32")
    assert sr == engine.sample_rate == 22050
    _check_speech(audio, sr, n_chars=16)
    json.dumps(result)


def test_same_seed_same_audio(engine):
    speaker = engine.speaker(REF_WAV, REF_TEXT)
    a = engine.synthesize("今天天氣真好。", speaker, seed=11).audio
    b = engine.synthesize("今天天氣真好。", speaker, seed=11).audio
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=1e-4)


def test_bad_request_is_reported_not_raised(engine, tmp_path):
    from breezyvoice.service import handle

    result = handle(engine, {"text": TEXT, "ref_wav": str(tmp_path / "missing.wav"), "ref_text": REF_TEXT,
                             "out": str(tmp_path / "x.wav")})
    assert result == {"ok": False, "error": f"ref_wav not found: {tmp_path / 'missing.wav'}", "kind": "bad_request"}

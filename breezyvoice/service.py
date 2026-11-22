"""Request handling shared by the batch (stdin/stdout) and HTTP front ends."""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import soundfile

from breezyvoice.audio import pauses
from breezyvoice.engine import BreezyVoice, Synthesis
from breezyvoice.pronunciation.readings import Unsupported
from breezyvoice.pronunciation.review import report as review_report


DEFAULT_MIN_PAUSE_S = 0.25
# Stop consonants close the mouth for up to about a tenth of a second; anything shorter is not a pause.
SHORTEST_PAUSE_S = 0.05


class BadRequest(ValueError):
    pass


@dataclass(frozen=True)
class Request:
    text: str
    ref_wav: Path
    ref_text: str
    seed: int
    out: Path
    min_pause_s: float
    tidy_pauses: bool

    @classmethod
    def from_json(cls, obj: object) -> "Request":
        if not isinstance(obj, dict):
            raise BadRequest("request must be a JSON object")
        missing = [k for k in ("text", "ref_wav", "ref_text", "out") if not obj.get(k)]
        if missing:
            raise BadRequest(f"missing field(s): {', '.join(missing)}")
        seed = obj.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise BadRequest("seed must be a non-negative integer")
        min_pause_s = obj.get("min_pause_s", DEFAULT_MIN_PAUSE_S)
        if isinstance(min_pause_s, bool) or not isinstance(min_pause_s, (int, float)) \
                or not SHORTEST_PAUSE_S <= min_pause_s <= 10:
            raise BadRequest(f"min_pause_s must be a number of seconds between {SHORTEST_PAUSE_S} and 10")
        tidy_pauses = obj.get("tidy_pauses", True)
        if not isinstance(tidy_pauses, bool):
            raise BadRequest("tidy_pauses must be true or false")
        ref_wav = Path(obj["ref_wav"]).expanduser()
        if not ref_wav.is_file():
            raise BadRequest(f"ref_wav not found: {ref_wav}")
        out = Path(obj["out"]).expanduser()
        if out.suffix.lower() != ".wav":
            raise BadRequest("out must end in .wav")
        return cls(text=str(obj["text"]), ref_wav=ref_wav, ref_text=str(obj["ref_text"]), seed=seed, out=out,
                   min_pause_s=float(min_pause_s), tidy_pauses=tidy_pauses)


def write_wav(path: Path, result: Synthesis) -> None:
    """16-bit PCM, written to a temporary file and renamed so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=path.parent)
    os.close(fd)
    try:
        soundfile.write(tmp, result.audio, result.sample_rate, subtype="PCM_16")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def handle(engine: BreezyVoice, payload: object, reviewer=None) -> dict:
    """Run one request; errors come back as {"ok": false, "error": ...} rather than raising.

    With a ``reviewer`` (``breezyvoice.pronunciation.review.Reviewer``) the take is checked against its
    text: the response reports wrong readings and stray sounds, which only another seed can fix, and
    unnatural pauses and pauses missing after a colon or semicolon, which are mended in the audio unless
    the request says ``tidy_pauses: false``.
    """
    try:
        req = Request.from_json(payload)
        start = time.monotonic()
        speaker = engine.speaker(req.ref_wav, req.ref_text)
        result = engine.synthesize(req.text, speaker, seed=req.seed)
        report = None
        if reviewer is not None:
            try:
                review = reviewer.review(result.audio, result.sample_rate, req.text)
            except Unsupported as e:
                report = {"unsupported": str(e)}
            else:
                report = review_report(review)
                report["tidied"] = bool(req.tidy_pauses and (review.unnatural_pauses
                                                              or any(m.at is not None for m in review.missing_pauses)))
                if report["tidied"]:
                    result = Synthesis(audio=reviewer.tidy(result.audio, result.sample_rate, review),
                                       sample_rate=result.sample_rate, text=result.text)
        write_wav(req.out, result)
        response = {
            "ok": True,
            "out": str(req.out),
            "duration_s": round(result.duration_s, 3),
            "sample_rate": result.sample_rate,
            "seed": req.seed,
            "text": result.text,
            # Silent stretches inside the speech, [start, end] in seconds: a caller that knows where
            # the words fall can tell a pause at a comma from one that splits a phrase.
            "pauses": [[round(a, 2), round(b, 2)]
                       for a, b in pauses(result.audio, result.sample_rate, min_seconds=req.min_pause_s)],
            "min_pause_s": req.min_pause_s,
            "elapsed_s": round(time.monotonic() - start, 3),
        }
        if report is not None:
            response["review"] = report
        return response
    except BadRequest as e:
        return {"ok": False, "error": str(e), "kind": "bad_request"}
    except Exception as e:  # noqa: BLE001 - reported to the caller, the process keeps serving
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "kind": "synthesis_failed"}

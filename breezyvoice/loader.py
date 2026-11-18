"""Wires the concrete pieces (files, ONNX sessions, g2pW, tokenizer, networks) into a BreezyVoice."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from breezyvoice import assets
from breezyvoice.engine import BreezyVoice
from breezyvoice.model import Placement, load_networks
from breezyvoice.speech_frontend import SpeechFrontend, onnx_session
from breezyvoice.text import word_stats
from breezyvoice.text.frontend import Annotator, TextFrontend
from cosyvoice.llm import sampling

DEVICES = ("cpu", "mps", "auto")
# "top-k" is the upstream decoding setting and reproduces takes made with it. The two give different
# takes for the same seed.
SAMPLERS = {"repetition-aware": lambda: sampling.repetition_aware(), "top-k": lambda: sampling.top_k(25)}
DEFAULT_SAMPLER = "repetition-aware"


def placement_for(device: str) -> Placement:
    """``auto`` puts each network where it is fastest: on Apple Silicon the LLM on CPU and the flow
    decoder and vocoder on MPS (see README for timings); without MPS, everything on CPU.

    The LLM decodes one position at a time, which is bound by reading its weights and by per-operator
    overhead, where the GPU has no edge; the flow decoder's wide convolutions and attention do gain.
    """
    if device == "auto":
        if not torch.backends.mps.is_available():
            device = "cpu"
        else:
            return Placement(llm=torch.device("cpu"), flow=torch.device("mps"), hift=torch.device("mps"))
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    d = torch.device(device)
    return Placement(llm=d, flow=d, hift=d)


def _g2p(g2pw_dir: Path, bert_dir: Path):
    from g2pw import G2PWConverter

    converter = G2PWConverter(model_dir=str(g2pw_dir), model_source=str(bert_dir), style="bopomofo")
    # 0 is falsy in G2PWConverter's constructor, so set it afterwards: no DataLoader worker processes.
    converter.num_workers = 0
    return lambda text: converter(text)[0]


def reading_tools(weights_dir: Path | None = None):
    """(normalise, predict readings, path of the polyphone table), as the synthesizer's text front end has them.

    ``normalise`` writes a text out the way it will be spoken; ``predict`` returns one bopomofo reading
    with tone per character of its input, or None for what is not spoken.
    """
    from breezyvoice.text.frontend import normalize

    found = assets.resolve(weights_dir)
    spell = _english_numbers()
    return (lambda text: normalize(text, spell)), _g2p(found.g2pw_dir, found.bert_dir), \
        found.g2pw_dir / "POLYPHONIC_CHARS.txt"


def _english_numbers():
    import inflect

    engine = inflect.engine()
    return lambda digits: engine.number_to_words(digits)


def _encoder():
    from breezyvoice._vendor.whisper.tokenizer import get_tokenizer

    tok = get_tokenizer(multilingual=True, num_languages=100, language="en", task="transcribe")
    return lambda text: tok.encode(text, allowed_special="all")


def load(device: str = "auto", weights_dir: Path | None = None, threads: int | None = None,
         sampler: str = DEFAULT_SAMPLER) -> BreezyVoice:
    found = assets.resolve(weights_dir)
    if threads:
        torch.set_num_threads(threads)
    ort_threads = threads or max(1, (os.cpu_count() or 2) // 2)
    annotator = Annotator(
        g2p=_g2p(found.g2pw_dir, found.bert_dir),
        char_frequency=word_stats.word_to_dataset_frequency,
        char_readings=word_stats.char2phn,
        always_annotate=word_stats.always_augment_chars,
    )
    return BreezyVoice(
        networks=load_networks(found.weights_dir, placement_for(device)),
        text_frontend=TextFrontend(annotator, _english_numbers()),
        encode=_encoder(),
        sampler=SAMPLERS[sampler](),
        speech_frontend=SpeechFrontend(
            campplus=onnx_session(found.weights_dir / "campplus.onnx", ort_threads),
            speech_tokenizer=onnx_session(found.weights_dir / "speech_tokenizer_v1.onnx", ort_threads),
        ),
    )

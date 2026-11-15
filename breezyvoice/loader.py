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

DEVICES = ("cpu", "mps", "auto")


def placement_for(device: str) -> Placement:
    """``auto`` = MPS when available (about 2x faster than CPU on an M4 Pro, see README), else CPU."""
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
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


def _english_numbers():
    import inflect

    engine = inflect.engine()
    return lambda digits: engine.number_to_words(digits)


def _encoder():
    from breezyvoice._vendor.whisper.tokenizer import get_tokenizer

    tok = get_tokenizer(multilingual=True, num_languages=100, language="en", task="transcribe")
    return lambda text: tok.encode(text, allowed_special="all")


def load(device: str = "auto", weights_dir: Path | None = None, threads: int | None = None) -> BreezyVoice:
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
        speech_frontend=SpeechFrontend(
            campplus=onnx_session(found.weights_dir / "campplus.onnx", ort_threads),
            speech_tokenizer=onnx_session(found.weights_dir / "speech_tokenizer_v1.onnx", ort_threads),
        ),
    )

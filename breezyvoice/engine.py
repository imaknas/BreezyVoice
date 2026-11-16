"""Zero-shot synthesis: reference voice + text + seed → waveform."""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from breezyvoice.model import SAMPLE_RATE, Networks
from cosyvoice.llm.sampling import Sampler
from breezyvoice.speech_frontend import SpeakerAudio, SpeechFrontend, load_wav_16k

# Upstream decoding settings (single_inference.CustomCosyVoiceModel.inference).
MAX_TOKEN_TEXT_RATIO = 30
MIN_TOKEN_TEXT_RATIO = 3

_SENTENCE_END = re.compile(r"(?<=[？！。.?!])\s*")
_CLAUSE_END = re.compile(r"(?<=[，；：、,;:])")
_ANNOTATION = re.compile(r"\[:[^\]]*\]")
# The model was trained on utterances of up to about thirty seconds; far beyond that it loses its
# place and repeats itself until the length limit. Sixty characters is about seventeen seconds.
MAX_SENTENCE_CHARS = 60


def _spoken_length(text: str) -> int:
    return sum(ch.isalnum() for ch in _ANNOTATION.sub("", text))


def split_long(sentence: str, limit: int = MAX_SENTENCE_CHARS) -> list[str]:
    """Break a sentence that is too long to synthesize in one go at its commas.

    Clauses are packed greedily into pieces of at most ``limit`` spoken characters. A single clause
    longer than that has no natural break and is left whole.
    """
    if _spoken_length(sentence) <= limit:
        return [sentence]
    pieces, current = [], ""
    for clause in _CLAUSE_END.split(sentence):
        if current and _spoken_length(current + clause) > limit:
            pieces.append(current)
            current = clause
        else:
            current += clause
    if current:
        pieces.append(current)
    return pieces


@dataclass(frozen=True)
class Speaker:
    transcript: str  # annotated transcript
    text_token: torch.Tensor  # (1, T) int32
    audio: SpeakerAudio


@dataclass(frozen=True)
class Synthesis:
    audio: np.ndarray  # float32 mono in [-1, 1]
    sample_rate: int
    text: str  # what was fed to the model, after normalisation and 注音 annotation

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


def _ids(encode: Callable[[str], list[int]], text: str) -> torch.Tensor:
    return torch.tensor([encode(text)], dtype=torch.int32)


def _len(t: torch.Tensor) -> torch.Tensor:
    return torch.tensor([t.shape[1]], dtype=torch.int32)


class BreezyVoice:
    """Holds the loaded networks. Calls are serialised, so any thread may use it."""

    def __init__(self, networks: Networks, text_frontend: Callable[[str], str],
                 encode: Callable[[str], list[int]], speech_frontend: SpeechFrontend,
                 sampler: Sampler, speaker_cache_size: int = 8):
        self._nets = networks
        self._text = text_frontend
        self._encode = encode
        self._speech = speech_frontend
        self._sampler = sampler
        self._speakers: OrderedDict[tuple, Speaker] = OrderedDict()
        self._cache_size = speaker_cache_size
        self._lock = threading.Lock()

    sample_rate = SAMPLE_RATE

    def prepare_text(self, text: str) -> str:
        return self._text(text)

    def text_tokens(self, text: str) -> torch.Tensor:
        """The (1, T) int32 token ids the LM reads for ``text``: normalised, annotated, encoded."""
        return _ids(self._encode, self._text(text))

    @property
    def networks(self) -> Networks:
        return self._nets

    @property
    def speech_frontend(self) -> SpeechFrontend:
        return self._speech

    def speaker(self, wav_path: Path, transcript: str) -> Speaker:
        """Reference features, cached by file identity and transcript."""
        stat = wav_path.stat()
        key = (str(wav_path.resolve()), stat.st_mtime_ns, stat.st_size, transcript)
        with self._lock:
            if key in self._speakers:
                self._speakers.move_to_end(key)
                return self._speakers[key]
            annotated = self._text(transcript)
            speaker = Speaker(
                transcript=annotated,
                text_token=_ids(self._encode, annotated),
                audio=self._speech.speaker_audio(load_wav_16k(wav_path)),
            )
            self._speakers[key] = speaker
            while len(self._speakers) > self._cache_size:
                self._speakers.popitem(last=False)
            return speaker

    def synthesize(self, text: str, speaker: Speaker, seed: int) -> Synthesis:
        annotated = self._text(text)
        sentences = [piece for s in _SENTENCE_END.split(annotated) if s for piece in split_long(s)]
        if not sentences:
            raise ValueError("nothing to synthesize after normalisation")
        with self._lock, torch.inference_mode():
            torch.manual_seed(seed)  # all sampling draws from the CPU generator (see cosyvoice/llm, flow, hifigan)
            pieces = [self._sentence(s, speaker) for s in sentences]
        audio = torch.cat(pieces, dim=1).squeeze(0).numpy().astype(np.float32)
        return Synthesis(audio=audio, sample_rate=SAMPLE_RATE, text=annotated)

    def _sentence(self, sentence: str, speaker: Speaker) -> torch.Tensor:
        nets, place, spk = self._nets, self._nets.placement, speaker.audio
        text = _ids(self._encode, sentence)
        d = place.llm
        tokens = nets.llm.inference(
            text=text.to(d), text_len=_len(text).to(d),
            prompt_text=speaker.text_token.to(d), prompt_text_len=_len(speaker.text_token).to(d),
            prompt_speech_token=spk.speech_token.to(d), prompt_speech_token_len=_len(spk.speech_token).to(d),
            embedding=spk.embedding.to(d),
            sampler=self._sampler,
            max_token_text_ratio=MAX_TOKEN_TEXT_RATIO, min_token_text_ratio=MIN_TOKEN_TEXT_RATIO,
        )
        d = place.flow
        mel = nets.flow.inference(
            token=tokens.to(d), token_len=torch.tensor([tokens.shape[1]], dtype=torch.int32, device=d),
            prompt_token=spk.speech_token.to(d), prompt_token_len=_len(spk.speech_token).to(d),
            prompt_feat=spk.speech_feat.to(d), prompt_feat_len=torch.tensor([spk.speech_feat.shape[1]], dtype=torch.int32, device=d),
            embedding=spk.embedding.to(d),
        )
        return nets.hift.inference(mel=mel.to(place.hift)).float().cpu()

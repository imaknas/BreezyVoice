"""Training examples: what the language model reads and what it should produce."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class Example:
    line: str  # id of the script line; takes of one line share it
    text_token: torch.Tensor  # (T,) int32
    speech_token: torch.Tensor  # (S,) int32, 50 per second of audio
    embedding: torch.Tensor  # (192,) speaker embedding of this clip

    @property
    def seconds(self) -> float:
        return len(self.speech_token) / 50


def extract(line: str, text: str, speech_16k: torch.Tensor,
            text_tokens: Callable[[str], torch.Tensor],
            speech_token: Callable[[torch.Tensor], torch.Tensor],
            embedding: Callable[[torch.Tensor], torch.Tensor]) -> Example:
    """One clip and its text -> an example. The three callables are the synthesizer's own front ends,
    so training sees text and audio exactly as inference does."""
    return Example(line=line, text_token=text_tokens(text).squeeze(0).to(torch.int32),
                   speech_token=speech_token(speech_16k).squeeze(0).to(torch.int32),
                   embedding=embedding(speech_16k).squeeze(0).to(torch.float32))


def split(examples: Sequence[Example], held_out_every: int = 8) -> tuple[list[Example], list[Example]]:
    """(train, held out). Lines are held out whole: a second take of a held-out line in training
    would let the model score well on it by memory."""
    lines = sorted({e.line for e in examples})
    held = set(lines[::held_out_every])
    return [e for e in examples if e.line not in held], [e for e in examples if e.line in held]


def speaker_embedding(examples: Sequence[Example]) -> torch.Tensor:
    """The mean embedding of the clips. Every training example is conditioned on it, so the model
    ties this speaker to one point instead of chasing clip-to-clip variation."""
    return torch.stack([e.embedding for e in examples]).mean(dim=0)


def collate(examples: Sequence[Example], embedding: torch.Tensor) -> dict:
    """The batch dictionary ``TransformerLM.forward`` takes."""
    return {
        "text_token": pad_sequence([e.text_token for e in examples], batch_first=True, padding_value=0),
        "text_token_len": torch.tensor([len(e.text_token) for e in examples], dtype=torch.int32),
        "speech_token": pad_sequence([e.speech_token for e in examples], batch_first=True, padding_value=0),
        "speech_token_len": torch.tensor([len(e.speech_token) for e in examples], dtype=torch.int32),
        "embedding": embedding.unsqueeze(0).repeat(len(examples), 1),
    }

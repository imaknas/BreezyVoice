"""Training the sound-unit recogniser."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from breezyvoice.pronunciation.model import BLANK, Recognizer
from breezyvoice.pronunciation.units import flatten


@dataclass(frozen=True)
class Utterance:
    key: str  # utterances with the same key (the same text) stay on one side of the train/held-out split
    wav: str
    readings: tuple[str, ...]


@dataclass(frozen=True)
class Settings:
    steps: int = 4000
    batch_size: int = 8
    encoder_learning_rate: float = 3e-5
    head_learning_rate: float = 1e-3
    grad_clip: float = 5.0
    evaluate_every: int = 250
    seed: int = 0


@dataclass
class Result:
    best_step: int = 0
    best_error: float = float("inf")
    best_state: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)


def vocabulary(utterances: Sequence[Utterance]) -> list[str]:
    return [BLANK] + sorted({unit for u in utterances for unit in flatten(u.readings)})


def split(utterances: Sequence[Utterance], held_out_share: float = 0.05, seed: int = 0) -> tuple[list[Utterance], list[Utterance]]:
    """(train, held out), split by key so that no held-out text is also trained on."""
    keys = sorted({u.key for u in utterances})
    random.Random(seed).shuffle(keys)
    held = set(keys[:max(1, int(len(keys) * held_out_share))])
    return [u for u in utterances if u.key not in held], [u for u in utterances if u.key in held]


def edit_distance(a: Sequence, b: Sequence) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def greedy(log_probs: torch.Tensor, vocab: Sequence[str]) -> list[str]:
    """Best unit per frame, repeats and blanks collapsed."""
    out, previous = [], 0
    for i in log_probs.argmax(dim=-1).tolist():
        if i != previous and i != 0:
            out.append(vocab[i])
        previous = i
    return out


def _batch(recognizer: Recognizer, items: Sequence[Utterance], load_audio: Callable[[str], torch.Tensor],
           device: torch.device):
    waves = [recognizer.normalise(load_audio(u.wav)) for u in items]
    lengths = torch.tensor([len(w) for w in waves])
    audio = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True).to(device)
    log_probs, frames = recognizer(audio, lengths)
    return log_probs.cpu(), frames  # the CTC loss is computed on the CPU


def error_rates(recognizer: Recognizer, items: Sequence[Utterance], load_audio: Callable[[str], torch.Tensor],
                device: torch.device, batch_size: int) -> tuple[float, float]:
    """(unit error rate, tone error rate) by greedy decoding. The tone rate compares the sequences of
    tone digits alone."""
    was_training = recognizer.training
    recognizer.eval()
    errors = total = tone_errors = tone_total = 0
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            log_probs, frames = _batch(recognizer, chunk, load_audio, device)
            for k, utterance in enumerate(chunk):
                expected = flatten(utterance.readings)
                heard = greedy(log_probs[k, :int(frames[k])], recognizer.vocabulary)
                errors += edit_distance(expected, heard)
                total += len(expected)
                tones = [[u[-1] for u in seq if u[-1].isdigit()] for seq in (expected, heard)]
                tone_errors += edit_distance(*tones)
                tone_total += len(tones[0])
    recognizer.train(was_training)
    return errors / total, tone_errors / tone_total


def train(recognizer: Recognizer, training: Sequence[Utterance], held_out: Sequence[Utterance],
          load_audio: Callable[[str], torch.Tensor], device: torch.device, settings: Settings = Settings(),
          log: Callable[[dict], None] = lambda entry: None) -> Result:
    """Train in place; return the weights at the lowest held-out unit error."""
    if not training or not held_out:
        raise ValueError("training needs both training and held-out utterances")
    recognizer.to(device).train()
    encoder_params = [p for p in recognizer.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": settings.encoder_learning_rate},
                                   {"params": recognizer.head.parameters(), "lr": settings.head_learning_rate}])
    rng = random.Random(settings.seed)
    index = recognizer.unit_index
    result = Result()
    for step in range(1, settings.steps + 1):
        items = rng.sample(list(training), min(settings.batch_size, len(training)))
        log_probs, frames = _batch(recognizer, items, load_audio, device)
        targets = [torch.tensor([index[u] for u in flatten(i.readings)]) for i in items]
        loss = torch.nn.functional.ctc_loss(log_probs.transpose(0, 1), torch.cat(targets), frames,
                                            torch.tensor([len(t) for t in targets]), blank=0, zero_infinity=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder_params + list(recognizer.head.parameters()), settings.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % settings.evaluate_every == 0 or step == settings.steps:
            unit_error, tone_error = error_rates(recognizer, held_out, load_audio, device, settings.batch_size)
            entry = {"step": step, "loss": float(loss.detach()), "unit_error": unit_error, "tone_error": tone_error}
            result.history.append(entry)
            log(entry)
            if unit_error < result.best_error:
                result.best_error, result.best_step = unit_error, step
                result.best_state = {k: v.detach().cpu().clone() for k, v in recognizer.state_dict().items()}
    recognizer.eval()
    return result

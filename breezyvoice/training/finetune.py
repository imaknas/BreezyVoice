"""The fine-tuning loop for the language model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from breezyvoice.training.data import Example, collate


@dataclass(frozen=True)
class Settings:
    steps: int = 400
    batch_size: int = 8
    learning_rate: float = 1e-5
    grad_clip: float = 5.0
    evaluate_every: int = 20
    seed: int = 0


@dataclass
class Result:
    best_step: int = 0
    best_loss: float = float("inf")
    best_state: dict = field(default_factory=dict)  # CPU copy of the weights at best_step
    history: list[dict] = field(default_factory=list)  # {"step", "train_loss", "held_out_loss", "held_out_accuracy"}


def held_out_loss(llm: torch.nn.Module, examples: Sequence[Example], embedding: torch.Tensor,
                  device: torch.device, batch_size: int) -> tuple[float, float]:
    """(loss, token accuracy) averaged over the examples, with the model in evaluation mode."""
    was_training = llm.training
    llm.eval()
    loss = accuracy = 0.0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            chunk = examples[i:i + batch_size]
            out = llm(collate(chunk, embedding), device)
            loss += float(out["loss"]) * len(chunk)
            accuracy += float(out["acc"]) * len(chunk)
    llm.train(was_training)
    return loss / len(examples), accuracy / len(examples)


def fine_tune(llm: torch.nn.Module, train: Sequence[Example], held_out: Sequence[Example],
              embedding: torch.Tensor, device: torch.device, settings: Settings = Settings(),
              log: Callable[[dict], None] = lambda entry: None) -> Result:
    """Train ``llm`` in place and return the weights at the lowest held-out loss.

    With minutes of data the model starts to memorise within a few passes: training loss keeps
    falling while held-out loss turns back up. The step where it turns is the one worth keeping,
    so the held-out lines decide, not the last step.
    """
    if not train or not held_out:
        raise ValueError("fine-tuning needs both training and held-out examples")
    llm.to(device).train()
    optimizer = torch.optim.Adam(llm.parameters(), lr=settings.learning_rate)
    generator = torch.Generator().manual_seed(settings.seed)
    result = Result()

    def evaluate(step: int, train_loss: float | None) -> None:
        loss, accuracy = held_out_loss(llm, held_out, embedding, device, settings.batch_size)
        entry = {"step": step, "train_loss": train_loss, "held_out_loss": loss, "held_out_accuracy": accuracy}
        result.history.append(entry)
        log(entry)
        if loss < result.best_loss:
            result.best_loss, result.best_step = loss, step
            result.best_state = {k: v.detach().cpu().clone() for k, v in llm.state_dict().items()}

    evaluate(0, None)
    for step in range(1, settings.steps + 1):
        picks = torch.randperm(len(train), generator=generator)[:settings.batch_size].tolist()
        out = llm(collate([train[i] for i in picks], embedding), device)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(llm.parameters(), settings.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % settings.evaluate_every == 0 or step == settings.steps:
            evaluate(step, float(out["loss"].detach()))
    llm.eval()
    return result

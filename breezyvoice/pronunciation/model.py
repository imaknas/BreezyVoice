"""The sound-unit recogniser: a speech encoder with a CTC head over initials and tonal finals."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

# A HuBERT encoder pre-trained on Mandarin speech (MIT licence). Any encoder that maps 16 kHz audio
# to frames with a `config.hidden_size` works; it is created by the caller.
BASE_ENCODER = "TencentGameMate/chinese-hubert-base"
BLANK = "<blank>"
SAMPLE_RATE = 16000


class Recognizer(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, vocabulary: Sequence[str]):
        super().__init__()
        if vocabulary[0] != BLANK:
            raise ValueError("the first unit must be the CTC blank")
        self.encoder = encoder
        self.vocabulary = list(vocabulary)
        self.unit_index = {unit: i for i, unit in enumerate(self.vocabulary)}
        self.head = torch.nn.Linear(encoder.config.hidden_size, len(self.vocabulary))

    @staticmethod
    def normalise(audio: torch.Tensor) -> torch.Tensor:
        return (audio - audio.mean()) / (audio.std() + 1e-7)

    def forward(self, audio: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Padded, normalised 16 kHz audio (batch, samples) -> log-probabilities (batch, frames, units)
        and the number of valid frames per item."""
        mask = torch.arange(audio.size(1), device=audio.device)[None, :] < lengths.to(audio.device)[:, None]
        hidden = self.encoder(audio, attention_mask=mask).last_hidden_state
        return self.head(hidden).log_softmax(dim=-1), self.encoder._get_feat_extract_output_lengths(lengths)

    @torch.no_grad()
    def log_probs(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """One utterance (samples,) -> (frames, units), on the CPU."""
        device = next(self.parameters()).device
        audio = self.normalise(audio_16k.to(torch.float32)).unsqueeze(0).to(device)
        out, _ = self.forward(audio, torch.tensor([audio.size(1)]))
        return out[0].cpu()


def save(recognizer: Recognizer, path: Path) -> None:
    torch.save({"vocabulary": recognizer.vocabulary, "encoder_config": recognizer.encoder.config.to_dict(),
                "state": {k: v.detach().cpu() for k, v in recognizer.state_dict().items()}}, path)


def load(path: Path) -> Recognizer:
    """Rebuild a saved recogniser without touching the network."""
    from transformers import HubertConfig, HubertModel

    saved = torch.load(path, map_location="cpu", weights_only=True)
    recognizer = Recognizer(HubertModel(HubertConfig(**saved["encoder_config"])), saved["vocabulary"])
    recognizer.load_state_dict(saved["state"])
    return recognizer.eval()


def pretrained_encoder() -> torch.nn.Module:
    """The base encoder, from the local Hugging Face cache or the network."""
    from transformers import HubertModel

    encoder = HubertModel.from_pretrained(BASE_ENCODER)
    encoder.feature_extractor._freeze_parameters()  # the convolutional front end stays as pre-trained
    return encoder

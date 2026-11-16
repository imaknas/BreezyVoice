"""Features extracted from the reference (speaker prompt) recording."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile
import torch

from breezyvoice._vendor.torchaudio import kaldi
from breezyvoice._vendor.torchaudio.resample import resample
from breezyvoice._vendor.whisper.audio import log_mel_spectrogram
from cosyvoice.matcha.audio import mel_spectrogram

PROMPT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 22050


@dataclass(frozen=True)
class SpeakerAudio:
    """Everything the networks need from the reference audio (CPU tensors, batch size 1)."""

    speech_token: torch.Tensor  # (1, T) int32, 50 Hz speech tokens
    speech_feat: torch.Tensor  # (1, T', 80) mel frames at 22.05 kHz
    embedding: torch.Tensor  # (1, 192) CAM++ speaker embedding


def load_wav_16k(path: Path) -> torch.Tensor:
    """Read a wav (any rate, mono or stereo) as a (1, N) float32 tensor at 16 kHz."""
    data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    speech = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    if sr != PROMPT_SAMPLE_RATE:
        speech = resample(speech, orig_freq=sr, new_freq=PROMPT_SAMPLE_RATE)
    return speech


def _mel_22k(speech: torch.Tensor) -> torch.Tensor:
    return mel_spectrogram(
        speech, n_fft=1024, num_mels=80, sampling_rate=OUTPUT_SAMPLE_RATE,
        hop_size=256, win_size=1024, fmin=0, fmax=8000,
    )


class SpeechFrontend:
    def __init__(self, campplus: onnxruntime.InferenceSession, speech_tokenizer: onnxruntime.InferenceSession):
        self._campplus = campplus
        self._speech_tokenizer = speech_tokenizer

    def speaker_audio(self, speech_16k: torch.Tensor) -> SpeakerAudio:
        return SpeakerAudio(
            speech_token=self.speech_token(speech_16k),
            speech_feat=self.speech_feat(speech_16k),
            embedding=self.embedding(speech_16k),
        )

    def speech_token(self, speech_16k: torch.Tensor) -> torch.Tensor:
        """(1, T) int32 speech tokens at 50 Hz."""
        feat = log_mel_spectrogram(speech_16k, n_mels=128)
        inputs = self._speech_tokenizer.get_inputs()
        tokens = self._speech_tokenizer.run(None, {
            inputs[0].name: feat.numpy(),
            inputs[1].name: np.array([feat.shape[2]], dtype=np.int32),
        })[0].flatten()
        return torch.tensor([tokens.tolist()], dtype=torch.int32)

    def speech_feat(self, speech_16k: torch.Tensor) -> torch.Tensor:
        """(1, T', 80) mel frames at 22.05 kHz."""
        speech_22k = resample(speech_16k, orig_freq=PROMPT_SAMPLE_RATE, new_freq=OUTPUT_SAMPLE_RATE)
        return _mel_22k(speech_22k).squeeze(0).transpose(0, 1).unsqueeze(0)

    def embedding(self, speech_16k: torch.Tensor) -> torch.Tensor:
        """(1, 192) CAM++ speaker embedding."""
        feat = kaldi.fbank(speech_16k, num_mel_bins=80, dither=0, sample_frequency=PROMPT_SAMPLE_RATE)
        feat = feat - feat.mean(dim=0, keepdim=True)
        inp = self._campplus.get_inputs()[0].name
        emb = self._campplus.run(None, {inp: feat.unsqueeze(0).numpy()})[0].flatten()
        return torch.tensor([emb.tolist()])


def onnx_session(path: Path, threads: int) -> onnxruntime.InferenceSession:
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    return onnxruntime.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

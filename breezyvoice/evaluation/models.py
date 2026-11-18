"""The recognisers and the speaker encoder used for evaluation: files, one-time download, loading.

Two recognisers from different model families are used, because each has its own blind spots, and
neither is the Whisper family the synthesizer's callers typically gate on. The speaker encoder
(ERes2NetV2) differs in architecture from the CAM++ encoder that conditions the synthesizer, so
similarity is not the model grading its own input.
"""
from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from breezyvoice import assets
from breezyvoice._vendor.torchaudio.resample import resample

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
_MODEL_RATE = 16000


@dataclass(frozen=True)
class Download:
    url: str
    sha256: str
    # Paths inside a tar archive to keep, mapped to their local names; empty = the file is used as is.
    members: tuple[tuple[str, str], ...] = ()

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[1]


PARAFORMER = Download(
    f"{_RELEASES}/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2",
    "8c6724d0a86bd867217d353db1eaa11f2f143bca446a1f2752e8c551a6f2bde0",
    (("sherpa-onnx-paraformer-zh-2024-03-09/model.onnx", "paraformer.onnx"),
     ("sherpa-onnx-paraformer-zh-2024-03-09/tokens.txt", "paraformer.tokens.txt")),
)
SENSE_VOICE = Download(
    f"{_RELEASES}/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
    "7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e",
    (("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx", "sense_voice.onnx"),
     ("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt", "sense_voice.tokens.txt")),
)
SPEAKER = Download(
    f"{_RELEASES}/speaker-recongition-models/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
    "bf1a75b9930474cf3389ef415e6e5d38ca96fea4a3a00f7e301d080a58ee2239",
)
_SPEAKER_NAME = "speaker_eres2netv2.onnx"
_ALL = (PARAFORMER, SENSE_VOICE, SPEAKER)


def home() -> Path:
    return assets.home() / "evaluation"


def _local_names(download: Download) -> tuple[str, ...]:
    return tuple(local for _, local in download.members) or (_SPEAKER_NAME,)


def resolve() -> Path:
    """The directory holding every evaluation model; never touches the network."""
    missing = [n for d in _ALL for n in _local_names(d) if not (home() / n).exists()]
    if missing:
        raise assets.AssetsMissing(
            f"evaluation models missing in {home()}: {', '.join(missing)}; "
            "run `breezyvoice evaluate setup` once (it needs the network)."
        )
    return home()


def from_network(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:  # noqa: S310 (pinned https URLs)
        shutil.copyfileobj(response, out)


def from_directory(directory: Path) -> Callable[[str, Path], None]:
    """A source that takes archives already downloaded into ``directory`` (same file names)."""
    def copy(url: str, dest: Path) -> None:
        shutil.copyfile(directory / url.rsplit("/", 1)[1], dest)
    return copy


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(source: Callable[[str, Path], None] = from_network, log=print) -> Path:
    """Get every model that is not there yet, checking each download against its pinned checksum."""
    target = home()
    target.mkdir(parents=True, exist_ok=True)
    for download in _ALL:
        if all((target / n).exists() for n in _local_names(download)):
            continue
        log(f"fetching {download.filename}")
        with tempfile.TemporaryDirectory(dir=target) as tmp:
            archive = Path(tmp) / download.filename
            source(download.url, archive)
            digest = _sha256(archive)
            if digest != download.sha256:
                raise RuntimeError(f"{download.filename}: checksum mismatch ({digest})")
            if not download.members:
                shutil.move(archive, target / _SPEAKER_NAME)
                continue
            with tarfile.open(archive) as tar:
                for member, local in download.members:
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"{download.filename}: {member} is not a file")
                    with open(Path(tmp) / local, "wb") as out:
                        shutil.copyfileobj(extracted, out)
            for _, local in download.members:
                shutil.move(Path(tmp) / local, target / local)
    return resolve()


def _mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == _MODEL_RATE:
        return audio.astype(np.float32)
    wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    return resample(wav, orig_freq=sample_rate, new_freq=_MODEL_RATE).squeeze(0).numpy()


Transcriber = Callable[[np.ndarray, int], str]  # (mono float32 audio, sample rate) -> text
SpeakerEncoder = Callable[[np.ndarray, int], np.ndarray]  # -> unit-length embedding


def _transcriber(recognizer) -> Transcriber:
    def transcribe(audio: np.ndarray, sample_rate: int) -> str:
        stream = recognizer.create_stream()
        stream.accept_waveform(_MODEL_RATE, _mono_16k(audio, sample_rate))
        recognizer.decode_stream(stream)
        return stream.result.text
    return transcribe


def paraformer(directory: Path, threads: int) -> Transcriber:
    import sherpa_onnx

    return _transcriber(sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=str(directory / "paraformer.onnx"), tokens=str(directory / "paraformer.tokens.txt"),
        num_threads=threads))


def sense_voice(directory: Path, threads: int) -> Transcriber:
    import sherpa_onnx

    return _transcriber(sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(directory / "sense_voice.onnx"), tokens=str(directory / "sense_voice.tokens.txt"),
        num_threads=threads, language="zh", use_itn=False))


def speaker_encoder(directory: Path, threads: int) -> SpeakerEncoder:
    import sherpa_onnx

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(directory / _SPEAKER_NAME), num_threads=threads))

    def embed(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        stream = extractor.create_stream()
        stream.accept_waveform(_MODEL_RATE, _mono_16k(audio, sample_rate))
        stream.input_finished()
        vector = np.asarray(extractor.compute(stream), dtype=np.float32)
        return vector / np.linalg.norm(vector)
    return embed


def to_simplified() -> Callable[[str], str]:
    import opencc

    return opencc.OpenCC("t2s").convert


_DIGITS = str.maketrans("0123456789", "零一二三四五六七八九")

# (expected, heard) syllables that are both right in Taiwan: the standard reading there differs from
# the one the syllable converter gives, or the character has two readings in free variation.
# Syllables are written as ``taiwan_syllables`` returns them.
TAIWAN_READINGS = frozenset({
    ("he", "han"),   # 和
    ("ye", "yi"),    # 液
    ("la", "le"), ("ji", "se"),  # 垃圾
    ("zu", "ze"), ("zu", "zao"), ("zu", "zuo"),  # 著 / 着
    ("na", "nei"), ("ze", "zei"), ("sui", "sei"),  # 那, 這, 誰
    ("su", "sou"),   # 熟
    ("xue", "xie"),  # 血
    ("bao", "pu"),   # 暴露
    ("jue", "jiao"),  # 角色
    ("wo", "gua"), ("kuo", "gua"),  # 蝸, 括
    ("xiao", "yao"),  # 混淆
    ("di", "ti"),    # 堤
})


def _taiwan_merger(syllable: str) -> str:
    """Drop the distinctions most speakers in Taiwan do not make (and recognisers therefore hear
    either way): retroflex against dental initials, and -ing/-eng against -in/-en."""
    import re

    syllable = re.sub(r"^(zh|ch|sh)", lambda m: m.group(1)[0], syllable)
    return re.sub(r"eng$", "en", re.sub(r"ing$", "in", syllable))


def taiwan_syllables() -> Callable[[str], list[str]]:
    """Text -> one toneless syllable per Chinese character, plus lower-cased Latin words.

    Both the expected text and what a recogniser returned must go through this. The text is first
    converted to Simplified script, because the converter picks the reading of a character with
    several readings from Simplified phrases: fed Traditional text it reads 著, 乾, 樂, 沒, 調 or 傳
    differently from the same word in a recogniser's Simplified output, and the two sides then
    disagree about speech that was right. Digits are read one by one; punctuation is dropped.
    """
    import re
    import unicodedata

    import opencc
    from pypinyin import Style, lazy_pinyin

    simplify = opencc.OpenCC("t2s").convert

    def latin_words(chunk: str) -> list[str]:
        return re.findall(r"[a-z]+", chunk.lower())

    def syllables(text: str) -> list[str]:
        text = simplify(unicodedata.normalize("NFKC", text)).translate(_DIGITS)
        return [_taiwan_merger(s) for s in lazy_pinyin(text, style=Style.NORMAL, errors=latin_words)]
    return syllables

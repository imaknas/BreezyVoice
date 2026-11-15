"""Model files: which ones, at which pinned revision, where they live, and the one-time download.

Loading (``resolve``) never touches the network: it only looks files up in the local Hugging Face
cache and in ``BREEZYVOICE_HOME``. Downloading happens only in ``fetch`` (``breezyvoice setup``).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HubFiles:
    repo_id: str
    revision: str  # a commit hash, so the files can never change underneath us
    filenames: tuple[str, ...]


WEIGHTS = HubFiles(
    "MediaTek-Research/BreezyVoice-300M",
    "e33b502e0ac21c16b0ee0d00df66ac3fa737393d",
    ("llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v1.onnx"),
)
# g2pW's tokenizer (vocabulary only).
BERT_TOKENIZER = HubFiles(
    "google-bert/bert-base-chinese",
    "8f23c25b06e129b6c986331a13d8d025a92cf0ea",
    ("vocab.txt", "tokenizer_config.json"),
)
G2PW_ARCHIVE = HubFiles(
    "YC-Chen/g2pW",
    "fb4784f7dce8466a53dd4f3552f5a4c5b657b90c",
    ("G2PWModel-v2-onnx.zip",),
)
G2PW_SHA256 = "699f3c1fd7fb0e2c2d49ed2486826fd5bff233fee7759350a91c3b49aedc4ed2"
G2PW_DIRNAME = "G2PWModel-v2-onnx"


class AssetsMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class Assets:
    weights_dir: Path
    bert_dir: Path
    g2pw_dir: Path


def home() -> Path:
    """Where extracted assets (g2pW) are kept: $BREEZYVOICE_HOME, else ~/.cache/breezyvoice."""
    env = os.environ.get("BREEZYVOICE_HOME")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "breezyvoice"


def _cached_dir(files: HubFiles) -> Path:
    from huggingface_hub import try_to_load_from_cache

    dirs = set()
    for name in files.filenames:
        found = try_to_load_from_cache(files.repo_id, name, revision=files.revision)
        if not isinstance(found, str):
            raise AssetsMissing(
                f"{files.repo_id}@{files.revision[:8]}: {name} is not in the local Hugging Face cache; "
                "run `breezyvoice setup` once (it needs the network)."
            )
        dirs.add(Path(found).parent)
    (d,) = dirs
    return d


def resolve(weights_dir: Path | None = None) -> Assets:
    """Locate every asset locally. ``weights_dir`` overrides the cached BreezyVoice weights."""
    g2pw_dir = home() / G2PW_DIRNAME
    if not (g2pw_dir / "version").exists():
        raise AssetsMissing(f"g2pW model not found in {g2pw_dir}; run `breezyvoice setup` once.")
    if weights_dir is not None:
        missing = [n for n in WEIGHTS.filenames if not (weights_dir / n).exists()]
        if missing:
            raise AssetsMissing(f"{weights_dir} lacks {', '.join(missing)}")
    return Assets(
        weights_dir=weights_dir if weights_dir is not None else _cached_dir(WEIGHTS),
        bert_dir=_cached_dir(BERT_TOKENIZER),
        g2pw_dir=g2pw_dir,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(log=print) -> Assets:
    """Download everything at the pinned revisions (skips files already cached) and unpack g2pW."""
    from huggingface_hub import hf_hub_download

    for files in (WEIGHTS, BERT_TOKENIZER, G2PW_ARCHIVE):
        for name in files.filenames:
            log(f"fetching {files.repo_id}/{name}")
            hf_hub_download(files.repo_id, name, revision=files.revision)

    target = home() / G2PW_DIRNAME
    if not (target / "version").exists():
        archive = Path(hf_hub_download(G2PW_ARCHIVE.repo_id, G2PW_ARCHIVE.filenames[0], revision=G2PW_ARCHIVE.revision))
        digest = _sha256(archive)
        if digest != G2PW_SHA256:
            raise RuntimeError(f"g2pW archive checksum mismatch: {digest}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    dest = (Path(tmp) / member).resolve()
                    if not dest.is_relative_to(Path(tmp).resolve()):
                        raise RuntimeError(f"unsafe path in g2pW archive: {member}")
                zf.extractall(tmp)
            (top,) = {Path(m).parts[0] for m in zf.namelist() if not m.startswith("__MACOSX")}
            shutil.move(str(Path(tmp) / top), str(target))
        log(f"g2pW unpacked to {target}")
    return resolve()

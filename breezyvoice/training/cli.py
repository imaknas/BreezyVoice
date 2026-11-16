"""``breezyvoice train``: clips -> features -> a weights directory with the adapted language model."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

LM_FILE = "llm.pt"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def add_parser(sub, model_args) -> None:
    p = sub.add_parser("train", help="adapt the language model to one speaker's recordings")
    actions = p.add_subparsers(dest="action", required=True)

    f = actions.add_parser("prepare", help="turn clips cut by `recording split` into training features")
    f.add_argument("--clips", required=True, type=Path, nargs="+", help="directories written by `recording split`")
    f.add_argument("--script", required=True, type=Path, nargs="+", help="the reading scripts the clips follow")
    f.add_argument("--out", required=True, type=Path, help="features file (.pt); keep it outside the repository")
    f.add_argument("--max-error", type=float, default=0.1,
                   help="skip takes whose recognised text differs from the line by more than this share")
    model_args(f)

    t = actions.add_parser("lm", help="fine-tune the language model on prepared features")
    t.add_argument("--features", required=True, type=Path, nargs="+")
    t.add_argument("--out", required=True, type=Path,
                   help="weights directory to create; use it with --weights-dir. Keep it outside the repository")
    t.add_argument("--train-device", choices=("auto", "cpu", "mps"), default="auto")
    t.add_argument("--steps", type=int, default=400)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--learning-rate", type=float, default=1e-5)
    t.add_argument("--evaluate-every", type=int, default=20)
    t.add_argument("--seed", type=int, default=0)
    model_args(t)


def main(args) -> int:
    return {"prepare": _prepare, "lm": _lm}[args.action](args)


def _prepare(args) -> int:
    import torch

    from breezyvoice.loader import load
    from breezyvoice.recording.script import load_script
    from breezyvoice.speech_frontend import load_wav_16k
    from breezyvoice.training.data import extract

    script: dict[str, str] = {}
    for path in args.script:
        script.update(load_script(path))
    engine = load(device="cpu", weights_dir=args.weights_dir, threads=args.threads)
    front = engine.speech_frontend
    examples, skipped = [], 0
    for directory in args.clips:
        for report in sorted(directory.glob("*.json")):
            for take in json.loads(report.read_text(encoding="utf-8"))["takes"]:
                if take["error_rate"] > args.max_error or take["line"] not in script:
                    skipped += 1
                    continue
                speech = load_wav_16k(directory / take["wav"])
                examples.append(extract(take["line"], script[take["line"]], speech,
                                        engine.text_tokens, front.speech_token, front.embedding))
    if not examples:
        _log("no usable takes found")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save([dataclasses.asdict(e) for e in examples], args.out)
    os.chmod(args.out, 0o600)
    minutes = sum(e.seconds for e in examples) / 60
    print(json.dumps({"examples": len(examples), "skipped": skipped, "minutes": round(minutes, 1),
                      "lines": len({e.line for e in examples}), "out": str(args.out)}, ensure_ascii=False))
    return 0


def _lm(args) -> int:
    import torch

    from breezyvoice import assets
    from breezyvoice.model import build_llm
    from breezyvoice.training.data import Example, speaker_embedding, split
    from breezyvoice.training.finetune import Settings, fine_tune

    examples = [Example(**row) for path in args.features for row in torch.load(path)]
    train, held_out = split(examples)
    source = assets.resolve(args.weights_dir).weights_dir
    llm = build_llm()
    llm.load_state_dict(torch.load(source / LM_FILE, map_location="cpu", weights_only=True))
    name = args.train_device
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    settings = Settings(steps=args.steps, batch_size=args.batch_size, learning_rate=args.learning_rate,
                        evaluate_every=args.evaluate_every, seed=args.seed)
    _log(f"{len(train)} training and {len(held_out)} held-out examples, "
         f"{sum(e.seconds for e in examples) / 60:.1f} minutes, on {name}")
    result = fine_tune(llm, train, held_out, speaker_embedding(examples), torch.device(name), settings,
                       log=lambda e: _log(f"step {e['step']:5d}  held-out loss {e['held_out_loss']:.3f}  "
                                          f"accuracy {e['held_out_accuracy']:.3f}"))

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(result.best_state, args.out / LM_FILE)
    for other in assets.WEIGHTS.filenames:
        link = args.out / other
        if other != LM_FILE and not link.exists():
            link.symlink_to((source / other).resolve())  # unchanged networks are shared, not copied
    summary = {"best_step": result.best_step, "best_held_out_loss": result.best_loss,
               "settings": dataclasses.asdict(settings), "examples": len(examples),
               "history": result.history}
    (args.out / "training.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("best_step", "best_held_out_loss")} | {"out": str(args.out)}))
    return 0

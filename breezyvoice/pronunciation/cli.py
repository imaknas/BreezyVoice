"""``breezyvoice pronunciation``: train the sound-unit recogniser, check takes with it."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path


MIN_SILENCE = 0.10  # shortest silence considered when looking for pauses


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def add_parser(sub) -> None:
    p = sub.add_parser("pronunciation", help="check that takes were pronounced as intended (tones, readings)")
    actions = p.add_subparsers(dest="action", required=True)

    t = actions.add_parser("train", help="train the sound-unit recogniser on recordings with known text")
    t.add_argument("--manifest", required=True, type=Path, nargs="+",
                   help='JSON lines of {"wav": path, "text": what is said}; relative paths are relative to the file')
    t.add_argument("--out", required=True, type=Path, help="where to write the recogniser (.pt)")
    t.add_argument("--train-device", choices=("auto", "cpu", "mps"), default="auto")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--evaluate-every", type=int, default=250)
    t.add_argument("--seed", type=int, default=0)

    c = actions.add_parser("check", help="score takes; prints one JSON line per take, most doubtful syllable first")
    c.add_argument("--model", required=True, type=Path)
    c.add_argument("--manifest", required=True, type=Path, help="JSON lines of {wav, text}")


def main(args) -> int:
    return {"train": _train, "check": _check}[args.action](args)


def _entries(manifests) -> list[dict]:
    out = []
    for path in manifests:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                entry["wav"] = str((path.parent / entry["wav"]).resolve())
                out.append(entry)
    return out


def _reader():
    from breezyvoice.loader import reading_tools
    from breezyvoice.pronunciation.readings import load_polyphones, read

    normalise, predict, table = reading_tools()
    polyphones = load_polyphones(table)
    return lambda text: read(text, normalise, predict, polyphones)


def _load_audio(path: str):
    from breezyvoice.speech_frontend import load_wav_16k

    return load_wav_16k(Path(path)).squeeze(0)


def _device(name: str):
    import torch

    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def _train(args) -> int:
    from breezyvoice.pronunciation import model
    from breezyvoice.pronunciation.readings import Unsupported
    from breezyvoice.pronunciation.train import Settings, Utterance, split, train, vocabulary

    reader = _reader()
    utterances, skipped = [], 0
    for entry in _entries(args.manifest):
        try:
            reading = reader(entry["text"])
        except Unsupported:
            skipped += 1
            continue
        utterances.append(Utterance(key=entry["text"], wav=entry["wav"], readings=reading.readings))
    training, held_out = split(utterances, seed=args.seed)
    recognizer = model.Recognizer(model.pretrained_encoder(), vocabulary(utterances))
    settings = Settings(steps=args.steps, batch_size=args.batch_size, evaluate_every=args.evaluate_every, seed=args.seed)
    _log(f"{len(training)} training and {len(held_out)} held-out utterances ({skipped} skipped), "
         f"{len(recognizer.vocabulary)} units")
    result = train(recognizer, training, held_out, _load_audio, _device(args.train_device), settings,
                   log=lambda e: _log(f"step {e['step']:6d}  loss {e['loss']:.3f}  held-out unit error "
                                      f"{e['unit_error']:.3f}  tone error {e['tone_error']:.3f}"))
    recognizer.load_state_dict(result.best_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(recognizer, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(
        {"best_step": result.best_step, "best_unit_error": result.best_error, "utterances": len(utterances),
         "settings": dataclasses.asdict(settings), "history": result.history}, indent=1), encoding="utf-8")
    print(json.dumps({"best_step": result.best_step, "best_unit_error": result.best_error, "out": str(args.out)}))
    return 0


def _check(args) -> int:
    from breezyvoice.pronunciation import model
    from breezyvoice.audio import pauses as silences
    from breezyvoice.pronunciation.align import stray_units, syllable_times
    from breezyvoice.pronunciation.check import score
    from breezyvoice.pronunciation.model import SAMPLE_RATE
    from breezyvoice.pronunciation.pauses import place, unnatural
    from breezyvoice.pronunciation.readings import Unsupported

    recognizer = model.load(args.model)
    reader = _reader()
    for entry in _entries([args.manifest]):
        try:
            reading = reader(entry["text"])
            missing = [r for r in reading.readings if any(u not in recognizer.unit_index for u in _units(r))]
            if missing:
                raise Unsupported(f"no unit for {missing[0]}")
        except Unsupported as e:
            print(json.dumps({"wav": entry["wav"], "text": entry["text"], "unsupported": str(e)}, ensure_ascii=False))
            continue
        audio_16k = _load_audio(entry["wav"])
        log_probs = recognizer.log_probs(audio_16k)
        syllables = score(log_probs, reading.readings, reading.others, recognizer.unit_index)
        ranked = sorted(syllables, key=lambda s: s.margin)
        times = syllable_times(log_probs, reading.readings, recognizer.unit_index)
        before, after = stray_units(log_probs, times, recognizer.vocabulary)
        placed = place(silences(audio_16k.numpy(), SAMPLE_RATE, MIN_SILENCE), times, reading.positions, reading.spoken)
        odd = unnatural(placed)
        print(json.dumps({
            "wav": entry["wav"], "text": entry["text"], "lowest_margin": round(ranked[0].margin, 2),
            "doubtful": [{"character": reading.characters[s.index], "position": s.index, "intended": s.reading,
                          "rival": s.rival, "margin": round(s.margin, 2)} for s in ranked[:3]],
            "extra_before": before, "extra_after": after,
            "unnatural_pauses": [{"after": reading.characters[p.after], "position": p.after, "seconds": round(p.length, 2),
                                  "at_punctuation": p.at_punctuation} for p in odd],
        }, ensure_ascii=False), flush=True)
    return 0


def _units(reading: str) -> list[str]:
    from breezyvoice.pronunciation.units import units

    return units(reading)

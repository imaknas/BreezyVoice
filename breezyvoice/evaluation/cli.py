"""``breezyvoice evaluate``: setup, run, compare. Wires files, models and the synthesizer together."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

SENTENCES = Path(__file__).with_name("sentences.txt")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def add_parser(sub, model_args) -> None:
    p = sub.add_parser("evaluate", help="measure intelligibility, speaker similarity and speed on fixed sentences")
    actions = p.add_subparsers(dest="action", required=True)

    s = actions.add_parser("setup", help="download the recognisers and the speaker encoder (pinned checksums)")
    s.add_argument("--archives", type=Path, default=None,
                   help="take the archives from this directory instead of the network")

    r = actions.add_parser("run", help="synthesize every sentence with every seed and measure the takes")
    r.add_argument("--ref-wav", required=True, type=Path)
    r.add_argument("--ref-text", required=True, help="transcript of --ref-wav")
    r.add_argument("--out", required=True, type=Path,
                   help="directory for result.json and the takes; keep it outside the repository")
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    r.add_argument("--sentences", type=Path, default=SENTENCES, help="one sentence per line")
    r.add_argument("--label", default="", help="what this run stands for, e.g. a commit or a setting")
    model_args(r)

    c = actions.add_parser("compare", help="pair two runs by sentence and seed and report the differences")
    c.add_argument("base", type=Path, help="directory of the run to compare against")
    c.add_argument("candidate", type=Path)


def main(args) -> int:
    return {"setup": _setup, "run": _run, "compare": _compare}[args.action](args)


def _setup(args) -> int:
    from breezyvoice.evaluation import models

    source = models.from_directory(args.archives) if args.archives else models.from_network
    _log(f"ready: {models.fetch(source=source, log=_log)}")
    return 0


def _commit() -> str | None:
    """The checkout this code runs from, when it is one."""
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _run(args) -> int:
    import soundfile
    import torch

    from breezyvoice.evaluation import models
    from breezyvoice.evaluation.run import evaluate, load_sentences, measure_reference, summarize
    from breezyvoice.loader import load

    directory = models.resolve()
    threads = args.threads or max(1, (os.cpu_count() or 2) // 2)
    transcribers = {"paraformer": models.paraformer(directory, threads),
                    "sense_voice": models.sense_voice(directory, threads)}
    embed = models.speaker_encoder(directory, threads)
    unify = models.to_simplified()

    reference, reference_rate = soundfile.read(str(args.ref_wav), dtype="float32", always_2d=True)
    reference = reference.mean(axis=1)
    engine = load(device=args.device, weights_dir=args.weights_dir, threads=args.threads, sampler=args.sampler)
    speaker = engine.speaker(args.ref_wav, args.ref_text)

    def synthesize(text: str, seed: int):
        result = engine.synthesize(text, speaker, seed)
        return result.audio, result.sample_rate

    args.out.mkdir(parents=True, exist_ok=True)

    def save_wav(name: str, audio, sample_rate: int) -> None:
        soundfile.write(str(args.out / name), audio, sample_rate, subtype="FLOAT")

    synthesize("暖機。", 0)  # first call pays one-time graph compilation; keep it out of the timings
    load_before = os.getloadavg()[0]
    records = evaluate(synthesize, load_sentences(args.sentences), args.seeds, transcribers, embed,
                       embed(reference, reference_rate), unify, models.toneless_pinyin(), save_wav, log=_log)
    result = {
        "meta": {
            "label": args.label, "commit": _commit(), "device": args.device, "sampler": args.sampler,
            "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "torch": torch.__version__, "seeds": args.seeds, "sentences": str(args.sentences),
            "reference_seconds": len(reference) / reference_rate,
            # Timings are only comparable between runs made on an equally idle machine.
            "load_average": [load_before, os.getloadavg()[0]],
        },
        "reference": measure_reference(reference, reference_rate, args.ref_text, transcribers, embed, unify),
        "summary": summarize(records),
        "records": records,
    }
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"meta": result["meta"], "reference": result["reference"], "summary": result["summary"]},
                     ensure_ascii=False, indent=1))
    return 0


def _compare(args) -> int:
    import soundfile

    from breezyvoice.evaluation.compare import compare
    from breezyvoice.evaluation.signal import mel_distance

    base = json.loads((args.base / "result.json").read_text(encoding="utf-8"))
    candidate = json.loads((args.candidate / "result.json").read_text(encoding="utf-8"))

    def load_pair(b: dict, c: dict):
        wav_b, rate = soundfile.read(str(args.base / b["wav"]), dtype="float32")
        wav_c, rate_c = soundfile.read(str(args.candidate / c["wav"]), dtype="float32")
        if rate != rate_c:
            raise ValueError(f"sample rates differ: {rate} vs {rate_c}")
        return wav_b, wav_c, rate

    report = compare(base["records"], candidate["records"], load_pair, mel_distance)
    rows = report.pop("rows")
    worst = sorted(rows, key=lambda r: max(r["cer_delta"].values()), reverse=True)[:5]
    print(json.dumps({
        "base": {"meta": base["meta"], "summary": base["summary"]},
        "candidate": {"meta": candidate["meta"], "summary": candidate["summary"]},
        "paired": report,
        "largest_cer_increases": worst,
    }, ensure_ascii=False, indent=1))
    return 0

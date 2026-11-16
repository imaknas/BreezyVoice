"""``breezyvoice recording``: turn a recorded script section into one clip per line."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Silence that separates two pieces. Shorter than the gap readers leave between sentences, longer
# than the closure of a stop consonant; pauses inside a sentence are rejoined by the alignment.
DEFAULT_MIN_GAP = 0.35
# Kept on each side of a take so its first and last sounds are not clipped.
PADDING = 0.15


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def add_parser(sub) -> None:
    p = sub.add_parser("recording", help="cut a recorded reading into one clip per script line")
    actions = p.add_subparsers(dest="action", required=True)
    s = actions.add_parser("split", help="cut one recorded section and match the pieces to its script lines")
    s.add_argument("--script", required=True, type=Path, help="reading script: one '<ID> <text>' per line")
    s.add_argument("--section", required=True, help="id prefix of the lines in this recording, e.g. B")
    s.add_argument("--wav", required=True, type=Path, nargs="+",
                   help="the recording (WAV or FLAC); several files are treated as one after another")
    s.add_argument("--out", required=True, type=Path,
                   help="directory for the clips and <section>.json; keep it outside the repository")
    s.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP, help="silence in seconds that separates pieces")
    s.add_argument("--threads", type=int, default=None)


def main(args) -> int:
    import numpy as np
    import soundfile

    from breezyvoice.audio import speech_pieces
    from breezyvoice.evaluation import models
    from breezyvoice.recording.align import assign, preferred
    from breezyvoice.recording.script import load_script, section

    lines = section(load_script(args.script), args.section)
    if not lines:
        _log(f"no lines with id prefix {args.section!r} in {args.script}")
        return 2
    directory = models.resolve()
    transcribe = models.sense_voice(directory, args.threads or max(1, (os.cpu_count() or 2) // 2))
    unify = models.to_simplified()

    pieces = []  # (file, start, end, audio, sample rate)
    for path in args.wav:
        data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        audio = data.mean(axis=1)
        for start, end in speech_pieces(audio, rate, args.min_gap):
            pieces.append((path, start, end, audio, rate))
    heard = [transcribe(audio[int(a * rate):int(b * rate)], rate) for _, a, b, audio, rate in pieces]
    takes = assign(heard, lines, unify)

    args.out.mkdir(parents=True, exist_ok=True)
    chosen = preferred(takes)
    counts: dict[str, int] = {}
    report = []
    for take in takes:
        path, start, _, audio, rate = pieces[take.first]
        end = pieces[take.last][2]
        if pieces[take.last][0] != path:
            continue  # a take cannot run from one file into the next
        counts[take.line] = counts.get(take.line, 0) + 1
        name = f"{take.line}_{counts[take.line]}.wav"
        a = max(0, int((start - PADDING) * rate))
        b = min(len(audio), int((end + PADDING) * rate))
        soundfile.write(str(args.out / name), audio[a:b], rate, subtype="PCM_24")
        report.append({"line": take.line, "take": counts[take.line], "wav": name, "source": path.name,
                       "start": round(start, 2), "end": round(end, 2), "error_rate": round(take.error_rate, 3),
                       "preferred": chosen[take.line] is take,
                       "heard": "".join(heard[take.first:take.last + 1])})
    used = {i for t in takes for i in range(t.first, t.last + 1)}
    leftover = [{"source": pieces[i][0].name, "start": round(pieces[i][1], 2), "end": round(pieces[i][2], 2),
                 "heard": heard[i]} for i in range(len(pieces)) if i not in used]
    missing = [k for k in lines if k not in counts]
    summary = {"section": args.section, "lines": len(lines), "takes": len(report), "missing": missing,
               "unmatched_pieces": len(leftover)}
    (args.out / f"{args.section}.json").write_text(
        json.dumps({"summary": summary, "takes": report, "unmatched": leftover}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0

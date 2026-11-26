"""``breezyvoice`` command line: setup, synth, batch, serve, evaluate, recording, train, pronunciation."""
from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _engine(args):
    from breezyvoice.loader import load

    return load(device=args.device, weights_dir=args.weights_dir, threads=args.threads, sampler=args.sampler,
                as_ended=args.as_ended)


def _reviewer(args):
    if args.checker is None:
        return None
    from breezyvoice.loader import reviewer

    return reviewer(args.checker, weights_dir=args.weights_dir)


def _recorder(args):
    """Appends one JSON line per take to ``--decoding-log``; safe to call from several threads."""
    if args.decoding_log is None:
        return None
    import threading
    from datetime import datetime, timezone

    path, lock = args.decoding_log, threading.Lock()
    path.parent.mkdir(parents=True, exist_ok=True)

    def record(take: dict) -> None:
        line = json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "sampler": args.sampler, **take}, ensure_ascii=False)
        with lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def reporting(take: dict) -> None:
        try:
            record(take)
        except Exception as e:  # noqa: BLE001 - said here because callers rarely read ``record_error``
            _log(f"could not record a take in {path}: {type(e).__name__}: {e}")
            raise

    return reporting


def cmd_setup(args) -> int:
    from breezyvoice import assets

    found = assets.fetch(log=_log)
    _log(f"ready: weights={found.weights_dir} g2pw={found.g2pw_dir}")
    return 0


def cmd_synth(args) -> int:
    from breezyvoice.service import handle

    engine = _engine(args)
    result = handle(engine, {"text": args.text, "ref_wav": str(args.ref_wav), "ref_text": args.ref_text,
                             "seed": args.seed, "out": str(args.out)}, _reviewer(args), _recorder(args))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["ok"] else 1


def cmd_batch(args) -> int:
    """One JSON request per stdin line, one JSON result per stdout line; the model loads once."""
    from breezyvoice.service import handle

    engine = _engine(args)
    review, record = _reviewer(args), _recorder(args)
    print(json.dumps({"ok": True, "ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as e:
            result = {"ok": False, "error": f"invalid JSON: {e}", "kind": "bad_request"}
        else:
            result = handle(engine, payload, review, record)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


def cmd_serve(args) -> int:
    from breezyvoice.service import handle

    if args.host not in ("127.0.0.1", "::1", "localhost") and not args.allow_remote:
        _log("refusing to listen on a non-loopback address without --allow-remote "
             "(requests name local file paths)")
        return 2
    engine = _engine(args)
    review, record = _reviewer(args), _recorder(args)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(HTTPStatus.OK, {"ok": True, "sample_rate": engine.sample_rate})
            else:
                self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/synthesize":
                self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"null")
            except json.JSONDecodeError as e:
                self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid JSON: {e}", "kind": "bad_request"})
                return
            result = handle(engine, payload, review, record)
            status = HTTPStatus.OK if result["ok"] else (
                HTTPStatus.BAD_REQUEST if result["kind"] == "bad_request" else HTTPStatus.INTERNAL_SERVER_ERROR)
            self._send(status, result)

        def log_message(self, fmt, *a):
            _log(f"{self.address_string()} {fmt % a}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    _log(f"listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto",
                   help="auto = LLM on CPU with flow decoder and vocoder on MPS when available, else all CPU")
    p.add_argument("--weights-dir", type=Path, default=None,
                   help="use BreezyVoice weights from this directory instead of the Hugging Face cache")
    p.add_argument("--threads", type=int, default=None, help="CPU threads for torch / onnxruntime")
    p.add_argument("--checker", type=Path, default=None,
                   help="sound-unit recogniser from `pronunciation train`: review each take and tidy its pauses")
    p.add_argument("--sampler", choices=("repetition-aware", "top-k"), default="repetition-aware",
                   help="how the next speech token is drawn; top-k reproduces takes made before the default changed")
    p.add_argument("--as-ended", action="store_true",
                   help="leave a sentence where the model ended it even if its last sound is still going (as upstream)")


def _take_args(p: argparse.ArgumentParser) -> None:
    _model_args(p)
    p.add_argument("--decoding-log", type=Path, default=None,
                   help="append to this file, one JSON line per take, how undecided the model was at each speech "
                        "token and (with --checker) when each character is said")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="breezyvoice", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="download models (pinned revisions) and unpack g2pW; the only networked step")

    p = sub.add_parser("synth", help="synthesize one text (loads the model each time; use batch/serve for many)")
    p.add_argument("--text", required=True)
    p.add_argument("--ref-wav", required=True, type=Path)
    p.add_argument("--ref-text", required=True, help="transcript of --ref-wav")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    _take_args(p)

    p = sub.add_parser("batch", help="JSON lines on stdin → JSON lines on stdout, model loaded once")
    _take_args(p)

    p = sub.add_parser("serve", help="localhost HTTP service, model loaded once")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--allow-remote", action="store_true")
    _take_args(p)

    from breezyvoice.evaluation import cli as evaluation

    evaluation.add_parser(sub, _model_args)
    from breezyvoice.recording import cli as recording

    recording.add_parser(sub)
    from breezyvoice.training import cli as training

    training.add_parser(sub, _model_args)
    from breezyvoice.pronunciation import cli as pronunciation

    pronunciation.add_parser(sub)

    args = parser.parse_args(argv)
    commands = {"setup": cmd_setup, "synth": cmd_synth, "batch": cmd_batch, "serve": cmd_serve,
                "evaluate": evaluation.main, "recording": recording.main,
                "train": training.main, "pronunciation": pronunciation.main}
    return commands[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

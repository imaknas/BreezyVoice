"""Reading scripts: one line per sentence, ``<ID> <text>``; ``#`` starts a comment."""
from __future__ import annotations

import re
from pathlib import Path

_LINE = re.compile(r"^([A-Z]+\d+)\s+(.+)$")


def load_script(path: Path) -> dict[str, str]:
    """Line id -> text, in file order."""
    lines: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = _LINE.match(raw.strip())
        if not match:
            continue
        if match.group(1) in lines:
            raise ValueError(f"{path}: line id {match.group(1)} appears twice")
        lines[match.group(1)] = match.group(2)
    return lines


def section(script: dict[str, str], prefix: str) -> dict[str, str]:
    """The lines whose id is ``prefix`` followed by digits (``B`` -> B01, B02, ...)."""
    return {k: v for k, v in script.items() if re.fullmatch(re.escape(prefix) + r"\d+", k)}

"""Whether a take stops while its last sound is still going."""
from __future__ import annotations

import numpy as np

WINDOW_S = 0.02
LEVEL_DB = -25.0


def ends_loud(audio: np.ndarray, sample_rate: int, window_s: float = WINDOW_S, level_db: float = LEVEL_DB) -> bool:
    """True when the last ``window_s`` of ``audio`` are within ``level_db`` of its peak.

    Speech that has finished decays into near silence; a take whose final stretch is still this
    loud was stopped during its last syllable and sounds cut off or abrupt. The level alone, with
    no alignment, agreed with a listener's verdict on 41 of 48 endings (5 of the 24 bad ones are
    quieter than this and are missed; lowering the level catches them and flags as many good ones).
    """
    if len(audio) == 0:
        return False
    peak = float(np.abs(audio).max())
    if peak <= 0:
        return False
    tail = audio[-max(1, int(window_s * sample_rate)):]
    level = 20 * np.log10(max(float(np.sqrt(np.mean(tail.astype(np.float64) ** 2))), 1e-9) / peak)
    return level > level_db

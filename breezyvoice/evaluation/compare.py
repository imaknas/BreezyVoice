"""Compare two evaluation runs take by take.

Takes are paired by (sentence, seed). A change to the acoustic stages leaves the speech tokens of a
seed untouched, so paired differences isolate its effect from the take-to-take variation that
dominates unpaired averages.
"""
from __future__ import annotations

import statistics
from typing import Callable, Sequence

import numpy as np


def compare(base: Sequence[dict], candidate: Sequence[dict],
            load_pair: Callable[[dict, dict], tuple[np.ndarray, np.ndarray, int]],
            mel_distance: Callable[[np.ndarray, np.ndarray, int], float | None]) -> dict:
    """``load_pair(base_record, candidate_record)`` returns both waveforms and their sample rate."""
    by_key = {(r["sentence"], r["seed"]): r for r in base}
    pairs = [(by_key[k], r) for r in candidate if (k := (r["sentence"], r["seed"])) in by_key]
    if not pairs:
        raise ValueError("the two runs share no (sentence, seed) pair")
    asrs = sorted(set(pairs[0][0]["cer"]) & set(pairs[0][1]["cer"]))
    rows = []
    for b, c in pairs:
        wav_b, wav_c, rate = load_pair(b, c)
        rows.append({
            "sentence": c["sentence"], "seed": c["seed"],
            "same_length": wav_b.shape == wav_c.shape,
            "mel_distance": mel_distance(wav_b, wav_c, rate),
            "cer_delta": {a: c["cer"][a] - b["cer"][a] for a in asrs},
            "similarity_delta": c["similarity"] - b["similarity"],
        })
    distances = [r["mel_distance"] for r in rows if r["mel_distance"] is not None]
    # Takes both runs could check: which of them stopped needing another take, and which started to.
    both = [(b["review"], c["review"]) for b, c in pairs
            if "needs_retake" in b.get("review", ()) and "needs_retake" in c.get("review", ())]
    return {
        "reviewed_retakes": {"pairs": len(both),
                    "base": sum(b["needs_retake"] for b, _ in both),
                    "candidate": sum(c["needs_retake"] for _, c in both),
                    "no_longer": sum(b["needs_retake"] and not c["needs_retake"] for b, c in both),
                    "newly": sum(c["needs_retake"] and not b["needs_retake"] for b, c in both)} if both else None,
        "pairs": len(rows),
        "unpaired": len(candidate) - len(rows),
        "same_length": sum(r["same_length"] for r in rows),
        "mel_distance_mean": statistics.fmean(distances) if distances else None,
        "mel_distance_max": max(distances) if distances else None,
        "cer_delta_mean": {a: statistics.fmean(r["cer_delta"][a] for r in rows) for a in asrs},
        "cer_worse": {a: sum(r["cer_delta"][a] > 0 for r in rows) for a in asrs},
        "cer_better": {a: sum(r["cer_delta"][a] < 0 for r in rows) for a in asrs},
        "similarity_delta_mean": statistics.fmean(r["similarity_delta"] for r in rows),
        "similarity_delta_min": min(r["similarity_delta"] for r in rows),
        "rows": rows,
    }

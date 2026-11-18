"""Synthesize a fixed set of sentences and measure each take."""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Callable, Collection, Mapping, Sequence

import numpy as np

from breezyvoice.audio import pauses
from breezyvoice.evaluation import signal
from breezyvoice.evaluation.text import character_error_rate, comparable, marked_breaks, syllable_match

# A take whose error rate is above this would be noticed by a listener (about two wrong characters
# in a twenty-character sentence).
BAD_TAKE_CER = 0.1
# A narration pipeline that listens to each take and redoes it below this syllable match pays one
# more synthesis for every take under it.
RETAKE_SYLLABLE_MATCH = 0.97

Synthesize = Callable[[str, int], tuple[np.ndarray, int]]  # (text, seed) -> (mono float32 audio, sample rate)


def load_sentences(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def measure_reference(audio: np.ndarray, sample_rate: int, transcript: str,
                      transcribers: Mapping[str, Callable[[np.ndarray, int], str]],
                      embed: Callable[[np.ndarray, int], np.ndarray],
                      unify_script: Callable[[str], str]) -> dict:
    """What the instruments say about real speech: the floor for error rate, the ceiling for similarity.

    The similarity ceiling is between the two halves of the recording, i.e. the same speaker saying
    different words.
    """
    half = len(audio) // 2
    return {
        "seconds": len(audio) / sample_rate,
        "cer": {name: character_error_rate(transcript, t(audio, sample_rate), unify_script)
                for name, t in transcribers.items()},
        "split_half_similarity": float(embed(audio[:half], sample_rate) @ embed(audio[half:], sample_rate)),
    }


def evaluate(synthesize: Synthesize, sentences: Sequence[str], seeds: Sequence[int],
             transcribers: Mapping[str, Callable[[np.ndarray, int], str]],
             embed: Callable[[np.ndarray, int], np.ndarray],
             reference_embedding: np.ndarray,
             unify_script: Callable[[str], str],
             to_syllables: Callable[[str], Sequence[str]],
             syllable_variants: Collection[tuple[str, str]],
             save_wav: Callable[[str, np.ndarray, int], None],
             clock: Callable[[], float] = time.perf_counter,
             log: Callable[[str], None] = print) -> list[dict]:
    """One record per (sentence, seed)."""
    records = []
    for index, sentence in enumerate(sentences):
        for seed in seeds:
            start = clock()
            audio, sample_rate = synthesize(sentence, seed)
            wall = clock() - start
            name = f"s{index:02d}_seed{seed}.wav"
            save_wav(name, audio, sample_rate)
            heard = {asr: t(audio, sample_rate) for asr, t in transcribers.items()}
            chars = len(comparable(sentence, unify_script))
            record = {
                "sentence": index, "seed": seed, "text": sentence, "chars": chars, "wav": name,
                "seconds": len(audio) / sample_rate, "wall_seconds": wall,
                "heard": heard,
                "cer": {asr: character_error_rate(sentence, text, unify_script) for asr, text in heard.items()},
                "syllable_match": {asr: syllable_match(sentence, text, to_syllables, syllable_variants)
                                   for asr, text in heard.items()},
                "similarity": float(embed(audio, sample_rate) @ reference_embedding),
                "seconds_per_char": len(audio) / sample_rate / chars,
                "clipped_fraction": signal.clipped_fraction(audio),
                "tail_level_db": signal.tail_level_db(audio, sample_rate),
                "pauses": pauses(audio, sample_rate),
            }
            # A pause where the text has no punctuation breaks a phrase apart. Without word timings
            # the pauses cannot be placed, so only their surplus over the marked breaks is counted.
            record["unmarked_pauses"] = max(0, len(record["pauses"]) - marked_breaks(sentence))
            records.append(record)
            worst = max(record["cer"].values())
            log(f"[{len(records)}/{len(sentences) * len(seeds)}] {name} {record['seconds']:.1f}s "
                f"in {wall:.1f}s cer<={worst:.2f} sim={record['similarity']:.3f}")
    return records


def summarize(records: Sequence[dict]) -> dict:
    """Averages, and the tails that averages hide: a bad take costs a retry, however good the mean is."""
    asrs = sorted(records[0]["cer"])
    audio = sum(r["seconds"] for r in records)
    wall = sum(r["wall_seconds"] for r in records)
    sims = [r["similarity"] for r in records]
    return {
        "takes": len(records),
        "audio_seconds": audio,
        "real_time_factor": wall / audio,
        "cer_mean": {a: statistics.fmean(r["cer"][a] for r in records) for a in asrs},
        "bad_takes": {a: sum(r["cer"][a] > BAD_TAKE_CER for r in records) for a in asrs},
        # Both recognisers agreeing that a take is bad is stronger evidence than either alone.
        "bad_takes_all_recognisers": sum(min(r["cer"].values()) > BAD_TAKE_CER for r in records),
        "retakes": {a: sum(r["syllable_match"][a] < RETAKE_SYLLABLE_MATCH for r in records) for a in asrs},
        "retakes_all_recognisers": sum(max(r["syllable_match"].values()) < RETAKE_SYLLABLE_MATCH for r in records),
        "similarity_mean": statistics.fmean(sims),
        "similarity_min": min(sims),
        "seconds_per_char_median": statistics.median(r["seconds_per_char"] for r in records),
        "takes_with_unmarked_pauses": sum(r["unmarked_pauses"] > 0 for r in records),
        "clipped_takes": sum(r["clipped_fraction"] > 0 for r in records),
        "loud_endings": sum(r["tail_level_db"] > -10 for r in records),
    }

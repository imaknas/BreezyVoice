# BreezyVoice (maintained fork)

Voice-cloning text-to-speech for **Taiwanese Mandarin**, packaged to run offline on CPU or Apple
Silicon (MPS). This is a maintained fork of MediaTek Research's
[BreezyVoice](https://github.com/mtkresearch/BreezyVoice) (Apache 2.0), whose upstream has been
inactive since June 2025 and targets Python 3.10, torch 2.3 and Linux + CUDA. The model itself is
unchanged: [MediaTek-Research/BreezyVoice-300M](https://huggingface.co/MediaTek-Research/BreezyVoice-300M)
(Apache 2.0), partially derived from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice).
Paper: [arXiv:2501.17790](https://arxiv.org/abs/2501.17790).

Given a short reference recording of a voice and its transcript, it reads new Traditional Chinese
text in that voice. Polyphonic and rare characters get 注音 hints from
[g2pW](https://github.com/GitYCC/g2pW); you can also force a reading inline: `今天天氣真好[:ㄏㄠ3]`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.14 is used by default (`.python-version`); 3.13 also works.

```sh
git clone https://github.com/imaknas/BreezyVoice.git breezyvoice
cd breezyvoice
uv sync
uv run breezyvoice setup      # the only step that uses the network
```

`setup` downloads, at pinned commit hashes, the BreezyVoice weights (five files) and the
`bert-base-chinese` vocabulary into the Hugging Face cache, and the g2pW model (checksum-verified)
into `$BREEZYVOICE_HOME` (default `~/.cache/breezyvoice`). After that, loading and synthesis make
no network calls at all (verified with networking denied by the macOS sandbox).

## Interface

Every mode takes the same request and writes a 22.05 kHz mono 16-bit wav:

| field      | type   | meaning                                                         |
|------------|--------|-----------------------------------------------------------------|
| `text`     | string | the sentence or chunk to speak                                  |
| `ref_wav`  | path   | reference recording of the voice (any sample rate; ~5–15 s works well) |
| `ref_text` | string | exact transcript of `ref_wav`                                   |
| `seed`     | int    | sampling seed, default 0; same seed + same inputs → same audio  |
| `out`      | path   | where to write the `.wav` (parent directories are created)      |
| `min_pause_s` | float | shortest silence reported in `pauses`, default 0.25 (optional) |

Use absolute paths: they are resolved by the process doing the synthesis. A successful result:

```json
{"ok": true, "out": "/abs/out.wav", "duration_s": 4.92, "sample_rate": 22050, "seed": 1,
 "text": "第一，你在監控情緒，大腦會放大不開心。", "pauses": [[1.02, 1.34], [2.7, 3.0]], "min_pause_s": 0.25, "elapsed_s": 8.5}
```

`text` in the result is what the model actually read, after normalisation and 注音 annotation.
`pauses` lists the silent stretches of `min_pause_s` or more inside the speech as `[start, end]` in seconds
(not the silence before the first or after the last sound), so a caller can reject a take that
pauses where the text has no punctuation.
Failures come back as `{"ok": false, "error": "...", "kind": "bad_request" | "synthesis_failed"}`.

Text containing several sentences is split on `。？！.?!` and the pieces are concatenated,
as upstream did; for retry-per-sentence workflows send one sentence per request.

### Batch (pipes, model loaded once)

```sh
uv run breezyvoice batch < requests.jsonl > results.jsonl
```

Prints `{"ok": true, "ready": true}` once the model is loaded, then one JSON result line per JSON
request line on stdin, in order. Logs go to stderr. A caller can keep the process open and write
requests one at a time.

### HTTP service (localhost, model loaded once)

```sh
uv run breezyvoice serve --port 8765
curl -s -X POST http://127.0.0.1:8765/synthesize -H 'Content-Type: application/json' \
  -d '{"text":"今天天氣真好。","ref_wav":"/abs/ref.wav","ref_text":"…","seed":0,"out":"/abs/out.wav"}'
curl -s http://127.0.0.1:8765/health
```

Binds to 127.0.0.1 only (requests name local files); requests are processed one at a time.
HTTP status: 200 ok, 400 bad request, 500 synthesis failed.

### One-off

```sh
uv run breezyvoice synth --text "今天天氣真好。" --ref-wav data/example.wav \
  --ref-text "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。只有擁有解密方法的對象，經由解密過程，才能將密文還原為正常可讀的內容。" \
  --seed 0 --out /tmp/out.wav
```

Loads the model on every call (~3 s); use `batch` or `serve` for more than a few sentences.

Common options: `--device auto|cpu|mps` (auto = LLM on CPU, flow decoder and vocoder on MPS when available), `--threads N`,
`--weights-dir DIR` (weights from a plain directory instead of the Hugging Face cache).

### Python

```python
from pathlib import Path
from breezyvoice.loader import load

tts = load("auto")
speaker = tts.speaker(Path("ref.wav"), "transcript of ref.wav")   # cached per file + transcript
result = tts.synthesize("今天天氣真好。", speaker, seed=0)          # result.audio: float32 numpy
```

## Speed (Mac mini M4 Pro)

Six short narration sentences (7–21 characters, 29 s of audio in total), model already loaded, one
process:

| device | per sentence | real-time factor | machine load avg |
|--------|--------------|------------------|------------------|
| MPS | 8.6 s (5.3–11.2) | 1.78 | 2.7 |
| CPU | 11.3 s | 2.35 | 6.8 |

Those figures predate the in-place key/value cache. With it, on three longer sentences (23 s of
audio) and a moderately busy machine (load average ~6), the same benchmark went from RTF 1.78 to
1.54 on MPS and from 2.07 to 1.71 on CPU, and the default placement (`auto`: LLM on CPU, flow
decoder and vocoder on MPS) reached 1.27. An LLM decode step on CPU fell from ~23 ms to ~13 ms.

Loading takes ~3 s; preparing a new reference voice ~0.4 s (then cached). Times roughly double when
the machine is busy (load average ~30), CPU more than MPS. Upstream on the same machine took 35 s for
a 3.4 s sentence. Most of the time is still the autoregressive LLM; the cost of a step grows with
the length of the reference clip, which sits in its attention cache.

## Sampling

Each speech token is drawn from the most likely tokens that together reach 80% of the probability, and
a token that already fills three of the last ten positions is not drawn again (`--sampler
repetition-aware`, the default). `--sampler top-k` is the upstream setting, a plain draw among the 25
most likely tokens; use it to reproduce a take made before the default changed, since the two give
different takes for the same seed.

On 150 narration sentences with one reference voice, counting takes in which two recognisers both
miss a syllable, and takes with a pause the text does not call for:

| sampler | syllable missed | unmarked pause | seconds per character |
|---------|-----------------|----------------|-----------------------|
| top-k 25 (upstream) | 43% | 24% | 0.258 |
| nucleus, repeat limit 1 of 10 (CosyVoice's setting) | 42% | 14% | 0.218 |
| nucleus, repeat limit 2 of 10 | 40% | 15% | 0.246 |
| nucleus, repeat limit 3 of 10 (default) | 33% | 23% | 0.261 |
| nucleus, no repeat limit | 32% | 20% | 0.325 |

The repeat limit sets the pace: this model produces 50 tokens per second, a held sound repeats its
token, and a strict limit cuts held sounds short. The stricter settings pause less but speak faster and,
to a listener, slur; the default keeps the pace of the upstream setting and misses fewer syllables.

## Text normalisation

`breezyvoice/text/normalize.py` replaces upstream's WeTextProcessing/pynini dependency (which does
not build on macOS) with rules for Taiwan Mandarin, e.g.:

| input | read as |
|-------|---------|
| `有2個人` / `第2名` / `2月` | 有兩個人 / 第二名 / 二月 |
| `20000元` / `200人` | 兩萬元 / 兩百人 |
| `50%` / `3.5％` / `1-3%` | 百分之五十 / 百分之三點五 / 百分之一到三 |
| `2025年` / `1990年代` / `2000年` / `民國114年` | 二零二五年 / 一九九零年代 / 兩千年 / 民國一百一十四年 |
| `2025/3/5` / `10:30` / `2:00` | 二零二五年三月五日 / 十點三十分 / 兩點 |
| `3-5天` / `1/3` / `1:2` | 三到五天 / 三分之一 / 一比二 |
| `0912345678` | 零九一二三四五六七八 |
| `30°C` / `5km` / `NT$300` | 三十度 / 五公里 / 三百元 |

Write dates as `3月5日`; a bare `3/5` is read as a fraction. See `tests/test_normalize.py`.

## Deployment

To keep a long-running service from picking up half-finished edits, run it from a separate production
checkout and update that checkout only with

```sh
bin/deploy.sh
```

It fast-forwards the production checkout to `origin/main`, runs `uv sync --locked` and the full test suite (including
the smoke test), and only then restarts the launchd service and waits for `/health`. It can refuse to restart while a
client is in the middle of a job (`--force` overrides). The first run clones the checkout. The paths and the service
label of a given machine go in `deploy.env`, which git ignores; the header of the script lists the settings.

## Evaluation

`breezyvoice evaluate` measures a build on fixed sentences and seeds, so two builds can be compared
take by take. It needs extra packages and models (about 1.2 GB, downloaded once, pinned by checksum):

```
uv sync --group evaluation
breezyvoice evaluate setup
breezyvoice evaluate run --ref-wav REF.wav --ref-text "…" --out RUNS/base --label "what this is"
breezyvoice evaluate compare RUNS/base RUNS/candidate
```

Keep `--out` outside the repository: it holds every take as a wav next to `result.json`.

Per take it records the character error rate against two recognisers of different families
(Paraformer and SenseVoice; neither is Whisper, which callers tend to gate on), the syllable match a
narration pipeline would gate retakes on, speaker similarity to the reference (ERes2NetV2, a different
architecture from the CAM++ encoder that conditions the model), seconds per character, clipping, the
level of the last 100 ms, and wall time with the machine's load average. `run` also measures the
reference recording itself, which gives the error-rate floor and a similarity ceiling for the instruments.

Three sentence files ship with it (`--sentences`): `sentences.txt` (38 narration sentences, the
default), `code_switch_sentences.txt` (Mandarin with English words; scored separately, recognisers are
weaker on it) and `accent_sentences.txt` (words read differently in Taiwan; for listening only, a
recogniser cannot tell the readings apart).

What the instruments can and cannot see, from runs with the flow decoder deliberately degraded
(76 takes, `data/example.wav` as reference):

| run | CER (two recognisers) | similarity, paired change | mel distance to baseline | loud endings |
|-----|-----------------------|---------------------------|--------------------------|--------------|
| baseline (10 flow steps, guidance 0.7) | 0.083 / 0.073 | – | – | 13 |
| 2 flow steps | 0.086 / 0.075 | −0.050 | 0.89 | 30 |
| no guidance | 0.082 / 0.077 | −0.015 | 0.42 | 14 |

- Error rate does not react to the acoustic stages: intelligibility is decided by the speech tokens.
  Judge a change to the flow decoder or vocoder by the paired similarity change, the mel distance and
  the endings, and by listening.
- Unpaired comparisons are noisy: the two seeds of the baseline alone differ by 0.024 against 0.060 in
  error rate and 0.876 against 0.855 in similarity. A change that alters the tokens needs many more
  takes than one that does not.
- Similarity is unreliable on takes under about two seconds (0.71–0.78 on four-character sentences).
- The recognisers do not hear accent. Listen to `accent_sentences.txt`.
- The syllable comparison is built for Taiwanese Mandarin. Both texts are converted to Simplified
  script first, because the syllable converter chooses among a character's readings from Simplified
  phrases and otherwise reads 著, 乾, 樂, 沒, 調 or 傳 differently on the two sides; initials and finals
  that speakers in Taiwan merge (zh/z, ch/c, sh/s, -ing/-in, -eng/-en) are merged; and readings that
  are standard in Taiwan (和 as hàn, 液 as yì) are accepted. Without this about a third of the takes
  flagged for a missed syllable were speech that was right: on 150 sentences the share of flagged takes
  fell from 33% to 22%. The percentages in the Sampling table were measured before this correction;
  the order of the settings is the same after it (33%, 32%, 25%, 22%, 18%).

Two findings about the model itself from the baseline: a 159-character sentence with commas only
ran away into repetition until the length limit, and very short sentences occasionally come out as
different words. The first is now avoided by breaking sentences longer than 60 characters at their
commas before synthesis (that sentence then takes 42 s with an error rate of 0.02–0.03, against 77 s
and 1.58); a clause that long without any comma is still at risk.

## Adapting the voice to your own recordings

A voice cloned from one reference clip varies from take to take. Recording more of the same speaker
and fine-tuning the language model on it is the route to a steadier voice; `recording/README.md`
(Traditional Chinese) explains what and how to record. The steps:

```
breezyvoice recording split --script recording/script-stage1.txt --section B --wav DATA/B.wav --out DATA/clips
breezyvoice train prepare --clips DATA/clips --script recording/script-stage1.txt --out DATA/features.pt
breezyvoice train lm --features DATA/features.pt --out DATA/model
breezyvoice evaluate run --weights-dir DATA/model --ref-wav … --ref-text … --out RUNS/tuned
```

`train lm` trains only the language model, keeps the weights at the lowest loss on script lines it
never trained on, and writes a weights directory (the unchanged networks are linked, not copied)
that every command accepts as `--weights-dir`. On a Mac mini M4 Pro a step with eight utterances
takes about 1.8 s on MPS, roughly 18 times faster than the audio it covers, and uses about 19 GB.

Ten minutes of recordings are enough to check the pipeline, not to improve the voice: held-out loss
turns back up after about seven passes over the data and the takes are no steadier than before.

Keep recordings, features and trained weights outside the repository. A model of a person's voice
lets anyone make that person say anything.

## Tests

```sh
uv run pytest tests/test_normalize.py   # fast, no model
uv run pytest                            # includes the smoke test (needs `breezyvoice setup`)
```

The smoke test synthesizes a sentence with `data/example.wav` (upstream's sample voice) and checks
the audio is finite, not silent, of plausible length, and identical for the same seed.

## Differences from upstream

- **Packaging:** `pyproject.toml` + `uv.lock`, Python ≥ 3.13, torch 2.9+, no CUDA or Linux-only
  dependencies. Training code, Docker/API scripts, the bundled Matcha-TTS tree and committed
  `__pycache__` were removed.
- **Fewer dependencies:** the parts of Matcha-TTS, openai-whisper (tokenizer + log-mel), torchaudio
  (Kaldi fbank, resampling) and librosa (mel filterbank) that inference uses are vendored under
  `cosyvoice/matcha/` and `breezyvoice/_vendor/` with their licences. torchaudio, librosa/numba,
  lightning, hydra, pyarrow, WeTextProcessing and ttsfrd are no longer needed.
- **Model built in code** (`breezyvoice/model.py`) from the same hyper-parameters as the weights'
  `cosyvoice.yaml`, loaded with `strict=True`.
- **Offline by construction:** pinned revisions, explicit `setup`, no Hugging Face calls at load time.
- **Text normalisation** of our own (above). Upstream's frontend also dropped the character before a
  manual `[:注音]` annotation (`真好[:ㄏㄠ3]` became `真[:ㄏㄠ3]`); fixed.
- **Speed:** no forced single thread; reference-audio features are cached per speaker; the
  relative-position projection is reused across decoding steps (bit-identical output on CPU).
- **Seeding:** each request seeds torch before synthesis, and all sampling (LLM top-k, flow noise,
  vocoder noise) draws from the CPU generator, so a seed gives the same take on CPU and MPS, and the
  same take as upstream on CPU.
- Whisper auto-transcription of the reference was dropped: `ref_text` is required.

Verified against upstream (torch 2.3.1, CPU) on `data/example.wav` with the same inputs and seed:
text annotations, text token ids, prompt speech tokens and the generated speech tokens are identical
on both CPU and MPS; the flow mel matches within 2e-5 and the vocoder output within 1.4e-4
(correlation > 0.99999). Reference features: mel within 0.006; the speaker embedding differs slightly
(cosine 0.9988) because newer onnxruntime computes the CAM++ network a little differently.
48 kHz stereo references load identically to upstream (within 1.2e-7).

## Licence

Apache 2.0 (see `LICENSE`); vendored third-party code keeps its own licence file next to it (see
`NOTICE`). Model weights: Apache 2.0 per their model card. Please cite the BreezyVoice paper:

```
@article{hsu2025breezyvoice,
  title={BreezyVoice: Adapting TTS for Taiwanese Mandarin with Enhanced Polyphone Disambiguation--Challenges and Insights},
  author={Hsu, Chan-Jan and Lin, Yi-Cheng and Lin, Chia-Chun and Chen, Wei-Chih and Chung, Ho Lam and Li, Chen-An and Chen, Yi-Chang and Yu, Chien-Yu and Lee, Ming-Ji and Chen, Chien-Cheng and others},
  journal={arXiv preprint arXiv:2501.17790},
  year={2025}
}
```

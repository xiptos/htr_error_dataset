# HTR Error Dataset

A pipeline that builds a handwritten "writing-error" dataset: a set of
(clean, corrupted) handwritten word-image pairs, where the corruption is a
**genuine handwritten glyph of one letter spliced over another letter**,
taken from the same writer. Nothing in the corrupted image is synthetically
rendered -- every pixel is real handwriting.

The dataset is designed to test whether handwriting OCR/VLM systems
transcribe faithfully what is actually on the page, or instead "repair" an
implausible stimulus toward a linguistically plausible reading -- following
the visual-grounding evaluation paradigm of Karamolegkou et al.,
*"Reading or Guessing? Visual Grounding Failures of Vision-Language Models
for OCR"* (arXiv:2605.27750), extended here from rendered printed text to
real handwriting.

## How it works

Source material is the [IAM Handwriting
Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database),
accessed through two derived, publicly hosted Hugging Face datasets rather
than the original release, combined to recover information neither one
exposes alone:

- [`Teklia/IAM-line`](https://huggingface.co/datasets/Teklia/IAM-line):
  line-image scans paired with their transcription, but without writer
  identifiers.
- [`AlhitawiMohammed22/En_IAM_lines`](https://huggingface.co/datasets/AlhitawiMohammed22/En_IAM_lines):
  transcriptions paired with the original IAM file name (e.g.
  `a01-000u-00.png`), which preserves the source form identifier
  (`a01-000u`), but without pixel data.

The pipeline then runs through these steps:

1. **Writer identification.** Neither corpus reports writer identity
   directly. The IAM *form* identifier (the file name prefix) is used as a
   writer proxy -- each form was completed by a single writer, so grouping
   lines by form is writer-homogeneous (it can only under-count, never mix,
   per-writer diversity).
2. **Image-transcription alignment.** The two corpora are joined on exact
   equality of their transcription string, keeping only strings that occur
   exactly once in *each* corpus, to avoid ambiguous one-to-many matches.
3. **Word segmentation.** Each aligned line image is binarized (Otsu) and
   cut at the widest inter-word ink gaps (one gap per word boundary implied
   by the transcription's whitespace tokens).
4. **OCR-based segmentation validation.** Gap-based segmentation is
   heuristic and can silently misassign a transcription token to the wrong
   image region. Every candidate word crop is transcribed by a pretrained
   handwriting-recognition model
   ([`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten))
   and is kept only if the model's reading is within a length-scaled
   Levenshtein-distance tolerance of the expected word.
5. **Letter segmentation.** Each validated word crop is re-cut at finer
   column gaps into per-letter slots, accepted only if every slot's width
   falls within a plausible fraction of the word's total width.
6. **Per-writer glyph bank.** Every accepted letter slot is cropped to its
   own ink extent and indexed by writer, letter identity, and case
   (upper/lower). A writer is kept for generation only if their bank has
   enough usable words and enough distinct letters.
7. **Corruption.** For each selected word, one letter position is chosen at
   random and replaced with a real glyph of a *different* letter from the
   *same* writer's bank (preferring matching case and a compatible size).
   The word image is rebuilt by horizontal splicing: unchanged left part +
   a new slot sized to the replacement glyph's own natural size + unchanged
   right part.
8. **Output.** Each generated pair is saved as `clean_image` /
   `error_image`, together with the position and identity of the
   substitution and the resulting corrupted word -- the ground truth to use
   when scoring a model's transcription of `error_image`.

### Known limitations

- Segmentation is unsupervised and heuristic (IAM's offline images carry no
  character-level ground truth); the whitespace-gap requirement biases the
  dataset toward shorter, more print-like words.
- Replacement glyphs are drawn from whichever letters a writer happened to
  produce elsewhere, not selected for visual confusability with the letter
  they replace.
- TrOCR validation reduces, but does not eliminate, segmentation error, so a
  small residual rate of mislabeled crops cannot be ruled out.

## Project layout

```
generate_dataset.py   build the dataset (IAM -> dataset/)
eda.ipynb              exploratory analysis of the dataset -> results/eda/
systems.py              registry of OCR/VLM systems to evaluate
backends.py              per-backend runners (tesseract, kraken, mlx-vlm)
evaluate.py               run one/all systems over the dataset -> results/predictions/
results.ipynb              aggregate predictions into metrics & figures -> results/
plotting.py                  shared figure style, used by both notebooks
textmetrics.py                 shared edit-distance / normalization / dictionary helpers

pyproject.toml   project metadata, dependencies and the optional extras
uv.lock          pinned dependency versions, used by `uv sync`
.gitignore       excludes generated outputs, caches and editor files

dataset/   generated by generate_dataset.py (gitignored)
results/   generated by eda.ipynb, evaluate.py, results.ipynb (gitignored)
```

The four stages run in order -- each reads only what the previous stage
wrote:

```
generate_dataset.py  -->  dataset/
                             |
                             +--> eda.ipynb      --> results/eda/
                             |
                             +--> evaluate.py    --> results/predictions/
                                                        |
                                                        +--> results.ipynb --> results/
```

## Installation

Requires Python >= 3.10.

```bash
# with uv (recommended)
uv sync --extra notebooks --extra evaluate

# or with plain pip
pip install -e ".[notebooks,evaluate]"
```

Install just `dataset` construction's dependencies (no extras) if you only
need `generate_dataset.py`; add `notebooks` for `eda.ipynb` / `results.ipynb`
and `evaluate` for `evaluate.py`; the `all` extra is shorthand for both.
`evaluate`'s VLM backend (`mlx-vlm`) only
installs on Apple Silicon (it depends on Apple's MLX framework) -- the two
traditional backends (Tesseract, Kraken) run anywhere. To evaluate VLMs on
CUDA/CPU instead, swap `run_mlx_vlm` in `backends.py` for a `transformers`-
or `vllm`-based loader; the rest of the pipeline (`evaluate.py`,
`results.ipynb`) is backend-agnostic.

The first `generate_dataset.py` run downloads the two IAM-derived datasets
and the TrOCR model from the Hugging Face Hub (cached under
`~/.cache/huggingface` afterward). `evaluate.py` additionally downloads
each VLM's weights from `mlx-community` on first use (several GB each).

## Workflow

### 1. Build the dataset

```bash
python generate_dataset.py
```

This writes `dataset/metadata.csv`, `dataset/metadata.jsonl`, and the paired
images under `dataset/images/`.

Useful flags (all optional, defaults match the settings used to build the
reference dataset):

| Flag | Default | Meaning |
|---|---|---|
| `--seed` | `42` | Random seed |
| `--max-lines` | `9600` | IAM lines to process (full safe/unambiguous pool is ~9532) |
| `--words-per-writer` | `15` | Max corrupted words generated per writer |
| `--max-writers` | all eligible | Cap on the number of writers used |
| `--min-word-len` | `3` | Minimum word length to be a corruption candidate |
| `--min-gap-letter` | `2` | Minimum gap width (px) to count as a letter boundary |
| `--min-slot-frac` | `0.12` | Minimum accepted letter-slot width (fraction of word width) |
| `--max-slot-frac` | `0.55` | Maximum accepted letter-slot width (fraction of word width) |
| `--min-words-per-writer` | `8` | Minimum usable words for a writer to be eligible |
| `--min-distinct-letters` | `4` | Minimum distinct letters in a writer's glyph bank |
| `--output-dir` | `dataset` | Output directory |

Running with the default `--max-lines` processes the IAM corpus through
TrOCR and takes roughly 35-45 minutes on a single GPU/MPS device (most of
the time is TrOCR inference); reduce `--max-lines` to iterate faster.

### 2. Explore the dataset (EDA)

```bash
jupyter notebook eda.ipynb
```

Computes dataset-level statistics and figures (word-length distribution,
position of the altered letter, letter-substitution confusion matrix,
letter frequency vs. a reference English corpus, the minimal-pair rate --
i.e. how often the corruption happens to land on another real word --
image geometry, writer distribution, and a qualitative sample grid).
Writes PDFs/PNGs, `summary_table.csv` and `minimal_pair_examples.csv` to
`results/eda/`. Depends only on
`dataset/metadata.csv`, so it can run before any evaluation.

### 3. Evaluate OCR/VLM systems

```bash
python evaluate.py --system tesseract              # one traditional engine
python evaluate.py --system qwen3_vl_8b --n 100     # one VLM, 100-pair sample
python evaluate.py --system all                     # every system, full dataset
python evaluate.py --system all --prompt-variant all # + both anti-correction prompt variants
```

Each system transcribes both the clean and the corrupted image of every
selected pair (a paired design) and predictions are appended incrementally
to `results/predictions/<system>.csv` (or `<system>__<variant>.csv`); an
interrupted run can simply be restarted, since already-recorded `(id,
condition)` pairs are skipped.

| Flag | Default | Meaning |
|---|---|---|
| `--system` | required | System name from `systems.py`, or `all` |
| `--n` | full dataset | Evaluate a random sample of N pairs |
| `--seed` | `42` | Random seed for the `--n` sample |
| `--prompt-variant` | each system's own prompt | `diplomatic`, `symbol_game`, or `all` (defined in `PROMPT_VARIANTS` in `systems.py`); only applies to prompt-driven VLMs, other systems are skipped |
| `--dataset-dir` | `dataset` | Directory containing `metadata.csv` |

The systems evaluated are registered in `systems.py`:

| System | Kind | Backend |
|---|---|---|
| Tesseract v5 | traditional | `pytesseract` |
| Kraken (Rev City English HTR) | traditional | `kraken` |
| LightOnOCR-1B | VLM | `mlx-vlm` |
| DeepSeek-OCR | VLM | `mlx-vlm` (see note below) |
| Qwen3-VL-2B/8B/30B-A3B-Instruct | VLM | `mlx-vlm` |
| olmOCR-2-7B | VLM | `mlx-vlm` |
| GLM-4.6V / GLM-4.6V-Flash | VLM | `mlx-vlm` |

> **Note:** DeepSeek-OCR only loads under `mlx-vlm` with
> `trust_remote_code=False` (set on its entry in `systems.py`, which also
> explains why). On isolated word crops it mostly answers in LaTeX math
> markup (e.g. `\[g o n e\]`) rather than plain text, so its entry sets
> `cleanup="latex_math"`: `results.ipynb` unwraps that markup
> (`strip_latex_math` in `textmetrics.py`) before the usual normalization.
> Raw predictions on disk are left untouched.
>
> **Note:** LightOnOCR-1B needs two workarounds under `mlx-vlm` (image
> placeholder tokens and escaped newlines in the repo's chat template, both
> set on its entry in `systems.py`); without them the model never sees the
> image. It is also the only system whose input is modified: each crop is
> shown with a border of half its height, in the crop's own paper colour
> (`image_pad_frac=0.5`), because on bare word crops this page-level model
> often hallucinates a markdown document instead of transcribing. On a
> 100-pair sample, clean-image word accuracy was 56% unpadded and 91% with
> the border; upscaling the crops instead made it worse.

Add, remove, or reconfigure a system by editing `SYSTEMS` in `systems.py`
-- `evaluate.py` and `results.ipynb` read that registry, nothing else needs
to change. To add a new backend (e.g. a CUDA-hosted VLM), add a
`run_*` function to `backends.py` and register it in `BACKEND_RUNNERS`.

### 4. Aggregate results

```bash
jupyter notebook results.ipynb
```

Reads every `results/predictions/*.csv`, scores each prediction against
`dataset/metadata.csv` (character error rate, exact match, and whether a
corrupted-image prediction matches the displayed text -- "faithful" -- or
the original uncorrupted word instead -- "repaired"), and writes:

- `results/predictions_scored.csv` -- long format, one row per prediction
- `results/metrics_summary.csv` -- one row per system (default prompt)
- `results/metrics_summary_variants.csv` -- one row per system x prompt variant
- `results/figures/*.pdf` / `*.png` -- CER, faithful-vs-repair, minimal-pair
  effect, and (if `--prompt-variant` runs exist) prompt-sensitivity figures

## Output schema

### `dataset/metadata.csv` (and the equivalent `.jsonl`)

Images referenced under `dataset/images/`:

| Column | Meaning |
|---|---|
| `id` | Unique identifier (`<writer_id>_<index>`) |
| `writer_id` | Writer proxy (IAM form id) |
| `word` | Original word |
| `error_word` | Word as it actually appears in `error_image` -- use this as ground truth when scoring a model's transcription of `error_image` |
| `letter_index` | 0-indexed position of the substituted letter |
| `original_letter` | Letter that was removed |
| `new_letter` | Letter that was spliced in |
| `clean_image` | Path to the unmodified control image |
| `error_image` | Path to the corrupted image -- the main test stimulus |

### `results/predictions/<system>[__<variant>].csv` (written by `evaluate.py`)

| Column | Meaning |
|---|---|
| `id` | Matches `dataset/metadata.csv`'s `id` |
| `condition` | `clean` or `error` |
| `image_path` | Path to the image shown to the system |
| `gt_text` | Ground truth for this condition (`word` or `error_word`) |
| `prediction` | Raw model/engine output |
| `elapsed_sec` | Wall-clock time for this single prediction |

### `results/predictions_scored.csv` (written by `results.ipynb`)

All of the above, plus `pred_norm`/`gt_norm` (normalized text; `pred_norm`
also applies a system's `cleanup`, if it has one), `cer`
(character error rate), `exact_match`, and `repaired_to_original` (true
when a corrupted-image prediction matches the *original* word instead of
the displayed, corrupted one).

## Citations

If you use this pipeline or the dataset it produces, please cite:

- Karamolegkou et al., *"Reading or Guessing? Visual Grounding Failures of
  Vision-Language Models for OCR"*, arXiv:2605.27750 -- the evaluation
  paradigm this dataset adapts to handwriting.
- Marti & Bunke, *"The IAM-database: an English sentence database for
  off-line handwriting recognition"*, IJDAR 2002 -- the source handwriting
  data.
- Li et al., *"TrOCR: Transformer-based Optical Character Recognition with
  Pre-trained Models"*, 2022 -- the model used for segmentation validation.
- Otsu, *"A threshold selection method from gray-level histograms"*, 1979 --
  the binarization method used for segmentation.

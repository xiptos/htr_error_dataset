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

## Installation

Requires Python >= 3.10.

```bash
# with uv (recommended)
uv sync

# or with plain pip
pip install -e .
```

The first run downloads the two IAM-derived datasets and the TrOCR model
from the Hugging Face Hub (cached under `~/.cache/huggingface` afterward).

## Usage

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

## Output schema

`dataset/metadata.csv` (and the equivalent `.jsonl`), with images under
`dataset/images/`:

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

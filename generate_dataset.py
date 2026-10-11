"""Build the handwritten writing-error dataset.

For a set of eligible IAM writers, this script splices a genuine handwritten
glyph of one letter over another letter in a real handwritten word, producing
(clean, corrupted) image pairs together with metadata describing the
substitution. Every pixel in the corrupted image is authentic handwriting
from the same writer -- nothing is synthetically rendered.

See README.md for the full methodology and rationale.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

IAM_LINES_META = "AlhitawiMohammed22/En_IAM_lines"  # text + original IAM filename (writer proxy)
IAM_LINES_IMG = "Teklia/IAM-line"  # image + text, no writer id
TROCR_MODEL = "microsoft/trocr-base-handwritten"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    seed: int = 42
    max_lines: int = 9600  # lines to process; the full safe/unambiguous pool is ~9532
    words_per_writer: int = 15  # max corrupted words generated per writer
    max_writers: int | None = None  # cap on eligible writers used (None = all)
    min_word_len: int = 3
    min_gap_letter: int = 2  # min gap width (px) to count as a letter boundary
    min_slot_frac: float = 0.12  # min letter-slot width, as a fraction of word width
    max_slot_frac: float = 0.55  # max letter-slot width, as a fraction of word width
    min_words_per_writer: int = 8  # min usable words for a writer to be eligible
    min_distinct_letters: int = 4  # min distinct letters in a writer's glyph bank
    output_dir: Path = Path("dataset")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-lines", type=int, default=9600,
        help="Number of IAM lines to process (full safe/unambiguous pool is ~9532)",
    )
    p.add_argument(
        "--words-per-writer", type=int, default=15,
        help="Max number of corrupted words generated per writer",
    )
    p.add_argument(
        "--max-writers", type=int, default=None,
        help="Cap the number of eligible writers used (default: all eligible)",
    )
    p.add_argument("--min-word-len", type=int, default=3)
    p.add_argument(
        "--min-gap-letter", type=int, default=2,
        help="Minimum gap width in pixels to count as a letter boundary",
    )
    p.add_argument(
        "--min-slot-frac", type=float, default=0.12,
        help="Minimum accepted letter-slot width, as a fraction of word width",
    )
    p.add_argument(
        "--max-slot-frac", type=float, default=0.55,
        help="Maximum accepted letter-slot width, as a fraction of word width",
    )
    p.add_argument(
        "--min-words-per-writer", type=int, default=8,
        help="Minimum number of usable words for a writer to be eligible",
    )
    p.add_argument(
        "--min-distinct-letters", type=int, default=4,
        help="Minimum number of distinct letters in a writer's glyph bank",
    )
    p.add_argument("--output-dir", type=Path, default=Path("dataset"))
    a = p.parse_args()
    return Config(
        seed=a.seed,
        max_lines=a.max_lines,
        words_per_writer=a.words_per_writer,
        max_writers=a.max_writers,
        min_word_len=a.min_word_len,
        min_gap_letter=a.min_gap_letter,
        min_slot_frac=a.min_slot_frac,
        max_slot_frac=a.max_slot_frac,
        min_words_per_writer=a.min_words_per_writer,
        min_distinct_letters=a.min_distinct_letters,
        output_dir=a.output_dir,
    )


# ---------------------------------------------------------------------------
# Image processing: binarization, gap detection, word/letter segmentation
# ---------------------------------------------------------------------------


def otsu_threshold(arr: np.ndarray) -> int:
    hist, _ = np.histogram(arr, bins=256, range=(0, 255))
    total = arr.size
    sum_all = np.dot(hist, np.arange(256))
    sum_b = 0.0
    weight_b = 0.0
    max_var = 0.0
    threshold = 128
    for i in range(256):
        weight_b += hist[i]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += i * hist[i]
        mean_b = sum_b / weight_b
        mean_f = (sum_all - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = i
    return threshold


def binarize(img: Image.Image) -> np.ndarray:
    """Grayscale + Otsu threshold. Returns a boolean mask, True = ink."""
    arr = np.array(img.convert("L"), dtype=np.float32)
    t = otsu_threshold(arr)
    return arr < t


def col_gap_runs(ink_mask: np.ndarray, min_gap: int = 1) -> list[tuple[int, int]]:
    """Runs of ink-free columns, i.e. candidate word/letter boundaries."""
    col_has_ink = ink_mask.any(axis=0)
    if len(col_has_ink) == 0:
        return []
    runs = []
    start = 0
    cur = col_has_ink[0]
    for i in range(1, len(col_has_ink)):
        if col_has_ink[i] != cur:
            runs.append((start, i, cur))
            start = i
            cur = col_has_ink[i]
    runs.append((start, len(col_has_ink), cur))
    return [(s, e) for (s, e, has_ink) in runs if not has_ink and (e - s) >= min_gap]


def tight_crop(
    img: Image.Image, ink_mask: np.ndarray
) -> tuple[Image.Image | None, np.ndarray | None]:
    cols = np.where(ink_mask.any(axis=0))[0]
    rows = np.where(ink_mask.any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0:
        return None, None
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    return img.crop((x0, y0, x1, y1)), ink_mask[y0:y1, x0:x1]


def segment_words(
    line_img: Image.Image, text: str
) -> list[tuple[str, Image.Image]] | None:
    """Split a line image into per-word crops using the widest ink gaps."""
    words = text.split()
    ink = binarize(line_img)
    gaps = col_gap_runs(ink, min_gap=1)
    gaps_sorted = sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)
    n_sep = len(words) - 1
    if n_sep < 0 or len(gaps_sorted) < n_sep:
        return None
    chosen = sorted(gaps_sorted[:n_sep], key=lambda g: g[0]) if n_sep > 0 else []
    bounds, prev, width = [], 0, line_img.width
    for s, e in chosen:
        mid = (s + e) // 2
        bounds.append((prev, mid))
        prev = mid
    bounds.append((prev, width))
    return [
        (w, line_img.crop((x0, 0, x1, line_img.height)))
        for w, (x0, x1) in zip(words, bounds)
    ]


def segment_letters(
    word_img: Image.Image, word: str, cfg: Config
) -> tuple[Image.Image, list[tuple[str, tuple[int, int, int, int]]]] | None:
    """Split an already-cropped word image into letters.

    Returns (tight_word_image, [(letter, (x0, y0, x1, y1)), ...]) or None if
    segmentation does not yield a plausible letter count / slot-width spread.
    """
    ink = binarize(word_img)
    word_t, ink_t = tight_crop(word_img, ink)
    if word_t is None or ink_t is None:
        return None
    n_sep = len(word) - 1
    gaps = col_gap_runs(ink_t, min_gap=cfg.min_gap_letter)
    if len(gaps) < n_sep:
        return None
    gaps_sorted = (
        sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:n_sep] if n_sep > 0 else []
    )
    chosen = sorted(gaps_sorted, key=lambda g: g[0])
    bounds, prev, width = [], 0, word_t.width
    for s, e in chosen:
        mid = (s + e) // 2
        bounds.append((prev, mid))
        prev = mid
    bounds.append((prev, width))
    widths = [x1 - x0 for (x0, x1) in bounds]
    if min(widths) < cfg.min_slot_frac * width or max(widths) > cfg.max_slot_frac * width:
        return None
    out = []
    for ch, (x0, x1) in zip(word, bounds):
        col_ink = ink_t[:, x0:x1]
        rows = np.where(col_ink.any(axis=1))[0]
        if len(rows) == 0:
            return None
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        out.append((ch, (x0, y0, x1, y1)))
    return word_t, out


def writer_id_from_filename(file_name: str) -> str:
    """IAM form id (writer proxy) is the filename minus its trailing line index."""
    return file_name.rsplit("-", 1)[0]


# ---------------------------------------------------------------------------
# TrOCR-based segmentation validation
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def word_is_plausible(pred: str, clean_word: str) -> bool:
    """Accept a segmented word crop only if TrOCR's reading is close enough.

    Exact match for words of <=3 characters, <=1 edit for 4-6 characters,
    <=2 edits for >=7 characters.
    """
    pred_norm = "".join(c for c in pred.lower() if c.isalpha())
    target = clean_word.lower()
    d = levenshtein(pred_norm, target)
    n = len(target)
    max_allowed = 0 if n <= 3 else (1 if n <= 6 else 2)
    return d <= max_allowed


class TrOCRValidator:
    """Thin wrapper around a pretrained handwriting-recognition model."""

    def __init__(self, model_name: str = TROCR_MODEL):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
            model_name
        )
        self.device = (
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"TrOCR loaded on {self.device}")

    def read(self, img: Image.Image) -> str:
        pixel_values = self.processor(
            images=img.convert("RGB")
        ).pixel_values.to(self.device)
        with torch.no_grad():
            ids = self.model.generate(pixel_values, max_new_tokens=20)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# Step 1-2: load the two IAM-derived corpora, align them, recover writer ids
# ---------------------------------------------------------------------------


def load_and_align(
    rng: random.Random,
) -> tuple[dict[str, Image.Image], defaultdict, list[str]]:
    """Join the two corpora on exact-text equality (unambiguous matches only).

    Returns:
        img_by_text: text -> PIL image, for every unambiguous line.
        writer_line_texts: writer_id -> [text, ...]
        writer_ids_shuffled: distinct writer ids, shuffled.
    """
    print("Loading IAM line corpora...")
    lines_meta = load_dataset(IAM_LINES_META, split="train")
    lines_img_dd = load_dataset(IAM_LINES_IMG)
    all_img = concatenate_datasets(
        [lines_img_dd["train"], lines_img_dd["validation"], lines_img_dd["test"]]
    )

    meta_text_counts = Counter(ex["text"] for ex in lines_meta)
    img_text_counts = Counter(all_img["text"])
    safe_texts = {
        t for t in meta_text_counts
        if meta_text_counts[t] == 1 and img_text_counts.get(t, 0) == 1
    }
    print(f"Unambiguously alignable lines: {len(safe_texts)}")

    print("Indexing images by text...")
    img_by_text: dict[str, Image.Image] = {}
    for ex in all_img:
        if ex["text"] in safe_texts:
            img_by_text[ex["text"]] = ex["image"]

    writer_line_texts = defaultdict(list)
    for ex in lines_meta:
        if ex["text"] in safe_texts:
            writer_line_texts[writer_id_from_filename(ex["file_name"])].append(ex["text"])

    writer_ids_shuffled = list(writer_line_texts.keys())
    rng.shuffle(writer_ids_shuffled)
    print(f"Distinct writers (forms): {len(writer_ids_shuffled)}")
    return img_by_text, writer_line_texts, writer_ids_shuffled


# ---------------------------------------------------------------------------
# Step 3-5: segment, validate, and build the per-writer glyph bank
# ---------------------------------------------------------------------------


def extract_glyph_banks(
    cfg: Config,
    rng: random.Random,
    validator: TrOCRValidator,
    writer_ids_shuffled: list[str],
    writer_line_texts: defaultdict,
    img_by_text: dict[str, Image.Image],
) -> tuple[defaultdict, defaultdict, list[str]]:
    """Segment lines into validated, letter-segmented words and glyph banks.

    This is the slowest step: TrOCR runs on every word candidate (for the
    default max_lines=9600, expect ~35-45 minutes, mostly TrOCR inference).
    Reduce --max-lines to speed up experimentation.
    """
    writer_glyph_bank = defaultdict(lambda: defaultdict(lambda: {"upper": [], "lower": []}))
    writer_words = defaultdict(list)  # writer_id -> [(word_tight_img, letters, clean_word), ...]

    n_lines_processed = n_word_candidates = n_trocr_ok = n_letter_seg_ok = 0

    for wid in writer_ids_shuffled:
        for text in writer_line_texts[wid]:
            if n_lines_processed >= cfg.max_lines:
                break
            n_lines_processed += 1
            line_img = img_by_text.get(text)
            if line_img is None:
                continue
            wseg = segment_words(line_img, text)
            if wseg is None:
                continue
            for raw_word, wimg in wseg:
                clean_word = "".join(c for c in raw_word if c.isalpha())
                if len(clean_word) < cfg.min_word_len or clean_word != raw_word or wimg.width < 2:
                    continue
                n_word_candidates += 1
                pred = validator.read(wimg)
                if not word_is_plausible(pred, clean_word):
                    continue
                n_trocr_ok += 1
                res = segment_letters(wimg, clean_word, cfg)
                if res is None:
                    continue
                n_letter_seg_ok += 1
                word_t, letters = res
                writer_words[wid].append((word_t, letters, clean_word))
                for ch, (x0, y0, x1, y1) in letters:
                    glyph = word_t.crop((x0, y0, x1, y1))
                    case = "upper" if ch.isupper() else "lower"
                    writer_glyph_bank[wid][ch.lower()][case].append(glyph)
        if n_lines_processed >= cfg.max_lines:
            break
        if n_lines_processed % 1000 == 0:
            print(f"...{n_lines_processed} lines, {n_letter_seg_ok} usable words so far")

    print("=== extraction statistics ===")
    print("lines processed:", n_lines_processed)
    print("candidate words:", n_word_candidates)
    print("passed TrOCR validation:", n_trocr_ok)
    print("passed letter segmentation:", n_letter_seg_ok)
    print("writers with at least 1 usable word:", len(writer_words))

    diverse_writers = [
        wid for wid, words in writer_words.items()
        if len(words) >= cfg.min_words_per_writer
        and len(writer_glyph_bank[wid]) >= cfg.min_distinct_letters
    ]
    if cfg.max_writers is not None and len(diverse_writers) > cfg.max_writers:
        diverse_writers = rng.sample(diverse_writers, cfg.max_writers)
    print(
        f"eligible writers (>= {cfg.min_words_per_writer} words, "
        f">= {cfg.min_distinct_letters} distinct letters): {len(diverse_writers)}"
    )
    return writer_glyph_bank, writer_words, diverse_writers


# ---------------------------------------------------------------------------
# Step 6: corrupt a word by splicing in a real glyph from the same writer
# ---------------------------------------------------------------------------


def pick_replacement(
    writer_glyph_bank: defaultdict,
    writer_id: str,
    orig_char: str,
    box_h: int,
    rng: random.Random,
) -> tuple[str, Image.Image] | None:
    """Pick a replacement glyph: same writer, different letter identity.

    Prefers (1) matching letter case and (2) a natural glyph height within
    [0.6x, 1.6x] of the original slot's height, relaxing case then size if no
    compatible candidate is found.
    """
    orig_lower = orig_char.lower()
    case = "upper" if orig_char.isupper() else "lower"
    bank = writer_glyph_bank[writer_id]

    def candidates_for(cases_keys):
        return [
            (letter, glyph)
            for letter, cases in bank.items()
            if letter != orig_lower
            for key in cases_keys
            for glyph in cases[key]
        ]

    pools = [candidates_for([case]), candidates_for(["upper", "lower"])]
    for pool in pools:  # 1st pass: matching case + compatible height
        if not pool:
            continue
        size_ok = [(l, g) for (l, g) in pool if 0.6 <= (box_h / g.size[1]) <= 1.6]
        if size_ok:
            return rng.choice(size_ok)
    for pool in pools:  # fallback: any candidate from this writer
        if pool:
            return rng.choice(pool)
    return None


def corrupt_word_image(
    word_t_img: Image.Image,
    letters: list[tuple[str, tuple[int, int, int, int]]],
    clean_word: str,
    writer_id: str,
    writer_glyph_bank: defaultdict,
    rng: random.Random,
) -> tuple[Image.Image, str, int, str, str] | None:
    """Splice a real replacement glyph into one letter position of a word.

    The word is rebuilt as [unchanged left part] + [new slot, sized to the
    replacement glyph's own natural size] + [unchanged right part], rather
    than stretching the replacement into the old letter's exact pixel box
    (which can leave a visible blank rectangle when proportions differ).
    """
    idx = rng.randrange(len(clean_word))
    orig_char = clean_word[idx]
    x0, y0, x1, y1 = letters[idx][1]
    box_w, box_h = x1 - x0, y1 - y0
    if box_w < 1 or box_h < 1:
        return None
    picked = pick_replacement(writer_glyph_bank, writer_id, orig_char, box_h, rng)
    if picked is None:
        return None
    new_letter_lower, glyph = picked
    new_char = new_letter_lower.upper() if orig_char.isupper() else new_letter_lower

    glyph_w, glyph_h = glyph.size
    scale = max(0.6, min(box_h / glyph_h, 1.6))
    new_w, new_h = max(1, round(glyph_w * scale)), max(1, round(glyph_h * scale))
    glyph_resized = glyph.resize((new_w, new_h), Image.Resampling.LANCZOS)

    height = word_t_img.height
    margin = max(2, round(0.05 * box_w))
    slot_w = new_w + 2 * margin
    left_part = word_t_img.crop((0, 0, x0, height))
    right_part = word_t_img.crop((x1, 0, word_t_img.width, height))
    slot = Image.new(word_t_img.mode, (slot_w, height), 255)
    slot.paste(glyph_resized, (margin, y1 - new_h))  # bottom-aligned (baseline approximation)

    corrupted = Image.new(
        word_t_img.mode, (left_part.width + slot_w + right_part.width, height), 255
    )
    corrupted.paste(left_part, (0, 0))
    corrupted.paste(slot, (left_part.width, 0))
    corrupted.paste(right_part, (left_part.width + slot_w, 0))

    error_word = clean_word[:idx] + new_char + clean_word[idx + 1 :]
    return corrupted, error_word, idx, orig_char, new_char


# ---------------------------------------------------------------------------
# Step 7: generate and save the final dataset
# ---------------------------------------------------------------------------


def generate_dataset(
    cfg: Config,
    rng: random.Random,
    diverse_writers: list[str],
    writer_words: defaultdict,
    writer_glyph_bank: defaultdict,
) -> pd.DataFrame:
    output_dir = cfg.output_dir
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    writer_order = list(diverse_writers)
    rng.shuffle(writer_order)

    n_attempted = n_ok = 0
    for wid in writer_order:
        words = writer_words[wid]
        idxs = list(range(len(words)))
        rng.shuffle(idxs)
        for wi in idxs[: cfg.words_per_writer]:
            word_t_img, letters, clean_word = words[wi]
            n_attempted += 1
            result = corrupt_word_image(
                word_t_img, letters, clean_word, wid, writer_glyph_bank, rng
            )
            if result is None:
                continue
            corrupted, error_word, idx, orig_char, new_char = result
            n_ok += 1
            rec_id = f"{wid}_{n_ok:05d}"
            clean_path = images_dir / f"{rec_id}_clean.png"
            error_path = images_dir / f"{rec_id}_error.png"
            word_t_img.save(clean_path)
            corrupted.save(error_path)
            records.append(
                {
                    "id": rec_id,
                    "writer_id": wid,
                    "word": clean_word,
                    "error_word": error_word,
                    "letter_index": idx,
                    "original_letter": orig_char,
                    "new_letter": new_char,
                    "clean_image": str(clean_path.relative_to(output_dir)),
                    "error_image": str(error_path.relative_to(output_dir)),
                }
            )

    print(
        f"attempted: {n_attempted}, generated: {n_ok}, "
        f"writers used: {len(set(r['writer_id'] for r in records))}"
    )

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_dir / "metadata.csv", index=False)
    df.to_json(output_dir / "metadata.jsonl", orient="records", lines=True, force_ascii=False)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()
    rng = random.Random(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    img_by_text, writer_line_texts, writer_ids_shuffled = load_and_align(rng)

    validator = TrOCRValidator()

    writer_glyph_bank, writer_words, diverse_writers = extract_glyph_banks(
        cfg, rng, validator, writer_ids_shuffled, writer_line_texts, img_by_text
    )

    df = generate_dataset(cfg, rng, diverse_writers, writer_words, writer_glyph_bank)

    print(df.head(10))
    print(f"\nDone. {len(df)} pairs written to {cfg.output_dir}/")


if __name__ == "__main__":
    main()

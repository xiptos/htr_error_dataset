"""Run one or more OCR/VLM systems over the writing-error dataset and save
their raw predictions.

Usage:
    python evaluate.py --system tesseract
    python evaluate.py --system qwen3_vl_8b --n 100 --seed 42
    python evaluate.py --system all --n 100                     # loop over every system
    python evaluate.py --system qwen3_vl_8b --prompt-variant diplomatic --n 100
    python evaluate.py --system all --prompt-variant all --n 100

Predictions are appended incrementally to results/predictions/<system>.csv
(or results/predictions/<system>__<variant>.csv when --prompt-variant is
given) with columns id, condition, image_path, gt_text, prediction,
elapsed_sec; already-computed (id, condition) pairs are skipped on re-run,
so an interrupted run can simply be restarted.

--prompt-variant overrides a system's configured prompt with one of the
alternatives in systems.PROMPT_VARIANTS, to compare prompting strategies
against the same system's results under its own default prompt. Systems
with no text prompt (e.g. LightOnOCR) or that are not prompt-driven at all
(Tesseract, Kraken) are skipped for variant runs, since there is nothing to
override.

Each VLM system (backend="mlx_vlm") requires Apple Silicon + mlx-vlm; see
README.md for CPU/CUDA alternatives.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import time
from pathlib import Path

import pandas as pd

from systems import SYSTEMS, SYSTEMS_BY_NAME, PROMPT_VARIANTS
from backends import run_system

BASE = Path("dataset")
RESULTS_DIR = Path("results/predictions")


def build_items(df: pd.DataFrame, base_dir: Path) -> list[tuple[str, str, str, str]]:
    """Returns (id, condition, image_path, ground_truth) for both the clean
    and the corrupted image of every row -- a paired design: each system
    transcribes the same word twice, once clean and once letter-corrupted."""
    items = []
    for _, row in df.iterrows():
        items.append((row["id"], "clean", str(base_dir / row["clean_image"]), row["word"]))
        items.append((row["id"], "error", str(base_dir / row["error_image"]), row["error_word"]))
    return items


def already_done(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    done = pd.read_csv(csv_path)
    return set(zip(done["id"], done["condition"]))


def run_one_system(
    system_name: str, df: pd.DataFrame, base_dir: Path, prompt_variant: str | None = None
) -> None:
    spec = SYSTEMS_BY_NAME[system_name]

    if prompt_variant is not None:
        if spec.backend != "mlx_vlm" or spec.prompt is None:
            print(f"[{system_name}] skipped for variant {prompt_variant!r}: "
                  f"not a prompt-driven mlx_vlm system")
            return
        spec = dataclasses.replace(spec, prompt=PROMPT_VARIANTS[prompt_variant])
        out_path = RESULTS_DIR / f"{system_name}__{prompt_variant}.csv"
    else:
        out_path = RESULTS_DIR / f"{system_name}.csv"

    items = build_items(df, base_dir)

    done = already_done(out_path)
    todo = [(i, c, p, gt) for (i, c, p, gt) in items if (i, c) not in done]
    print(f"[{system_name}] {len(items)} total, {len(done)} already done, {len(todo)} to run")
    if not todo:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new_file = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["id", "condition", "image_path", "gt_text", "prediction", "elapsed_sec"])

        id_to_meta = {(i, c): (p, gt) for (i, c, p, gt) in todo}
        gen_items = [(f"{i}\t{c}", p) for (i, c, p, gt) in todo]  # pack id+condition into one key

        t_start = time.time()
        n = 0
        for pred in run_system(spec, gen_items):
            key = pred.id
            item_id, condition = key.split("\t")
            image_path, gt_text = id_to_meta[(item_id, condition)]
            writer.writerow([item_id, condition, image_path, gt_text, pred.prediction, f"{pred.elapsed_sec:.3f}"])
            f.flush()
            n += 1
            if n % 25 == 0:
                elapsed = time.time() - t_start
                rate = n / elapsed
                remaining = (len(todo) - n) / rate if rate > 0 else float("inf")
                print(f"[{system_name}] {n}/{len(todo)} ({rate:.2f} img/s, ~{remaining/60:.1f} min left)")

    print(f"[{system_name}] done: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, help="system name (see systems.py), or 'all'")
    ap.add_argument("--n", type=int, default=None, help="sample N rows (default: full dataset)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-variant", default=None,
                     help=f"one of {list(PROMPT_VARIANTS)}, or 'all'; omit to use each "
                          f"system's own configured prompt (default behaviour)")
    ap.add_argument("--dataset-dir", type=Path, default=BASE,
                     help="directory containing metadata.csv (default: dataset/)")
    args = ap.parse_args()

    metadata_path = args.dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        raise SystemExit(
            f"{metadata_path} not found. Run `python generate_dataset.py` first."
        )
    df = pd.read_csv(metadata_path)
    if args.n is not None and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    system_names = [s.name for s in SYSTEMS] if args.system == "all" else [args.system]
    for name in system_names:
        if name not in SYSTEMS_BY_NAME:
            raise SystemExit(f"unknown system {name!r}. Known: {[s.name for s in SYSTEMS]}")

    if args.prompt_variant is None:
        variants = [None]
    elif args.prompt_variant == "all":
        variants = list(PROMPT_VARIANTS)
    else:
        if args.prompt_variant not in PROMPT_VARIANTS:
            raise SystemExit(f"unknown prompt variant {args.prompt_variant!r}. "
                              f"Known: {list(PROMPT_VARIANTS)}")
        variants = [args.prompt_variant]

    failures = []
    for variant in variants:
        for name in system_names:
            try:
                run_one_system(name, df, args.dataset_dir, prompt_variant=variant)
            except Exception as e:
                print(f"[{name}/{variant}] FAILED: {e}", file=sys.stderr)
                failures.append((name, variant))
    if failures:
        print(f"\nRuns that failed: {failures}", file=sys.stderr)


if __name__ == "__main__":
    main()

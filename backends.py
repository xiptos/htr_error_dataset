"""Per-backend runners for the systems registered in systems.py.

Each `run_*` function loads its model/engine once and yields a prediction
string for every (id, image_path) pair it is given, so a system is only
ever loaded once per evaluation run regardless of how many images it
processes.
"""

from __future__ import annotations

import glob
import os
import time
import uuid
from typing import Iterable, Iterator, NamedTuple

from systems import SystemSpec


class Prediction(NamedTuple):
    id: str
    prediction: str
    elapsed_sec: float


def _find_kraken_model(model_ref: str) -> str:
    """Resolve a kraken model DOI to a local .mlmodel path, downloading it
    via `kraken get` if it is not already cached."""
    cache_root = os.path.expanduser("~/Library/Application Support/htrmopo")
    hits = glob.glob(os.path.join(cache_root, "**", "*.mlmodel"), recursive=True)
    if hits:
        return hits[0]
    import subprocess
    subprocess.run(["kraken", "get", model_ref], check=True)
    hits = glob.glob(os.path.join(cache_root, "**", "*.mlmodel"), recursive=True)
    if not hits:
        raise RuntimeError(f"could not locate a kraken model after `kraken get {model_ref}`")
    return hits[0]


def run_tesseract(spec: SystemSpec, items: Iterable[tuple[str, str]]) -> Iterator[Prediction]:
    import pytesseract
    from PIL import Image

    for item_id, image_path in items:
        t0 = time.time()
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=spec.model_ref, config="--psm 8").strip()
        yield Prediction(item_id, text, time.time() - t0)


def run_kraken(spec: SystemSpec, items: Iterable[tuple[str, str]]) -> Iterator[Prediction]:
    from PIL import Image
    from kraken.containers import BBoxLine, Segmentation
    from kraken.lib import models
    from kraken import rpred

    model_path = _find_kraken_model(spec.model_ref)
    net = models.load_any(model_path)

    for item_id, image_path in items:
        t0 = time.time()
        im = Image.open(image_path)
        bounds = Segmentation(
            type="bbox",
            text_direction="horizontal-lr",
            imagename=str(image_path),
            script_detection=False,
            lines=[BBoxLine(id=f"_{uuid.uuid4()}", bbox=(0, 0, im.width, im.height))],
        )
        preds = list(rpred.rpred(net, im, bounds))
        text = preds[0].prediction if preds else ""
        yield Prediction(item_id, text, time.time() - t0)


def run_mlx_vlm(spec: SystemSpec, items: Iterable[tuple[str, str]]) -> Iterator[Prediction]:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model, processor = load(spec.model_ref)
    config = load_config(spec.model_ref)
    prompt_text = spec.prompt or ""
    prompt = apply_chat_template(processor, config, prompt_text, num_images=1)

    for item_id, image_path in items:
        t0 = time.time()
        result = generate(
            model, processor, prompt, image=[image_path],
            verbose=False, max_tokens=spec.max_tokens, temperature=0.0,
        )
        yield Prediction(item_id, result.text.strip(), time.time() - t0)


BACKEND_RUNNERS = {
    "tesseract": run_tesseract,
    "kraken": run_kraken,
    "mlx_vlm": run_mlx_vlm,
}


def run_system(spec: SystemSpec, items: list[tuple[str, str]]) -> Iterator[Prediction]:
    runner = BACKEND_RUNNERS[spec.backend]
    yield from runner(spec, items)

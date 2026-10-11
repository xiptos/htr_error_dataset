"""Registry of the OCR / VLM systems to evaluate on the writing-error dataset.

Nine systems total: five VLMs and two traditional recognizers used in
Karamolegkou et al. (2026, arXiv:2605.27750) -- adapted here from printed
text to single handwritten English word crops -- plus two additional VLMs.
Edit SYSTEMS below to add, remove, or reconfigure a system; evaluate.py and
the results notebook read this registry, nothing else needs to change.

Every VLM entry names an exact mlx-community repo + quantisation so results
are reproducible. Traditional systems specify the local engine/model used
instead of a Hugging Face repo. VLM systems run through MLX (Apple Silicon
only, via mlx-vlm) -- see README.md if you need a CUDA/CPU backend instead.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SystemSpec:
    name: str                      # short id, used in filenames and tables
    display_name: str              # human-readable name for tables/figures
    kind: str                      # "vlm" | "traditional"
    backend: str                   # "mlx_vlm" | "tesseract" | "kraken"
    model_ref: str                 # HF repo (mlx_vlm) or local model ref (kraken/tesseract)
    prompt: Optional[str] = None   # None => no text prompt (image-only)
    max_tokens: int = 24
    notes: str = ""


GENERIC_OCR_PROMPT = (
    "Transcribe the handwritten word in this image exactly as written, "
    "character for character. Respond with only the transcribed word and "
    "nothing else."
)

# --- Prompt-variant experiment ---
# Two additional prompt strategies, explicitly instructing the model not to
# normalize/correct what it reads, to test whether prompt engineering alone
# can suppress "repair" behaviour (returning the original word instead of
# the corrupted one actually shown).

DIPLOMATIC_TRANSCRIPTION_PROMPT = (
    "You are performing a diplomatic transcription of a handwritten document\n"
    "for linguistic error analysis. Your task is NOT to reconstruct, correct,\n"
    "or normalize words — it is to report the graphic evidence visible in the\n"
    "image, exactly as written.\n"
    "\n"
    "Rules:\n"
    "1. Transcribe exactly what is visually present, including misspellings,\n"
    "   non-words, and any deviation from standard orthography. A sequence\n"
    "   that is not a real word in the target language must still be\n"
    "   transcribed as written.\n"
    "2. Do not complete, normalize, or replace a character sequence with a\n"
    "   more probable or standard word, even when the written form looks\n"
    "   like a plausible typo for a common word.\n"
    "3. Do not use lexical, grammatical, or semantic plausibility to resolve\n"
    "   visual ambiguity. Base every decision only on the visible strokes.\n"
    "4. Preserve original spacing, capitalization, punctuation, and line\n"
    "   breaks as closely as possible.\n"
    "5. Output only the transcription. Do not add explanations, comments,\n"
    "   corrections, or a \"corrected version\" alongside it.\n"
    "\n"
    "Return only the transcribed text, line by line."
)

SYMBOL_GAME_PROMPT = (
    "We are playing a visual symbol game. The image contains sequences of\n"
    "graphic symbols. Some sequences may resemble words, but they\n"
    "are not necessarily words and do not need to have any meaning.\n"
    "\n"
    "Your score depends only on how accurately you reproduce the visible\n"
    "letters, in exactly the same order in which they appear.\n"
    "\n"
    "Rules:\n"
    "- Do not correct spelling.\n"
    "- Do not complete a sequence.\n"
    "- Do not replace an unusual or meaningless sequence with a more\n"
    "  familiar or probable word.\n"
    "- If the image shows an unusual or meaningless sequence, preserve it\n"
    "  exactly as it appears.\n"
    "- You lose points whenever you replace what is visually present with\n"
    "  a linguistically more plausible sequence.\n"
    "\n"
    "Meaning is irrelevant to this game. Only visual correspondence\n"
    "matters.\n"
    "\n"
    "Examples (these are invented, not real images — for illustration only):\n"
    "Image/string: hopes\n"
    "Correct: hopes\n"
    "Incorrect: hojes\n"
    "\n"
    "Image/string: power\n"
    "Correct: power\n"
    "Incorrect: powed\n"
    "\n"
    "Now reproduce the sequence of letters visible in the image. Output\n"
    "only the sequence."
)

PROMPT_VARIANTS = {
    "diplomatic": DIPLOMATIC_TRANSCRIPTION_PROMPT,
    "symbol_game": SYMBOL_GAME_PROMPT,
}

SYSTEMS: list[SystemSpec] = [
    # --- Traditional recognizers ---
    SystemSpec(
        name="tesseract",
        display_name="Tesseract v5",
        kind="traditional",
        backend="tesseract",
        model_ref="eng",
        prompt=None,
        notes="pytesseract, --psm 8 (single word), since stimuli are "
              "isolated word crops.",
    ),
    SystemSpec(
        name="kraken",
        display_name="Kraken (Rev City English HTR)",
        kind="traditional",
        backend="kraken",
        model_ref="10.5281/zenodo.19238205",  # revcity_july_2025.mlmodel
        prompt=None,
        notes="Closest available public English HTR model; recognition "
              "runs on a whole-crop bbox (no baseline segmentation), which "
              "the model itself warns degrades accuracy -- a documented "
              "limitation, not a bug.",
    ),
    # --- VLMs ---
    SystemSpec(
        name="lightonocr_1b",
        display_name="LightOnOCR-1B",
        kind="vlm",
        backend="mlx_vlm",
        model_ref="mlx-community/LightOnOCR-1B-1025-bf16",
        prompt=None,  # runs image-only with no text prompt
        max_tokens=64,
    ),
    SystemSpec(
        name="deepseek_ocr",
        display_name="DeepSeek-OCR",
        kind="vlm",
        backend="mlx_vlm",
        model_ref="mlx-community/DeepSeek-OCR-bf16",
        prompt="Free OCR.",  # the model card's default prompt
        max_tokens=64,
        notes="As of mlx-vlm 0.7.1, loading this repo raises "
              "'Unrecognized processing class': "
              "mlx_vlm.models.deepseekocr.processing_deepseekocr registers "
              "an AutoProcessor patch for model_type=='deepseekocr', but "
              "the patch is not active when load() calls "
              "AutoProcessor.from_pretrained, even if the submodule is "
              "force-imported first. Reproduced on both the -bf16 and "
              "-2-bf16 conversions. Looks like an upstream bug in "
              "mlx-vlm's chained AutoProcessor monkey-patch "
              "(github.com/Blaizzy/mlx-vlm) -- retry once a fix ships. "
              "The official transformers path (deepseek-ai/DeepSeek-OCR) "
              "is CUDA-only and would need local patching to run on Apple "
              "Silicon.",
    ),
    SystemSpec(
        name="qwen3_vl_2b",
        display_name="Qwen3-VL-2B-Instruct",
        kind="vlm",
        backend="mlx_vlm",
        model_ref="mlx-community/Qwen3-VL-2B-Instruct-bf16",
        prompt=GENERIC_OCR_PROMPT,
        max_tokens=24,
    ),
    SystemSpec(
        name="olmocr_2_7b",
        display_name="olmOCR-2-7B",
        kind="vlm",
        backend="mlx_vlm",
        model_ref="mlx-community/olmOCR-2-7B-1025-bf16",
        prompt=GENERIC_OCR_PROMPT,
        max_tokens=64,
        notes="The olmOCR toolkit's own prompt assumes full-page anchoring "
              "metadata, not applicable to isolated word crops, so a "
              "generic transcription instruction is used instead.",
    ),
    SystemSpec(
        name="qwen3_vl_8b",
        display_name="Qwen3-VL-8B-Instruct",
        kind="vlm",
        backend="mlx_vlm",
        model_ref="mlx-community/Qwen3-VL-8B-Instruct-bf16",
        prompt=GENERIC_OCR_PROMPT,
        max_tokens=24,
    ),
    SystemSpec(
        name="qwen3_vl_30b_a3b",
        display_name="Qwen3-VL-30B-A3B-Instruct",
        kind="vlm",
        backend="mlx_vlm",
        model_ref="mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit",
        prompt=GENERIC_OCR_PROMPT,
        max_tokens=24,
        notes="Mixture-of-experts scale-up of Qwen3-VL-8B (30B total / "
              "~3B active params).",
    ),
    SystemSpec(
        name="glm_4_6v_flash",
        display_name="GLM-4.6V-Flash",
        kind="vlm",
        backend="mlx_vlm",
        model_ref="mlx-community/GLM-4.6V-Flash-8bit",
        prompt=GENERIC_OCR_PROMPT,
        max_tokens=24,
    ),
    SystemSpec(
        name="glm_4_6v",
        display_name="GLM-4.6V",
        kind="vlm",
        backend="mlx_vlm",
        model_ref="mlx-community/GLM-4.6V-nvfp4",
        prompt=GENERIC_OCR_PROMPT,
        max_tokens=24,
        notes="Full (non-Flash) GLM-4.6V, nvfp4 quantisation.",
    ),
]

SYSTEMS_BY_NAME = {s.name: s for s in SYSTEMS}

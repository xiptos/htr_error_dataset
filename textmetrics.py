"""Shared text-comparison utilities.

Used by dataset generation (TrOCR segmentation validation) and by the EDA /
results notebooks (minimal-pair detection, prediction scoring).
"""

from __future__ import annotations

import re

_ALPHA_RUN = re.compile(r"[A-Za-z]+")
_LATEX_DELIM = re.compile(r"\\[\[\]()]|\$")
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_wordset_cache: set[str] | None = None


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


def strip_latex_math(raw: str) -> str:
    """Unwrap a transcription that was emitted as LaTeX math markup.

    DeepSeek-OCR often renders an isolated word crop as display/inline math
    with the letters spaced out, e.g. ``\\[g o n e\\]`` for "gone". When math
    delimiters are present this drops them, drops LaTeX commands and
    grouping/script characters, and re-joins the letters if what remains is
    a run of single characters. Outputs without math delimiters are returned
    unchanged.
    """
    if not _LATEX_DELIM.search(raw):
        return raw
    text = _LATEX_DELIM.sub(" ", raw)
    text = _LATEX_COMMAND.sub(" ", text)
    text = re.sub(r"[{}_^]", "", text)
    tokens = text.split()
    if tokens and all(len(t) == 1 for t in tokens):
        return "".join(tokens)
    return " ".join(tokens)


# System-specific clean-ups, selected per system via SystemSpec.cleanup.
PREDICTION_CLEANERS = {
    "latex_math": strip_latex_math,
}


def normalize_prediction(raw: str, cleanup: str | None = None) -> str:
    """Extract the first alphabetic token from a raw model output, lowercased.

    VLMs occasionally wrap the answer in punctuation/whitespace despite being
    asked for the bare word; traditional engines rarely do, so this is a
    no-op for them in practice. `cleanup` names an optional system-specific
    clean-up from PREDICTION_CLEANERS, applied first.
    """
    if not isinstance(raw, str):
        return ""
    if cleanup is not None:
        raw = PREDICTION_CLEANERS[cleanup](raw)
    m = _ALPHA_RUN.search(raw)
    return m.group(0).lower() if m else ""


def load_english_wordset() -> set[str]:
    """Lowercased English dictionary, from nltk's `words` corpus (cached)."""
    global _wordset_cache
    if _wordset_cache is None:
        import nltk
        from nltk.corpus import words as nltk_words

        nltk.download("words", quiet=True)
        _wordset_cache = {w.lower() for w in nltk_words.words()}
    return _wordset_cache

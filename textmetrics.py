"""Shared text-comparison utilities.

Used by dataset generation (TrOCR segmentation validation) and by the EDA /
results notebooks (minimal-pair detection, prediction scoring).
"""

from __future__ import annotations

import re

_ALPHA_RUN = re.compile(r"[A-Za-z]+")
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


def normalize_prediction(raw: str) -> str:
    """Extract the first alphabetic token from a raw model output, lowercased.

    VLMs occasionally wrap the answer in punctuation/whitespace despite being
    asked for the bare word; traditional engines rarely do, so this is a
    no-op for them in practice.
    """
    if not isinstance(raw, str):
        return ""
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

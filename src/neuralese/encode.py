"""Deterministic local text encoders.

``char_trigram`` hashes each row on its own. ``word_sentence_svd`` builds a
TF-IDF word-sentence matrix from the texts passed in, then takes a truncated
SVD. Both are pure functions of that text. Neither is a model hidden state,
a logit lens, or an SAE, and neither reads topic labels.
"""
from __future__ import annotations

import math
import re
from typing import List, Sequence

import numpy as np

from neuralese.adapters import hashed_ngram_vector

ENCODER_CHAR_TRIGRAM = "char_trigram"
ENCODER_WORD_SENTENCE_SVD = "word_sentence_svd"
ENCODERS = (ENCODER_CHAR_TRIGRAM, ENCODER_WORD_SENTENCE_SVD)

# Closed class of function words. Not topic vocabulary and not fit to a corpus.
FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "was",
        "were",
        "been",
        "this",
        "that",
        "with",
        "while",
        "after",
        "before",
        "from",
        "its",
        "as",
        "at",
        "by",
        "it",
        "had",
        "have",
        "has",
        "they",
        "their",
        "them",
        "his",
        "her",
        "she",
        "he",
        "we",
        "you",
        "but",
        "so",
        "if",
        "when",
        "into",
        "over",
        "under",
        "than",
        "then",
        "there",
        "here",
        "not",
        "no",
        "do",
        "did",
        "does",
    }
)

_TOKEN = re.compile(r"[a-z0-9']+")


def content_tokens(text: str) -> List[str]:
    """Lowercase word tokens with function words and one-character tokens removed."""
    return [
        token
        for token in _TOKEN.findall(text.lower())
        if token not in FUNCTION_WORDS and len(token) > 1
    ]


def word_sentence_svd(texts: Sequence[str], *, rank: int = 8) -> List[List[float]]:
    """TF-IDF word-sentence SVD.

    The matrix is built only from ``texts``, in order. Term frequency is the
    raw count. IDF is ``ln(N / df) + 1`` with document frequency ``df``.
    Rows are L2-normalized, factored with a truncated SVD of width ``rank``,
    scaled by the singular values, and L2-normalized again. Each left singular
    vector is signed so its largest-magnitude coordinate is positive.

    This is a deterministic local encoder. It is not a hidden state.
    """
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    rows = [content_tokens(text or "") for text in texts]
    n_rows = len(rows)
    if n_rows == 0:
        return []
    document_frequency: dict[str, int] = {}
    for tokens in rows:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    if not document_frequency:
        return [[0.0] * rank for _ in rows]

    vocabulary = sorted(document_frequency)
    index = {token: position for position, token in enumerate(vocabulary)}
    matrix = np.zeros((n_rows, len(vocabulary)), dtype=np.float64)
    for row_index, tokens in enumerate(rows):
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            idf = math.log(n_rows / document_frequency[token]) + 1.0
            matrix[row_index, index[token]] = float(count) * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)

    width = min(rank, matrix.shape[0], matrix.shape[1])
    left, singular, _right = np.linalg.svd(matrix, full_matrices=False)
    left = np.array(left[:, :width], dtype=np.float64, copy=True)
    singular = np.asarray(singular[:width], dtype=np.float64)
    for column in range(width):
        pivot = int(np.argmax(np.abs(left[:, column])))
        if left[pivot, column] < 0.0:
            left[:, column] *= -1.0
    embedded = left * singular
    if width < rank:
        embedded = np.pad(embedded, ((0, 0), (0, rank - width)))
    row_norms = np.linalg.norm(embedded, axis=1, keepdims=True)
    embedded = np.divide(
        embedded,
        row_norms,
        out=np.zeros_like(embedded),
        where=row_norms > 0.0,
    )
    return embedded.tolist()


def embed_texts(
    texts: Sequence[str],
    encoder: str,
    *,
    rank: int = 8,
    dim: int = 32,
    ngram: int = 3,
) -> List[List[float]]:
    """Embed ``texts`` with a named local encoder."""
    if encoder == ENCODER_CHAR_TRIGRAM:
        return [hashed_ngram_vector(text or "", dim=dim, n=ngram) for text in texts]
    if encoder == ENCODER_WORD_SENTENCE_SVD:
        return word_sentence_svd(texts, rank=rank)
    known = ", ".join(ENCODERS)
    raise ValueError(f"unknown encoder {encoder!r}; expected one of {known}")

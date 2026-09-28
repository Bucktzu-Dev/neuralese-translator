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

# Glasgow Information Retrieval Group English stop list, as shipped by
# scikit-learn (sklearn.feature_extraction._stop_words). Used whole: it is
# not edited against topic names and it is not fit by document frequency on
# a corpus. The list is broader than closed-class function words; content-like
# entries such as "fire" and "bill" stay because the list is not trimmed.
# "did" and "does" are kept from the previous closed-class list. The Glasgow
# list has "do" and "done" and not those two forms.
_GLASGOW_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "across",
        "after",
        "afterwards",
        "again",
        "against",
        "all",
        "almost",
        "alone",
        "along",
        "already",
        "also",
        "although",
        "always",
        "am",
        "among",
        "amongst",
        "amoungst",
        "amount",
        "an",
        "and",
        "another",
        "any",
        "anyhow",
        "anyone",
        "anything",
        "anyway",
        "anywhere",
        "are",
        "around",
        "as",
        "at",
        "back",
        "be",
        "became",
        "because",
        "become",
        "becomes",
        "becoming",
        "been",
        "before",
        "beforehand",
        "behind",
        "being",
        "below",
        "beside",
        "besides",
        "between",
        "beyond",
        "bill",
        "both",
        "bottom",
        "but",
        "by",
        "call",
        "can",
        "cannot",
        "cant",
        "co",
        "con",
        "could",
        "couldnt",
        "cry",
        "de",
        "describe",
        "detail",
        "do",
        "done",
        "down",
        "due",
        "during",
        "each",
        "eg",
        "eight",
        "either",
        "eleven",
        "else",
        "elsewhere",
        "empty",
        "enough",
        "etc",
        "even",
        "ever",
        "every",
        "everyone",
        "everything",
        "everywhere",
        "except",
        "few",
        "fifteen",
        "fifty",
        "fill",
        "find",
        "fire",
        "first",
        "five",
        "for",
        "former",
        "formerly",
        "forty",
        "found",
        "four",
        "from",
        "front",
        "full",
        "further",
        "get",
        "give",
        "go",
        "had",
        "has",
        "hasnt",
        "have",
        "he",
        "hence",
        "her",
        "here",
        "hereafter",
        "hereby",
        "herein",
        "hereupon",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "however",
        "hundred",
        "i",
        "ie",
        "if",
        "in",
        "inc",
        "indeed",
        "interest",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "keep",
        "last",
        "latter",
        "latterly",
        "least",
        "less",
        "ltd",
        "made",
        "many",
        "may",
        "me",
        "meanwhile",
        "might",
        "mill",
        "mine",
        "more",
        "moreover",
        "most",
        "mostly",
        "move",
        "much",
        "must",
        "my",
        "myself",
        "name",
        "namely",
        "neither",
        "never",
        "nevertheless",
        "next",
        "nine",
        "no",
        "nobody",
        "none",
        "noone",
        "nor",
        "not",
        "nothing",
        "now",
        "nowhere",
        "of",
        "off",
        "often",
        "on",
        "once",
        "one",
        "only",
        "onto",
        "or",
        "other",
        "others",
        "otherwise",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "part",
        "per",
        "perhaps",
        "please",
        "put",
        "rather",
        "re",
        "same",
        "see",
        "seem",
        "seemed",
        "seeming",
        "seems",
        "serious",
        "several",
        "she",
        "should",
        "show",
        "side",
        "since",
        "sincere",
        "six",
        "sixty",
        "so",
        "some",
        "somehow",
        "someone",
        "something",
        "sometime",
        "sometimes",
        "somewhere",
        "still",
        "such",
        "system",
        "take",
        "ten",
        "than",
        "that",
        "the",
        "their",
        "them",
        "themselves",
        "then",
        "thence",
        "there",
        "thereafter",
        "thereby",
        "therefore",
        "therein",
        "thereupon",
        "these",
        "they",
        "thick",
        "thin",
        "third",
        "this",
        "those",
        "though",
        "three",
        "through",
        "throughout",
        "thru",
        "thus",
        "to",
        "together",
        "too",
        "top",
        "toward",
        "towards",
        "twelve",
        "twenty",
        "two",
        "un",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "very",
        "via",
        "was",
        "we",
        "well",
        "were",
        "what",
        "whatever",
        "when",
        "whence",
        "whenever",
        "where",
        "whereafter",
        "whereas",
        "whereby",
        "wherein",
        "whereupon",
        "wherever",
        "whether",
        "which",
        "while",
        "whither",
        "who",
        "whoever",
        "whole",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "yet",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)
FUNCTION_WORDS = _GLASGOW_STOP_WORDS | frozenset({"did", "does"})

_TOKEN = re.compile(r"[a-z0-9']+")


def content_tokens(text: str) -> List[str]:
    """Lowercase word tokens with stop words and one-character tokens removed."""
    return [
        token
        for token in _TOKEN.findall(text.lower())
        if token not in FUNCTION_WORDS and len(token) > 1
    ]


def word_sentence_svd(texts: Sequence[str], *, rank: int = 8) -> List[List[float]]:
    """TF-IDF word-sentence SVD.

    The matrix is built only from ``texts``, in order. Tokens in
    ``FUNCTION_WORDS`` are dropped before counting. Term frequency is the
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

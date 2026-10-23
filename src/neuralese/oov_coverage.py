"""Train-independent text features on the frozen generalization split.

Production ``word_sentence_svd`` stays a pure function of whatever texts it
is given. This module does not replace it. Arm 1 refits nothing of its own:
it calls the train-only encoder from the encoder-isolation trial. Arms 2 and
3 hash each row with a fixed width and a fixed hash. They fit no vocabulary,
no document frequency, no IDF, and no SVD. Arm 4 is a later call. It runs
only when the arm-2 and arm-3 tables already justify it and the written note
cites those tables.

Admission constants below are the gates. They are fixed before any hashed-arm
purity is computed. A hashed arm that only removes zero vectors is not admitted.
"""
from __future__ import annotations

import hashlib
import itertools
import re
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neuralese.adapters import hashed_ngram_vector
from neuralese.clustering_generalization import (
    CLUSTER_TIME_REPEATS,
    FIXTURE_SPECS,
    LABEL_SHUFFLE_SEED,
    ROW_SHUFFLE_SEED,
    TIME_RUNS,
    _peak_bytes,
    _per_call_median_ms,
    _production_spec,
    load_fixture,
)
from neuralese.clustering_isolation import (
    _assign,
    _fit_algorithm,
    _prototype_residual,
)
from neuralese.encode import content_tokens, word_sentence_svd
from neuralese.encoder_isolation import (
    FROZEN_SPLIT_SHA256,
    RANK,
    _git_revision,
    _inductive_controls,
    _score_inductive,
    fit_train_only_encoder,
    numerical_library,
    split_fingerprint,
    transform_train_only,
)
from neuralese.eval_control_integrity import score_against_map, train_only_majority_map
from neuralese.factorization import pin_blas_threads

# Status of the trial this experiment starts from. Not this run's decision.
STATUS = "RESEARCH_ONLY"
BASELINE_COMMIT = "9040bb4"
ENTERING_VERDICT = "UNRESOLVED"
ENTERING_RECOMMENDATION = "NONE"
RUNTIME_AUTHORITY = "RUNTIME_NOT_AUTHORIZED"

# Admission gates. Fixed before any hashed-arm purity is computed.
# An arm is RECOMMENDED for a later isolated encoder PR only when every
# generalization fixture meets every gate. Coverage by itself is not a gate.
PURITY_ABOVE_ARM1_GATE = (
    "eval_size_weighted_purity_strictly_greater_than_arm1_inductive_eval_purity"
)
PURITY_ABOVE_BASE_GATE = (
    "eval_size_weighted_purity_strictly_greater_than_eval_majority_base_rate"
)
LABEL_CONTROL_GATE = (
    "shuffled_label_purity_strictly_below_real_eval_purity"
)
ROW_CONTROL_GATE = (
    "shuffled_row_purity_strictly_below_real_eval_purity"
)
ZERO_RATE_GATE = "zero_vector_rate_strictly_below_arm1"
DISTINCTION_GATE = "pairwise_distinction_retention_at_least_0.95"
RUNTIME_GATE = "feature_extraction_plus_fit_median_at_most_5x_arm1_embed_median"
DISTINCTION_MIN = 0.95
RUNTIME_FACTOR = 5.0

# Pairwise distinction retention is the fraction of distinct eval texts whose
# vector equals no other distinct eval text. Shared zero vectors count as one
# vector. The bar above is this fraction. It is not the fraction of text pairs.
ARM_RECOMMENDED = "RECOMMENDED"
ARM_EXPERIMENTAL = "EXPERIMENTAL"
ARM_UNRESOLVED = "UNRESOLVED"
ARM_CONTROL = "CONTROL"
ARM_STATUSES = (ARM_RECOMMENDED, ARM_EXPERIMENTAL, ARM_UNRESOLVED, ARM_CONTROL)

DECISION_NO_ADMISSION = "NO_ADMISSION"
DECISION_UNRESOLVED = "UNRESOLVED"
DECISION_RECOMMENDED = "RECOMMENDED"
DECISION_STATUSES = (DECISION_NO_ADMISSION, DECISION_UNRESOLVED, DECISION_RECOMMENDED)
PATTERN_COVERAGE_WITHOUT_DISTINCTION = "COVERAGE_WITHOUT_DISTINCTION"
PATTERN_CONTROL = "CONTROL_DID_NOT_FALL"
PATTERN_MIXED = "MIXED"
PATTERN_ADMITTED = "ADMITTED_ARM"
PATTERN_BASELINE = "BASELINE_NOT_REPRODUCED"

# Hash width and character order. Chosen before any hashed-arm purity.
# 32 is the width ``hashed_ngram_vector`` already uses in this repository.
HASH_DIM = 32
CHAR_NGRAM = 3
WORD_TOKEN_PATTERN = "[a-z0-9']+"

ARM1_ID = "tfidf_svd"
ARM2_ID = "char_hash"
ARM3_ID = "word_hash"
ARM4_ID = "combined_hash"
HASH_ARM_IDS = (ARM2_ID, ARM3_ID, ARM4_ID)

# The note must contain this marker, and the precondition sentence, before
# arm 4 is allowed to embed a generalization row.
ARM4_RUN_MARKER = "ARM4_JUSTIFICATION_PRESENT"

# Control seeds. The generalization trial's constants. Not retuned.
INDUCTIVE_LABEL_SHUFFLE_SEED = LABEL_SHUFFLE_SEED
INDUCTIVE_ROW_SHUFFLE_SEED = ROW_SHUFFLE_SEED

# Published inductive numbers from the encoder-isolation trial. Arm 1 is
# recomputed and compared to this table. The table is not a purity target
# for the hashed arms; it is the control those arms are measured against.
PUBLISHED_ARM1: Dict[str, Dict[str, int]] = {
    "uneven": {
        "eval_hits": 5,
        "n_eval": 10,
        "base_hits": 5,
        "zero_rows": 3,
        "label_hits": 5,
        "row_hits": 4,
        "train_hits": 26,
        "n_train": 40,
        "occurrences": 72,
        "absent_occurrences": 56,
        "absent_types": 55,
    },
    "overlapping": {
        "eval_hits": 3,
        "n_eval": 9,
        "base_hits": 2,
        "zero_rows": 3,
        "label_hits": 0,
        "row_hits": 1,
        "train_hits": 27,
        "n_train": 39,
        "occurrences": 66,
        "absent_occurrences": 52,
        "absent_types": 51,
    },
    "ambiguous": {
        "eval_hits": 4,
        "n_eval": 9,
        "base_hits": 3,
        "zero_rows": 4,
        "label_hits": 1,
        "row_hits": 1,
        "train_hits": 22,
        "n_train": 39,
        "occurrences": 58,
        "absent_occurrences": 44,
        "absent_types": 43,
    },
    "balanced": {
        "eval_hits": 3,
        "n_eval": 16,
        "base_hits": 2,
        "zero_rows": 8,
        "label_hits": 2,
        "row_hits": 1,
        "train_hits": 33,
        "n_train": 64,
        "occurrences": 94,
        "absent_occurrences": 74,
        "absent_types": 72,
    },
}

# Synthetic strings. They are not read from the generalization files.
# The collision search walks this alphabet in product order and stops at
# the first shared bucket. The bound is fixed before any purity is computed.
COLLISION_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
COLLISION_TOKEN_LENGTH = 3
COLLISION_SEARCH_LIMIT = 4096
DISJOINT_SEARCH_LIMIT = 256
REPEATED_TEXT = "qxvblort mzzpfkye"
NONSENSE_TEXTS: Tuple[str, ...] = (
    "qxvblort",
    "mzzpfkye",
    "jjwdnquv",
    "htrgclox",
    "bnvysaei",
    "wplkdzuc",
    "feyomqar",
    "tcuhjnsx",
)
UNSEEN_TRAIN_TEXTS: Tuple[str, ...] = (
    "harbor skiff crossed the tide beside the wharf",
    "kiln glaze cooled beside the wheel",
    "orchard ladder reached the ripe fruit",
    "apiary smoke calmed the hive frames",
)
UNSEEN_EVAL_TEXTS: Tuple[str, ...] = (
    "qxvblort mzzpfkye",
    "jjwdnquv htrgclox",
    "bnvysaei wplkdzuc",
    "feyomqar tcuhjnsx",
)

_WORD = re.compile(WORD_TOKEN_PATTERN)


def decide_arm(
    *,
    fixtures: Sequence[Mapping],
    controls_fall: bool,
    repeated_input_identical: bool,
) -> str:
    """Apply the admission rule. Does not embed text and does not compute a purity.

    A control that does not fall on every fixture yields ``UNRESOLVED``.
    A representation that maps one string to two vectors yields ``UNRESOLVED``.
    Every fixture clearing every gate yields ``RECOMMENDED``. Any other
    scored arm yields ``EXPERIMENTAL``.
    """
    if not fixtures or not repeated_input_identical or not controls_fall:
        return ARM_UNRESOLVED
    if all(bool(row["admitted"]) for row in fixtures):
        return ARM_RECOMMENDED
    return ARM_EXPERIMENTAL


def arm_pattern(fixtures: Sequence[Mapping]) -> str:
    """Name the shared failure shape. Does not decide admission.

    ``COVERAGE_WITHOUT_DISTINCTION`` is only the case where the zero-vector
    rate fell and pairwise distinction still missed its bar. A zero-vector
    drop whose distinct texts keep distinct vectors, while purity misses, is
    ``PURITY_MISS``.
    """
    if not fixtures:
        return PATTERN_BASELINE
    zero_cut = all(bool(row["zero_below_arm1"]) for row in fixtures)
    distinction_miss = any(not bool(row["distinction_ok"]) for row in fixtures)
    purity_miss = any(
        (not bool(row["above_arm1"])) or (not bool(row["above_base"])) for row in fixtures
    )
    if zero_cut and distinction_miss:
        return PATTERN_COVERAGE_WITHOUT_DISTINCTION
    if zero_cut and purity_miss:
        return "PURITY_MISS"
    if any(not bool(row["admitted"]) for row in fixtures):
        return "GATE_MISS"
    return PATTERN_ADMITTED


def decide_trial(
    arm_rows: Sequence[Mapping],
    *,
    split_reproduced: bool,
    baseline_reproduced: bool,
) -> dict:
    """Choose the trial decision from arm statuses that are already computed.

    ``NO_ADMISSION`` is the decision when every hashed arm's controls fell
    and no hashed arm cleared the rule. A control that did not fall, a split
    that did not reproduce, or a baseline that did not match the published
    inductive table stays ``UNRESOLVED``. Arms that disagree about that
    coverage pattern also stay ``UNRESOLVED``. The decision is never a
    pass of the encoder-isolation purity rule.
    """
    hashed = [row for row in arm_rows if bool(row.get("candidate"))]
    recommended = [
        str(row["arm_id"]) for row in hashed if row.get("status") == ARM_RECOMMENDED
    ]
    if not split_reproduced or not baseline_reproduced:
        return {
            "decision_status": DECISION_UNRESOLVED,
            "pattern": PATTERN_BASELINE,
            "recommended_arms": [],
            "recommendation": ENTERING_RECOMMENDATION,
            "inductive_pass": False,
        }
    if recommended:
        return {
            "decision_status": DECISION_RECOMMENDED,
            "pattern": PATTERN_ADMITTED,
            "recommended_arms": recommended,
            "recommendation": ",".join(recommended),
            "inductive_pass": False,
        }
    if any(row.get("status") == ARM_UNRESOLVED for row in hashed):
        return {
            "decision_status": DECISION_UNRESOLVED,
            "pattern": PATTERN_CONTROL,
            "recommended_arms": [],
            "recommendation": ENTERING_RECOMMENDATION,
            "inductive_pass": False,
        }
    patterns = [str(row.get("pattern")) for row in hashed]
    if len(set(patterns)) > 1:
        return {
            "decision_status": DECISION_UNRESOLVED,
            "pattern": PATTERN_MIXED,
            "recommended_arms": [],
            "recommendation": ENTERING_RECOMMENDATION,
            "inductive_pass": False,
        }
    pattern = patterns[0] if patterns else PATTERN_BASELINE
    return {
        "decision_status": DECISION_NO_ADMISSION,
        "pattern": pattern,
        "recommended_arms": [],
        "recommendation": ENTERING_RECOMMENDATION,
        "inductive_pass": False,
    }


def arm4_precondition(
    arm2_fixtures: Sequence[Mapping],
    arm3_fixtures: Sequence[Mapping],
    arm2_synthetic: Mapping,
    arm3_synthetic: Mapping,
) -> dict:
    """Whether arm 4 may be scored. Uses arms 2 and 3 only.

    One of those arms must cut the zero-vector rate against arm 1 on every
    generalization fixture. The other must retain a synthetic distinction
    the first arm loses. Repeated-input is not such a distinction: identical
    vectors on that case are required.
    """

    def _cuts(fixtures: Sequence[Mapping]) -> bool:
        return bool(fixtures) and all(bool(row["zero_below_arm1"]) for row in fixtures)

    def _lost(synthetic: Mapping) -> List[str]:
        lost = []
        for name, case in synthetic.items():
            if name == "repeated_input":
                continue
            collapsed = bool(case["collapsed"])
            retention = float(case["pairwise_distinction_retention"])
            if collapsed or retention < 1.0:
                lost.append(str(name))
        return lost

    def _retained(synthetic: Mapping) -> List[str]:
        kept = []
        for name, case in synthetic.items():
            if name == "repeated_input":
                continue
            if (
                not bool(case["collapsed"])
                and float(case["pairwise_distinction_retention"]) >= 1.0
                and int(case["n_distinct"]) >= 2
            ):
                kept.append(str(name))
        return kept

    def _kept_losses(first: Mapping, other: Mapping) -> List[str]:
        return sorted(set(_lost(first)) & set(_retained(other)))

    char_cuts = _cuts(arm2_fixtures)
    word_cuts = _cuts(arm3_fixtures)
    word_keeps = _kept_losses(arm2_synthetic, arm3_synthetic)
    char_keeps = _kept_losses(arm3_synthetic, arm2_synthetic)
    char_then_word = bool(char_cuts and word_keeps)
    word_then_char = bool(word_cuts and char_keeps)
    justified = bool(char_then_word or word_then_char)
    if char_then_word and word_then_char:
        sentence = (
            f"{ARM2_ID} and {ARM3_ID} each cut the zero-vector rate versus {ARM1_ID} "
            "on every generalization fixture, and each retains a synthetic distinction "
            f"the other loses ({ARM2_ID} loses {','.join(_lost(arm2_synthetic))}; "
            f"{ARM3_ID} loses {','.join(_lost(arm3_synthetic))})."
        )
    elif char_then_word:
        sentence = (
            f"{ARM2_ID} cuts the zero-vector rate versus {ARM1_ID} on every "
            f"generalization fixture, and {ARM3_ID} retains the synthetic distinction "
            f"{','.join(word_keeps)} that {ARM2_ID} loses."
        )
    elif word_then_char:
        sentence = (
            f"{ARM3_ID} cuts the zero-vector rate versus {ARM1_ID} on every "
            f"generalization fixture, and {ARM2_ID} retains the synthetic distinction "
            f"{','.join(char_keeps)} that {ARM3_ID} loses."
        )
    else:
        sentence = (
            f"Arm 4 is not justified. {ARM2_ID} zero-rate cut on every fixture is "
            f"{str(char_cuts).lower()}. {ARM3_ID} zero-rate cut on every fixture is "
            f"{str(word_cuts).lower()}. Distinctions {ARM2_ID} loses and {ARM3_ID} retains: "
            f"{', '.join(word_keeps) if word_keeps else 'none'}. "
            f"Distinctions {ARM3_ID} loses and {ARM2_ID} retains: "
            f"{', '.join(char_keeps) if char_keeps else 'none'}."
        )
    return {
        "justified": justified,
        "char_cuts_zero_rate": char_cuts,
        "word_cuts_zero_rate": word_cuts,
        "word_retains_distinction_char_loses": word_keeps,
        "char_retains_distinction_word_loses": char_keeps,
        "justification_sentence": sentence,
    }


def hashed_word_vector(text: str, dim: int = HASH_DIM) -> np.ndarray:
    """Signed hashed word vector. The tokenizer is a fixed regex.

    No vocabulary is stored. No document frequency is counted. Each token
    adds one signed count in a bucket chosen by SHA-256, the same construction
    ``hashed_ngram_vector`` uses for character grams. The row is L2-normalized
    from its own norm.
    """
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")
    vec = np.zeros(dim, dtype=np.float64)
    for token in _WORD.findall((text or "").lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign
    return _l2_normalize(vec)


def combined_vector(text: str, dim: int = HASH_DIM, n: int = CHAR_NGRAM) -> np.ndarray:
    """Per-row sum of the character hash and the word hash, then L2.

    Both inputs are already row-normalized. Their sum is normalized from
    this row alone. Nothing is fit on a corpus.
    """
    char = np.asarray(hashed_ngram_vector(text or "", dim=dim, n=n), dtype=np.float64)
    word = hashed_word_vector(text or "", dim=dim)
    return _l2_normalize(char + word)


def embed_char(texts: Sequence[str]) -> np.ndarray:
    """Character n-gram hash. ``hashed_ngram_vector`` already L2-normalizes."""
    if not texts:
        return np.zeros((0, HASH_DIM), dtype=np.float64)
    return np.ascontiguousarray(
        [hashed_ngram_vector(text or "", dim=HASH_DIM, n=CHAR_NGRAM) for text in texts],
        dtype=np.float64,
    )


def embed_word(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, HASH_DIM), dtype=np.float64)
    return np.ascontiguousarray(
        np.stack([hashed_word_vector(text or "", dim=HASH_DIM) for text in texts]),
        dtype=np.float64,
    )


def embed_combined(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, HASH_DIM), dtype=np.float64)
    return np.ascontiguousarray(
        np.stack([combined_vector(text or "", dim=HASH_DIM, n=CHAR_NGRAM) for text in texts]),
        dtype=np.float64,
    )


def distinction_retention(texts: Sequence[str], matrix: np.ndarray) -> dict:
    """Fraction of distinct texts whose vector is unique among those texts.

    Copies of one text must already share a vector. They count once.
    ``collapsed`` is true when more than one distinct text exists and every
    one of them has the same vector. A nonzero collapse is coverage with no
    distinction.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if len(texts) != int(matrix.shape[0]):
        raise ValueError("texts and matrix differ in length")
    vectors: Dict[str, bytes] = {}
    order: List[str] = []
    for text, row in zip(texts, matrix):
        blob = np.ascontiguousarray(row, dtype=np.float64).tobytes()
        if text in vectors:
            if vectors[text] != blob:
                raise RuntimeError("one text produced two vectors")
            continue
        vectors[text] = blob
        order.append(text)
    counts: Dict[bytes, int] = {}
    for blob in vectors.values():
        counts[blob] = counts.get(blob, 0) + 1
    n_distinct = len(order)
    unique_texts = sum(1 for text in order if counts[vectors[text]] == 1)
    n_unique_vectors = len(counts)
    pair_total = n_distinct * (n_distinct - 1) // 2
    pair_collisions = sum(count * (count - 1) // 2 for count in counts.values())
    collapsed = bool(n_distinct > 1 and n_unique_vectors == 1)
    nonzero_collapse = bool(
        collapsed and n_distinct > 0 and not _is_zero_bytes(vectors[order[0]], matrix.shape[1])
    )
    return {
        "n_rows": int(len(texts)),
        "n_distinct": int(n_distinct),
        "n_unique_vectors": int(n_unique_vectors),
        "unique_text_count": int(unique_texts),
        "pairwise_distinction_retention": (
            float(unique_texts / n_distinct) if n_distinct else 0.0
        ),
        "unique_vector_fraction": (
            float(n_unique_vectors / n_distinct) if n_distinct else 0.0
        ),
        "distinct_pair_fraction": (
            float((pair_total - pair_collisions) / pair_total) if pair_total else 1.0
        ),
        "collapsed": collapsed,
        "nonzero_collapse": nonzero_collapse,
    }


def repeated_input_case(embed: Callable[[Sequence[str]], np.ndarray]) -> dict:
    """The same string twice. The two rows must be identical."""
    matrix = embed([REPEATED_TEXT, REPEATED_TEXT])
    identical = bool(np.array_equal(matrix[0], matrix[1]))
    return {
        "text": REPEATED_TEXT,
        "identical": identical,
        "zero": bool(np.all(matrix[0] == 0.0)),
        "n_copies": 2,
        "collapsed": False,
        "n_distinct": 1,
        "pairwise_distinction_retention": 1.0 if identical else 0.0,
    }


def nonsense_case(embed: Callable[[Sequence[str]], np.ndarray]) -> dict:
    """Distinct strings with no shared token and no topic label."""
    _assert_nonsense_shape()
    matrix = embed(NONSENSE_TEXTS)
    report = distinction_retention(NONSENSE_TEXTS, matrix)
    zeros = _zero_count(matrix)
    report.update(
        {
            "zero_rows": zeros,
            "zero_rate": float(zeros / matrix.shape[0]),
            "nonzero_rows": int(matrix.shape[0] - zeros),
            "topic": None,
        }
    )
    return report


def unseen_token_case(embed: Callable[[Sequence[str]], np.ndarray], *, vocab_fitted: bool) -> dict:
    """Eval tokens are absent from the train token set."""
    overlap = _unseen_overlap()
    if overlap["content_overlap"] or overlap["regex_overlap"]:
        raise RuntimeError(f"unseen-token fixture shares tokens: {overlap}")
    eval_matrix = embed(UNSEEN_EVAL_TEXTS)
    report = distinction_retention(UNSEEN_EVAL_TEXTS, eval_matrix)
    zeros = _zero_count(eval_matrix)
    report.update(
        {
            "zero_rows": zeros,
            "zero_rate": float(zeros / eval_matrix.shape[0]) if eval_matrix.shape[0] else 0.0,
            "nonzero_rows": int(eval_matrix.shape[0] - zeros),
            "vocab_fitted": bool(vocab_fitted),
            "content_token_absence_fraction": 1.0,
            "train_content_types": overlap["train_content_types"],
            "eval_content_types": overlap["eval_content_types"],
            "topic": None,
        }
    )
    return report


def hash_collision_fixture() -> dict:
    """Two strings that share a hash bucket, and two that share none.

    The scan is lexicographic and bounded. A hash that never repeats a bucket
    inside the bound is reported as impossible for that search, which is the
    explicit alternative to a detected collision.
    """
    word_hit = _first_shared_bucket("word")
    char_hit = _first_shared_bucket("char")
    disjoint = _first_disjoint_pair()
    strings = []
    for hit in (word_hit, char_hit):
        if hit["detected"]:
            strings.extend([hit["left"], hit["right"]])
    if disjoint is not None:
        strings.extend([disjoint["left"], disjoint["right"]])
    deduped = []
    for text in strings:
        if text not in deduped:
            deduped.append(text)
    return {
        "word": word_hit,
        "char": char_hit,
        "disjoint": disjoint,
        "texts": deduped,
    }


def hash_collision_case(
    embed: Callable[[Sequence[str]], np.ndarray],
    fixture: Mapping,
    *,
    kind: str,
) -> dict:
    """Distinction on the planted strings, plus whether the planted bucket collided."""
    texts = list(fixture["texts"])
    if len(texts) < 2:
        report = {
            "n_rows": len(texts),
            "n_distinct": len(texts),
            "n_unique_vectors": len(texts),
            "unique_text_count": len(texts),
            "pairwise_distinction_retention": 1.0 if len(texts) <= 1 else 0.0,
            "unique_vector_fraction": 1.0,
            "distinct_pair_fraction": 1.0,
            "collapsed": False,
            "nonzero_collapse": False,
            "zero_rows": 0,
            "zero_rate": 0.0,
        }
    else:
        matrix = embed(texts)
        report = distinction_retention(texts, matrix)
        zeros = _zero_count(matrix)
        report.update(
            {
                "zero_rows": zeros,
                "zero_rate": float(zeros / matrix.shape[0]),
            }
        )
    hit = fixture[kind]
    report.update(
        {
            "collision_detected": bool(hit["detected"]),
            "collision_impossible": bool(hit["impossible"]),
            "collision_bucket": hit["bucket"],
            "collision_left": hit["left"],
            "collision_right": hit["right"],
            "collision_vectors_equal": hit["vectors_equal"],
            "disjoint_found": fixture["disjoint"] is not None,
            "disjoint_shares_bucket": False if fixture["disjoint"] is not None else None,
        }
    )
    return report


def arm1_synthetic_report(collision: Mapping) -> dict:
    """Train-only behavior on the synthetic strings.

    Repeated input is two copies of one string. Unseen-token, nonsense, and
    the planted collision strings are projected with an encoder fit on
    ``UNSEEN_TRAIN_TEXTS`` only. Those eval tokens are absent from that
    vocabulary, so this arm has no column for them.
    """
    repeated_matrix = np.asarray(
        word_sentence_svd([REPEATED_TEXT, REPEATED_TEXT], rank=RANK),
        dtype=np.float64,
    )
    repeated_encoder = fit_train_only_encoder([REPEATED_TEXT], rank=RANK)
    repeated_projected = transform_train_only(
        repeated_encoder,
        [REPEATED_TEXT, REPEATED_TEXT],
    )
    train_encoder = fit_train_only_encoder(list(UNSEEN_TRAIN_TEXTS), rank=RANK)
    unseen_matrix = transform_train_only(train_encoder, list(UNSEEN_EVAL_TEXTS))
    nonsense_matrix = transform_train_only(train_encoder, list(NONSENSE_TEXTS))
    collision_texts = list(collision["texts"])
    collision_matrix = transform_train_only(train_encoder, collision_texts)
    unseen = distinction_retention(UNSEEN_EVAL_TEXTS, unseen_matrix)
    nonsense = distinction_retention(NONSENSE_TEXTS, nonsense_matrix)
    collided = distinction_retention(collision_texts, collision_matrix)
    unseen_zeros = _zero_count(unseen_matrix)
    nonsense_zeros = _zero_count(nonsense_matrix)
    collision_zeros = _zero_count(collision_matrix)
    unseen.update(
        {
            "zero_rows": unseen_zeros,
            "zero_rate": float(unseen_zeros / unseen_matrix.shape[0]),
            "nonzero_rows": int(unseen_matrix.shape[0] - unseen_zeros),
            "vocab_fitted": True,
            "content_token_absence_fraction": 1.0,
            "topic": None,
        }
    )
    nonsense.update(
        {
            "zero_rows": nonsense_zeros,
            "zero_rate": float(nonsense_zeros / nonsense_matrix.shape[0]),
            "nonzero_rows": int(nonsense_matrix.shape[0] - nonsense_zeros),
            "topic": None,
        }
    )
    collided.update(
        {
            "zero_rows": collision_zeros,
            "zero_rate": float(collision_zeros / collision_matrix.shape[0]) if collision_matrix.shape[0] else 0.0,
            "collision_detected": False,
            "collision_impossible": False,
            "hash_buckets": "not_used",
            "disjoint_found": collision["disjoint"] is not None,
        }
    )
    return {
        "repeated_input": {
            "text": REPEATED_TEXT,
            "identical": bool(np.array_equal(repeated_matrix[0], repeated_matrix[1])),
            "max_abs_diff": float(np.max(np.abs(repeated_matrix[0] - repeated_matrix[1]))),
            "projected_identical": bool(
                np.array_equal(repeated_projected[0], repeated_projected[1])
            ),
            "zero": bool(np.all(repeated_matrix[0] == 0.0)),
            "n_copies": 2,
            "collapsed": False,
            "n_distinct": 1,
            "pairwise_distinction_retention": 1.0,
        },
        "nonsense_text": nonsense,
        "unseen_token": unseen,
        "hash_collision": collided,
    }


def synthetic_report(
    embed: Callable[[Sequence[str]], np.ndarray],
    *,
    vocab_fitted: bool,
    collision_kind: str,
    collision: Mapping,
) -> dict:
    """Representation checks. These strings are not generalization rows."""
    return {
        "repeated_input": repeated_input_case(embed),
        "nonsense_text": nonsense_case(embed),
        "unseen_token": unseen_token_case(embed, vocab_fitted=vocab_fitted),
        "hash_collision": hash_collision_case(embed, collision, kind=collision_kind),
    }


def run_oov_coverage(
    root: Optional[Path] = None,
    *,
    time_runs: int = TIME_RUNS,
) -> dict:
    """Score arm 1, then arm 2, then arm 3. Does not score arm 4.

    The split fingerprint is checked before any fixture is embedded. Arm 1
    is compared to the published inductive table before a hashed purity is
    used in the trial decision. Arm 4's precondition is recorded from the
    arm-2 and arm-3 tables and is not itself a score of arm 4.
    """
    if time_runs < 0:
        raise ValueError(f"time_runs must be >= 0, got {time_runs}")
    if HASH_DIM != 32 or CHAR_NGRAM != 3:
        raise RuntimeError("hash width changed after the constants were fixed")
    if DISTINCTION_MIN != 0.95 or RUNTIME_FACTOR != 5.0:
        raise RuntimeError("admission thresholds changed")
    if INDUCTIVE_LABEL_SHUFFLE_SEED != 0 or INDUCTIVE_ROW_SHUFFLE_SEED != 1:
        raise RuntimeError("control seeds changed")
    started = time.perf_counter()
    fingerprint = split_fingerprint(root)
    if not fingerprint["reproduced"] or fingerprint["sha256"] != FROZEN_SPLIT_SHA256:
        decision = decide_trial(
            [],
            split_reproduced=False,
            baseline_reproduced=False,
        )
        return _empty_report(
            decision,
            fingerprint,
            elapsed_s=time.perf_counter() - started,
            time_runs=time_runs,
            reason="split fingerprint was not reproduced",
        )
    pin_blas_threads(1)
    fixture_root = Path(root) if root is not None else _default_root()
    loaded = _load_splits(fixture_root, fingerprint)
    collision = hash_collision_fixture()
    arm1_rows = [
        _score_arm1(fixture, time_runs=time_runs) for fixture in loaded
    ]
    baseline_reproduced = all(_matches_published(row) for row in arm1_rows)
    arm1 = _arm_record(
        ARM1_ID,
        candidate=False,
        kind="train_only_tfidf_svd",
        fixtures=arm1_rows,
        status=ARM_CONTROL,
        pattern="published_inductive_control",
        synthetic=arm1_synthetic_report(collision),
        vocab_fitted=True,
        fit_on_eval=False,
    )
    by_id = {row["fixture_id"]: row for row in arm1_rows}
    arm2_rows = [
        _score_hashed(
            embed_char,
            fixture,
            by_id[fixture["fixture_id"]],
            time_runs=time_runs,
            fit_parameters=False,
        )
        for fixture in loaded
    ]
    arm2_synthetic = synthetic_report(
        embed_char,
        vocab_fitted=False,
        collision_kind="char",
        collision=collision,
    )
    arm2_status = decide_arm(
        fixtures=arm2_rows,
        controls_fall=all(_controls_fell(row) for row in arm2_rows),
        repeated_input_identical=bool(arm2_synthetic["repeated_input"]["identical"]),
    )
    arm2 = _arm_record(
        ARM2_ID,
        candidate=True,
        kind="character_ngram_hash",
        fixtures=arm2_rows,
        status=arm2_status,
        pattern=arm_pattern(arm2_rows),
        synthetic=arm2_synthetic,
        vocab_fitted=False,
        fit_on_eval=False,
    )
    arm3_rows = [
        _score_hashed(
            embed_word,
            fixture,
            by_id[fixture["fixture_id"]],
            time_runs=time_runs,
            fit_parameters=False,
        )
        for fixture in loaded
    ]
    arm3_synthetic = synthetic_report(
        embed_word,
        vocab_fitted=False,
        collision_kind="word",
        collision=collision,
    )
    arm3_status = decide_arm(
        fixtures=arm3_rows,
        controls_fall=all(_controls_fell(row) for row in arm3_rows),
        repeated_input_identical=bool(arm3_synthetic["repeated_input"]["identical"]),
    )
    arm3 = _arm_record(
        ARM3_ID,
        candidate=True,
        kind="word_hash",
        fixtures=arm3_rows,
        status=arm3_status,
        pattern=arm_pattern(arm3_rows),
        synthetic=arm3_synthetic,
        vocab_fitted=False,
        fit_on_eval=False,
    )
    precondition = arm4_precondition(
        arm2_rows,
        arm3_rows,
        arm2_synthetic,
        arm3_synthetic,
    )
    decision = decide_trial(
        [arm2, arm3],
        split_reproduced=True,
        baseline_reproduced=baseline_reproduced,
    )
    return _report(
        decision=decision,
        fingerprint=fingerprint,
        arms=[arm1, arm2, arm3],
        precondition=precondition,
        arm4={"ran": False, "reason": "arm 4 is scored only after the note cites the tables"},
        baseline_reproduced=baseline_reproduced,
        elapsed_s=time.perf_counter() - started,
        time_runs=time_runs,
        collision=collision,
    )


def score_arm4(
    phase1: Mapping,
    doc_text: str,
    root: Optional[Path] = None,
    *,
    time_runs: int = TIME_RUNS,
) -> dict:
    """Score the combined hash only when the note already justifies it.

    The function returns before any generalization row is embedded when the
    precondition is false or the note does not cite the arm-2 and arm-3 tables.
    """
    precondition = phase1["arm4_precondition"]
    if not bool(precondition["justified"]):
        return {
            "ran": False,
            "reason": "justification to run arm 4 is absent",
            "precondition": dict(precondition),
        }
    sentence = str(precondition["justification_sentence"])
    if not justification_in_doc(doc_text, phase1):
        return {
            "ran": False,
            "reason": "justification absent from the doc",
            "precondition": dict(precondition),
            "expected_sentence": sentence,
        }
    if time_runs < 0:
        raise ValueError(f"time_runs must be >= 0, got {time_runs}")
    fingerprint = split_fingerprint(root)
    if not fingerprint["reproduced"]:
        raise RuntimeError("split fingerprint was not reproduced")
    pin_blas_threads(1)
    fixture_root = Path(root) if root is not None else _default_root()
    loaded = _load_splits(fixture_root, fingerprint)
    arm1 = _find_arm(phase1, ARM1_ID)
    by_id = {row["fixture_id"]: row for row in arm1["fixtures"]}
    rows = [
        _score_hashed(
            embed_combined,
            fixture,
            by_id[fixture["fixture_id"]],
            time_runs=time_runs,
            fit_parameters=False,
        )
        for fixture in loaded
    ]
    collision = phase1["hash_collision_fixture"]
    synthetic = synthetic_report(
        embed_combined,
        vocab_fitted=False,
        collision_kind="char",
        collision=collision,
    )
    status = decide_arm(
        fixtures=rows,
        controls_fall=all(_controls_fell(row) for row in rows),
        repeated_input_identical=bool(synthetic["repeated_input"]["identical"]),
    )
    arm = _arm_record(
        ARM4_ID,
        candidate=True,
        kind="char_plus_word_hash",
        fixtures=rows,
        status=status,
        pattern=arm_pattern(rows),
        synthetic=synthetic,
        vocab_fitted=False,
        fit_on_eval=False,
    )
    hashed = [_find_arm(phase1, ARM2_ID), _find_arm(phase1, ARM3_ID), arm]
    decision = decide_trial(
        hashed,
        split_reproduced=bool(phase1["split_reproduced"]),
        baseline_reproduced=bool(phase1["arm1_matches_published"]),
    )
    return {
        "ran": True,
        "reason": sentence,
        "arm": arm,
        "decision": decision,
    }


def justification_in_doc(doc_text: str, phase1: Mapping) -> bool:
    """The note cites the computed arm-2 and arm-3 zero counts and the sentence."""
    precondition = phase1["arm4_precondition"]
    if not bool(precondition["justified"]):
        return False
    if ARM4_RUN_MARKER not in doc_text:
        return False
    if str(precondition["justification_sentence"]) not in doc_text:
        return False
    for arm_id in (ARM2_ID, ARM3_ID):
        arm = _find_arm(phase1, arm_id)
        for fixture in arm["fixtures"]:
            token = (
                f"{arm_id} {fixture['fixture_id']} zero "
                f"{fixture['zero_rows']}/{fixture['n_eval']}"
            )
            if token not in doc_text:
                return False
    return True


def _score_arm1(fixture: Mapping, *, time_runs: int) -> dict:
    inductive = _score_inductive(
        fixture["train_texts"],
        fixture["eval_texts"],
        fixture["train_topics"],
        fixture["eval_topics"],
    )
    eval_matrix = np.asarray(inductive["eval_matrix"], dtype=np.float64)
    train_matrix = np.asarray(inductive["train_matrix"], dtype=np.float64)
    distinction = distinction_retention(fixture["eval_texts"], eval_matrix)
    zeros = int(inductive["coverage"]["zero_rows"])
    n_eval = int(inductive["eval_rows"])
    if zeros != _zero_count(eval_matrix):
        raise RuntimeError("arm 1 zero-row count drifted")
    tokens = inductive["tokens"]
    costs = _costs_arm1(
        fixture["train_texts"],
        fixture["eval_texts"],
        train_matrix,
        inductive["encoder"],
        eval_matrix,
        time_runs,
    )
    published = PUBLISHED_ARM1[fixture["fixture_id"]]
    return {
        "fixture_id": fixture["fixture_id"],
        "n_train": int(inductive["train_rows"]),
        "n_eval": n_eval,
        "eval_hits": int(inductive["eval_hits"]),
        "eval_purity": float(inductive["eval_purity"]),
        "base_hits": int(inductive["base_hits"]),
        "base_rate": float(inductive["base_rate"]),
        "train_hits": int(inductive["train_hits"]),
        "train_purity": float(inductive["train_purity"]),
        "train_residual": float(inductive["train_residual"]),
        "eval_residual": float(inductive["eval_residual"]),
        "zero_rows": zeros,
        "zero_rate": float(zeros / n_eval) if n_eval else 0.0,
        "distinction": distinction,
        "controls": _public_controls(inductive["controls"]),
        "oov": {
            "vocab_fitted": True,
            "occurrence_absence_fraction": float(tokens["absence_fraction"]),
            "absent_occurrences": int(tokens["absent_occurrences"]),
            "eval_content_token_occurrences": int(tokens["eval_content_token_occurrences"]),
            "absent_types": int(tokens["absent_types"]),
            "eval_content_token_types": int(tokens["eval_content_token_types"]),
            "nonzero_rows": int(n_eval - zeros),
            "nonzero_row_fraction": float((n_eval - zeros) / n_eval) if n_eval else 0.0,
            "eval_rows_with_train_vocab_overlap": int(tokens["eval_rows_with_train_vocab_overlap"]),
        },
        "costs": costs,
        "matches_published": _published_equal(fixture["fixture_id"], inductive, zeros),
        "published_eval_hits": int(published["eval_hits"]),
        "published_base_hits": int(published["base_hits"]),
        "admitted": False,
        "above_arm1": False,
        "above_base": bool(inductive["above_base_rate"]),
        "zero_below_arm1": False,
        "distinction_ok": bool(
            distinction["pairwise_distinction_retention"] >= DISTINCTION_MIN
        ),
        "runtime_ok": False,
        "label_below": bool(inductive["controls"]["label_below"]),
        "row_below": bool(inductive["controls"]["row_below"]),
    }


def _score_hashed(
    embed: Callable[[Sequence[str]], np.ndarray],
    fixture: Mapping,
    arm1: Mapping,
    *,
    time_runs: int,
    fit_parameters: bool,
) -> dict:
    if fit_parameters:
        raise RuntimeError("hashed arms fit no corpus parameters")
    train_texts = fixture["train_texts"]
    eval_texts = fixture["eval_texts"]
    train_topics = fixture["train_topics"]
    eval_topics = fixture["eval_topics"]
    train_matrix = embed(train_texts)
    eval_matrix = embed(eval_texts)
    if train_matrix.shape[1] != HASH_DIM or eval_matrix.shape[1] != HASH_DIM:
        raise RuntimeError("hashed width drifted from HASH_DIM")
    spec = _production_spec()
    train_labels, centroids, _indices = _fit_algorithm(train_matrix, spec)
    eval_labels = _assign(eval_matrix, centroids, "euclidean")
    mapping = train_only_majority_map(train_labels, train_topics)
    train_score = score_against_map(train_labels, train_topics, mapping)
    eval_score = score_against_map(eval_labels, eval_topics, mapping)
    controls = _inductive_controls(
        eval_matrix,
        centroids,
        train_labels,
        eval_labels,
        train_topics,
        eval_topics,
        mapping,
        int(eval_score["hits"]),
    )
    distinction = distinction_retention(eval_texts, eval_matrix)
    zeros = _zero_count(eval_matrix)
    n_eval = int(eval_matrix.shape[0])
    if int(arm1["n_eval"]) != n_eval:
        raise RuntimeError("eval length drifted from arm 1")
    costs = _costs_hashed(
        embed,
        train_texts,
        eval_texts,
        train_matrix,
        eval_matrix,
        time_runs,
    )
    eval_hits = int(eval_score["hits"])
    above_arm1 = bool(eval_hits > int(arm1["eval_hits"]))
    above_base = bool(eval_hits > int(arm1["base_hits"]))
    zero_below = bool(zeros < int(arm1["zero_rows"]))
    distinction_ok = bool(
        float(distinction["pairwise_distinction_retention"]) >= DISTINCTION_MIN
    )
    label_below = bool(controls["label_below"])
    row_below = bool(controls["row_below"])
    runtime_ok = _runtime_ok(
        costs["feature_plus_fit_median_ms"],
        arm1["costs"]["feature_plus_fit_median_ms"],
    )
    admitted = bool(
        above_arm1
        and above_base
        and label_below
        and row_below
        and zero_below
        and distinction_ok
        and runtime_ok
    )
    return {
        "fixture_id": fixture["fixture_id"],
        "n_train": int(train_matrix.shape[0]),
        "n_eval": n_eval,
        "eval_hits": eval_hits,
        "eval_purity": float(eval_score["size_weighted"]),
        "base_hits": int(arm1["base_hits"]),
        "base_rate": float(arm1["base_rate"]),
        "train_hits": int(train_score["hits"]),
        "train_purity": float(train_score["size_weighted"]),
        "train_residual": float(_prototype_residual(train_matrix, train_labels, centroids)),
        "eval_residual": float(_prototype_residual(eval_matrix, eval_labels, centroids)),
        "zero_rows": int(zeros),
        "zero_rate": float(zeros / n_eval) if n_eval else 0.0,
        "arm1_zero_rows": int(arm1["zero_rows"]),
        "arm1_zero_rate": float(arm1["zero_rate"]),
        "arm1_eval_hits": int(arm1["eval_hits"]),
        "distinction": distinction,
        "controls": _public_controls(controls),
        "oov": {
            "vocab_fitted": False,
            "occurrence_absence_fraction": None,
            "absent_occurrences": None,
            "eval_content_token_occurrences": None,
            "absent_types": None,
            "eval_content_token_types": None,
            "nonzero_rows": int(n_eval - zeros),
            "nonzero_row_fraction": float((n_eval - zeros) / n_eval) if n_eval else 0.0,
            "eval_rows_with_train_vocab_overlap": None,
        },
        "costs": costs,
        "above_arm1": above_arm1,
        "above_base": above_base,
        "zero_below_arm1": zero_below,
        "distinction_ok": distinction_ok,
        "runtime_ok": runtime_ok,
        "label_below": label_below,
        "row_below": row_below,
        "admitted": admitted,
    }


def _costs_arm1(
    train_texts: Sequence[str],
    eval_texts: Sequence[str],
    train_matrix: np.ndarray,
    encoder: Mapping,
    eval_matrix: np.ndarray,
    time_runs: int,
) -> dict:
    spec = _production_spec()

    def _combined() -> None:
        fitted = fit_train_only_encoder(train_texts, rank=RANK)
        transform_train_only(fitted, eval_texts)

    def _fit() -> None:
        fit_train_only_encoder(train_texts, rank=RANK)

    frozen = fit_train_only_encoder(train_texts, rank=RANK)

    def _extract() -> None:
        transform_train_only(frozen, eval_texts)

    def _cluster() -> None:
        _fit_algorithm(train_matrix, spec)

    plus_ms, _plus_samples = _per_call_median_ms(_combined, time_runs, 1)
    fit_ms, _fit_samples = _per_call_median_ms(_fit, time_runs, 1)
    extract_ms, _extract_samples = _per_call_median_ms(_extract, time_runs, 1)
    cluster_ms, _cluster_samples = _per_call_median_ms(_cluster, time_runs, CLUSTER_TIME_REPEATS)
    pack = _arm1_pack_nbytes(encoder)
    eval_nbytes = int(np.asarray(eval_matrix).nbytes)
    return {
        "feature_extraction_median_ms": extract_ms,
        "fit_median_ms": fit_ms,
        "feature_plus_fit_median_ms": plus_ms,
        "cluster_median_ms": cluster_ms,
        "feature_plus_fit_peak_bytes": _peak_bytes(_combined, time_runs),
        "cluster_peak_bytes": _peak_bytes(_cluster, time_runs),
        "parameter_pack_nbytes": pack,
        "eval_matrix_nbytes": eval_nbytes,
        "serialized_nbytes": int(pack + eval_nbytes),
        "time_runs": int(time_runs),
        "fit_parameters": True,
    }


def _costs_hashed(
    embed: Callable[[Sequence[str]], np.ndarray],
    train_texts: Sequence[str],
    eval_texts: Sequence[str],
    train_matrix: np.ndarray,
    eval_matrix: np.ndarray,
    time_runs: int,
) -> dict:
    spec = _production_spec()

    def _extract() -> None:
        embed(train_texts)
        embed(eval_texts)

    def _cluster() -> None:
        _fit_algorithm(train_matrix, spec)

    extract_ms, _samples = _per_call_median_ms(_extract, time_runs, 1)
    cluster_ms, _cluster_samples = _per_call_median_ms(_cluster, time_runs, CLUSTER_TIME_REPEATS)
    eval_nbytes = int(np.asarray(eval_matrix).nbytes)
    fit_ms: Optional[float] = 0.0 if time_runs else None
    return {
        "feature_extraction_median_ms": extract_ms,
        "fit_median_ms": fit_ms,
        "feature_plus_fit_median_ms": extract_ms,
        "cluster_median_ms": cluster_ms,
        "feature_plus_fit_peak_bytes": _peak_bytes(_extract, time_runs),
        "cluster_peak_bytes": _peak_bytes(_cluster, time_runs),
        "parameter_pack_nbytes": 0,
        "eval_matrix_nbytes": eval_nbytes,
        "serialized_nbytes": int(eval_nbytes),
        "time_runs": int(time_runs),
        "fit_parameters": False,
    }


def _runtime_ok(
    median_ms: Optional[float],
    arm1_median_ms: Optional[float],
) -> bool:
    if median_ms is None or arm1_median_ms is None:
        return False
    return float(median_ms) <= RUNTIME_FACTOR * float(arm1_median_ms)


def _arm1_pack_nbytes(encoder: Mapping) -> int:
    vocabulary = list(encoder["vocabulary"])
    text = "\n".join(vocabulary).encode("utf-8")
    idf = np.asarray(encoder["idf"], dtype=np.float64)
    document_frequency = encoder["document_frequency"]
    counts = np.asarray(
        [int(document_frequency[token]) for token in vocabulary],
        dtype=np.int64,
    )
    factors = encoder["factors"]
    basis = 0
    if factors is not None:
        basis = int(np.asarray(factors["right"]).nbytes + np.asarray(factors["singular"]).nbytes)
    return int(len(text) + idf.nbytes + counts.nbytes + basis)


def _matches_published(row: Mapping) -> bool:
    return bool(row["matches_published"])


def _published_equal(fixture_id: str, inductive: Mapping, zeros: int) -> bool:
    published = PUBLISHED_ARM1[fixture_id]
    tokens = inductive["tokens"]
    checks = (
        int(inductive["eval_hits"]) == int(published["eval_hits"]),
        int(inductive["eval_rows"]) == int(published["n_eval"]),
        int(inductive["base_hits"]) == int(published["base_hits"]),
        int(zeros) == int(published["zero_rows"]),
        int(inductive["controls"]["label_hits"]) == int(published["label_hits"]),
        int(inductive["controls"]["row_hits"]) == int(published["row_hits"]),
        int(inductive["train_hits"]) == int(published["train_hits"]),
        int(inductive["train_rows"]) == int(published["n_train"]),
        int(tokens["eval_content_token_occurrences"]) == int(published["occurrences"]),
        int(tokens["absent_occurrences"]) == int(published["absent_occurrences"]),
        int(tokens["absent_types"]) == int(published["absent_types"]),
    )
    return all(checks)


def _controls_fell(row: Mapping) -> bool:
    return bool(row["label_below"] and row["row_below"])


def _public_controls(controls: Mapping) -> dict:
    return {
        "label_shuffle_seed": int(controls["label_shuffle_seed"]),
        "label_hits": int(controls["label_hits"]),
        "label_rows": int(controls["label_rows"]),
        "label_purity": float(controls["label_purity"]),
        "label_below": bool(controls["label_below"]),
        "row_shuffle_seed": int(controls["row_shuffle_seed"]),
        "row_hits": int(controls["row_hits"]),
        "row_rows": int(controls["row_rows"]),
        "row_purity": float(controls["row_purity"]),
        "row_below": bool(controls["row_below"]),
        "map_fit_on_eval": False,
        "centroids_refit": False,
    }


def _arm_record(
    arm_id: str,
    *,
    candidate: bool,
    kind: str,
    fixtures: Sequence[Mapping],
    status: str,
    pattern: str,
    synthetic: Optional[Mapping],
    vocab_fitted: bool,
    fit_on_eval: bool,
) -> dict:
    return {
        "arm_id": arm_id,
        "candidate": bool(candidate),
        "kind": kind,
        "status": status,
        "pattern": pattern,
        "vocab_fitted": bool(vocab_fitted),
        "fit_on_eval": bool(fit_on_eval),
        "hash_dim": None if arm_id == ARM1_ID else HASH_DIM,
        "fixtures": list(fixtures),
        "synthetic": synthetic,
    }


def _load_splits(root: Path, fingerprint: Mapping) -> List[dict]:
    loaded_rows = []
    for spec, finger in zip(FIXTURE_SPECS, fingerprint["fixtures"]):
        loaded = load_fixture(root / str(spec["dirname"]))
        texts = list(loaded["texts"])
        topics = list(loaded["topics"])
        observation_ids = list(loaded["observation_ids"])
        eval_index = [int(index) for index in finger["eval_index"]]
        eval_ids = [observation_ids[index] for index in eval_index]
        if eval_ids != list(finger["eval_ids"]):
            raise RuntimeError(f"{spec['id']} eval ids drifted")
        eval_set = set(eval_index)
        train_index = [index for index in range(len(texts)) if index not in eval_set]
        loaded_rows.append(
            {
                "fixture_id": str(spec["id"]),
                "train_texts": [texts[index] for index in train_index],
                "eval_texts": [texts[index] for index in eval_index],
                "train_topics": [topics[index] for index in train_index],
                "eval_topics": [topics[index] for index in eval_index],
                "eval_ids": eval_ids,
            }
        )
    return loaded_rows


def _first_shared_bucket(kind: str) -> dict:
    seen: Dict[int, str] = {}
    for count, token in enumerate(_probe_tokens()):
        if count >= COLLISION_SEARCH_LIMIT:
            break
        buckets = _buckets(kind, token)
        for bucket in sorted(buckets):
            previous = seen.get(bucket)
            if previous is not None and previous != token:
                return {
                    "detected": True,
                    "impossible": False,
                    "left": previous,
                    "right": token,
                    "bucket": int(bucket),
                    "vectors_equal": _vectors_equal(kind, previous, token),
                }
            seen.setdefault(bucket, token)
    return {
        "detected": False,
        "impossible": True,
        "left": None,
        "right": None,
        "bucket": None,
        "vectors_equal": None,
    }


def _first_disjoint_pair() -> Optional[dict]:
    chosen: List[dict] = []
    for token in _probe_tokens():
        word_buckets = _buckets("word", token)
        char_buckets = _buckets("char", token)
        for previous in chosen:
            if word_buckets.isdisjoint(previous["word"]) and char_buckets.isdisjoint(previous["char"]):
                return {
                    "left": previous["token"],
                    "right": token,
                    "shares_word_bucket": False,
                    "shares_char_bucket": False,
                }
        chosen.append({"token": token, "word": word_buckets, "char": char_buckets})
        if len(chosen) >= DISJOINT_SEARCH_LIMIT:
            break
    return None


def _probe_tokens() -> Iterator[str]:
    for chars in itertools.product(COLLISION_ALPHABET, repeat=COLLISION_TOKEN_LENGTH):
        yield "".join(chars)


def _buckets(kind: str, token: str) -> set:
    text = token
    if kind == "word":
        found = set()
        for word in _WORD.findall(text.lower()):
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            found.add(int.from_bytes(digest[:4], "little") % HASH_DIM)
        return found
    if kind == "char":
        return _char_buckets(text)
    raise ValueError(f"unknown hash kind {kind!r}")


def _char_buckets(text: str) -> set:
    blob = f"^{text.lower()}$"
    if len(blob) < CHAR_NGRAM:
        blob = blob.ljust(CHAR_NGRAM, "_")
    found = set()
    for index in range(len(blob) - CHAR_NGRAM + 1):
        gram = blob[index : index + CHAR_NGRAM].encode("utf-8")
        digest = hashlib.sha256(gram).digest()
        found.add(int.from_bytes(digest[:4], "little") % HASH_DIM)
    return found


def _vectors_equal(kind: str, left: str, right: str) -> bool:
    if kind == "word":
        return bool(np.array_equal(hashed_word_vector(left), hashed_word_vector(right)))
    if kind == "char":
        return bool(
            np.array_equal(
                np.asarray(hashed_ngram_vector(left, dim=HASH_DIM, n=CHAR_NGRAM)),
                np.asarray(hashed_ngram_vector(right, dim=HASH_DIM, n=CHAR_NGRAM)),
            )
        )
    raise ValueError(f"unknown hash kind {kind!r}")


def _assert_nonsense_shape() -> None:
    if len(NONSENSE_TEXTS) < 2 or len(set(NONSENSE_TEXTS)) != len(NONSENSE_TEXTS):
        raise RuntimeError("nonsense texts are not distinct")
    tokens = [_WORD.findall(text.lower()) for text in NONSENSE_TEXTS]
    flat = [token for row in tokens for token in row]
    if len(flat) != len(set(flat)):
        raise RuntimeError("nonsense texts share a token")


def _unseen_overlap() -> dict:
    train_content = set()
    eval_content = set()
    train_regex = set()
    eval_regex = set()
    for text in UNSEEN_TRAIN_TEXTS:
        train_content.update(content_tokens(text))
        train_regex.update(_WORD.findall(text.lower()))
    for text in UNSEEN_EVAL_TEXTS:
        eval_content.update(content_tokens(text))
        eval_regex.update(_WORD.findall(text.lower()))
    return {
        "content_overlap": sorted(train_content & eval_content),
        "regex_overlap": sorted(train_regex & eval_regex),
        "train_content_types": len(train_content),
        "eval_content_types": len(eval_content),
    }


def _zero_count(matrix: np.ndarray) -> int:
    return int(np.all(np.asarray(matrix) == 0.0, axis=1).sum())


def _is_zero_bytes(blob: bytes, dim: int) -> bool:
    return blob == np.zeros(dim, dtype=np.float64).tobytes()


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        return vec / norm
    return vec


def _find_arm(phase1: Mapping, arm_id: str) -> Mapping:
    for arm in phase1["arms"]:
        if arm["arm_id"] == arm_id:
            return arm
    raise KeyError(arm_id)


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "examples" / "generalization"


def _empty_report(
    decision: Mapping,
    fingerprint: Mapping,
    *,
    elapsed_s: float,
    time_runs: int,
    reason: str,
) -> dict:
    return _report(
        decision=decision,
        fingerprint=fingerprint,
        arms=[],
        precondition={
            "justified": False,
            "justification_sentence": reason,
            "char_cuts_zero_rate": False,
            "word_cuts_zero_rate": False,
            "word_retains_distinction_char_loses": [],
            "char_retains_distinction_word_loses": [],
        },
        arm4={"ran": False, "reason": reason},
        baseline_reproduced=False,
        elapsed_s=elapsed_s,
        time_runs=time_runs,
        collision=None,
    )


def _report(
    *,
    decision: Mapping,
    fingerprint: Mapping,
    arms: Sequence[Mapping],
    precondition: Mapping,
    arm4: Mapping,
    baseline_reproduced: bool,
    elapsed_s: float,
    time_runs: int,
    collision: Optional[Mapping],
) -> dict:
    public_fingerprint = {
        "rule": fingerprint["rule"],
        "sha256": fingerprint["sha256"],
        "reproduced": bool(fingerprint["reproduced"]),
        "fixtures": [
            {
                "fixture_id": fixture["fixture_id"],
                "n_train": fixture["n_train"],
                "n_eval": fixture["n_eval"],
                "eval_ids": list(fixture["eval_ids"]),
            }
            for fixture in fingerprint["fixtures"]
        ],
    }
    return {
        "status": STATUS,
        "baseline_commit": BASELINE_COMMIT,
        "entering_verdict": ENTERING_VERDICT,
        "entering_recommendation": ENTERING_RECOMMENDATION,
        "decision_status": decision["decision_status"],
        "pattern": decision["pattern"],
        "recommendation": decision["recommendation"],
        "recommended_arms": list(decision["recommended_arms"]),
        "runtime_authority": RUNTIME_AUTHORITY,
        "production_replaced": False,
        "inductive_pass": False,
        "split_reproduced": bool(fingerprint["reproduced"]),
        "arm1_matches_published": bool(baseline_reproduced),
        "gates": {
            "purity_above_arm1": PURITY_ABOVE_ARM1_GATE,
            "purity_above_base": PURITY_ABOVE_BASE_GATE,
            "label_control": LABEL_CONTROL_GATE,
            "row_control": ROW_CONTROL_GATE,
            "zero_rate": ZERO_RATE_GATE,
            "distinction": DISTINCTION_GATE,
            "runtime": RUNTIME_GATE,
            "distinction_min": DISTINCTION_MIN,
            "runtime_factor": RUNTIME_FACTOR,
        },
        "hash_dim": HASH_DIM,
        "char_ngram": CHAR_NGRAM,
        "word_token_pattern": WORD_TOKEN_PATTERN,
        "split_fingerprint": public_fingerprint,
        "arms": list(arms),
        "arm4_precondition": dict(precondition),
        "arm4": dict(arm4),
        "hash_collision_fixture": _public_collision(collision),
        "costs": {
            "elapsed_s": float(elapsed_s),
            "time_runs": int(time_runs),
        },
        "numerical_library": numerical_library(),
        "git_revision": _git_revision(),
        "production_svd_modified": False,
        "learn_pack_called": False,
        "package_version_modified": False,
    }


def _public_collision(collision: Optional[Mapping]) -> Optional[dict]:
    if collision is None:
        return None
    def _hit(hit: Mapping) -> dict:
        return {
            "detected": bool(hit["detected"]),
            "impossible": bool(hit["impossible"]),
            "left": hit["left"],
            "right": hit["right"],
            "bucket": hit["bucket"],
            "vectors_equal": hit["vectors_equal"],
        }
    disjoint = collision["disjoint"]
    return {
        "word": _hit(collision["word"]),
        "char": _hit(collision["char"]),
        "disjoint": None
        if disjoint is None
        else {
            "left": disjoint["left"],
            "right": disjoint["right"],
            "shares_word_bucket": False,
            "shares_char_bucket": False,
        },
        "texts": list(collision["texts"]),
    }


def _format_report(report: Mapping) -> str:
    lines = [
        f"decision_status {report['decision_status']}",
        f"pattern {report['pattern']}",
        f"recommendation {report['recommendation']}",
        f"runtime_authority {report['runtime_authority']}",
        f"production_replaced {report['production_replaced']}",
        f"inductive_pass {report['inductive_pass']}",
        f"split_reproduced {report['split_reproduced']}",
        f"arm1_matches_published {report['arm1_matches_published']}",
        f"sha256 {report['split_fingerprint']['sha256']}",
        f"git {report['git_revision']}",
        f"elapsed_s {report['costs']['elapsed_s']:.3f}",
        f"arm4_justified {report['arm4_precondition']['justified']}",
        f"arm4_sentence {report['arm4_precondition']['justification_sentence']}",
        f"numpy {report['numerical_library']['numpy']}",
        f"libraries {report['numerical_library']['loaded_libraries']}",
    ]
    for arm in report["arms"]:
        lines.append(f"arm {arm['arm_id']} status {arm['status']} pattern {arm['pattern']}")
        for fixture in arm["fixtures"]:
            controls = fixture["controls"]
            distinction = fixture["distinction"]
            costs = fixture["costs"]
            lines.append(
                " ".join(
                    [
                        arm["arm_id"],
                        fixture["fixture_id"],
                        f"eval {fixture['eval_hits']}/{fixture['n_eval']}",
                        f"base {fixture['base_hits']}/{fixture['n_eval']}",
                        f"train {fixture['train_hits']}/{fixture['n_train']}",
                        f"zero {fixture['zero_rows']}/{fixture['n_eval']}",
                        f"retain {distinction['pairwise_distinction_retention']:.4f}",
                        f"unique {distinction['n_unique_vectors']}/{distinction['n_distinct']}",
                        f"label {controls['label_hits']}",
                        f"row {controls['row_hits']}",
                        f"train_resid {fixture['train_residual']:.4f}",
                        f"eval_resid {fixture['eval_residual']:.4f}",
                        f"extract_ms {costs['feature_extraction_median_ms']}",
                        f"fit_ms {costs['fit_median_ms']}",
                        f"plus_ms {costs['feature_plus_fit_median_ms']}",
                        f"cluster_ms {costs['cluster_median_ms']}",
                        f"peak {costs['feature_plus_fit_peak_bytes']}",
                        f"cluster_peak {costs['cluster_peak_bytes']}",
                        f"pack {costs['parameter_pack_nbytes']}",
                        f"eval_bytes {costs['eval_matrix_nbytes']}",
                        f"serial {costs['serialized_nbytes']}",
                        f"above_arm1 {fixture['above_arm1']}",
                        f"above_base {fixture['above_base']}",
                        f"zero_below {fixture['zero_below_arm1']}",
                        f"distinction_ok {fixture['distinction_ok']}",
                        f"runtime_ok {fixture['runtime_ok']}",
                        f"admitted {fixture['admitted']}",
                        f"oov_vocab {fixture['oov']['vocab_fitted']}",
                        f"nonzero {fixture['oov']['nonzero_rows']}/{fixture['n_eval']}",
                    ]
                )
            )
        synthetic = arm["synthetic"]
        if synthetic:
            for name, case in synthetic.items():
                if name == "repeated_input":
                    lines.append(
                        f"{arm['arm_id']} synthetic {name} identical {case['identical']} "
                        f"projected_identical {case.get('projected_identical')} "
                        f"max_abs_diff {case.get('max_abs_diff')} zero {case['zero']}"
                    )
                else:
                    lines.append(
                        f"{arm['arm_id']} synthetic {name} "
                        f"retain {case['pairwise_distinction_retention']:.4f} "
                        f"collapsed {case['collapsed']} "
                        f"nonzero_collapse {case.get('nonzero_collapse')} "
                        f"zero {case.get('zero_rows')}"
                    )
    collision = report["hash_collision_fixture"]
    if collision:
        lines.append(
            "collision "
            f"word {collision['word']['detected']} {collision['word']['left']} {collision['word']['right']} "
            f"bucket {collision['word']['bucket']} equal {collision['word']['vectors_equal']} "
            f"char {collision['char']['detected']} {collision['char']['left']} {collision['char']['right']} "
            f"bucket {collision['char']['bucket']} equal {collision['char']['vectors_equal']} "
            f"disjoint {collision['disjoint']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(_format_report(run_oov_coverage(time_runs=0)))

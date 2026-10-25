"""Train-only encoder isolation on the frozen generalization split.

Production ``word_sentence_svd`` stays a pure function of whatever texts it
is given. This module does not replace it. The inductive arm fits vocabulary,
document frequency, IDF, and the SVD basis on training rows, then projects
eval rows with those frozen parameters. Clustering stays production
farthest-first squared-Euclidean k-means, fit on the training representation
only. The whole-fixture SVD plus k-means on that full matrix's train rows is
kept as a labeled transductive control. It is not a pass gate.

Admission constants below are the gates. They are fixed before any inductive
purity is computed. The full-matrix agreement figure from the evaluation-control audit is
not read and is not a gate.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neuralese.clustering_generalization import (
    CLUSTER_TIME_REPEATS,
    FIXTURE_SPECS,
    HOLDOUT_MODULUS,
    HOLDOUT_REMAINDER,
    K,
    LABEL_SHUFFLE_SEED,
    RANK,
    ROW_SHUFFLE_SEED,
    TIME_RUNS,
    _assert_production_ignores_rng,
    _peak_bytes,
    _per_call_median_ms,
    _production_spec,
    _score_assignment,
    holdout_mask,
    load_fixture,
)
from neuralese.clustering_isolation import (
    PRODUCTION_FAMILY,
    _assign,
    _fit_algorithm,
    _prototype_residual,
    adjusted_rand_index,
)
from neuralese.encode import content_tokens, word_sentence_svd, word_sentence_tfidf
from neuralese.eval_control_integrity import (
    permute_rows,
    permute_sequence,
    score_against_map,
    train_only_majority_map,
)
from neuralese.factorization import pin_blas_threads

# Status of the audit this experiment starts from. Not this run's verdict.
STATUS = "RESEARCH_ONLY"
BASELINE_COMMIT = "12407d3"
ENTERING_VERDICT = "UNRESOLVED"
ENTERING_RECOMMENDATION = "NONE"
RUNTIME_AUTHORITY = "RUNTIME_NOT_AUTHORIZED"
RECOMMENDATION = "NONE"

# Admission gates. Requirement text, fixed before inductive purity.
COVERAGE_GATE = "every_eval_row_has_a_finite_train_only_vector"
PURITY_GATE = "eval_size_weighted_purity_strictly_greater_than_eval_majority_base_rate"
LABEL_CONTROL_GATE = (
    "shuffled_train_label_map_purity_strictly_below_real_inductive_eval_purity"
)
ROW_CONTROL_GATE = (
    "eval_embedding_permutation_purity_strictly_below_real_inductive_eval_purity"
)
LEAKAGE_GATE = "no_shared_ids_or_texts_between_train_and_eval"

VERDICT_PASS = "INDUCTIVE_PASS"
VERDICT_FAIL = "INDUCTIVE_FAIL"
VERDICT_UNRESOLVED = "UNRESOLVED"
VERDICTS = (VERDICT_PASS, VERDICT_FAIL, VERDICT_UNRESOLVED)

# Coverage reasons this runner can emit. Any other string is undefined.
COVERAGE_COMPLETE = "complete"
COVERAGE_NON_FINITE = "non_finite_coordinates"
DEFINED_COVERAGE_REASONS = (COVERAGE_COMPLETE, COVERAGE_NON_FINITE)

# Control seeds. The generalization trial's constants. Not retuned.
# The label seed permutes the train topic list. The row seed permutes eval
# embedding rows. Neither seed is a clustering seed.
INDUCTIVE_LABEL_SHUFFLE_SEED = LABEL_SHUFFLE_SEED
INDUCTIVE_ROW_SHUFFLE_SEED = ROW_SHUFFLE_SEED

# Frozen split. File order, index modulo 5 equals 4. Counts are the
# evaluation-control audit's train and eval lengths. The sha256 is the
# eval observation ids in fixture order, one fixture per line,
# ``id:id1,id2,...``. A mismatch means the fingerprint was not reproduced.
FROZEN_SPLIT_RULE = "file-order index % 5 == 4"
FROZEN_SPLIT_COUNTS: Tuple[Tuple[str, int, int], ...] = (
    ("uneven", 40, 10),
    ("overlapping", 39, 9),
    ("ambiguous", 39, 9),
    ("balanced", 64, 16),
)
FROZEN_SPLIT_SHA256 = "55f561c832ef47d453827b3fc9704cf4d98e66d0bda2e4ff4aa9603c5923c20e"

# What the production encoder learns, and what it computes per row.
# Centering is absent from ``word_sentence_svd``. Per-row L2 uses that row's
# own norm after the frozen IDF weights. It is not a corpus moment.
LEARNED_ON_TRAIN = (
    "vocabulary",
    "document_frequency",
    "idf",
    "svd_basis",
    "singular_values",
    "sign_pivots",
)
DETERMINISTIC_PER_ROW = (
    "content_tokens",
    "raw_counts",
    "row_l2_of_tfidf",
    "row_l2_after_projection",
)
CENTERING = "not_a_production_statistic"
L2_SCOPE = "per_row_not_a_corpus_moment"
UNKNOWN_TOKEN_RULE = "zero_contribution_no_new_column"

# Diagnostic cutoffs. Not admission gates. Not a semantic-quality bar.
# Fixed before purity. A single cause is named only when exactly one fires.
DIAGNOSTIC_MAX_PRINCIPAL_ANGLE_DEG = 15.0
DIAGNOSTIC_CENTROID_MEAN_DISPLACEMENT = 0.50
DIAGNOSTIC_EVAL_FAR_FRACTION = 0.50
CAUSE_SUBSPACE = "TRAIN_SUBSPACE_MOVED"
CAUSE_CENTROIDS = "CENTROIDS_MOVED"
CAUSE_EVAL_FAR = "EVAL_FAR_FROM_TRAIN_CENTROIDS"
CAUSE_UNRESOLVED = "UNRESOLVED"
CAUSE_NOT_APPLICABLE = "NOT_APPLICABLE"

# Analytical blobs. Same draw as the evaluation-control audit. The label
# control here shuffles train labels, which that audit did not.
ANALYTICAL_DATA_SEED = 11
ANALYTICAL_DIM = 8
ANALYTICAL_BLOB_GAP = 30.0
ANALYTICAL_BLOB_SCALE = 1.0
ANALYTICAL_BLOB_TRAIN_PER_CLASS = 200
ANALYTICAL_BLOB_EVAL_PER_CLASS = 100

# Train TF-IDF must match production before an eval row is projected.
TRAIN_TFIDF_ATOL = 1e-12
TRAIN_EMBED_ATOL = 1e-8
# A projected row at or below this Euclidean norm is stored as zero.
# The floor sits above double-precision dust and below a real singular-value
# coordinate. It is not a purity threshold. Production ``word_sentence_svd``
# does not use it; the train matrix that k-means sees is that function's
# own output.
PROJECTION_NORM_FLOOR = 1e-8

TRANSDUCTIVE_LABEL = "TRANSDUCTIVE"
TRANSDUCTIVE_ENCODER = "word_sentence_svd on every row of the fixture"
TRANSDUCTIVE_CLUSTERING = (
    "production farthest-first k-means on the train rows of that full matrix"
)


def inductive_verdict(
    *,
    split_reproduced: bool,
    controls_fall: bool,
    coverage_complete: bool,
    coverage_reason_defined: bool,
    every_fixture_passes: bool,
    any_base_rate_miss: bool,
) -> str:
    """Apply the admission rule. Does not compute a purity.

    ``INDUCTIVE_FAIL`` is reachable only when the controls fell and at least
    one fixture missed the base-rate gate. A control that does not fall, a
    split fingerprint that does not reproduce, or partial coverage with an
    undefined reason stays ``UNRESOLVED``.
    """
    if every_fixture_passes and (
        any_base_rate_miss or not coverage_complete or not controls_fall or not split_reproduced
    ):
        raise ValueError("passing fixtures contradict a failed gate")
    if not split_reproduced:
        return VERDICT_UNRESOLVED
    if not controls_fall:
        return VERDICT_UNRESOLVED
    if not coverage_complete and not coverage_reason_defined:
        return VERDICT_UNRESOLVED
    if every_fixture_passes:
        return VERDICT_PASS
    if any_base_rate_miss:
        return VERDICT_FAIL
    return VERDICT_UNRESOLVED


def authority_disposition(verdict: str) -> dict:
    """A pass does not replace production and does not grant runtime authority."""
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    return {
        "verdict": verdict,
        "recommendation": RECOMMENDATION,
        "runtime_authority": RUNTIME_AUTHORITY,
        "production_replaced": False,
    }


def isolate_cause(
    *,
    purity_failed: bool,
    subspace_moved: bool,
    centroids_moved: bool,
    eval_far: bool,
) -> str:
    """Name one diagnostic cause, or ``UNRESOLVED`` when the numbers do not isolate it."""
    if not purity_failed:
        return CAUSE_NOT_APPLICABLE
    fired = []
    if subspace_moved:
        fired.append(CAUSE_SUBSPACE)
    if centroids_moved:
        fired.append(CAUSE_CENTROIDS)
    if eval_far:
        fired.append(CAUSE_EVAL_FAR)
    if len(fired) == 1:
        return fired[0]
    return CAUSE_UNRESOLVED


def cross_split_duplicates(
    train_ids: Sequence[str],
    eval_ids: Sequence[str],
    train_texts: Sequence[str],
    eval_texts: Sequence[str],
) -> dict:
    """Ids and texts that sit on both sides of the split.

    Comparison is exact string equality. The pass gate requires both
    intersections to be empty.
    """
    shared_ids = set(train_ids) & set(eval_ids)
    shared_texts = set(train_texts) & set(eval_texts)
    return {
        "shared_ids": int(len(shared_ids)),
        "shared_texts": int(len(shared_texts)),
        "shared_id_values": sorted(shared_ids),
        "empty": bool(len(shared_ids) == 0 and len(shared_texts) == 0),
    }


def fit_train_only_encoder(train_texts: Sequence[str], *, rank: int = RANK) -> dict:
    """Fit vocabulary, IDF, and the SVD basis on ``train_texts`` only.

    The returned object has no method that refits. ``transform_train_only``
    reads these parameters and does not update them. Train rows are checked
    against ``word_sentence_svd(train_texts)``, which is the production
    function on the training texts and does not see eval text.
    """
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    n_train = len(train_texts)
    vocabulary, document_frequency = _document_frequency(train_texts)
    built = _tfidf_matrix(train_texts, vocabulary, document_frequency, n_train)
    reference = word_sentence_tfidf(list(train_texts))
    tfidf_diff = _max_abs_diff(built, reference)
    if built.shape != reference.shape or tfidf_diff > TRAIN_TFIDF_ATOL:
        raise RuntimeError(
            f"train TF-IDF differs from word_sentence_tfidf by {tfidf_diff}"
        )
    if reference.shape[1] == 0:
        official = np.zeros((n_train, rank), dtype=np.float64)
        return {
            "vocabulary": tuple(vocabulary),
            "document_frequency": dict(document_frequency),
            "idf": np.zeros((0,), dtype=np.float64),
            "n_train": n_train,
            "rank": rank,
            "factors": None,
            "official_train": official,
            "train_tfidf_max_abs_diff": tfidf_diff,
            "train_embed_max_abs_diff": 0.0,
            "train_rows_at_or_below_floor": int(n_train),
        }
    factors = _svd_factors(reference, rank)
    from_left = _embed_from_left(factors, rank)
    raw_projection = _project_raw(reference, factors, rank)
    raw_left = factors["left"] * factors["singular"]
    if raw_left.shape[1] < rank:
        raw_left = np.pad(raw_left, ((0, 0), (0, rank - raw_left.shape[1])))
    raw_diff = _max_abs_diff(raw_projection, raw_left)
    official = np.asarray(word_sentence_svd(list(train_texts), rank=rank), dtype=np.float64)
    embed_diff = _max_abs_diff(from_left, official)
    if raw_diff > TRAIN_EMBED_ATOL or embed_diff > TRAIN_EMBED_ATOL:
        raise RuntimeError(
            "train SVD differs from word_sentence_svd "
            f"(left {embed_diff}, projection {raw_diff})"
        )
    left_norms = np.linalg.norm(raw_left, axis=1)
    stable = left_norms > 1e-6
    projected = _project(reference, factors, rank)
    if stable.any():
        stable_diff = _max_abs_diff(projected[stable], official[stable])
        if stable_diff > 1e-6:
            raise RuntimeError(
                f"stable train rows differ after projection by {stable_diff}"
            )
    idf = np.array(
        [
            math.log(n_train / document_frequency[token]) + 1.0
            for token in vocabulary
        ],
        dtype=np.float64,
    )
    return {
        "vocabulary": tuple(vocabulary),
        "document_frequency": dict(document_frequency),
        "idf": idf,
        "n_train": n_train,
        "rank": rank,
        "factors": factors,
        "official_train": official,
        "train_tfidf_max_abs_diff": tfidf_diff,
        "train_embed_max_abs_diff": float(max(embed_diff, raw_diff)),
        "train_rows_at_or_below_floor": int(np.sum(left_norms <= PROJECTION_NORM_FLOOR)),
    }


def transform_train_only(encoder: Mapping, texts: Sequence[str]) -> np.ndarray:
    """Project ``texts`` with a frozen train encoder.

    Tokens absent from the train vocabulary contribute nothing. They do not
    add a column and they do not change document frequency. Per-row L2 uses
    the row being transformed.
    """
    vocabulary = list(encoder["vocabulary"])
    document_frequency = encoder["document_frequency"]
    n_train = int(encoder["n_train"])
    rank = int(encoder["rank"])
    matrix = _tfidf_matrix(texts, vocabulary, document_frequency, n_train)
    factors = encoder["factors"]
    if factors is None:
        return np.zeros((len(texts), rank), dtype=np.float64)
    return _project(matrix, factors, rank)


def run_analytical_controls() -> dict:
    """Inequalities on a separated pair of blobs and on the duplicate detector.

    These cases do not read the fixtures. Both controls must land strictly
    under the real eval purity, and the real purity must land strictly above
    the eval majority base rate.
    """
    blobs = _two_blobs()
    duplicates = _duplicate_detector_check()
    checks = {
        "blob_real_above_base_rate": int(blobs["real_hits"]) > int(blobs["base_hits"]),
        "blob_label_control_below": int(blobs["label_hits"]) < int(blobs["real_hits"]),
        "blob_embedding_control_below": int(blobs["embedding_hits"]) < int(blobs["real_hits"]),
        "duplicate_empty_is_empty": bool(duplicates["empty_ids"] == 0 and duplicates["empty_texts"] == 0),
        "duplicate_planted_id_detected": int(duplicates["planted_ids"]) > 0,
        "duplicate_planted_text_detected": int(duplicates["planted_texts"]) > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "blobs": blobs,
        "duplicates": duplicates,
    }


def split_fingerprint(root: Optional[Path] = None) -> dict:
    """Eval observation ids under the frozen file-order rule."""
    fixture_root = Path(root) if root is not None else _default_root()
    fixtures = []
    for spec in FIXTURE_SPECS:
        loaded = load_fixture(fixture_root / str(spec["dirname"]))
        n_rows = len(loaded["texts"])
        mask = holdout_mask(n_rows)
        if not np.array_equal(mask, (np.arange(n_rows) % HOLDOUT_MODULUS) == HOLDOUT_REMAINDER):
            raise RuntimeError("holdout mask drifted from the frozen modulus")
        eval_index = np.flatnonzero(mask)
        train_index = np.flatnonzero(~mask)
        fixtures.append(
            {
                "fixture_id": str(spec["id"]),
                "n_rows": n_rows,
                "n_train": int(train_index.size),
                "n_eval": int(eval_index.size),
                "eval_ids": [loaded["observation_ids"][int(index)] for index in eval_index],
                "train_index_head": [int(index) for index in train_index[:3]],
                "eval_index": [int(index) for index in eval_index],
            }
        )
    payload = _fingerprint_payload(fixtures)
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    counts_ok = all(
        fixture["fixture_id"] == expected[0]
        and fixture["n_train"] == expected[1]
        and fixture["n_eval"] == expected[2]
        for fixture, expected in zip(fixtures, FROZEN_SPLIT_COUNTS)
    )
    constants_ok = (
        HOLDOUT_MODULUS == 5
        and HOLDOUT_REMAINDER == 4
        and len(fixtures) == len(FROZEN_SPLIT_COUNTS)
    )
    return {
        "rule": FROZEN_SPLIT_RULE,
        "modulus": HOLDOUT_MODULUS,
        "remainder": HOLDOUT_REMAINDER,
        "sha256": sha,
        "payload": payload,
        "fixtures": fixtures,
        "reproduced": bool(constants_ok and counts_ok and sha == FROZEN_SPLIT_SHA256),
    }


def run_encoder_isolation(
    root: Optional[Path] = None,
    *,
    time_runs: int = TIME_RUNS,
) -> dict:
    """Score every generalization fixture with a train-only encoder.

    Analytical inequalities and the split fingerprint run before any
    inductive purity. A failure of either one returns ``UNRESOLVED`` and
    does not embed the fixtures.
    """
    if time_runs < 0:
        raise ValueError(f"time_runs must be >= 0, got {time_runs}")
    started = time.perf_counter()
    analytical = run_analytical_controls()
    fingerprint = split_fingerprint(root)
    if not analytical["passed"] or not fingerprint["reproduced"]:
        disposition = authority_disposition(VERDICT_UNRESOLVED)
        return _report(
            disposition=disposition,
            analytical=analytical,
            fingerprint=fingerprint,
            fixtures=[],
            controls_fall=False,
            coverage_complete=False,
            coverage_reason_defined=True,
            every_fixture_passes=False,
            any_base_rate_miss=False,
            elapsed_s=time.perf_counter() - started,
            time_runs=time_runs,
        )
    fixture_root = Path(root) if root is not None else _default_root()
    pin_blas_threads(1)
    fixtures = []
    checked_rng = False
    for spec, finger in zip(FIXTURE_SPECS, fingerprint["fixtures"]):
        loaded = load_fixture(fixture_root / str(spec["dirname"]))
        if not checked_rng:
            mask = holdout_mask(len(loaded["texts"]))
            train_texts = [loaded["texts"][int(index)] for index in np.flatnonzero(~mask)]
            probe = np.asarray(word_sentence_svd(train_texts, rank=RANK), dtype=np.float64)
            _assert_production_ignores_rng(probe)
            checked_rng = True
        fixtures.append(
            _score_fixture(
                str(spec["id"]),
                loaded,
                finger,
                time_runs=time_runs,
            )
        )
    controls_fall = all(_controls_fell(fixture) for fixture in fixtures)
    coverage_complete = all(bool(fixture["coverage"]["complete"]) for fixture in fixtures)
    coverage_reason_defined = all(
        fixture["coverage"]["reason"] in DEFINED_COVERAGE_REASONS for fixture in fixtures
    )
    every_fixture_passes = all(bool(fixture["gates"]["pass"]) for fixture in fixtures)
    any_base_rate_miss = any(not bool(fixture["gates"]["above_base_rate"]) for fixture in fixtures)
    verdict = inductive_verdict(
        split_reproduced=True,
        controls_fall=controls_fall,
        coverage_complete=coverage_complete,
        coverage_reason_defined=coverage_reason_defined,
        every_fixture_passes=every_fixture_passes,
        any_base_rate_miss=any_base_rate_miss,
    )
    failing = [
        str(fixture["diagnostic"]["cause"])
        for fixture in fixtures
        if not fixture["gates"]["above_base_rate"]
    ]
    return _report(
        disposition=authority_disposition(verdict),
        analytical=analytical,
        fingerprint=fingerprint,
        fixtures=fixtures,
        controls_fall=controls_fall,
        coverage_complete=coverage_complete,
        coverage_reason_defined=coverage_reason_defined,
        every_fixture_passes=every_fixture_passes,
        any_base_rate_miss=any_base_rate_miss,
        elapsed_s=time.perf_counter() - started,
        time_runs=time_runs,
        diagnostic_cause=_aggregate_cause(failing),
    )


def _score_fixture(
    fixture_id: str,
    loaded: Mapping,
    finger: Mapping,
    *,
    time_runs: int,
) -> dict:
    texts = list(loaded["texts"])
    topics = list(loaded["topics"])
    observation_ids = list(loaded["observation_ids"])
    eval_index = [int(index) for index in finger["eval_index"]]
    train_index = [index for index in range(len(texts)) if index not in set(eval_index)]
    if [observation_ids[index] for index in eval_index] != list(finger["eval_ids"]):
        raise RuntimeError(f"{fixture_id} eval ids drifted from the fingerprint")
    train_texts = [texts[index] for index in train_index]
    eval_texts = [texts[index] for index in eval_index]
    train_topics = [topics[index] for index in train_index]
    eval_topics = [topics[index] for index in eval_index]
    train_ids = [observation_ids[index] for index in train_index]
    eval_ids = [observation_ids[index] for index in eval_index]

    inductive = _score_inductive(train_texts, eval_texts, train_topics, eval_topics)
    # Transductive control and the subspace diagnostic run after the inductive
    # score is fixed. They do not update the frozen encoder.
    transductive = _transductive_control(texts, train_index, eval_index, train_topics, eval_topics)
    diagnostic = _diagnostic(
        texts,
        train_index,
        inductive,
        transductive,
        purity_failed=not bool(inductive["above_base_rate"]),
    )
    duplicates = cross_split_duplicates(train_ids, eval_ids, train_texts, eval_texts)
    file_duplicate_ids = int(len(observation_ids) - len(set(observation_ids)))
    file_duplicate_texts = int(len(texts) - len(set(texts)))
    costs = _costs(train_texts, eval_texts, inductive["train_matrix"], time_runs)
    gates = _fixture_gates(inductive, duplicates)
    base_counts = Counter(eval_topics)
    return {
        "fixture_id": fixture_id,
        "n_train": len(train_texts),
        "n_eval": len(eval_texts),
        "eval_ids": eval_ids,
        "train_class_counts": dict(sorted(Counter(train_topics).items())),
        "eval_class_counts": dict(sorted(base_counts.items())),
        "eval_majority_hits": int(inductive["base_hits"]),
        "eval_majority_base_rate": float(inductive["base_rate"]),
        "inductive": _public_inductive(inductive),
        "transductive": transductive,
        "controls": inductive["controls"],
        "duplicates": {
            "shared_ids": duplicates["shared_ids"],
            "shared_texts": duplicates["shared_texts"],
            "duplicate_ids_in_file": file_duplicate_ids,
            "duplicate_texts_in_file": file_duplicate_texts,
            "empty_cross_split": duplicates["empty"],
        },
        "coverage": inductive["coverage"],
        "tokens": inductive["tokens"],
        "diagnostic": diagnostic,
        "costs": costs,
        "gates": gates,
        "train_tfidf_max_abs_diff": inductive["train_tfidf_max_abs_diff"],
        "train_embed_max_abs_diff": inductive["train_embed_max_abs_diff"],
        "production_train_labels_match_frozen_basis": True,
    }


def _score_inductive(
    train_texts: Sequence[str],
    eval_texts: Sequence[str],
    train_topics: Sequence[str],
    eval_topics: Sequence[str],
) -> dict:
    encoder = fit_train_only_encoder(train_texts, rank=RANK)
    eval_matrix = transform_train_only(encoder, eval_texts)
    train_matrix = np.asarray(encoder["official_train"], dtype=np.float64)
    if train_matrix.shape != (len(train_texts), RANK):
        raise RuntimeError(f"train embedding shape {train_matrix.shape}")
    if eval_matrix.shape != (len(eval_texts), RANK):
        raise RuntimeError(f"eval embedding shape {eval_matrix.shape}")
    vocab = set(encoder["vocabulary"])
    train_token_set = set()
    for text in train_texts:
        train_token_set.update(content_tokens(text))
    if not vocab <= train_token_set:
        raise RuntimeError("train vocabulary contains a token absent from train")
    spec = _production_spec()
    train_labels, centroids, _indices = _fit_algorithm(train_matrix, spec)
    eval_labels = _assign(eval_matrix, centroids, "euclidean")
    # Topic strings start here. The encoder and the centroid fit are done.
    mapping = train_only_majority_map(train_labels, train_topics)
    train_score = score_against_map(train_labels, train_topics, mapping)
    eval_score = score_against_map(eval_labels, eval_topics, mapping)
    base_hits = int(max(Counter(eval_topics).values()))
    base_rate = base_hits / float(len(eval_topics))
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
    coverage = _coverage(eval_matrix)
    tokens = _token_coverage(eval_texts, vocab)
    train_sq = _row_squared_distance(train_matrix, train_labels, centroids)
    eval_sq = _row_squared_distance(eval_matrix, eval_labels, centroids)
    nearest = _nearest_squared_distance(eval_matrix, centroids)
    if float(np.max(np.abs(eval_sq - nearest))) > 1e-8:
        raise RuntimeError("eval assignment is not the nearest train centroid")
    return {
        "encoder": encoder,
        "train_matrix": train_matrix,
        "eval_matrix": eval_matrix,
        "train_labels": train_labels,
        "eval_labels": eval_labels,
        "centroids": centroids,
        "mapping": mapping,
        "train_hits": int(train_score["hits"]),
        "train_rows": int(train_score["rows"]),
        "train_purity": float(train_score["size_weighted"]),
        "eval_hits": int(eval_score["hits"]),
        "eval_rows": int(eval_score["rows"]),
        "eval_purity": float(eval_score["size_weighted"]),
        "base_hits": base_hits,
        "base_rate": float(base_rate),
        "above_base_rate": bool(int(eval_score["hits"]) > base_hits),
        "train_residual": float(_prototype_residual(train_matrix, train_labels, centroids)),
        "eval_residual": float(_prototype_residual(eval_matrix, eval_labels, centroids)),
        "train_max_squared_distance": float(np.max(train_sq)) if train_sq.size else 0.0,
        "eval_squared_distance": eval_sq,
        "vocabulary_size": len(vocab),
        "map_fit_rows": len(train_topics),
        "controls": controls,
        "coverage": coverage,
        "tokens": tokens,
        "train_tfidf_max_abs_diff": float(encoder["train_tfidf_max_abs_diff"]),
        "train_embed_max_abs_diff": float(encoder["train_embed_max_abs_diff"]),
        "train_rows_at_or_below_floor": int(encoder["train_rows_at_or_below_floor"]),
    }


def _inductive_controls(
    eval_matrix: np.ndarray,
    centroids: np.ndarray,
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    train_topics: Sequence[str],
    eval_topics: Sequence[str],
    mapping: Mapping[int, str],
    real_hits: int,
) -> dict:
    """Shuffled train map, and eval rows permuted against the real map.

    The label control fits the majority map on the shuffled train topics and
    scores the real eval topics. The row control assigns permuted eval
    embeddings to the real train centroids and scores them with the real map.
    Centroids are not refit. The map is not refit on eval topics.
    """
    shuffled_train = permute_sequence(list(train_topics), INDUCTIVE_LABEL_SHUFFLE_SEED)
    shuffled_map = train_only_majority_map(train_labels, shuffled_train)
    label_score = score_against_map(eval_labels, eval_topics, shuffled_map)
    shuffled_eval = permute_rows(eval_matrix, INDUCTIVE_ROW_SHUFFLE_SEED)
    permuted_labels = _assign(shuffled_eval, centroids, "euclidean")
    row_score = score_against_map(permuted_labels, eval_topics, mapping)
    label_hits = int(label_score["hits"])
    row_hits = int(row_score["hits"])
    return {
        "label_shuffle_seed": INDUCTIVE_LABEL_SHUFFLE_SEED,
        "label_shuffle_scope": "train_topics_only",
        "label_hits": label_hits,
        "label_rows": int(label_score["rows"]),
        "label_purity": float(label_score["size_weighted"]),
        "label_below": bool(label_hits < real_hits),
        "row_shuffle_seed": INDUCTIVE_ROW_SHUFFLE_SEED,
        "row_shuffle_scope": "eval_embeddings_assigned_to_real_train_centroids",
        "row_hits": row_hits,
        "row_rows": int(row_score["rows"]),
        "row_purity": float(row_score["size_weighted"]),
        "row_below": bool(row_hits < real_hits),
        "map_fit_on_eval": False,
        "centroids_refit": False,
    }


def _transductive_control(
    texts: Sequence[str],
    train_index: Sequence[int],
    eval_index: Sequence[int],
    train_topics: Sequence[str],
    eval_topics: Sequence[str],
) -> dict:
    """Whole-fixture SVD and k-means on the train rows of that matrix.

    Eval rows change the vocabulary, the IDF, and the basis. They do not
    enter the centroid update. The reported eval purity uses the majority
    map of those train labels. Self-majority on the eval slice is recorded
    beside it and is not the control score. This block is not a pass gate.
    """
    matrix = np.asarray(word_sentence_svd(list(texts), rank=RANK), dtype=np.float64)
    train_matrix = matrix[np.asarray(train_index, dtype=int)]
    eval_matrix = matrix[np.asarray(eval_index, dtype=int)]
    spec = _production_spec()
    train_labels, centroids, _indices = _fit_algorithm(train_matrix, spec)
    eval_labels = _assign(eval_matrix, centroids, "euclidean")
    mapping = train_only_majority_map(train_labels, train_topics)
    train_score = score_against_map(train_labels, train_topics, mapping)
    eval_score = score_against_map(eval_labels, eval_topics, mapping)
    self_majority = _score_assignment(eval_labels, eval_topics)
    return {
        "label": TRANSDUCTIVE_LABEL,
        "encoder": TRANSDUCTIVE_ENCODER,
        "clustering": TRANSDUCTIVE_CLUSTERING,
        "map": "train_rows_of_the_transductive_assignment",
        "is_pass_gate": False,
        "is_inductive_result": False,
        "train_hits": int(train_score["hits"]),
        "train_rows": int(train_score["rows"]),
        "train_purity": float(train_score["size_weighted"]),
        "eval_hits": int(eval_score["hits"]),
        "eval_rows": int(eval_score["rows"]),
        "eval_purity": float(eval_score["size_weighted"]),
        "train_residual": float(_prototype_residual(train_matrix, train_labels, centroids)),
        "eval_residual": float(_prototype_residual(eval_matrix, eval_labels, centroids)),
        "self_majority_eval_hits": int(self_majority["hits"]),
        "self_majority_eval_purity": float(self_majority["size_weighted"]),
        "self_majority_note": "eval labels name the eval clusters; not the control score",
        "train_labels": train_labels,
        "eval_labels": eval_labels,
        "train_matrix": train_matrix,
        "centroids": centroids,
    }


def _diagnostic(
    texts: Sequence[str],
    train_index: Sequence[int],
    inductive: Mapping,
    transductive: Mapping,
    *,
    purity_failed: bool,
) -> dict:
    """Ask which geometry moved. The answer is not an admission gate."""
    angles = _train_subspace_angles(texts, train_index)
    rotation = _procrustes_rotation(inductive["train_matrix"], transductive["train_matrix"])
    aligned = inductive["train_matrix"] @ rotation
    unaligned_cosine = _paired_mean_cosine(inductive["train_matrix"], transductive["train_matrix"])
    aligned_cosine = _paired_mean_cosine(aligned, transductive["train_matrix"])
    target_norm = float(np.linalg.norm(transductive["train_matrix"]))
    procrustes_relative = (
        float(np.linalg.norm(aligned - transductive["train_matrix"]) / target_norm)
        if target_norm
        else float("nan")
    )
    displacement = _matched_centroid_displacement(
        inductive["centroids"],
        transductive["centroids"],
        rotation,
    )
    far_fraction = float(
        np.mean(inductive["eval_squared_distance"] > inductive["train_max_squared_distance"])
    )
    distance_correlation = _pairwise_distance_correlation(
        inductive["train_matrix"],
        transductive["train_matrix"],
    )
    train_ari = float(
        adjusted_rand_index(transductive["train_labels"], inductive["train_labels"])
    )
    eval_ari = float(
        adjusted_rand_index(transductive["eval_labels"], inductive["eval_labels"])
    )
    max_angle = float(np.max(angles)) if angles.size else float("nan")
    # Procrustes residual uses the centroid displacement scale. A cloud that
    # still disagrees after the best rotation has moved, even when the
    # shared-vocabulary principal angle stays under its cutoff.
    subspace_moved = bool(
        (np.isfinite(max_angle) and max_angle > DIAGNOSTIC_MAX_PRINCIPAL_ANGLE_DEG)
        or (
            np.isfinite(procrustes_relative)
            and procrustes_relative > DIAGNOSTIC_CENTROID_MEAN_DISPLACEMENT
        )
    )
    centroids_moved = bool(displacement > DIAGNOSTIC_CENTROID_MEAN_DISPLACEMENT)
    eval_far = bool(far_fraction > DIAGNOSTIC_EVAL_FAR_FRACTION)
    measured = bool(np.isfinite(max_angle) and np.isfinite(displacement) and np.isfinite(far_fraction))
    if not measured:
        cause = CAUSE_UNRESOLVED if purity_failed else CAUSE_NOT_APPLICABLE
    else:
        cause = isolate_cause(
            purity_failed=purity_failed,
            subspace_moved=subspace_moved,
            centroids_moved=centroids_moved,
            eval_far=eval_far,
        )
    return {
        "purity_failed": bool(purity_failed),
        "cause": cause,
        "is_gate": False,
        "max_principal_angle_deg": max_angle,
        "principal_angles_deg": [float(angle) for angle in angles.tolist()],
        "subspace_moved": subspace_moved if measured else None,
        "subspace_space": "shared_train_vocabulary_right_singular_subspace",
        "unaligned_mean_cosine": unaligned_cosine,
        "aligned_mean_cosine": aligned_cosine,
        "procrustes_relative_residual": procrustes_relative,
        "centroid_mean_l2_displacement": displacement,
        "centroids_moved": centroids_moved if measured else None,
        "eval_far_fraction": far_fraction,
        "eval_far": eval_far if measured else None,
        "train_max_squared_distance": float(inductive["train_max_squared_distance"]),
        "train_pairwise_distance_correlation": distance_correlation,
        "train_ari": train_ari,
        "eval_ari": eval_ari,
        "cutoffs": {
            "max_principal_angle_deg": DIAGNOSTIC_MAX_PRINCIPAL_ANGLE_DEG,
            "centroid_mean_l2_displacement": DIAGNOSTIC_CENTROID_MEAN_DISPLACEMENT,
            "procrustes_relative_residual": DIAGNOSTIC_CENTROID_MEAN_DISPLACEMENT,
            "eval_far_fraction": DIAGNOSTIC_EVAL_FAR_FRACTION,
        },
    }


def _train_subspace_angles(texts: Sequence[str], train_index: Sequence[int]) -> np.ndarray:
    """Principal angles between train-only and full-matrix train subspaces.

    Both bases live in the train vocabulary's coordinates. The full-matrix
    side is the production TF-IDF of every row, restricted to train rows and
    to train-vocabulary columns. IDF weights on that side saw eval rows.
    Angles are degrees, largest first.
    """
    train_texts = [texts[int(index)] for index in train_index]
    train_vocab = _document_frequency(train_texts)[0]
    full_vocab = _document_frequency(texts)[0]
    train_tfidf = word_sentence_tfidf(train_texts)
    full_tfidf = word_sentence_tfidf(list(texts))
    if train_tfidf.shape[1] != len(train_vocab) or full_tfidf.shape[1] != len(full_vocab):
        raise RuntimeError("vocabulary width drifted from word_sentence_tfidf")
    full_index = {token: position for position, token in enumerate(full_vocab)}
    columns = [full_index[token] for token in train_vocab]
    restricted = full_tfidf[np.asarray(list(train_index), dtype=int)][:, columns]
    left = _orthonormal_right_basis(train_tfidf, RANK)
    right = _orthonormal_right_basis(restricted, RANK)
    angles = _principal_angles_deg(left, right)
    return np.sort(angles)[::-1]


def _fixture_gates(inductive: Mapping, duplicates: Mapping) -> dict:
    coverage_ok = bool(inductive["coverage"]["complete"])
    above = bool(inductive["above_base_rate"])
    label_ok = bool(inductive["controls"]["label_below"])
    row_ok = bool(inductive["controls"]["row_below"])
    leakage_ok = bool(duplicates["empty"])
    return {
        "coverage": coverage_ok,
        "above_base_rate": above,
        "label_control_below": label_ok,
        "row_control_below": row_ok,
        "no_shared_ids_or_texts": leakage_ok,
        "pass": bool(coverage_ok and above and label_ok and row_ok and leakage_ok),
    }


def _controls_fell(fixture: Mapping) -> bool:
    controls = fixture["controls"]
    duplicates = fixture["duplicates"]
    return bool(
        controls["label_below"]
        and controls["row_below"]
        and duplicates["empty_cross_split"]
    )


def _coverage(matrix: np.ndarray) -> dict:
    finite = np.isfinite(matrix).all(axis=1)
    rows = int(matrix.shape[0])
    finite_rows = int(finite.sum())
    reason = COVERAGE_COMPLETE if finite_rows == rows else COVERAGE_NON_FINITE
    return {
        "finite_rows": finite_rows,
        "rows": rows,
        "fraction": float(finite_rows / rows) if rows else 0.0,
        "complete": bool(finite_rows == rows and rows > 0),
        "reason": reason,
        "reason_defined": bool(reason in DEFINED_COVERAGE_REASONS),
        "zero_rows": int(np.all(matrix == 0.0, axis=1).sum()),
    }


def _token_coverage(eval_texts: Sequence[str], vocabulary: set) -> dict:
    occurrences = 0
    absent_occurrences = 0
    types: set = set()
    absent_types: set = set()
    rows_with_overlap = 0
    for text in eval_texts:
        tokens = content_tokens(text)
        if any(token in vocabulary for token in tokens):
            rows_with_overlap += 1
        for token in tokens:
            occurrences += 1
            types.add(token)
            if token not in vocabulary:
                absent_occurrences += 1
                absent_types.add(token)
    return {
        "eval_content_token_occurrences": int(occurrences),
        "absent_occurrences": int(absent_occurrences),
        "absence_fraction": float(absent_occurrences / occurrences) if occurrences else 0.0,
        "eval_content_token_types": int(len(types)),
        "absent_types": int(len(absent_types)),
        "type_absence_fraction": float(len(absent_types) / len(types)) if types else 0.0,
        "eval_rows_with_train_vocab_overlap": int(rows_with_overlap),
    }


def _costs(
    train_texts: Sequence[str],
    eval_texts: Sequence[str],
    train_matrix: np.ndarray,
    time_runs: int,
) -> dict:
    spec = _production_spec()

    def _embed() -> None:
        encoder = fit_train_only_encoder(train_texts, rank=RANK)
        transform_train_only(encoder, eval_texts)

    def _cluster() -> None:
        _fit_algorithm(train_matrix, spec)

    embed_ms, _embed_samples = _per_call_median_ms(_embed, time_runs, 1)
    cluster_ms, _cluster_samples = _per_call_median_ms(_cluster, time_runs, CLUSTER_TIME_REPEATS)
    embed_peak = _peak_bytes(_embed, time_runs)
    cluster_peak = _peak_bytes(_cluster, time_runs)
    return {
        "embed_median_ms": embed_ms,
        "cluster_median_ms": cluster_ms,
        "embed_peak_bytes": embed_peak,
        "cluster_peak_bytes": cluster_peak,
        "peak_bytes": _max_optional(embed_peak, cluster_peak),
        "time_runs": int(time_runs),
        "cluster_time_repeats": int(CLUSTER_TIME_REPEATS) if time_runs else 0,
    }


def _public_inductive(inductive: Mapping) -> dict:
    return {
        "train_hits": inductive["train_hits"],
        "train_rows": inductive["train_rows"],
        "train_purity": inductive["train_purity"],
        "train_residual": inductive["train_residual"],
        "eval_hits": inductive["eval_hits"],
        "eval_rows": inductive["eval_rows"],
        "eval_purity": inductive["eval_purity"],
        "eval_residual": inductive["eval_residual"],
        "vocabulary_size": inductive["vocabulary_size"],
        "map_fit_rows": inductive["map_fit_rows"],
        "map_source": "train_rows_only",
        "clustering": PRODUCTION_FAMILY,
        "train_rows_at_or_below_floor": inductive["train_rows_at_or_below_floor"],
    }


def _report(
    *,
    disposition: Mapping,
    analytical: Mapping,
    fingerprint: Mapping,
    fixtures: Sequence[Mapping],
    controls_fall: bool,
    coverage_complete: bool,
    coverage_reason_defined: bool,
    every_fixture_passes: bool,
    any_base_rate_miss: bool,
    elapsed_s: float,
    time_runs: int,
    diagnostic_cause: str = CAUSE_NOT_APPLICABLE,
) -> dict:
    public_fingerprint = {
        "rule": fingerprint["rule"],
        "modulus": fingerprint["modulus"],
        "remainder": fingerprint["remainder"],
        "sha256": fingerprint["sha256"],
        "reproduced": fingerprint["reproduced"],
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
    public_fixtures = [_public_fixture(fixture) for fixture in fixtures]
    return {
        "status": STATUS,
        "baseline_commit": BASELINE_COMMIT,
        "entering_verdict": ENTERING_VERDICT,
        "entering_recommendation": ENTERING_RECOMMENDATION,
        "verdict": disposition["verdict"],
        "recommendation": disposition["recommendation"],
        "runtime_authority": disposition["runtime_authority"],
        "production_replaced": disposition["production_replaced"],
        "analytical_passed": bool(analytical["passed"]),
        "split_reproduced": bool(fingerprint["reproduced"]),
        "controls_fall": bool(controls_fall),
        "coverage_complete": bool(coverage_complete),
        "coverage_reason_defined": bool(coverage_reason_defined),
        "every_fixture_passes": bool(every_fixture_passes),
        "any_base_rate_miss": bool(any_base_rate_miss),
        "diagnostic_cause": diagnostic_cause,
        "gates": {
            "coverage": COVERAGE_GATE,
            "purity": PURITY_GATE,
            "label_control": LABEL_CONTROL_GATE,
            "row_control": ROW_CONTROL_GATE,
            "leakage": LEAKAGE_GATE,
        },
        "learned_on_train": list(LEARNED_ON_TRAIN),
        "deterministic_per_row": list(DETERMINISTIC_PER_ROW),
        "centering": CENTERING,
        "l2_scope": L2_SCOPE,
        "unknown_token_rule": UNKNOWN_TOKEN_RULE,
        "analytical": {
            "passed": analytical["passed"],
            "checks": analytical["checks"],
            "blobs": analytical["blobs"],
            "duplicates": analytical["duplicates"],
        },
        "split_fingerprint": public_fingerprint,
        "fixtures": public_fixtures,
        "costs": {
            "elapsed_s": float(elapsed_s),
            "time_runs": int(time_runs),
            "sum_of_embed_medians_ms": _sum_optional(
                fixture["costs"]["embed_median_ms"] for fixture in public_fixtures
            ),
            "sum_of_cluster_medians_ms": _sum_optional(
                fixture["costs"]["cluster_median_ms"] for fixture in public_fixtures
            ),
            "peak_bytes": _max_optional(
                *(
                    value
                    for fixture in public_fixtures
                    for value in (
                        fixture["costs"]["embed_peak_bytes"],
                        fixture["costs"]["cluster_peak_bytes"],
                    )
                )
            ),
        },
        "numerical_library": numerical_library(),
        "git_revision": _git_revision(),
        "rank": RANK,
        "k": K,
        "production_family": PRODUCTION_FAMILY,
        "production_svd_modified": False,
        "learn_pack_called": False,
        "candidates_rerun": False,
    }


def _public_fixture(fixture: Mapping) -> dict:
    """Drop embedding matrices and label arrays from the returned fixture."""
    transductive = {
        key: value
        for key, value in fixture["transductive"].items()
        if key not in {"train_labels", "eval_labels", "train_matrix", "centroids"}
    }
    diagnostic = dict(fixture["diagnostic"])
    return {
        "fixture_id": fixture["fixture_id"],
        "n_train": fixture["n_train"],
        "n_eval": fixture["n_eval"],
        "eval_ids": list(fixture["eval_ids"]),
        "train_class_counts": fixture["train_class_counts"],
        "eval_class_counts": fixture["eval_class_counts"],
        "eval_majority_hits": fixture["eval_majority_hits"],
        "eval_majority_base_rate": fixture["eval_majority_base_rate"],
        "inductive": fixture["inductive"],
        "transductive": transductive,
        "controls": fixture["controls"],
        "duplicates": fixture["duplicates"],
        "coverage": fixture["coverage"],
        "tokens": fixture["tokens"],
        "diagnostic": diagnostic,
        "costs": _finalize_costs(fixture),
        "gates": fixture["gates"],
        "train_tfidf_max_abs_diff": fixture["train_tfidf_max_abs_diff"],
        "train_embed_max_abs_diff": fixture["train_embed_max_abs_diff"],
    }


def _finalize_costs(fixture: Mapping) -> dict:
    costs = dict(fixture["costs"])
    costs["peak_bytes"] = _max_optional(costs["embed_peak_bytes"], costs["cluster_peak_bytes"])
    return costs


def _two_blobs() -> dict:
    """Two isotropic blobs. Production k-means, then the inductive controls."""
    rng = np.random.default_rng(ANALYTICAL_DATA_SEED)
    dim = ANALYTICAL_DIM
    center_b = np.zeros(dim)
    center_b[0] = ANALYTICAL_BLOB_GAP

    def _draw(count: int, center: np.ndarray) -> np.ndarray:
        return center + rng.normal(scale=ANALYTICAL_BLOB_SCALE, size=(count, dim))

    n_train = ANALYTICAL_BLOB_TRAIN_PER_CLASS
    n_eval = ANALYTICAL_BLOB_EVAL_PER_CLASS
    train_x = np.vstack([_draw(n_train, np.zeros(dim)), _draw(n_train, center_b)])
    eval_x = np.vstack([_draw(n_eval, np.zeros(dim)), _draw(n_eval, center_b)])
    train_y = ["alpha"] * n_train + ["beta"] * n_train
    eval_y = ["alpha"] * n_eval + ["beta"] * n_eval
    train_labels, centroids, _indices = _fit_algorithm(train_x, _production_spec())
    eval_labels = _assign(eval_x, centroids, "euclidean")
    mapping = train_only_majority_map(train_labels, train_y)
    real = score_against_map(eval_labels, eval_y, mapping)
    base_hits = n_eval
    shuffled_train = permute_sequence(train_y, INDUCTIVE_LABEL_SHUFFLE_SEED)
    shuffled_map = train_only_majority_map(train_labels, shuffled_train)
    label_score = score_against_map(eval_labels, eval_y, shuffled_map)
    shuffled_eval = permute_rows(eval_x, INDUCTIVE_ROW_SHUFFLE_SEED)
    embed_labels = _assign(shuffled_eval, centroids, "euclidean")
    embed_score = score_against_map(embed_labels, eval_y, mapping)
    duplicates = cross_split_duplicates(
        [f"train-{index}" for index in range(train_x.shape[0])],
        [f"eval-{index}" for index in range(eval_x.shape[0])],
        [f"train-text-{index}" for index in range(train_x.shape[0])],
        [f"eval-text-{index}" for index in range(eval_x.shape[0])],
    )
    return {
        "n_eval": int(real["rows"]),
        "base_hits": int(base_hits),
        "base_rate": float(base_hits / real["rows"]),
        "real_hits": int(real["hits"]),
        "real_purity": float(real["size_weighted"]),
        "label_hits": int(label_score["hits"]),
        "label_purity": float(label_score["size_weighted"]),
        "embedding_hits": int(embed_score["hits"]),
        "embedding_purity": float(embed_score["size_weighted"]),
        "duplicates_empty": bool(duplicates["empty"]),
    }


def _duplicate_detector_check() -> dict:
    empty = cross_split_duplicates(["a"], ["b"], ["text-a"], ["text-b"])
    planted = cross_split_duplicates(
        ["a", "c"],
        ["b", "a"],
        ["text-1", "text-2"],
        ["text-3", "text-1"],
    )
    return {
        "empty_ids": int(empty["shared_ids"]),
        "empty_texts": int(empty["shared_texts"]),
        "planted_ids": int(planted["shared_ids"]),
        "planted_texts": int(planted["shared_texts"]),
    }


def _document_frequency(texts: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
    document_frequency: Dict[str, int] = {}
    for text in texts:
        for token in set(content_tokens(text or "")):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    return sorted(document_frequency), document_frequency


def _tfidf_matrix(
    texts: Sequence[str],
    vocabulary: Sequence[str],
    document_frequency: Mapping[str, int],
    n_documents: int,
) -> np.ndarray:
    """TF-IDF with a frozen vocabulary and frozen document counts.

    ``n_documents`` is the training row count. It is not ``len(texts)`` when
    the texts are eval rows. Unknown tokens are skipped.
    """
    if n_documents < 1:
        raise ValueError(f"n_documents must be positive, got {n_documents}")
    index = {token: position for position, token in enumerate(vocabulary)}
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float64)
    for row_index, text in enumerate(texts):
        counts: Dict[str, int] = {}
        for token in content_tokens(text or ""):
            if token not in index:
                continue
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            idf = math.log(n_documents / document_frequency[token]) + 1.0
            matrix[row_index, index[token]] = float(count) * idf
    return _renorm_rows(matrix)


def _svd_factors(matrix: np.ndarray, rank: int) -> dict:
    width = min(rank, matrix.shape[0], matrix.shape[1])
    pin_blas_threads(1)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    left = np.array(left[:, :width], dtype=np.float64, copy=True)
    singular = np.asarray(singular[:width], dtype=np.float64)
    right = np.array(right[:width, :], dtype=np.float64, copy=True)
    for column in range(width):
        pivot = int(np.argmax(np.abs(left[:, column])))
        if left[pivot, column] < 0.0:
            left[:, column] *= -1.0
            right[column, :] *= -1.0
    return {"left": left, "singular": singular, "right": right, "width": width}


def _embed_from_left(factors: Mapping, rank: int) -> np.ndarray:
    embedded = factors["left"] * factors["singular"]
    width = int(factors["width"])
    if width < rank:
        embedded = np.pad(embedded, ((0, 0), (0, rank - width)))
    return _renorm_rows(embedded)


def _project_raw(matrix: np.ndarray, factors: Mapping, rank: int) -> np.ndarray:
    """``X @ Vt.T``, padded to ``rank``. This equals ``U * S`` on the fit matrix."""
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], rank), dtype=np.float64)
    embedded = matrix @ np.ascontiguousarray(factors["right"].T)
    if embedded.shape[1] < rank:
        embedded = np.pad(embedded, ((0, 0), (0, rank - embedded.shape[1])))
    return np.asarray(embedded, dtype=np.float64)


def _project(matrix: np.ndarray, factors: Mapping, rank: int) -> np.ndarray:
    """Project rows into the production ``U * S`` coordinates, then L2-normalize.

    ``np.linalg.svd`` returns ``Vt`` with ``X @ Vt.T == U * S`` on the
    matrix that was factored. The singular values are already inside that
    product. Rows at or below ``PROJECTION_NORM_FLOOR`` stay zero so
    roundoff is not renormalized into a direction.
    """
    embedded = _project_raw(matrix, factors, rank)
    norms = np.linalg.norm(embedded, axis=1)
    embedded = np.array(embedded, dtype=np.float64, copy=True)
    embedded[norms <= PROJECTION_NORM_FLOOR] = 0.0
    return _renorm_rows(embedded)


def _renorm_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)


def _orthonormal_right_basis(matrix: np.ndarray, rank: int) -> np.ndarray:
    width = min(rank, matrix.shape[0], matrix.shape[1])
    pin_blas_threads(1)
    _left, _singular, right = np.linalg.svd(matrix, full_matrices=False)
    basis = np.asarray(right[:width, :], dtype=np.float64).T
    q_factor, _r_factor = np.linalg.qr(basis)
    return np.asarray(q_factor[:, :width], dtype=np.float64)


def _principal_angles_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    k = min(int(left.shape[1]), int(right.shape[1]))
    singular = np.linalg.svd(left[:, :k].T @ right[:, :k], compute_uv=False)
    singular = np.clip(np.asarray(singular, dtype=np.float64), 0.0, 1.0)
    return np.degrees(np.arccos(singular))


def _procrustes_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Orthogonal map from ``source`` rows toward ``target`` rows. Reflections allowed."""
    u_factor, _singular, vt = np.linalg.svd(source.T @ target, full_matrices=False)
    return u_factor @ vt


def _matched_centroid_displacement(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
) -> float:
    aligned = source @ rotation
    delta = aligned[:, None, :] - target[None, :, :]
    cost = np.sqrt(np.sum(delta ** 2, axis=2))
    if cost.shape[0] != cost.shape[1]:
        raise RuntimeError("centroid counts differ")
    best = None
    for perm in itertools.permutations(range(cost.shape[0])):
        total = float(sum(cost[index, perm[index]] for index in range(cost.shape[0])))
        if best is None or total < best:
            best = total
    return float(best / cost.shape[0])


def _paired_mean_cosine(left: np.ndarray, right: np.ndarray) -> float:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    mask = denominator > 0.0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(numerator[mask] / denominator[mask]))


def _pairwise_distance_correlation(left: np.ndarray, right: np.ndarray) -> float:
    def _upper(matrix: np.ndarray) -> np.ndarray:
        distances = ((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2)
        index = np.triu_indices(matrix.shape[0], 1)
        return distances[index]

    return float(np.corrcoef(_upper(left), _upper(right))[0, 1])


def _row_squared_distance(
    matrix: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    assigned = centroids[np.asarray(labels)]
    return np.sum((matrix - assigned) ** 2, axis=1)


def _nearest_squared_distance(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dist = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return dist.min(axis=1)


def _fingerprint_payload(fixtures: Sequence[Mapping]) -> str:
    return "\n".join(
        fixture["fixture_id"] + ":" + ",".join(fixture["eval_ids"]) for fixture in fixtures
    )


def _aggregate_cause(causes: Sequence[str]) -> str:
    if not causes:
        return CAUSE_NOT_APPLICABLE
    if len(set(causes)) == 1 and causes[0] != CAUSE_UNRESOLVED:
        return causes[0]
    return CAUSE_UNRESOLVED


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def _sum_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    items = list(values)
    if not items or any(value is None for value in items):
        return None
    return float(sum(float(value) for value in items))


def _max_optional(*values: Optional[int]) -> Optional[int]:
    numbers = [int(value) for value in values if value is not None]
    if not numbers:
        return None
    return int(max(numbers))


def numerical_library() -> dict:
    """Numpy build config and the loaded BLAS/LAPACK shared objects."""
    loaded: List[str] = []
    maps = Path("/proc/self/maps")
    if maps.exists():
        for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
            lower = line.lower()
            if not any(token in lower for token in ("libblas", "liblapack", "libopenblas", "libmkl")):
                continue
            path = line.split()[-1]
            if path.startswith("/") and path not in loaded:
                loaded.append(path)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_config()
    return {
        "numpy": np.__version__,
        "loaded_libraries": loaded,
        "show_config": buffer.getvalue(),
    }


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _default_root() -> Path:
    return _repo_root() / "examples" / "generalization"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _format_report(report: Mapping) -> str:
    lines = [
        f"verdict {report['verdict']}",
        f"recommendation {report['recommendation']}",
        f"runtime_authority {report['runtime_authority']}",
        f"production_replaced {report['production_replaced']}",
        f"analytical_passed {report['analytical_passed']}",
        f"split_reproduced {report['split_reproduced']}",
        f"controls_fall {report['controls_fall']}",
        f"coverage_complete {report['coverage_complete']}",
        f"any_base_rate_miss {report['any_base_rate_miss']}",
        f"diagnostic_cause {report['diagnostic_cause']}",
        f"sha256 {report['split_fingerprint']['sha256']}",
        f"git {report['git_revision']}",
        f"elapsed_s {report['costs']['elapsed_s']:.3f}",
        f"libraries {report['numerical_library']['loaded_libraries']}",
        f"numpy {report['numerical_library']['numpy']}",
        (
            "blobs "
            f"real {report['analytical']['blobs']['real_hits']}/"
            f"{report['analytical']['blobs']['n_eval']} "
            f"label {report['analytical']['blobs']['label_hits']} "
            f"embed {report['analytical']['blobs']['embedding_hits']} "
            f"base {report['analytical']['blobs']['base_hits']}"
        ),
    ]
    for fixture in report["fixtures"]:
        inductive = fixture["inductive"]
        transductive = fixture["transductive"]
        controls = fixture["controls"]
        diagnostic = fixture["diagnostic"]
        tokens = fixture["tokens"]
        costs = fixture["costs"]
        lines.append(
            " ".join(
                [
                    fixture["fixture_id"],
                    f"inductive {inductive['eval_hits']}/{inductive['eval_rows']}",
                    f"train {inductive['train_hits']}/{inductive['train_rows']}",
                    f"base {fixture['eval_majority_hits']}/{fixture['n_eval']}",
                    f"transductive {transductive['eval_hits']}/{transductive['eval_rows']}",
                    f"trans_train {transductive['train_hits']}/{transductive['train_rows']}",
                    f"self {transductive['self_majority_eval_hits']}",
                    f"train_resid {inductive['train_residual']:.6f}",
                    f"eval_resid {inductive['eval_residual']:.6f}",
                    f"trans_train_resid {transductive['train_residual']:.10f}",
                    f"coverage {fixture['coverage']['finite_rows']}/{fixture['coverage']['rows']}",
                    f"zero {fixture['coverage']['zero_rows']}",
                    f"absent {tokens['absent_occurrences']}/{tokens['eval_content_token_occurrences']}",
                    f"types {tokens['absent_types']}/{tokens['eval_content_token_types']}",
                    f"label {controls['label_hits']}",
                    f"row {controls['row_hits']}",
                    f"shared {fixture['duplicates']['shared_ids']}/{fixture['duplicates']['shared_texts']}",
                    f"train_ari {diagnostic['train_ari']:.10f}",
                    f"eval_ari {diagnostic['eval_ari']:.10f}",
                    f"corr {diagnostic['train_pairwise_distance_correlation']:.10f}",
                    f"angle {diagnostic['max_principal_angle_deg']:.4f}",
                    f"cos {diagnostic['unaligned_mean_cosine']:.4f}",
                    f"aligned_cos {diagnostic['aligned_mean_cosine']:.4f}",
                    f"procrustes {diagnostic['procrustes_relative_residual']:.4f}",
                    f"centroid {diagnostic['centroid_mean_l2_displacement']:.4f}",
                    f"far {diagnostic['eval_far_fraction']:.4f}",
                    f"cause {diagnostic['cause']}",
                    f"embed_ms {costs['embed_median_ms']}",
                    f"cluster_ms {costs['cluster_median_ms']}",
                    f"embed_peak {costs['embed_peak_bytes']}",
                    f"cluster_peak {costs['cluster_peak_bytes']}",
                    f"tfidf_diff {fixture['train_tfidf_max_abs_diff']}",
                    f"embed_diff {fixture['train_embed_max_abs_diff']}",
                    f"floor_rows {inductive['train_rows_at_or_below_floor']}",
                    f"gates {fixture['gates']}",
                ]
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(_format_report(run_encoder_isolation()))

"""Generalization probe for four frozen clustering candidates.

Model selection happened in the clustering-isolation trial. This module
does not add seeds, drop a candidate, or replace ``clustering.kmeans``.
``learn_pack`` is not called. Topic strings are a probe: they are not
passed to the encoder, the initializer, or the Lloyd update.

The advantage rule, the split, and the control seeds below are module
constants. They are fixed before any eval score is computed.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import subprocess
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neuralese.clustering import kmeans
from neuralese.clustering_isolation import (
    PRODUCTION_FAMILY,
    _assign,
    _fit_algorithm,
    _majority,
    _prototype_residual,
    adjusted_rand_index,
)
from neuralese.encode import ENCODER_WORD_SENTENCE_SVD, word_sentence_svd
from neuralese.factorization import pin_blas_threads

# Product cluster count. Not retuned per fixture. A fixture may have fewer
# topics than this; k stays here anyway.
K = 8
RANK = 8
ENCODER = ENCODER_WORD_SENTENCE_SVD

# Isolation trial that froze the four candidates. This module does not
# reopen that search.
ISOLATION_COMMIT = "c1179c90fb202aefc6979363847e9bfd33e1809c"

# Frozen candidates. Initialization and seed are the isolation selections.
# Do not append a seed and do not drop a row because a new fixture dislikes it.
FROZEN_CANDIDATES: Tuple[Mapping[str, object], ...] = (
    {
        "id": "euclidean_random_subset",
        "distance": "euclidean",
        "init": "random_subset",
        "seed": 2,
    },
    {
        "id": "euclidean_kmeanspp",
        "distance": "euclidean",
        "init": "kmeanspp",
        "seed": 2,
    },
    {
        "id": "spherical_random_subset",
        "distance": "spherical",
        "init": "random_subset",
        "seed": 2,
    },
    {
        "id": "spherical_kmeanspp",
        "distance": "spherical",
        "init": "kmeanspp",
        "seed": 3,
    },
)
FROZEN_CANDIDATE_IDS: Tuple[str, ...] = tuple(str(row["id"]) for row in FROZEN_CANDIDATES)

# Same file-order rule on each new fixture. It is not the varied_prose
# index list. Every 5th row, 0-based index congruent to 4.
HOLDOUT_MODULUS = 5
HOLDOUT_REMAINDER = 4

# A fixture under this many rows is too small for a 20% holdout to be scored.
MIN_ROWS = 40

# Advantage on one fixture, all three required:
# eval size-weighted purity strictly above production on the same split,
# prototype residual at most this multiple of production residual,
# clustering median runtime at most this multiple of production clustering time.
RESIDUAL_WORSE_FACTOR = 1.10
CLUSTER_RUNTIME_FACTOR = 2.0

# Timed samples. Clustering medians are per call, averaged inside each
# sample over CLUSTER_TIME_REPEATS so a sub-millisecond fit is not the
# timer. The median is the element at index len // 2 of the sorted samples.
TIME_RUNS = 5
# Each clustering sample repeats the fit so the median is not a few
# milliseconds of scheduler noise. The 2× gate compares those per-call medians.
CLUSTER_TIME_REPEATS = 100

# Negative controls. These are not candidate seeds and they are not searched.
LABEL_SHUFFLE_SEED = 0
ROW_SHUFFLE_SEED = 1

# Singleton clusters score purity 1 under every labeling. A control is
# informative only when the production partition has a cluster at least
# this large. The train partition of n > k always is. Eval counts when it is.
INFORMATIVE_CLUSTER_SIZE = 2

# Eval rows whose cluster majority topic is not the row topic. Short sample.
FAILURE_SAMPLE_CAP = 6

# Cause names. A single cause is reported only when every fixture agrees.
# OBJECTIVE is in the set so the aggregate can name it, but the per-fixture
# classifier does not emit it: the four frozen rows are not a factorial
# that holds the objective fixed against a quiet init and a quiet distance.
CAUSES = (
    "INIT_SENSITIVITY",
    "ANGULAR_DISTANCE",
    "OBJECTIVE",
    "UNRESOLVED",
)

# Matched contrasts available without a new seed.
# Angular: random-subset seed 2, Euclidean versus spherical.
# Init: Euclidean seed 2, random-subset versus k-means++.
# Spherical k-means++ is seed 3 and is not part of either pair.
ANGULAR_PAIR = ("euclidean_random_subset", "spherical_random_subset")
INIT_PAIR = ("euclidean_random_subset", "euclidean_kmeanspp")

FIXTURE_SPECS: Tuple[Mapping[str, str], ...] = (
    {"id": "uneven", "kind": "uneven_topic_sizes", "dirname": "uneven"},
    {"id": "overlapping", "kind": "overlapping_concepts", "dirname": "overlapping"},
    {"id": "ambiguous", "kind": "ambiguous_rows", "dirname": "ambiguous"},
    {"id": "balanced", "kind": "balanced_control", "dirname": "balanced"},
)

RUNTIME_AUTHORITY = "RUNTIME_NOT_AUTHORIZED"


def holdout_mask(n_rows: int) -> np.ndarray:
    """True on eval positions. Does not read labels or the varied_prose file."""
    if n_rows < 0:
        raise ValueError(f"n_rows must be >= 0, got {n_rows}")
    index = np.arange(n_rows)
    return (index % HOLDOUT_MODULUS) == HOLDOUT_REMAINDER


def advantage_on_fixture(
    *,
    eval_size_weighted: float,
    production_eval_size_weighted: float,
    residual: float,
    production_residual: float,
    median_ms: Optional[float],
    production_median_ms: Optional[float],
) -> dict:
    """Apply the predeclared advantage rule. Does not choose a seed."""
    purity_ok = float(eval_size_weighted) > float(production_eval_size_weighted)
    residual_ok = (
        math.isfinite(float(residual))
        and math.isfinite(float(production_residual))
        and float(residual) <= RESIDUAL_WORSE_FACTOR * float(production_residual)
    )
    if median_ms is None or production_median_ms is None:
        runtime_ok = False
        runtime_ratio = None
    else:
        runtime_ok = float(median_ms) <= CLUSTER_RUNTIME_FACTOR * float(production_median_ms)
        runtime_ratio = (
            float(median_ms) / float(production_median_ms)
            if float(production_median_ms)
            else None
        )
    return {
        "purity_ok": bool(purity_ok),
        "residual_ok": bool(residual_ok),
        "runtime_ok": bool(runtime_ok),
        "runtime_ratio": runtime_ratio,
        "advantage": bool(purity_ok and residual_ok and runtime_ok),
    }


def recommend(fixture_advantages: Sequence[Mapping[str, bool]]) -> str:
    """A candidate that has the advantage on every fixture, or ``NONE``.

    More than one clearer is listed in frozen order. That list is not a
    new selection among them.
    """
    if not fixture_advantages:
        return "NONE"
    cleared = [
        candidate_id
        for candidate_id in FROZEN_CANDIDATE_IDS
        if all(bool(row[candidate_id]) for row in fixture_advantages)
    ]
    if not cleared:
        return "NONE"
    if len(cleared) == 1:
        return cleared[0]
    return ",".join(cleared)


def classify_fixture_cause(advantages: Mapping[str, bool]) -> str:
    """Name init or angular distance only from the two matched pairs.

    Both pairs disagreeing is ``UNRESOLVED``. Both pairs agreeing is also
    ``UNRESOLVED``: four frozen procedures moving together does not isolate
    the objective, and four of them failing does not either.
    """
    angular = bool(advantages[ANGULAR_PAIR[0]]) != bool(advantages[ANGULAR_PAIR[1]])
    init = bool(advantages[INIT_PAIR[0]]) != bool(advantages[INIT_PAIR[1]])
    if angular and init:
        return "UNRESOLVED"
    if angular:
        return "ANGULAR_DISTANCE"
    if init:
        return "INIT_SENSITIVITY"
    return "UNRESOLVED"


def aggregate_cause(causes: Sequence[str]) -> str:
    """One cause only when every fixture names that same cause."""
    if not causes:
        return "UNRESOLVED"
    for cause in causes:
        if cause not in CAUSES:
            raise ValueError(f"unknown cause {cause!r}")
    first = causes[0]
    if first == "UNRESOLVED":
        return "UNRESOLVED"
    if all(cause == first for cause in causes):
        return first
    return "UNRESOLVED"


def generalization_verdict(
    *,
    scorable: bool,
    controls_ok: bool,
    recommendation: str,
) -> str:
    """``UNRESOLVED`` when the probe cannot be scored, else the trial result.

    ``RESEARCH_FAIL`` means no frozen candidate has the advantage on every
    fixture. ``CONSISTENT_ADVANTAGE`` means at least one does. That value
    is not a certificate pass and it does not grant runtime authority.
    """
    if not scorable or not controls_ok:
        return "UNRESOLVED"
    if recommendation == "NONE":
        return "RESEARCH_FAIL"
    return "CONSISTENT_ADVANTAGE"


def load_fixture(directory: Path) -> dict:
    """Load texts and a sidecar of primary topic labels.

    Observation JSONL may contain only ``observation_id`` and ``text``.
    The sidecar may contain ``secondary_topic``. That field is returned
    for the failure notes and is not part of ``topics``.
    """
    observations = _read_jsonl(directory / "observations.jsonl")
    sidecar = _read_jsonl(directory / "topics.jsonl")
    if len(observations) != len(sidecar):
        raise ValueError(f"{directory} observations and topics differ in length")
    texts: List[str] = []
    topics: List[str] = []
    observation_ids: List[str] = []
    secondary: Dict[str, str] = {}
    for obs, side in zip(observations, sidecar):
        extra = set(obs) - {"observation_id", "text"}
        if extra:
            raise ValueError(f"{directory} observation has extra fields {sorted(extra)}")
        if "topic" not in side or "observation_id" not in side:
            raise ValueError(f"{directory} sidecar row is missing observation_id or topic")
        if obs["observation_id"] != side["observation_id"]:
            raise ValueError(f"{directory} sidecar order does not match observations")
        observation_ids.append(str(obs["observation_id"]))
        texts.append(str(obs["text"]))
        topics.append(str(side["topic"]))
        if side.get("secondary_topic"):
            secondary[str(obs["observation_id"])] = str(side["secondary_topic"])
    return {
        "texts": texts,
        "topics": topics,
        "observation_ids": observation_ids,
        "secondary": secondary,
    }


def diagnose_fixture(
    fixture_id: str,
    texts: Sequence[str],
    topics: Sequence[str],
    observation_ids: Sequence[str],
    *,
    time_runs: int = TIME_RUNS,
) -> dict:
    """Embed one fixture, fit on train rows, and score eval rows.

    ``topics`` is unused until after every centroid fit. ``time_runs`` of
    0 skips the clock and therefore cannot grant the runtime clause.
    """
    if not (len(texts) == len(topics) == len(observation_ids)):
        raise ValueError("texts, topics, and observation_ids differ in length")
    if time_runs < 0:
        raise ValueError(f"time_runs must be >= 0, got {time_runs}")
    n_rows = len(texts)
    mask = holdout_mask(n_rows)
    train = np.flatnonzero(~mask)
    test = np.flatnonzero(mask)
    scorable = bool(n_rows >= MIN_ROWS and test.size >= 1 and train.size >= K)
    pin_blas_threads(1)
    matrix = np.asarray(word_sentence_svd(list(texts), rank=RANK), dtype=np.float64)
    if matrix.shape != (n_rows, RANK):
        raise RuntimeError(f"{fixture_id} embedding has shape {matrix.shape}")
    histogram = dict(sorted(Counter(topics).items()))
    n_topics = len(histogram)
    base = {
        "fixture_id": fixture_id,
        "n_rows": n_rows,
        "n_topics": n_topics,
        "topic_counts": histogram,
        "k": K,
        "k_note": _k_note(n_topics),
        "scorable": scorable,
        "n_train": int(train.size),
        "n_eval": int(test.size),
        "eval_indices_head": [int(index) for index in test[:3]],
        "embedding_sha256": hashlib.sha256(
            np.ascontiguousarray(matrix, dtype=np.float64).tobytes()
        ).hexdigest(),
        "max_abs_row_norm_error": float(
            np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0))
        ),
        "stability_retuned": False,
    }
    if train.size < K or test.size < 1:
        base.update(
            {
                "candidates": [],
                "production": None,
                "negative_controls": {"probe_ok": False, "reason": "too_small"},
                "cause": "UNRESOLVED",
            }
        )
        return base

    train_matrix = matrix[train]
    test_matrix = matrix[test]
    train_topics = [topics[int(index)] for index in train]
    test_topics = [topics[int(index)] for index in test]
    test_ids = [observation_ids[int(index)] for index in test]
    production_spec = _production_spec()
    _assert_production_ignores_rng(train_matrix)
    fitted = [_fit_one(train_matrix, test_matrix, spec) for spec in _specs()]
    production = fitted[0]
    for run in fitted[1:]:
        run["train_ari_vs_production"] = adjusted_rand_index(
            run["train_labels"], production["train_labels"]
        )
        run["eval_ari_vs_production"] = adjusted_rand_index(
            run["eval_labels"], production["eval_labels"]
        )
    production["train_ari_vs_production"] = 1.0
    production["eval_ari_vs_production"] = 1.0

    # Topic probe starts here, after the fits.
    for run in fitted:
        train_score = _score_assignment(run["train_labels"], train_topics)
        eval_score = _score_assignment(run["eval_labels"], test_topics)
        run["train_size_weighted"] = train_score["size_weighted"]
        run["train_hits"] = train_score["hits"]
        run["train_rows"] = train_score["rows"]
        run["eval_size_weighted"] = eval_score["size_weighted"]
        run["eval_hits"] = eval_score["hits"]
        run["eval_rows"] = eval_score["rows"]
        run["failures"] = _failure_sample(
            eval_score,
            test_ids,
            test_topics,
            [texts[int(index)] for index in test],
        )
        run["failure_count"] = int(eval_score["failure_count"])

    _attach_timings(train_matrix, texts, train, fitted, time_runs)
    for run in fitted[1:]:
        run.update(
            advantage_on_fixture(
                eval_size_weighted=run["eval_size_weighted"],
                production_eval_size_weighted=production["eval_size_weighted"],
                residual=run["residual"],
                production_residual=production["residual"],
                median_ms=run["cluster_median_ms"],
                production_median_ms=production["cluster_median_ms"],
            )
        )
    advantages = {str(run["id"]): bool(run["advantage"]) for run in fitted[1:]}
    controls = _negative_controls(
        matrix,
        topics,
        train,
        test,
        production,
    )
    base.update(
        {
            "production": _public_run(production),
            "candidates": [_public_run(run) for run in fitted[1:]],
            "negative_controls": controls,
            "cause": classify_fixture_cause(advantages),
            "advantages": advantages,
        }
    )
    return base


def diagnose_clustering_generalization(
    root: Optional[Path] = None,
    *,
    time_runs: int = TIME_RUNS,
) -> dict:
    """Score the four frozen candidates on each new fixture, then aggregate."""
    fixture_root = Path(root) if root is not None else _default_root()
    fixtures = []
    for spec in FIXTURE_SPECS:
        loaded = load_fixture(fixture_root / spec["dirname"])
        report = diagnose_fixture(
            spec["id"],
            loaded["texts"],
            loaded["topics"],
            loaded["observation_ids"],
            time_runs=time_runs,
        )
        report["kind"] = spec["kind"]
        report["n_secondary"] = len(loaded["secondary"])
        _annotate_secondary(report, loaded["secondary"])
        fixtures.append(report)
    advantages = [
        {
            candidate_id: bool(fixture["advantages"][candidate_id])
            for candidate_id in FROZEN_CANDIDATE_IDS
        }
        for fixture in fixtures
        if fixture.get("advantages")
    ]
    scorable = all(bool(fixture["scorable"]) for fixture in fixtures) and len(fixtures) == len(
        FIXTURE_SPECS
    )
    controls_ok = all(
        bool(fixture["negative_controls"].get("probe_ok")) for fixture in fixtures
    )
    advantage_recommendation = (
        recommend(advantages) if len(advantages) == len(fixtures) else "NONE"
    )
    # A broken control or a fixture too small to score cannot support a
    # recommendation, even if the advantage bits happen to line up.
    if not scorable or not controls_ok:
        recommendation = "NONE"
    else:
        recommendation = advantage_recommendation
    causes = [str(fixture["cause"]) for fixture in fixtures]
    return {
        "verdict": generalization_verdict(
            scorable=scorable,
            controls_ok=controls_ok,
            recommendation=recommendation,
        ),
        "recommendation": recommendation,
        "advantage_recommendation": advantage_recommendation,
        "runtime_authority": RUNTIME_AUTHORITY,
        "isolated_cause": aggregate_cause(causes),
        "scorable": scorable,
        "controls_ok": controls_ok,
        "fixtures": fixtures,
        "receipts": _receipts(fixtures),
        "bar": {
            "residual_worse_factor": RESIDUAL_WORSE_FACTOR,
            "cluster_runtime_factor": CLUSTER_RUNTIME_FACTOR,
            "purity": "eval size-weighted purity strictly greater than production",
            "recommendation": "advantage on every fixture, otherwise NONE",
        },
    }


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "examples" / "generalization"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _k_note(n_topics: int) -> str:
    if n_topics < K:
        return (
            f"k stays {K} even though this fixture has {n_topics} topics. "
            "The topic count was set with the file, not from a score."
        )
    if n_topics == K:
        return (
            f"k stays {K}. This fixture also has {n_topics} topics, one block "
            "per subject. That match was set with the file, not from a score."
        )
    return f"k stays {K} on this fixture, which has {n_topics} topics."


def _production_spec() -> dict:
    return {
        "id": PRODUCTION_FAMILY,
        "distance": "euclidean",
        "init": "farthest_first",
        "seed": None,
    }


def _specs() -> List[dict]:
    rows = [_production_spec()]
    for candidate in FROZEN_CANDIDATES:
        rows.append(
            {
                "id": str(candidate["id"]),
                "distance": str(candidate["distance"]),
                "init": str(candidate["init"]),
                "seed": int(candidate["seed"]),
            }
        )
    return rows


def _assert_production_ignores_rng(matrix: np.ndarray) -> None:
    first, _ = kmeans(matrix, K, rng=np.random.default_rng(0))
    second, _ = kmeans(matrix, K, rng=np.random.default_rng(4))
    if not np.array_equal(first, second):
        raise RuntimeError("production kmeans changed with the unused rng")


def _fit_one(train_matrix: np.ndarray, test_matrix: np.ndarray, spec: Mapping) -> dict:
    train_labels, centroids, _indices = _fit_algorithm(train_matrix, spec)
    residual = _prototype_residual(train_matrix, train_labels, centroids)
    eval_labels = _assign(test_matrix, centroids, str(spec["distance"]))
    return {
        "id": spec["id"],
        "distance": spec["distance"],
        "init": spec["init"],
        "seed": spec["seed"],
        "train_labels": train_labels,
        "eval_labels": eval_labels,
        "residual": float(residual),
    }


def _score_assignment(labels: np.ndarray, topics: Sequence[str]) -> dict:
    """Size-weighted majority purity. Tie break matches the isolation probe."""
    labels = np.asarray(labels)
    if int(labels.size) != len(topics):
        raise ValueError("labels and topics differ in length")
    majority_of: Dict[int, Optional[str]] = {}
    hits = 0
    for code in np.unique(labels):
        members = np.flatnonzero(labels == code)
        member_topics = [topics[int(index)] for index in members]
        majority, count = _majority(member_topics)
        majority_of[int(code)] = majority
        hits += int(count)
    rows = int(labels.size)
    match = []
    for index, code in enumerate(labels.tolist()):
        match.append(topics[index] == majority_of[int(code)])
    failures = int(sum(1 for flag in match if not flag))
    return {
        "hits": int(hits),
        "rows": rows,
        "size_weighted": float(hits / rows) if rows else 0.0,
        "majority_of": majority_of,
        "match": match,
        "failure_count": failures,
        "labels": labels,
    }


def _failure_sample(
    score: Mapping,
    observation_ids: Sequence[str],
    topics: Sequence[str],
    texts: Sequence[str],
) -> List[dict]:
    labels = score["labels"]
    sample = []
    for index, flag in enumerate(score["match"]):
        if flag:
            continue
        code = int(labels[index])
        sample.append(
            {
                "observation_id": observation_ids[index],
                "topic": topics[index],
                "majority_topic": score["majority_of"].get(code),
                "cluster": code,
                "text": texts[index],
            }
        )
        if len(sample) >= FAILURE_SAMPLE_CAP:
            break
    return sample


def _negative_controls(
    matrix: np.ndarray,
    topics: Sequence[str],
    train: np.ndarray,
    test: np.ndarray,
    production: Mapping,
) -> dict:
    """Shuffled labels and shuffled rows must lose to the real assignment.

    Both are scored on the train partition. Eval is scored too, and it
    counts only when the real eval partition has a cluster of at least
    ``INFORMATIVE_CLUSTER_SIZE`` rows. A singleton partition cannot collapse.
    """
    label_rng = np.random.default_rng(LABEL_SHUFFLE_SEED)
    permutation = label_rng.permutation(len(topics))
    shuffled_topics = [topics[int(index)] for index in permutation]
    train_topics = [topics[int(index)] for index in train]
    train_topics_shuffled = [shuffled_topics[int(index)] for index in train]
    eval_topics = [topics[int(index)] for index in test]
    eval_topics_shuffled = [shuffled_topics[int(index)] for index in test]

    real_train = _score_assignment(production["train_labels"], train_topics)
    label_train = _score_assignment(production["train_labels"], train_topics_shuffled)
    real_eval = _score_assignment(production["eval_labels"], eval_topics)
    label_eval = _score_assignment(production["eval_labels"], eval_topics_shuffled)

    row_rng = np.random.default_rng(ROW_SHUFFLE_SEED)
    order = row_rng.permutation(matrix.shape[0])
    shuffled_matrix = matrix[order]
    spec = _production_spec()
    shuffled_train_labels, centroids, _ = _fit_algorithm(shuffled_matrix[train], spec)
    row_train = _score_assignment(shuffled_train_labels, train_topics)
    shuffled_eval_labels = _assign(shuffled_matrix[test], centroids, "euclidean")
    row_eval = _score_assignment(shuffled_eval_labels, eval_topics)

    train_informative = _informative(production["train_labels"])
    eval_informative = _informative(production["eval_labels"])
    label_train_ok = float(label_train["size_weighted"]) < float(real_train["size_weighted"])
    row_train_ok = float(row_train["size_weighted"]) < float(real_train["size_weighted"])
    label_eval_ok = float(label_eval["size_weighted"]) < float(real_eval["size_weighted"])
    row_eval_ok = float(row_eval["size_weighted"]) < float(real_eval["size_weighted"])
    probe_ok = bool(train_informative and label_train_ok and row_train_ok)
    if eval_informative:
        probe_ok = bool(probe_ok and label_eval_ok and row_eval_ok)
    return {
        "probe_ok": probe_ok,
        "train_informative": train_informative,
        "eval_informative": eval_informative,
        "label_shuffle": {
            "seed": LABEL_SHUFFLE_SEED,
            "train_size_weighted": label_train["size_weighted"],
            "train_hits": label_train["hits"],
            "train_rows": label_train["rows"],
            "train_below_production": label_train_ok,
            "eval_size_weighted": label_eval["size_weighted"],
            "eval_hits": label_eval["hits"],
            "eval_rows": label_eval["rows"],
            "eval_below_production": label_eval_ok,
            "eval_counted": eval_informative,
        },
        "row_shuffle": {
            "seed": ROW_SHUFFLE_SEED,
            "train_size_weighted": row_train["size_weighted"],
            "train_hits": row_train["hits"],
            "train_rows": row_train["rows"],
            "train_below_production": row_train_ok,
            "eval_size_weighted": row_eval["size_weighted"],
            "eval_hits": row_eval["hits"],
            "eval_rows": row_eval["rows"],
            "eval_below_production": row_eval_ok,
            "eval_counted": eval_informative,
        },
        "production_train_size_weighted": real_train["size_weighted"],
        "production_train_hits": real_train["hits"],
        "production_eval_size_weighted": real_eval["size_weighted"],
        "production_eval_hits": real_eval["hits"],
    }


def _informative(labels: np.ndarray) -> bool:
    if int(np.asarray(labels).size) == 0:
        return False
    _codes, counts = np.unique(labels, return_counts=True)
    return bool(int(counts.max()) >= INFORMATIVE_CLUSTER_SIZE)


def _attach_timings(
    train_matrix: np.ndarray,
    texts: Sequence[str],
    train: np.ndarray,
    runs: List[dict],
    time_runs: int,
) -> None:
    for run in runs:
        spec = {
            "id": run["id"],
            "distance": run["distance"],
            "init": run["init"],
            "seed": run["seed"],
        }

        def _cluster(spec: Mapping = spec) -> None:
            _fit_algorithm(train_matrix, spec)

        def _embed_cluster(spec: Mapping = spec) -> None:
            embedded = np.asarray(word_sentence_svd(list(texts), rank=RANK), dtype=np.float64)
            _fit_algorithm(embedded[train], spec)

        cluster_ms, cluster_samples = _per_call_median_ms(
            _cluster, time_runs, CLUSTER_TIME_REPEATS
        )
        embed_ms, embed_samples = _per_call_median_ms(_embed_cluster, time_runs, 1)
        run["cluster_median_ms"] = cluster_ms
        run["cluster_samples_ms"] = cluster_samples
        run["cluster_peak_bytes"] = _peak_bytes(_cluster, time_runs)
        run["embed_cluster_median_ms"] = embed_ms
        run["embed_cluster_samples_ms"] = embed_samples
        run["embed_cluster_peak_bytes"] = _peak_bytes(_embed_cluster, time_runs)


def _per_call_median_ms(fn, runs: int, repeats: int) -> Tuple[Optional[float], List[float]]:
    if runs <= 0:
        return None, []
    fn()
    samples: List[float] = []
    for _ in range(runs):
        gc.disable()
        try:
            start = time.perf_counter()
            for _rep in range(repeats):
                fn()
            elapsed = time.perf_counter() - start
        finally:
            gc.enable()
        samples.append(elapsed * 1000.0 / repeats)
    ordered = sorted(samples)
    return ordered[len(ordered) // 2], samples


def _peak_bytes(fn, runs: int) -> Optional[int]:
    if runs <= 0:
        return None
    peaks: List[int] = []
    for _ in range(runs):
        tracemalloc.start()
        try:
            fn()
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        peaks.append(int(peak))
    ordered = sorted(peaks)
    return int(ordered[len(ordered) // 2])


def _annotate_secondary(report: dict, secondary: Mapping[str, str]) -> None:
    """Copy the sidecar's second topic onto failure notes. Not a score input."""
    blocks = []
    if report.get("production"):
        blocks.append(report["production"])
    blocks.extend(report.get("candidates") or [])
    for block in blocks:
        for failure in block.get("failures") or []:
            extra = secondary.get(failure["observation_id"])
            if extra:
                failure["secondary_topic"] = extra


def _receipts(fixtures: Sequence[Mapping]) -> dict:
    return {
        "isolation_commit": ISOLATION_COMMIT,
        "git_revision": _git_revision(),
        "candidates": [
            {
                "id": str(row["id"]),
                "init": str(row["init"]),
                "distance": str(row["distance"]),
                "seed": int(row["seed"]),
            }
            for row in FROZEN_CANDIDATES
        ],
        "split": f"file-order index % {HOLDOUT_MODULUS} == {HOLDOUT_REMAINDER}",
        "holdout_modulus": HOLDOUT_MODULUS,
        "holdout_remainder": HOLDOUT_REMAINDER,
        "encoder": ENCODER,
        "rank": RANK,
        "k": K,
        "min_rows": MIN_ROWS,
        "time_runs": TIME_RUNS,
        "cluster_time_repeats": CLUSTER_TIME_REPEATS,
        "label_shuffle_seed": LABEL_SHUFFLE_SEED,
        "row_shuffle_seed": ROW_SHUFFLE_SEED,
        "failure_sample_cap": FAILURE_SAMPLE_CAP,
        "numpy": np.__version__,
        "blas_threads": 1,
        "production_init": "farthest_first",
        "production_distance": "euclidean",
        "production_ignores_rng": True,
        "stability_retuned": False,
        "varied_prose_split_reused": False,
        "learn_pack_called": False,
        "embedding_scope": (
            "word_sentence_svd on every row of that fixture only; "
            "eval text is in the TF-IDF and the SVD; topic labels are not"
        ),
        "fit_scope": "centroids fit on train rows only; eval rows are assigned and do not move centroids",
        "fixture_embeddings": [
            {
                "fixture_id": fixture["fixture_id"],
                "embedding_sha256": fixture.get("embedding_sha256"),
                "n_rows": fixture["n_rows"],
                "n_train": fixture["n_train"],
                "n_eval": fixture["n_eval"],
            }
            for fixture in fixtures
        ],
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


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} invalid JSON") from exc
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _public_run(run: Mapping) -> dict:
    skip = {"train_labels", "eval_labels"}
    public = {}
    for key, value in run.items():
        if key in skip:
            continue
        if isinstance(value, np.ndarray):
            public[key] = value.tolist()
        elif isinstance(value, np.floating):
            public[key] = float(value)
        elif isinstance(value, np.integer):
            public[key] = int(value)
        else:
            public[key] = value
    return public

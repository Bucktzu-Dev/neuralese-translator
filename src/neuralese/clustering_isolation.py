"""Clustering-isolation probe on the frozen rank-8 word-sentence SVD.

This module does not replace ``clustering.kmeans`` and it is not called by
``learn_pack``. Production k-means stays farthest-first initialization,
squared Euclidean assignment, and a mean update. Its ``rng`` argument is
not read. The same matrix order therefore yields the same partition.

Topic strings are a probe. They are not an argument of the fitters, and
they are not read until after the unsupervised seed choice.

The admission bar and the isolation thresholds below are module constants.
They are fixed before any held-out score is computed.
"""
from __future__ import annotations

import hashlib
import time
import tracemalloc
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify
from neuralese.clustering import kmeans
from neuralese.contracts import Observation
from neuralese.encode import (
    ENCODER_WORD_SENTENCE_SVD,
    word_sentence_svd,
    word_sentence_tfidf,
)
from neuralese.factorization import pin_blas_threads
from neuralese.rank_boundary import PRODUCT_K, PRODUCT_RANK, singular_spectrum

# Product cluster count. Not an adaptive rank and not a new default.
K = PRODUCT_K
RANK = PRODUCT_RANK
PROBE_TOPIC = "weather"

# Same cutoff ``cluster_majority_diagnostic`` uses. Diagnostic only.
MIXED_PURITY = 0.5

# Lloyd iteration cap. Matches ``clustering.kmeans``.
MAX_ITER = 64

# Fixed seeds for the non-production initializers. Production farthest-first
# does not use a seed. Five is the minimum this probe runs.
SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

# Frozen before any held-out score. Every 5th row in file order: 0-based
# indices congruent to 4 (mod 5), which are 1-based rows 5, 10, 15, ...
# On 320 rows that is 64 held out (20%) and 256 train. The rule does not
# read topic labels. Contiguous blocks of 40 happen to make the split
# balanced; that is a property of the file layout, not a stratified sample.
HOLDOUT_MODULUS = 5
HOLDOUT_REMAINDER = 4

# Negative controls. Fixed before the split is scored.
LABEL_SHUFFLE_SEED = 0
ROW_SHUFFLE_SEED = 1

# Timed samples after the fit that produces the official labels. Median is
# the element at index ``len // 2`` of the sorted samples, the same index
# the rank-boundary learn timer uses.
TIME_RUNS = 5

# Admission bar. A candidate is ADMISSIBLE only when every clause holds.
# (a) unsupervised residual no worse than this multiple of production.
RESIDUAL_WORSE_FACTOR = 1.10
# (b) median clustering runtime at most this multiple of production
# clustering, and learn-equivalent cost strictly under this multiple of
# the documented 222.369 ms learn_pack median (a few times 222 ms).
CLUSTER_RUNTIME_FACTOR = 2.0
LEARN_PACK_BASELINE_MS = 222.369
LEARN_EQUIVALENT_FACTOR = 3.0

# Isolation thresholds. Also fixed before held-out scores.
# Weather purity (diagnostic) must move by at least this absolute amount
# before initialization or distance counts as explaining the mixture.
PURITY_MOVE = 0.05
# Weather-majority cluster is large relative to the smallest non-empty
# cluster at this multiple. This is a size cutoff, not the rank-boundary
# distance cutoff.
SIZE_RATIO_IMBALANCE = 2.0

FAMILY_TIE_ORDER: Tuple[str, ...] = (
    "euclidean_random_subset",
    "euclidean_kmeanspp",
    "spherical_farthest_first",
    "spherical_random_subset",
    "spherical_kmeanspp",
)
PRODUCTION_FAMILY = "production_farthest_first_euclidean"

ISOLATED_CAUSES = (
    "INIT_SENSITIVITY",
    "ANGULAR_DISTANCE",
    "SIZE_IMBALANCE",
    "OBJECTIVE",
    "UNRESOLVED",
)


def holdout_mask(n_rows: int) -> np.ndarray:
    """True on the frozen held-out positions. Does not read labels."""
    if n_rows < 0:
        raise ValueError(f"n_rows must be >= 0, got {n_rows}")
    index = np.arange(n_rows)
    return (index % HOLDOUT_MODULUS) == HOLDOUT_REMAINDER


def adjusted_rand_index(left: np.ndarray, right: np.ndarray) -> float:
    """Hubert-Arabie adjusted Rand index. Labels are not topic ids.

    Identical partitions score 1, including a relabeling of the same sets.
    """
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("adjusted Rand index requires aligned assignments")
    n = int(left.size)
    if n < 2:
        return 1.0
    _, left_inv = np.unique(left, return_inverse=True)
    _, right_inv = np.unique(right, return_inverse=True)
    table = np.zeros((int(left_inv.max()) + 1, int(right_inv.max()) + 1), dtype=np.int64)
    np.add.at(table, (left_inv, right_inv), 1)

    def _pairs(counts: np.ndarray) -> float:
        values = counts.astype(np.float64)
        return float(np.sum(values * (values - 1.0) / 2.0))

    sum_ij = _pairs(table)
    sum_a = _pairs(table.sum(axis=1))
    sum_b = _pairs(table.sum(axis=0))
    total = n * (n - 1) / 2.0
    expected = sum_a * sum_b / total
    maximum = 0.5 * (sum_a + sum_b)
    denom = maximum - expected
    if denom == 0.0:
        return 1.0
    return float((sum_ij - expected) / denom)


def init_indices(
    matrix: np.ndarray,
    k: int,
    init: str,
    seed: Optional[int],
) -> List[int]:
    """Deterministic row indices. Topic labels are not an argument.

    ``farthest_first`` ignores ``seed`` and matches ``clustering._farthest_first``.
    ``random_subset`` is a partial Fisher-Yates shuffle from
    ``np.random.default_rng(seed)``. ``kmeanspp`` is Arthur and Vassilvitskii:
    the first center is uniform, and each later center is drawn with
    probability proportional to squared Euclidean distance to the nearest
    center already chosen, via the inverse CDF of ``Generator.random``.
    """
    n = int(matrix.shape[0])
    if k < 1 or k > n:
        raise ValueError(f"k must be in 1..{n}, got {k}")
    if init == "farthest_first":
        return _farthest_first_indices(matrix, k)
    if seed is None:
        raise ValueError(f"{init} requires a seed")
    rng = np.random.default_rng(seed)
    if init == "random_subset":
        return _random_subset_indices(n, k, rng)
    if init == "kmeanspp":
        return _kmeanspp_indices(matrix, k, rng)
    raise ValueError(f"unknown init {init!r}")


def euclidean_size_contrast(runs: Sequence[Mapping]) -> bool:
    """True when Euclidean size tracks the weather mixture.

    Balanced means ``size_ratio`` is strictly below ``SIZE_RATIO_IMBALANCE``.
    The contrast holds when both sides are present and the lowest balanced
    weather purity exceeds the highest imbalanced weather purity by at least
    ``PURITY_MOVE``. Runs missing either number are ignored. This is how size
    can be separated from the Euclidean-versus-spherical switch: the
    comparison stays inside Euclidean runs.
    """
    balanced: List[float] = []
    imbalanced: List[float] = []
    for run in runs:
        ratio = run.get("size_ratio")
        purity = run.get("weather_purity")
        if ratio is None or purity is None:
            continue
        if float(ratio) < SIZE_RATIO_IMBALANCE:
            balanced.append(float(purity))
        else:
            imbalanced.append(float(purity))
    if not balanced or not imbalanced:
        return False
    return min(balanced) - max(imbalanced) >= PURITY_MOVE


def classify_clustering_isolation(
    *,
    probe_ok: bool,
    init_purity_span: float,
    angular_purity_span: float,
    size_contrast: bool,
    baseline_size_ratio: Optional[float],
    best_size_ratio: Optional[float],
    baseline_weather_purity: Optional[float],
    best_weather_purity: Optional[float],
    best_fails_held_out_bar: bool,
) -> str:
    """Name the cause the table isolates, or ``UNRESOLVED``.

    Initialization explains the mixture when Euclidean weather purity,
    including production, spans at least ``PURITY_MOVE``. Angular distance
    explains it when a matched-init pair spans at least ``PURITY_MOVE``.
    Size explains it only when ``size_contrast`` is true inside Euclidean
    runs. Two of those flags, or a broken negative control, is
    ``UNRESOLVED``.

    ``OBJECTIVE`` is claimed only when none of those three explain the
    mixture, both the baseline and the best unsupervised candidate are
    below ``SIZE_RATIO_IMBALANCE``, both stay mixed, and that candidate
    fails held-out clause (c). A size ratio at or above the cutoff blocks
    ``OBJECTIVE`` even without a size contrast, because the gap is then
    still a live alternative and the table did not separate it.
    """
    if not probe_ok:
        return "UNRESOLVED"
    flags: List[str] = []
    if float(init_purity_span) >= PURITY_MOVE:
        flags.append("INIT_SENSITIVITY")
    if float(angular_purity_span) >= PURITY_MOVE:
        flags.append("ANGULAR_DISTANCE")
    if size_contrast:
        flags.append("SIZE_IMBALANCE")
    if len(flags) == 1:
        return flags[0]
    if len(flags) > 1:
        return "UNRESOLVED"
    baseline_mixed = (
        baseline_weather_purity is not None and float(baseline_weather_purity) < MIXED_PURITY
    )
    best_mixed = best_weather_purity is not None and float(best_weather_purity) < MIXED_PURITY
    size_gap_absent = _ratio_below(baseline_size_ratio) and _ratio_below(best_size_ratio)
    if size_gap_absent and baseline_mixed and best_mixed and best_fails_held_out_bar:
        return "OBJECTIVE"
    return "UNRESOLVED"


def diagnose_clustering_isolation(
    texts: Sequence[str],
    topics: Sequence[str],
    *,
    time_runs: int = TIME_RUNS,
) -> Dict[str, object]:
    """Compare clustering procedures on one frozen rank-8 matrix.

    ``topics`` is aligned with ``texts`` and is unused until the seed
    choice has been made from prototype residual. The matrix is the single
    ``word_sentence_svd`` call at ``RANK``. It is not truncated again.
    """
    if len(texts) != len(topics):
        raise ValueError(f"texts ({len(texts)}) and topics ({len(topics)}) differ in length")
    if time_runs < 0:
        raise ValueError(f"time_runs must be >= 0, got {time_runs}")
    labels = [str(topic) for topic in topics]
    pin_blas_threads(1)
    tfidf = word_sentence_tfidf(texts)
    spectrum = singular_spectrum(tfidf)
    matrix = np.asarray(word_sentence_svd(texts, rank=RANK), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != RANK:
        raise RuntimeError(f"frozen matrix has shape {matrix.shape}, expected (*, {RANK})")
    if matrix.shape[0] != len(labels):
        raise RuntimeError("frozen matrix row count does not match the texts")
    if matrix.shape[0] < K:
        raise ValueError(f"need at least {K} rows, got {matrix.shape[0]}")

    # Fits do not receive ``labels``. Seed choice reads residual only.
    runs = [_fit_record(matrix, spec) for spec in _specs()]
    _attach_stability(runs)
    representatives = _select_representatives(runs)
    best = _best_unsupervised(representatives)
    # Topic diagnostics start here, after the seed choice.
    for run in runs:
        run.update(_score_topics(run["labels"], labels))
    _attach_timings(matrix, runs, time_runs)

    mask = holdout_mask(matrix.shape[0])
    train = np.flatnonzero(~mask)
    test = np.flatnonzero(mask)
    production = _find_family(runs, PRODUCTION_FAMILY, None)
    held_production = _held_out_score(matrix, production, train, test, labels)
    held_candidates = [
        _held_out_score(matrix, run, train, test, labels) for run in representatives
    ]

    controls = _negative_controls(
        matrix,
        labels,
        production["labels"],
        train,
        test,
        baseline_size_weighted=float(production["size_weighted"]),
        baseline_weather_purity=production["weather_purity"],
        production_held_out_size_weighted=float(held_production["size_weighted"]),
    )
    cost = _time_learn_and_svd(texts, matrix, time_runs)
    evaluated = [
        _evaluate_candidate(run, production, held_row, held_production, cost)
        for run, held_row in zip(representatives, held_candidates)
    ]
    best_eval = next(
        row
        for row in evaluated
        if row["family"] == best["family"] and row["seed"] == best["seed"]
    )
    euclidean_runs = [run for run in runs if run["distance"] == "euclidean"]
    init_span = _purity_span([run["weather_purity"] for run in euclidean_runs])
    pairs = _matched_pairs(runs)
    angular_span = _purity_span_from_pairs(pairs)
    size_contrast = euclidean_size_contrast(euclidean_runs)
    best_fails_c = not bool(best_eval["held_out_weather_ok"] and best_eval["held_out_weighted_ok"])
    isolated = classify_clustering_isolation(
        probe_ok=bool(controls["probe_ok"]),
        init_purity_span=init_span,
        angular_purity_span=angular_span,
        size_contrast=size_contrast,
        baseline_size_ratio=production["size_ratio"],
        best_size_ratio=best["size_ratio"],
        baseline_weather_purity=production["weather_purity"],
        best_weather_purity=best["weather_purity"],
        best_fails_held_out_bar=best_fails_c,
    )
    if not controls["probe_ok"]:
        verdict = "UNRESOLVED"
    elif isolated == "UNRESOLVED":
        verdict = "RESEARCH_FAIL"
    else:
        verdict = isolated
    any_admissible = any(bool(row["admissible"]) for row in evaluated)
    identity = _production_identity(texts, matrix, production["labels"], labels)
    s7 = float(spectrum[7])
    s8 = float(spectrum[8])
    return {
        "verdict": verdict,
        "isolated_cause": isolated,
        "admission": "ADMISSIBLE" if any_admissible else "NOT_ADMISSIBLE",
        "runtime_authority": "RUNTIME_NOT_AUTHORIZED",
        "any_admissible": any_admissible,
        "certify_scope": "structural_only",
        "baseline": _public_run(production),
        "best_unsupervised": {
            "family": best["family"],
            "init": best["init"],
            "seed": best["seed"],
            "residual": best["residual"],
        },
        "candidates": evaluated,
        "runs": [_public_run(run) for run in runs],
        "matched_pairs": pairs,
        "held_out": {
            "production": held_production,
            "candidates": held_candidates,
        },
        "negative_controls": controls,
        "contrasts": {
            "init_purity_span": init_span,
            "angular_purity_span": angular_span,
            "euclidean_size_contrast": size_contrast,
            "baseline_size_ratio": production["size_ratio"],
            "best_size_ratio": best["size_ratio"],
            "baseline_weather_rows": production["weather_rows"],
            "baseline_smallest_rows": production["smallest_rows"],
            "best_weather_rows": best["weather_rows"],
            "best_smallest_rows": best["smallest_rows"],
            "best_fails_held_out_bar": best_fails_c,
            "init_explains": init_span >= PURITY_MOVE,
            "angular_explains": angular_span >= PURITY_MOVE,
            "size_explains": size_contrast,
        },
        "cost": cost,
        "identity": identity,
        "failures": [
            {
                "family": row["family"],
                "seed": row["seed"],
                "failed": [
                    name
                    for name, key in (
                        ("residual", "residual_ok"),
                        ("runtime", "runtime_ok"),
                        ("learn_equivalent", "learn_equivalent_ok"),
                        ("held_out_weather", "held_out_weather_ok"),
                        ("held_out_weighted", "held_out_weighted_ok"),
                    )
                    if not row[key]
                ],
            }
            for row in evaluated
            if not row["admissible"]
        ],
        "receipts": {
            "seeds": list(SEEDS),
            "split": (
                f"file-order index % {HOLDOUT_MODULUS} == {HOLDOUT_REMAINDER}"
            ),
            "holdout_modulus": HOLDOUT_MODULUS,
            "holdout_remainder": HOLDOUT_REMAINDER,
            "n_rows": int(matrix.shape[0]),
            "n_train": int(train.size),
            "n_holdout": int(test.size),
            "holdout_indices_head": [int(index) for index in test[:3]],
            "s7": s7,
            "s8": s8,
            "gap_s7_minus_s8": s7 - s8,
            "ratio_s8_over_s7": s8 / s7,
            "embedding_sha256": hashlib.sha256(
                np.ascontiguousarray(matrix, dtype=np.float64).tobytes()
            ).hexdigest(),
            "max_abs_row_norm_error": float(
                np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0))
            ),
            "rank": RANK,
            "k": K,
            "max_iter": MAX_ITER,
            "blas_threads": 1,
            "numpy": np.__version__,
            "production_init": "farthest_first",
            "production_distance": "euclidean",
            "production_ignores_rng": bool(production["seed_invariant"]),
            "production_lloyd_matches": bool(production["lloyd_matches"]),
            "label_shuffle_seed": LABEL_SHUFFLE_SEED,
            "row_shuffle_seed": ROW_SHUFFLE_SEED,
            "selection": (
                "lowest full-matrix prototype residual, then lowest seed; "
                "topic labels and held-out purity are not read"
            ),
        },
        "bar": {
            "residual_worse_factor": RESIDUAL_WORSE_FACTOR,
            "cluster_runtime_factor": CLUSTER_RUNTIME_FACTOR,
            "learn_pack_baseline_ms": LEARN_PACK_BASELINE_MS,
            "learn_equivalent_factor": LEARN_EQUIVALENT_FACTOR,
            "purity_move": PURITY_MOVE,
            "size_ratio_imbalance": SIZE_RATIO_IMBALANCE,
            "mixed_purity": MIXED_PURITY,
        },
    }


def _ratio_below(ratio: Optional[float]) -> bool:
    return ratio is not None and float(ratio) < SIZE_RATIO_IMBALANCE


def _specs() -> List[dict]:
    rows = [
        {
            "family": PRODUCTION_FAMILY,
            "distance": "euclidean",
            "init": "farthest_first",
            "seed": None,
        }
    ]
    for init in ("random_subset", "kmeanspp"):
        for seed in SEEDS:
            rows.append(
                {
                    "family": f"euclidean_{init}",
                    "distance": "euclidean",
                    "init": init,
                    "seed": seed,
                }
            )
    rows.append(
        {
            "family": "spherical_farthest_first",
            "distance": "spherical",
            "init": "farthest_first",
            "seed": None,
        }
    )
    for init in ("random_subset", "kmeanspp"):
        for seed in SEEDS:
            rows.append(
                {
                    "family": f"spherical_{init}",
                    "distance": "spherical",
                    "init": init,
                    "seed": seed,
                }
            )
    return rows


def _fit_record(matrix: np.ndarray, spec: Mapping) -> dict:
    labels, centroids, indices = _fit_algorithm(matrix, spec)
    lloyd_matches = True
    seed_invariant = spec["init"] == "farthest_first"
    if spec["distance"] == "euclidean" and spec["init"] == "farthest_first":
        local_labels, local_centroids = _lloyd_euclidean(matrix, indices)
        lloyd_matches = bool(
            np.array_equal(labels, local_labels) and np.allclose(centroids, local_centroids)
        )
        other, _ = kmeans(matrix, K, rng=np.random.default_rng(SEEDS[-1]))
        seed_invariant = bool(np.array_equal(labels, other))
        if not lloyd_matches:
            raise RuntimeError("local Euclidean Lloyd diverged from production kmeans")
        if not seed_invariant:
            raise RuntimeError("production kmeans changed with the unused rng")
    residual = _prototype_residual(matrix, labels, centroids)
    return {
        "family": spec["family"],
        "distance": spec["distance"],
        "init": spec["init"],
        "seed": spec["seed"],
        "labels": labels,
        "centroids": centroids,
        "init_indices": indices,
        "residual": residual,
        "lloyd_matches": lloyd_matches,
        "seed_invariant": seed_invariant,
    }


def _fit_algorithm(
    matrix: np.ndarray,
    spec: Mapping,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """One initializer plus one Lloyd run.

    Euclidean ``farthest_first`` calls production ``kmeans``. Timed runs
    and held-out refits use this path, so a check that production ignores
    ``rng`` is not folded into the runtime.
    """
    init = str(spec["init"])
    seed = spec["seed"]
    indices = init_indices(matrix, K, init, None if seed is None else int(seed))
    if spec["distance"] == "euclidean" and init == "farthest_first":
        labels, centroids = kmeans(matrix, K, rng=np.random.default_rng(0))
        return labels, centroids, indices
    if spec["distance"] == "euclidean":
        labels, centroids = _lloyd_euclidean(matrix, indices)
        return labels, centroids, indices
    labels, centroids = _lloyd_spherical(matrix, indices)
    return labels, centroids, indices


def _lloyd_euclidean(
    matrix: np.ndarray,
    seeds: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Squared-Euclidean Lloyd. Same update as ``clustering.kmeans``."""
    centroids = matrix[np.asarray(list(seeds), dtype=int)].copy()
    k = int(centroids.shape[0])
    labels = np.zeros(matrix.shape[0], dtype=int)
    for _ in range(MAX_ITER):
        dist = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        new_centroids = centroids.copy()
        for class_id in range(k):
            members = matrix[labels == class_id]
            if members.shape[0]:
                new_centroids[class_id] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids


def _lloyd_spherical(
    matrix: np.ndarray,
    seeds: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine assignment, then mean and L2 renorm.

    Squared Euclidean distance to a renormed centroid is ``2 - 2 cos``
    when both vectors are unit, so the recorded prototype residual stays
    in the same units as production.
    """
    centroids = matrix[np.asarray(list(seeds), dtype=int)].copy()
    centroids = _renorm_rows(centroids)
    k = int(centroids.shape[0])
    labels = np.zeros(matrix.shape[0], dtype=int)
    for _ in range(MAX_ITER):
        labels = (matrix @ centroids.T).argmax(axis=1)
        new_centroids = centroids.copy()
        for class_id in range(k):
            members = matrix[labels == class_id]
            if members.shape[0]:
                mean = members.mean(axis=0)
                norm = float(np.linalg.norm(mean))
                if norm > 0.0:
                    new_centroids[class_id] = mean / norm
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids


def _renorm_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)


def _assign(matrix: np.ndarray, centroids: np.ndarray, distance: str) -> np.ndarray:
    if distance == "euclidean":
        dist = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        return dist.argmin(axis=1)
    return (matrix @ centroids.T).argmax(axis=1)


def _prototype_residual(
    matrix: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> float:
    """Mean squared Euclidean distance to the assigned centroid."""
    if matrix.shape[0] == 0:
        return 0.0
    assigned = centroids[labels]
    return float(np.mean(np.sum((matrix - assigned) ** 2, axis=1)))


def _farthest_first_indices(matrix: np.ndarray, k: int) -> List[int]:
    norms = np.linalg.norm(matrix, axis=1)
    start = int(np.argmax(norms))
    chosen = [start]
    dmin = ((matrix - matrix[start]) ** 2).sum(axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dnew = ((matrix - matrix[nxt]) ** 2).sum(axis=1)
        dmin = np.minimum(dmin, dnew)
    return chosen


def _random_subset_indices(n: int, k: int, rng: np.random.Generator) -> List[int]:
    pool = list(range(n))
    for i in range(k):
        j = int(rng.integers(i, n))
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:k]


def _kmeanspp_indices(matrix: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    n = int(matrix.shape[0])
    first = int(rng.integers(0, n))
    chosen = [first]
    closest = ((matrix - matrix[first]) ** 2).sum(axis=1)
    chosen_set = {first}
    for _ in range(1, k):
        weights = closest.copy()
        for index in chosen:
            weights[index] = 0.0
        total = float(weights.sum())
        if total <= 0.0:
            unused = [index for index in range(n) if index not in chosen_set]
            nxt = unused[0]
        else:
            draw = float(rng.random()) * total
            cdf = np.cumsum(weights)
            nxt = int(np.searchsorted(cdf, draw, side="right"))
            if nxt >= n:
                nxt = n - 1
            if nxt in chosen_set:
                positive = np.flatnonzero(weights > 0.0)
                nxt = int(positive[0])
        chosen.append(nxt)
        chosen_set.add(nxt)
        closest = np.minimum(closest, ((matrix - matrix[nxt]) ** 2).sum(axis=1))
    return chosen


def _attach_stability(runs: List[dict]) -> None:
    production = _find_family(runs, PRODUCTION_FAMILY, None)
    by_family: Dict[str, List[dict]] = {}
    for run in runs:
        by_family.setdefault(run["family"], []).append(run)
    for run in runs:
        run["ari_vs_production"] = adjusted_rand_index(run["labels"], production["labels"])
        siblings = by_family[run["family"]]
        run["stability_ari"] = _median_pairwise_ari([item["labels"] for item in siblings])


def _median_pairwise_ari(groups: Sequence[np.ndarray]) -> Optional[float]:
    if len(groups) < 2:
        return None
    scores = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            scores.append(adjusted_rand_index(groups[i], groups[j]))
    return _median(scores)


def _select_representatives(runs: Sequence[dict]) -> List[dict]:
    """Lowest residual, then lowest seed. Labels are not consulted."""
    grouped: Dict[str, List[dict]] = {}
    for run in runs:
        if run["family"] == PRODUCTION_FAMILY:
            continue
        grouped.setdefault(run["family"], []).append(run)
    chosen = []
    for family in FAMILY_TIE_ORDER:
        group = grouped[family]
        group.sort(key=lambda run: (run["residual"], _seed_key(run["seed"])))
        chosen.append(group[0])
    return chosen


def _best_unsupervised(representatives: Sequence[dict]) -> dict:
    order = {name: index for index, name in enumerate(FAMILY_TIE_ORDER)}
    return min(
        representatives,
        key=lambda run: (run["residual"], order[run["family"]], _seed_key(run["seed"])),
    )


def _seed_key(seed: Optional[int]) -> int:
    return -1 if seed is None else int(seed)


def _score_topics(labels: np.ndarray, topics: Sequence[str]) -> dict:
    """Probe scores for one assignment. Not used to fit or to pick a seed."""
    codes = [int(code) for code in np.unique(labels)]
    clusters = []
    for code in codes:
        members = np.flatnonzero(labels == code)
        member_topics = [topics[int(index)] for index in members]
        majority, count = _majority(member_topics)
        rows = int(members.size)
        topic_counts: Dict[str, int] = {}
        for topic in member_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        weather_count = int(topic_counts.get(PROBE_TOPIC, 0))
        purity = count / rows if rows else 0.0
        clusters.append(
            {
                "code": code,
                "rows": rows,
                "majority_topic": majority,
                "majority_count": int(count),
                "purity": float(purity),
                "weather_count": weather_count,
                "non_weather": int(rows - weather_count),
                "topic_counts": dict(sorted(topic_counts.items())),
            }
        )
    # Cluster holding the most probe rows, even when that topic is not the majority.
    # Used to describe a run that has no weather-majority cluster. Not a selection input.
    weather_mass = max(
        clusters,
        key=lambda cluster: (cluster["weather_count"], cluster["purity"], -int(cluster["code"])),
    )
    weather_clusters = [cluster for cluster in clusters if cluster["majority_topic"] == PROBE_TOPIC]
    if weather_clusters:
        weather = max(
            weather_clusters,
            key=lambda cluster: (
                cluster["weather_count"],
                cluster["purity"],
                -int(cluster["code"]),
            ),
        )
        weather_purity = float(weather["purity"])
        weather_rows = int(weather["rows"])
        weather_count = int(weather["weather_count"])
        non_weather = int(weather["non_weather"])
        contamination = float(1.0 - weather_purity)
        weather_code = int(weather["code"])
        contaminant_counts = {
            topic: count
            for topic, count in weather["topic_counts"].items()
            if topic != PROBE_TOPIC
        }
    else:
        weather_purity = None
        weather_rows = None
        weather_count = None
        non_weather = None
        contamination = None
        weather_code = None
        contaminant_counts = None
    weight_rows = sum(cluster["rows"] for cluster in clusters)
    weight_hits = sum(cluster["majority_count"] for cluster in clusters)
    smallest = min((cluster["rows"] for cluster in clusters), default=0)
    size_ratio = None
    if weather_rows is not None and smallest:
        size_ratio = float(weather_rows) / float(smallest)
    return {
        "weather_code": weather_code,
        "weather_contaminants": contaminant_counts,
        "weather_purity": weather_purity,
        "weather_rows": weather_rows,
        "weather_count": weather_count,
        "non_weather": non_weather,
        "contamination": contamination,
        "size_weighted_hits": int(weight_hits),
        "size_weighted_rows": int(weight_rows),
        "size_weighted": float(weight_hits / weight_rows) if weight_rows else 0.0,
        "smallest_rows": int(smallest),
        "size_ratio": size_ratio,
        "cluster_sizes": [int(cluster["rows"]) for cluster in clusters],
        "weather_mass_code": int(weather_mass["code"]),
        "weather_mass_rows": int(weather_mass["rows"]),
        "weather_mass_count": int(weather_mass["weather_count"]),
        "weather_mass_majority": weather_mass["majority_topic"],
    }


def _majority(topics: Sequence[str]) -> Tuple[Optional[str], int]:
    counts: Dict[str, int] = {}
    for topic in topics:
        counts[topic] = counts.get(topic, 0) + 1
    if not counts:
        return None, 0
    topic, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return topic, int(count)


def _held_out_score(
    matrix: np.ndarray,
    run: Mapping,
    train: np.ndarray,
    test: np.ndarray,
    topics: Sequence[str],
) -> dict:
    """Fit on train rows only and assign held-out rows. No refit on test."""
    train_labels, centroids, _ = _fit_algorithm(matrix[train], run)
    assigned = _assign(matrix[test], centroids, str(run["distance"]))
    test_topics = [topics[int(index)] for index in test]
    scored = _score_topics(assigned, test_topics)
    return {
        "family": run["family"],
        "init": run["init"],
        "seed": run["seed"],
        "train_residual": _prototype_residual(matrix[train], train_labels, centroids),
        **scored,
    }


def _negative_controls(
    matrix: np.ndarray,
    topics: Sequence[str],
    production_labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    baseline_size_weighted: float,
    baseline_weather_purity: Optional[float],
    production_held_out_size_weighted: float,
) -> dict:
    label_rng = np.random.default_rng(LABEL_SHUFFLE_SEED)
    shuffled_topics = [topics[int(index)] for index in label_rng.permutation(len(topics))]
    label_score = _score_topics(production_labels, shuffled_topics)
    weather_code = _weather_code(production_labels, topics)
    if weather_code is None:
        majority, count = None, 0
        weather_under_shuffle = None
        weather_member_rows = 0
    else:
        weather_members = np.flatnonzero(production_labels == weather_code)
        shuffled_on_weather = [shuffled_topics[int(index)] for index in weather_members]
        majority, count = _majority(shuffled_on_weather)
        weather_under_shuffle = float(count / len(shuffled_on_weather))
        weather_member_rows = int(len(shuffled_on_weather))
    label_below_weighted = float(label_score["size_weighted"]) < baseline_size_weighted
    label_below_weather = (
        baseline_weather_purity is not None
        and float(label_score["size_weighted"]) < float(baseline_weather_purity)
    )
    weather_set_below = (
        baseline_weather_purity is not None
        and weather_under_shuffle is not None
        and weather_under_shuffle < float(baseline_weather_purity)
    )

    row_rng = np.random.default_rng(ROW_SHUFFLE_SEED)
    order = row_rng.permutation(matrix.shape[0])
    shuffled_matrix = matrix[order]
    spec = {
        "family": PRODUCTION_FAMILY,
        "distance": "euclidean",
        "init": "farthest_first",
        "seed": None,
    }
    _fitted, centroids, _ = _fit_algorithm(shuffled_matrix[train], spec)
    del _fitted
    assigned = _assign(shuffled_matrix[test], centroids, "euclidean")
    test_topics = [topics[int(index)] for index in test]
    row_score = _score_topics(assigned, test_topics)
    row_below = float(row_score["size_weighted"]) < production_held_out_size_weighted
    probe_ok = bool(
        label_below_weighted and label_below_weather and weather_set_below and row_below
    )
    return {
        "probe_ok": probe_ok,
        "label_shuffle": {
            "seed": LABEL_SHUFFLE_SEED,
            "size_weighted": label_score["size_weighted"],
            "size_weighted_hits": label_score["size_weighted_hits"],
            "size_weighted_rows": label_score["size_weighted_rows"],
            "below_production_size_weighted": label_below_weighted,
            "below_production_weather_purity": label_below_weather,
            "weather_cluster_majority": majority,
            "weather_cluster_purity": weather_under_shuffle,
            "weather_cluster_rows": weather_member_rows,
            "weather_cluster_majority_count": int(count),
            "weather_cluster_below_baseline": weather_set_below,
        },
        "row_shuffle": {
            "seed": ROW_SHUFFLE_SEED,
            "held_out_size_weighted": row_score["size_weighted"],
            "held_out_size_weighted_hits": row_score["size_weighted_hits"],
            "held_out_size_weighted_rows": row_score["size_weighted_rows"],
            "held_out_weather_purity": row_score["weather_purity"],
            "held_out_weather_rows": row_score["weather_rows"],
            "held_out_weather_count": row_score["weather_count"],
            "held_out_non_weather": row_score["non_weather"],
            "below_production_held_out": row_below,
        },
    }


def _weather_code(labels: np.ndarray, topics: Sequence[str]) -> Optional[int]:
    scored = _score_topics(labels, topics)
    if scored["weather_code"] is None:
        return None
    return int(scored["weather_code"])


def _attach_timings(matrix: np.ndarray, runs: List[dict], time_runs: int) -> None:
    for run in runs:
        spec = run

        def _call(spec: Mapping = spec) -> None:
            _fit_algorithm(matrix, spec)

        median_ms, samples = _median_ms(_call, time_runs)
        run["median_ms"] = median_ms
        run["runtime_samples_ms"] = samples
        run["peak_bytes"] = _peak_bytes(_call, time_runs)


def _time_learn_and_svd(
    texts: Sequence[str],
    matrix: np.ndarray,
    time_runs: int,
) -> dict:
    observations = [
        Observation(observation_id=f"row-{index:04d}", text=text or "", metadata={})
        for index, text in enumerate(texts)
    ]
    config = LearnConfig(n_symbols=K, seed=0, encoder=ENCODER_WORD_SENTENCE_SVD)

    def _learn() -> None:
        learn_pack(observations, config=config)

    def _svd() -> None:
        word_sentence_svd(texts, rank=RANK)

    def _cluster() -> None:
        kmeans(matrix, K, rng=np.random.default_rng(0))

    learn_median, learn_samples = _median_ms(_learn, time_runs)
    svd_median, svd_samples = _median_ms(_svd, time_runs)
    cluster_median, cluster_samples = _median_ms(_cluster, time_runs)
    return {
        "learn_ms_median": learn_median,
        "learn_ms_samples": learn_samples,
        "svd_ms_median": svd_median,
        "svd_ms_samples": svd_samples,
        "production_cluster_ms_median": cluster_median,
        "production_cluster_ms_samples": cluster_samples,
        "documented_learn_ms": LEARN_PACK_BASELINE_MS,
    }


def _evaluate_candidate(
    run: Mapping,
    production: Mapping,
    held: Mapping,
    held_production: Mapping,
    cost: Mapping,
) -> dict:
    production_residual = float(production["residual"])
    residual_ok = float(run["residual"]) <= RESIDUAL_WORSE_FACTOR * production_residual
    # Same harness as the candidate: one algorithm call, median of the samples.
    prod_ms = production["median_ms"]
    cand_ms = run["median_ms"]
    learn_ms = cost["learn_ms_median"]
    if prod_ms is None or cand_ms is None or learn_ms is None:
        runtime_ok = False
        learn_equivalent_ok = False
        runtime_ratio = None
        learn_equivalent_ms = None
    else:
        runtime_ok = float(cand_ms) <= CLUSTER_RUNTIME_FACTOR * float(prod_ms)
        runtime_ratio = float(cand_ms) / float(prod_ms) if float(prod_ms) else None
        non_clustering = max(0.0, float(learn_ms) - float(prod_ms))
        learn_equivalent_ms = non_clustering + float(cand_ms)
        learn_equivalent_ok = (
            learn_equivalent_ms < LEARN_EQUIVALENT_FACTOR * LEARN_PACK_BASELINE_MS
        )
    candidate_weather = held["weather_purity"]
    production_weather = held_production["weather_purity"]
    weather_ok = (
        candidate_weather is not None
        and production_weather is not None
        and float(candidate_weather) > float(production_weather)
    )
    weighted_ok = float(held["size_weighted"]) > float(held_production["size_weighted"])
    held_production_weather = production_weather
    admissible = bool(
        residual_ok and runtime_ok and learn_equivalent_ok and weather_ok and weighted_ok
    )
    return {
        "family": run["family"],
        "init": run["init"],
        "distance": run["distance"],
        "seed": run["seed"],
        "residual": run["residual"],
        "median_ms": cand_ms,
        "runtime_ratio": runtime_ratio,
        "peak_bytes": run["peak_bytes"],
        "learn_equivalent_ms": learn_equivalent_ms,
        "weather_purity": run["weather_purity"],
        "weather_rows": run["weather_rows"],
        "weather_count": run["weather_count"],
        "non_weather": run["non_weather"],
        "contamination": run["contamination"],
        "size_weighted": run["size_weighted"],
        "size_weighted_hits": run["size_weighted_hits"],
        "size_weighted_rows": run["size_weighted_rows"],
        "smallest_rows": run["smallest_rows"],
        "size_ratio": run["size_ratio"],
        "cluster_sizes": list(run["cluster_sizes"]),
        "stability_ari": run["stability_ari"],
        "ari_vs_production": run["ari_vs_production"],
        "held_out_weather_purity": held["weather_purity"],
        "held_out_weather_rows": held["weather_rows"],
        "held_out_weather_count": held["weather_count"],
        "held_out_non_weather": held["non_weather"],
        "held_out_contamination": held["contamination"],
        "held_out_size_weighted": held["size_weighted"],
        "held_out_size_weighted_hits": held["size_weighted_hits"],
        "held_out_size_weighted_rows": held["size_weighted_rows"],
        "residual_ok": residual_ok,
        "runtime_ok": runtime_ok,
        "learn_equivalent_ok": learn_equivalent_ok,
        "held_out_weather_ok": weather_ok,
        "held_out_weighted_ok": weighted_ok,
        "admissible": admissible,
        "production_held_out_weather_purity": held_production_weather,
    }


def _production_identity(
    texts: Sequence[str],
    matrix: np.ndarray,
    production_labels: np.ndarray,
    topics: Sequence[str],
) -> dict:
    """One ``learn_pack`` must reproduce the production labels.

    ``certify`` is the structural contract on that pack. It does not grow
    a purity field. Topic strings are applied to the sealed membership
    only for the live size-weighted receipt.
    """
    observations = [
        Observation(observation_id=f"row-{index:04d}", text=text or "", metadata={})
        for index, text in enumerate(texts)
    ]
    pack = learn_pack(
        observations,
        config=LearnConfig(n_symbols=K, seed=0, encoder=ENCODER_WORD_SENTENCE_SVD),
    )
    stored = np.asarray(
        [pack.observations[obs.observation_id].embedding for obs in observations],
        dtype=np.float64,
    )
    if not np.array_equal(matrix, stored):
        raise RuntimeError("frozen rank-8 matrix diverged from learn_pack")
    for symbol in pack.symbols:
        for obs_id in symbol.observation_ids:
            index = int(obs_id.split("-", 1)[1])
            if int(production_labels[index]) != int(symbol.code):
                raise RuntimeError("learn_pack assignment diverged from production kmeans")
    certificate = certify(pack)
    live = _live_from_pack(pack, topics)
    return {
        "learn_pack_matches": True,
        "certify_passed": bool(certificate.passed),
        "certify_tau_residual": certificate.details["tau_residual"],
        "certify_has_cluster_purity": "cluster_purity" in certificate.details,
        "certify_has_failure_mode": "failure_mode" in certificate.details,
        "live_size_weighted_hits": live["hits"],
        "live_size_weighted_rows": live["rows"],
        "live_size_weighted": live["purity"],
        "metadata_empty": all(not obs.metadata for obs in observations),
    }


def _live_from_pack(pack, topics: Sequence[str]) -> dict:
    """Size-weighted majority purity on symbols the pack left unquarantined.

    Quarantine stays the pack's ``min_cluster_size`` and ``tau_kappa``.
    This receipt is not the admission score. The admission score counts
    every non-empty cluster, so ``tau_kappa`` does not choose a candidate.
    """
    hits = 0
    rows = 0
    for symbol in pack.symbols:
        if symbol.quarantined or not symbol.observation_ids:
            continue
        members = [int(obs_id.split("-", 1)[1]) for obs_id in symbol.observation_ids]
        _, count = _majority([topics[index] for index in members])
        hits += int(count)
        rows += len(members)
    return {
        "hits": hits,
        "rows": rows,
        "purity": float(hits / rows) if rows else 0.0,
    }


def _find_family(runs: Sequence[dict], family: str, seed: Optional[int]) -> dict:
    for run in runs:
        if run["family"] == family and run["seed"] == seed:
            return run
    raise KeyError((family, seed))


def _matched_pairs(runs: Sequence[dict]) -> List[dict]:
    pairs = []
    euclidean = {
        (run["init"], run["seed"]): run for run in runs if run["distance"] == "euclidean"
    }
    for run in runs:
        if run["distance"] != "spherical":
            continue
        match = euclidean[(run["init"], run["seed"])]
        shared = list(run["init_indices"]) == list(match["init_indices"])
        left = match["weather_purity"]
        right = run["weather_purity"]
        if left is None or right is None:
            delta = 1.0
        else:
            delta = abs(float(left) - float(right))
        pairs.append(
            {
                "init": run["init"],
                "seed": run["seed"],
                "shared_init": shared,
                "ari": adjusted_rand_index(run["labels"], match["labels"]),
                "euclidean_weather_purity": left,
                "spherical_weather_purity": right,
                "purity_delta": float(delta),
                "euclidean_family": match["family"],
                "spherical_family": run["family"],
            }
        )
    return pairs


def _purity_span(purities: Sequence[Optional[float]]) -> float:
    numeric = [float(value) for value in purities if value is not None]
    missing = sum(1 for value in purities if value is None)
    if not numeric:
        return 0.0
    span = max(numeric) - min(numeric)
    if missing:
        span = max(span, 1.0)
    return float(span)


def _purity_span_from_pairs(pairs: Sequence[Mapping]) -> float:
    if not pairs:
        return 0.0
    return float(max(float(pair["purity_delta"]) for pair in pairs))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[len(ordered) // 2]


def _median_ms(fn, runs: int) -> Tuple[Optional[float], List[float]]:
    if runs <= 0:
        return None, []
    samples: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return _median(samples), samples


def _peak_bytes(fn, runs: int) -> Optional[int]:
    if runs <= 0:
        return None
    peaks: List[int] = []
    for _ in range(runs):
        tracemalloc.start()
        try:
            fn()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        peaks.append(int(peak))
    return int(_median([float(peak) for peak in peaks]))


def _public_run(run: Mapping) -> dict:
    """Drop assignments and centroids from the returned record."""
    skip = {"labels", "centroids"}
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

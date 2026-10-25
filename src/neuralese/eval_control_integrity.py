"""Evaluation-control integrity audit.

The clustering-generalization trial scored shuffled eval labels by refitting
the majority topic inside each eval cluster. That uses the eval labels
twice. This module freezes a different scorer and does not change
``clustering.kmeans``, ``word_sentence_svd``, ``learn_pack``, ``certify``,
the four frozen candidates, or the admission thresholds.

Analytical controls run before any candidate is scored. If they fail, the
candidate rerun does not happen and the audit verdict is ``FAIL``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neuralese.clustering_generalization import (
    CLUSTER_RUNTIME_FACTOR,
    CLUSTER_TIME_REPEATS,
    FIXTURE_SPECS,
    FROZEN_CANDIDATES,
    LABEL_SHUFFLE_SEED,
    RANK,
    RESIDUAL_WORSE_FACTOR,
    ROW_SHUFFLE_SEED,
    RUNTIME_AUTHORITY,
    TIME_RUNS,
    _fit_one,
    _negative_controls,
    _per_call_median_ms,
    _production_spec,
    _score_assignment,
    advantage_on_fixture,
    holdout_mask,
    load_fixture,
    recommend,
)
from neuralese.clustering_isolation import _assign, _fit_algorithm, _majority, adjusted_rand_index
from neuralese.encode import content_tokens, word_sentence_svd, word_sentence_tfidf
from neuralese.factorization import pin_blas_threads

# Revised control definition. These constants are the definition. They are
# not adjusted after a candidate score.
REVISED_DEFINITION = "train_only_majority_map"
MAP_SOURCE = "train_rows_only"
LABEL_CONTROL = "permute_eval_labels"
EMBEDDING_CONTROL = "permute_eval_embeddings_relative_to_eval_labels"
# Same seeds as the generalization trial. Not retuned, and not candidate seeds.
REVISED_LABEL_SHUFFLE_SEED = LABEL_SHUFFLE_SEED
REVISED_EMBEDDING_SHUFFLE_SEED = ROW_SHUFFLE_SEED

# Analytical gates. A shuffle scored with the train-only map must land at or
# under SHUFFLE_MAX_PURITY and strictly under the real purity. The real
# two-blob purity must clear BLOB_REAL_MIN_PURITY. A single Gaussian scored
# with the train-only map must land within GAUSSIAN_ABS_TOLERANCE of the
# eval majority base rate.
ANALYTICAL_DATA_SEED = 11
ANALYTICAL_DIM = 8
ANALYTICAL_BLOB_GAP = 30.0
ANALYTICAL_BLOB_SCALE = 1.0
ANALYTICAL_BLOB_TRAIN_PER_CLASS = 200
ANALYTICAL_BLOB_EVAL_PER_CLASS = 100
BLOB_REAL_MIN_PURITY = 0.99
SHUFFLE_MAX_PURITY = 0.60
ANALYTICAL_GAUSS_TRAIN = 2000
ANALYTICAL_GAUSS_EVAL = 2000
ANALYTICAL_GAUSS_MAJORITY_RATE = 0.75
GAUSSIAN_ABS_TOLERANCE = 0.02
SINGLETON_EVAL_ROWS = 8
SINGLETON_TRAIN_PER_CLUSTER = 4

# Full-matrix SVD versus a train-only basis. Train adjusted Rand below this
# means the leak moves the clustering. The bar is not an admission threshold.
SVD_MATERIAL_TRAIN_ARI = 0.90

BASELINE_COMMIT = "75f26cb"


def train_only_majority_map(labels: np.ndarray, topics: Sequence[str]) -> Dict[int, str]:
    """Majority topic of each cluster, from these rows only.

    The tie break is ``clustering_isolation._majority``: higher count, then
    the topic string in ascending order. Clusters that do not appear in
    ``labels`` are absent from the map.
    """
    labels = np.asarray(labels)
    if int(labels.size) != len(topics):
        raise ValueError("labels and topics differ in length")
    mapping: Dict[int, str] = {}
    for code in np.unique(labels):
        members = [topics[int(index)] for index in np.flatnonzero(labels == code)]
        topic, _count = _majority(members)
        if topic is None:
            continue
        mapping[int(code)] = topic
    return mapping


def score_against_map(
    labels: np.ndarray,
    topics: Sequence[str],
    mapping: Mapping[int, str],
) -> dict:
    """Count rows whose topic equals the mapped label of their cluster.

    A cluster id that is not in ``mapping`` is a miss. This function does
    not recompute a majority from ``topics``.
    """
    labels = np.asarray(labels)
    if int(labels.size) != len(topics):
        raise ValueError("labels and topics differ in length")
    hits = 0
    for index, code in enumerate(labels.tolist()):
        if mapping.get(int(code)) == topics[index]:
            hits += 1
    rows = int(labels.size)
    return {
        "hits": int(hits),
        "rows": rows,
        "size_weighted": float(hits / rows) if rows else 0.0,
    }


def permute_sequence(items: Sequence[str], seed: int) -> List[str]:
    """Return ``items`` in the order drawn by ``numpy`` Generator ``seed``."""
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    order = np.random.default_rng(seed).permutation(len(items))
    return [items[int(index)] for index in order]


def permute_rows(matrix: np.ndarray, seed: int) -> np.ndarray:
    """Permute rows. Row ``i`` of the result is the old row ``order[i]``."""
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    order = np.random.default_rng(seed).permutation(int(matrix.shape[0]))
    return np.asarray(matrix)[order]


def expected_label_permutation_hits(predictions: Sequence[str], labels: Sequence[str]) -> float:
    """Expected hits of a uniform permutation of ``labels`` against ``predictions``.

    For each row the chance the permuted label matches the fixed prediction
    is the count of that prediction in ``labels``, divided by the length.
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels differ in length")
    rows = len(labels)
    if rows == 0:
        return 0.0
    counts = Counter(labels)
    return float(sum(counts[prediction] / rows for prediction in predictions))


def integrity_verdict(
    *,
    analytical_passed: bool,
    candidates_rerun: bool,
    fixture_controls_collapse: bool,
    svd_material: bool,
) -> str:
    """``PASS`` only when the revised method holds on the fixtures too.

    ``FAIL`` means the analytical controls did not behave, so the candidates
    were not rerun. ``UNRESOLVED`` means those controls passed and the
    candidates were rerun, but the fixture result is not a clean holdout:
    a corrected control did not strictly drop, or the full-matrix SVD moves
    the train clustering. ``PASS`` means the evaluation method is internally
    valid. It is not a clustering-quality claim and it does not authorize
    a runtime change.
    """
    if not analytical_passed or not candidates_rerun:
        return "FAIL"
    if fixture_controls_collapse and not svd_material:
        return "PASS"
    return "UNRESOLVED"


def run_analytical_controls() -> dict:
    """Synthetic cases whose expected scores do not depend on the fixtures."""
    blobs = _two_blobs()
    gaussian = _one_gaussian()
    singleton = _singleton_contrast()
    checks = {
        "blob_real_near_one": blobs["real_purity"] >= BLOB_REAL_MIN_PURITY,
        "blob_label_shuffle_near_chance": blobs["label_shuffle_purity"] <= SHUFFLE_MAX_PURITY,
        "blob_label_shuffle_drops": blobs["label_shuffle_purity"] < blobs["real_purity"],
        "blob_embedding_shuffle_near_chance": (
            blobs["embedding_shuffle_purity"] <= SHUFFLE_MAX_PURITY
        ),
        "blob_embedding_shuffle_drops": (
            blobs["embedding_shuffle_purity"] < blobs["real_purity"]
        ),
        "gaussian_near_base_rate": gaussian["abs_gap"] <= GAUSSIAN_ABS_TOLERANCE,
        "singleton_self_majority_stays_perfect": (
            singleton["self_majority_hits_after_shuffle"] == singleton["rows"]
        ),
        "singleton_train_map_drops": (
            singleton["train_map_hits_after_shuffle"] < singleton["rows"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "blobs": blobs,
        "gaussian": gaussian,
        "singleton": singleton,
        "definition": REVISED_DEFINITION,
    }


def audit_eval_controls(
    root: Optional[Path] = None,
    *,
    time_runs: int = TIME_RUNS,
) -> dict:
    """Run the analytical controls, then score fixtures only if they pass."""
    if time_runs < 0:
        raise ValueError(f"time_runs must be >= 0, got {time_runs}")
    analytical = run_analytical_controls()
    if not analytical["passed"]:
        return {
            "verdict": "FAIL",
            "recommendation": "NONE",
            "runtime_authority": RUNTIME_AUTHORITY,
            "candidates_rerun": False,
            "analytical_passed": False,
            "fixture_controls_collapse": False,
            "svd_material": None,
            "definition": _definition_record(),
            "analytical": analytical,
            "fixtures": [],
            "baseline_commit": BASELINE_COMMIT,
        }
    fixture_root = Path(root) if root is not None else _default_root()
    pin_blas_threads(1)
    fixtures = [
        _audit_fixture(fixture_root / str(spec["dirname"]), str(spec["id"]), time_runs=time_runs)
        for spec in FIXTURE_SPECS
    ]
    advantages = [
        {row["id"]: bool(row["advantage"]) for row in fixture["candidates"]}
        for fixture in fixtures
    ]
    recommendation = recommend(advantages)
    collapse = all(bool(fixture["corrected_controls"]["both_below"]) for fixture in fixtures)
    material = any(bool(fixture["leakage"]["material"]) for fixture in fixtures)
    return {
        "verdict": integrity_verdict(
            analytical_passed=True,
            candidates_rerun=True,
            fixture_controls_collapse=collapse,
            svd_material=material,
        ),
        "recommendation": recommendation,
        "runtime_authority": RUNTIME_AUTHORITY,
        "candidates_rerun": True,
        "analytical_passed": True,
        "fixture_controls_collapse": collapse,
        "svd_material": material,
        "definition": _definition_record(),
        "analytical": analytical,
        "fixtures": fixtures,
        "baseline_commit": BASELINE_COMMIT,
        "bar": {
            "residual_worse_factor": RESIDUAL_WORSE_FACTOR,
            "cluster_runtime_factor": CLUSTER_RUNTIME_FACTOR,
            "purity": "corrected eval size-weighted purity strictly greater than production",
            "recommendation": "advantage on every fixture, otherwise NONE",
        },
    }


def _definition_record() -> dict:
    return {
        "id": REVISED_DEFINITION,
        "map_source": MAP_SOURCE,
        "label_control": LABEL_CONTROL,
        "embedding_control": EMBEDDING_CONTROL,
        "label_shuffle_seed": REVISED_LABEL_SHUFFLE_SEED,
        "embedding_shuffle_seed": REVISED_EMBEDDING_SHUFFLE_SEED,
        "blob_real_min_purity": BLOB_REAL_MIN_PURITY,
        "shuffle_max_purity": SHUFFLE_MAX_PURITY,
        "gaussian_abs_tolerance": GAUSSIAN_ABS_TOLERANCE,
        "svd_material_train_ari": SVD_MATERIAL_TRAIN_ARI,
        "residual_worse_factor": RESIDUAL_WORSE_FACTOR,
        "cluster_runtime_factor": CLUSTER_RUNTIME_FACTOR,
        "admission_thresholds_changed": False,
    }


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "examples" / "generalization"


def _two_blobs() -> dict:
    """Two isotropic blobs in 8-d, separated on the first axis.

    Production k-means, k = 8, is the clustering. Each blob is pure, so a
    correct train map scores every eval row. A label permutation scored
    with that map is a class match against a fixed prediction, which for
    equal classes is near one half. An embedding permutation does the same
    once each point is still assigned to its own blob.
    """
    rng = np.random.default_rng(ANALYTICAL_DATA_SEED)
    dim = ANALYTICAL_DIM
    center_b = np.zeros(dim)
    center_b[0] = ANALYTICAL_BLOB_GAP

    def _draw(count: int, center: np.ndarray) -> np.ndarray:
        return center + rng.normal(
            scale=ANALYTICAL_BLOB_SCALE,
            size=(count, dim),
        )

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
    shuffled_labels = permute_sequence(eval_y, REVISED_LABEL_SHUFFLE_SEED)
    label_shuffle = score_against_map(eval_labels, shuffled_labels, mapping)
    shuffled_eval = permute_rows(eval_x, REVISED_EMBEDDING_SHUFFLE_SEED)
    embed_labels = _assign(shuffled_eval, centroids, "euclidean")
    embed_shuffle = score_against_map(embed_labels, eval_y, mapping)
    return {
        "n_eval": real["rows"],
        "real_hits": real["hits"],
        "real_purity": real["size_weighted"],
        "label_shuffle_hits": label_shuffle["hits"],
        "label_shuffle_purity": label_shuffle["size_weighted"],
        "embedding_shuffle_hits": embed_shuffle["hits"],
        "embedding_shuffle_purity": embed_shuffle["size_weighted"],
        "balanced_chance": 0.5,
    }


def _one_gaussian() -> dict:
    """One standard Gaussian. Labels are independent of position.

    Both clusters of a large draw take the majority label as their train
    map, so every eval row is scored as that label. Purity equals the eval
    majority base rate.
    """
    rng = np.random.default_rng(ANALYTICAL_DATA_SEED)
    train_x = rng.normal(size=(ANALYTICAL_GAUSS_TRAIN, ANALYTICAL_DIM))
    eval_x = rng.normal(size=(ANALYTICAL_GAUSS_EVAL, ANALYTICAL_DIM))
    train_y = _majority_labels(rng, ANALYTICAL_GAUSS_TRAIN, ANALYTICAL_GAUSS_MAJORITY_RATE)
    eval_y = _majority_labels(rng, ANALYTICAL_GAUSS_EVAL, ANALYTICAL_GAUSS_MAJORITY_RATE)
    base_hits = sum(1 for topic in eval_y if topic == "maj")
    base_rate = base_hits / float(len(eval_y))
    train_labels, centroids, _indices = _fit_algorithm(train_x, _production_spec())
    eval_labels = _assign(eval_x, centroids, "euclidean")
    mapping = train_only_majority_map(train_labels, train_y)
    scored = score_against_map(eval_labels, eval_y, mapping)
    return {
        "n_eval": scored["rows"],
        "majority_base_hits": int(base_hits),
        "majority_base_rate": float(base_rate),
        "hits": scored["hits"],
        "purity": scored["size_weighted"],
        "abs_gap": abs(float(scored["size_weighted"]) - float(base_rate)),
        "map_values": sorted(set(mapping.values())),
    }


def _majority_labels(rng: np.random.Generator, count: int, rate: float) -> List[str]:
    return ["maj" if float(rng.random()) < rate else "min" for _ in range(count)]


def _singleton_contrast() -> dict:
    """Eight eval singletons. Self-majority stays 1 after any relabeling.

    The train map is fixed before the shuffle. The same shuffle scored
    with that map drops. This is the case the old eval scorer cannot fail.
    """
    rows = SINGLETON_EVAL_ROWS
    per = SINGLETON_TRAIN_PER_CLUSTER
    train_labels = np.repeat(np.arange(rows), per)
    train_topics = [
        "alpha" if int(code) % 2 == 0 else "beta" for code in train_labels.tolist()
    ]
    eval_labels = np.arange(rows)
    eval_topics = ["alpha" if code % 2 == 0 else "beta" for code in range(rows)]
    mapping = train_only_majority_map(train_labels, train_topics)
    real = score_against_map(eval_labels, eval_topics, mapping)
    shuffled = permute_sequence(eval_topics, REVISED_LABEL_SHUFFLE_SEED)
    mapped = score_against_map(eval_labels, shuffled, mapping)
    refit = _score_assignment(eval_labels, shuffled)
    return {
        "rows": rows,
        "real_hits": real["hits"],
        "self_majority_hits_after_shuffle": int(refit["hits"]),
        "train_map_hits_after_shuffle": mapped["hits"],
    }


def _audit_fixture(directory: Path, fixture_id: str, *, time_runs: int) -> dict:
    loaded = load_fixture(directory)
    texts = loaded["texts"]
    topics = loaded["topics"]
    observation_ids = loaded["observation_ids"]
    n_rows = len(texts)
    mask = holdout_mask(n_rows)
    train = np.flatnonzero(~mask)
    test = np.flatnonzero(mask)
    matrix = np.asarray(word_sentence_svd(list(texts), rank=RANK), dtype=np.float64)
    train_topics = [topics[int(index)] for index in train]
    eval_topics = [topics[int(index)] for index in test]
    specs = [_production_spec()]
    for candidate in FROZEN_CANDIDATES:
        specs.append(
            {
                "id": str(candidate["id"]),
                "distance": str(candidate["distance"]),
                "init": str(candidate["init"]),
                "seed": int(candidate["seed"]),
            }
        )
    fitted = [_fit_one(matrix[train], matrix[test], spec) for spec in specs]
    scored = []
    for run in fitted:
        mapping = train_only_majority_map(run["train_labels"], train_topics)
        eval_score = score_against_map(run["eval_labels"], eval_topics, mapping)
        self_score = _score_assignment(run["eval_labels"], eval_topics)
        scored.append(
            {
                "id": run["id"],
                "distance": run["distance"],
                "init": run["init"],
                "seed": run["seed"],
                "residual": float(run["residual"]),
                "eval_hits": eval_score["hits"],
                "eval_rows": eval_score["rows"],
                "eval_size_weighted": eval_score["size_weighted"],
                "self_majority_hits": self_score["hits"],
                "train_map": {str(key): value for key, value in mapping.items()},
            }
        )
    _attach_cluster_times(matrix[train], scored, time_runs)
    production = scored[0]
    for run in scored[1:]:
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
    controls = _corrected_controls(
        matrix,
        train,
        test,
        train_topics,
        eval_topics,
        fitted[0],
    )
    original = _negative_controls(matrix, topics, train, test, fitted[0])
    alignment = _original_row_shuffle_alignment(matrix, topics, train)
    _codes, eval_counts = np.unique(fitted[0]["eval_labels"], return_counts=True)
    return {
        "fixture_id": fixture_id,
        "n_rows": n_rows,
        "n_train": int(train.size),
        "n_eval": int(test.size),
        "train_class_counts": dict(sorted(Counter(train_topics).items())),
        "eval_class_counts": dict(sorted(Counter(eval_topics).items())),
        "eval_majority_count": int(max(Counter(eval_topics).values())),
        "eval_cluster_sizes": sorted(int(count) for count in eval_counts.tolist()),
        "production": production,
        "candidates": scored[1:],
        "corrected_controls": controls,
        "original_controls": {
            "probe_ok": bool(original["probe_ok"]),
            "production_train_hits": int(original["production_train_hits"]),
            "production_eval_hits": int(original["production_eval_hits"]),
            "label_shuffle_train_hits": int(original["label_shuffle"]["train_hits"]),
            "label_shuffle_eval_hits": int(original["label_shuffle"]["eval_hits"]),
            "row_shuffle_train_hits": int(original["row_shuffle"]["train_hits"]),
            "row_shuffle_eval_hits": int(original["row_shuffle"]["eval_hits"]),
            "row_permutation_fixed_points": alignment["fixed_points"],
            "row_permutation_train_same_topic": alignment["train_same_topic"],
            "row_permutation_train_rows": alignment["train_rows"],
        },
        "leakage": _leakage(
            texts,
            observation_ids,
            matrix,
            train,
            test,
        ),
    }


def _original_row_shuffle_alignment(
    matrix: np.ndarray,
    topics: Sequence[str],
    train: np.ndarray,
) -> dict:
    """The original row control's permutation, not the revised one.

    ``shuffled_matrix[i]`` is the embedding from row ``order[i]``. The label
    scored at i stays ``topics[i]``. A fixed point is a row whose embedding
    did not move. ``train_same_topic`` counts train positions where the
    embedding's source row happens to carry the same topic string.
    """
    order = np.random.default_rng(ROW_SHUFFLE_SEED).permutation(matrix.shape[0])
    fixed_points = int(np.sum(order == np.arange(matrix.shape[0])))
    same_topic = 0
    for index in train.tolist():
        if topics[int(order[int(index)])] == topics[int(index)]:
            same_topic += 1
    return {
        "fixed_points": fixed_points,
        "train_same_topic": same_topic,
        "train_rows": int(train.size),
    }


def _corrected_controls(
    matrix: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    train_topics: Sequence[str],
    eval_topics: Sequence[str],
    production: Mapping,
) -> dict:
    """Eval label permutation and eval embedding permutation, train map fixed.

    Centroids stay the production centroids fit on the real train rows.
    The label control permutes eval topics only. The embedding control
    permutes eval rows of the embedding and leaves eval topics in place.
    """
    mapping = train_only_majority_map(production["train_labels"], train_topics)
    real = score_against_map(production["eval_labels"], eval_topics, mapping)
    shuffled_topics = permute_sequence(eval_topics, REVISED_LABEL_SHUFFLE_SEED)
    label_score = score_against_map(production["eval_labels"], shuffled_topics, mapping)
    _train_labels, centroids, _indices = _fit_algorithm(matrix[train], _production_spec())
    shuffled_eval = permute_rows(matrix[test], REVISED_EMBEDDING_SHUFFLE_SEED)
    embed_labels = _assign(shuffled_eval, centroids, "euclidean")
    embed_score = score_against_map(embed_labels, eval_topics, mapping)
    predictions = [mapping[int(code)] for code in np.asarray(production["eval_labels"]).tolist()]
    label_below = int(label_score["hits"]) < int(real["hits"])
    embed_below = int(embed_score["hits"]) < int(real["hits"])
    return {
        "real_hits": int(real["hits"]),
        "real_rows": int(real["rows"]),
        "real_size_weighted": float(real["size_weighted"]),
        "label_shuffle_hits": int(label_score["hits"]),
        "label_shuffle_size_weighted": float(label_score["size_weighted"]),
        "label_shuffle_below": bool(label_below),
        "embedding_shuffle_hits": int(embed_score["hits"]),
        "embedding_shuffle_size_weighted": float(embed_score["size_weighted"]),
        "embedding_shuffle_below": bool(embed_below),
        "both_below": bool(label_below and embed_below),
        "label_shuffle_expected_hits": expected_label_permutation_hits(predictions, eval_topics),
    }


def _attach_cluster_times(train_matrix: np.ndarray, runs: List[dict], time_runs: int) -> None:
    for run in runs:
        spec = {
            "id": run["id"],
            "distance": run["distance"],
            "init": run["init"],
            "seed": run["seed"],
        }

        def _cluster(spec: Mapping = spec) -> None:
            _fit_algorithm(train_matrix, spec)

        median_ms, _samples = _per_call_median_ms(_cluster, time_runs, CLUSTER_TIME_REPEATS)
        run["cluster_median_ms"] = median_ms


def _leakage(
    texts: Sequence[str],
    observation_ids: Sequence[str],
    full_matrix: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> dict:
    """Duplicate rows, and whether the full-matrix SVD moves k-means."""
    n_rows = len(texts)
    train_ids = {observation_ids[int(index)] for index in train}
    eval_ids = {observation_ids[int(index)] for index in test}
    train_texts = {texts[int(index)] for index in train}
    eval_texts = {texts[int(index)] for index in test}
    train_tokens = set()
    for index in train:
        train_tokens.update(content_tokens(texts[int(index)]))
    eval_token_occurrences = 0
    eval_only_occurrences = 0
    eval_only_types = set()
    for index in test:
        for token in content_tokens(texts[int(index)]):
            eval_token_occurrences += 1
            if token not in train_tokens:
                eval_only_occurrences += 1
                eval_only_types.add(token)
    train_only_matrix, eval_only_projected = _train_only_embedding(texts, train, test)
    spec = _production_spec()
    full_train, full_centroids, _indices = _fit_algorithm(full_matrix[train], spec)
    full_eval = _assign(full_matrix[test], full_centroids, "euclidean")
    clean_train, clean_centroids, _indices = _fit_algorithm(train_only_matrix, spec)
    clean_eval = _assign(eval_only_projected, clean_centroids, "euclidean")
    train_ari = float(adjusted_rand_index(full_train, clean_train))
    eval_ari = float(adjusted_rand_index(full_eval, clean_eval))
    distance_correlation = _pairwise_distance_correlation(full_matrix[train], train_only_matrix)
    return {
        "duplicate_observation_ids_in_file": int(n_rows - len(set(observation_ids))),
        "duplicate_texts_in_file": int(n_rows - len(set(texts))),
        "shared_observation_ids": int(len(train_ids & eval_ids)),
        "shared_texts": int(len(train_texts & eval_texts)),
        "eval_token_occurrences": int(eval_token_occurrences),
        "eval_only_token_occurrences": int(eval_only_occurrences),
        "eval_only_token_types": int(len(eval_only_types)),
        "train_ari_full_vs_train_only": train_ari,
        "eval_ari_full_vs_train_only": eval_ari,
        "train_pairwise_distance_correlation": distance_correlation,
        "material": bool(train_ari < SVD_MATERIAL_TRAIN_ARI),
    }


def _train_only_embedding(
    texts: Sequence[str],
    train: np.ndarray,
    test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """TF-IDF and SVD fit on train texts. Eval rows are projected onto that basis.

    Tokens that never occur in train are dropped. Document frequency and the
    singular vectors do not see eval rows. The sign of each singular vector
    follows the train left vectors, the same pivot rule as ``word_sentence_svd``.
    """
    train_texts = [texts[int(index)] for index in train]
    eval_texts = [texts[int(index)] for index in test]
    rows = [content_tokens(text) for text in train_texts]
    document_frequency: Dict[str, int] = {}
    for tokens in rows:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    vocabulary = sorted(document_frequency)
    index = {token: position for position, token in enumerate(vocabulary)}
    n_train = len(train_texts)

    def _transform(text_list: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(text_list), len(vocabulary)), dtype=np.float64)
        for row_index, text in enumerate(text_list):
            counts: Dict[str, int] = {}
            for token in content_tokens(text):
                if token not in index:
                    continue
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                # np.log follows NumPy's CPU dispatch. word_sentence_tfidf uses
                # math.log, and allclose below accepts one ulp. The overlapping
                # train-ARI and eval-ARI locks are this call's AVX512 result
                # on the AVX512 hex, and the recorded libm images when
                # np.log(39/5) matches math.log. See
                # docs/PR1_EVAL_CONTROL_INTEGRITY.md.
                idf = float(np.log(n_train / document_frequency[token]) + 1.0)
                matrix[row_index, index[token]] = float(count) * idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)

    train_tfidf = _transform(train_texts)
    eval_tfidf = _transform(eval_texts)
    # word_sentence_tfidf is the production transformer. The train block of a
    # train-only call must match it, or this comparison is not the encoder.
    reference = word_sentence_tfidf(train_texts)
    if train_tfidf.shape != reference.shape or not np.allclose(train_tfidf, reference):
        raise RuntimeError("train-only TF-IDF diverged from word_sentence_tfidf")
    width = min(RANK, train_tfidf.shape[0], train_tfidf.shape[1])
    pin_blas_threads(1)
    left, singular, right = np.linalg.svd(train_tfidf, full_matrices=False)
    left = np.array(left[:, :width], dtype=np.float64, copy=True)
    singular = np.asarray(singular[:width], dtype=np.float64)
    right = np.array(right[:width, :], dtype=np.float64, copy=True)
    for column in range(width):
        pivot = int(np.argmax(np.abs(left[:, column])))
        if left[pivot, column] < 0.0:
            left[:, column] *= -1.0
            right[column, :] *= -1.0
    train_embedded = left * singular
    eval_embedded = (eval_tfidf @ right.T) * singular
    if width < RANK:
        pad = RANK - width
        train_embedded = np.pad(train_embedded, ((0, 0), (0, pad)))
        eval_embedded = np.pad(eval_embedded, ((0, 0), (0, pad)))
    return _renorm_rows(train_embedded), _renorm_rows(eval_embedded)


def _renorm_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)


def _pairwise_distance_correlation(left: np.ndarray, right: np.ndarray) -> float:
    def _upper(matrix: np.ndarray) -> np.ndarray:
        distances = ((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2)
        index = np.triu_indices(matrix.shape[0], 1)
        return distances[index]

    return float(np.corrcoef(_upper(left), _upper(right))[0, 1])

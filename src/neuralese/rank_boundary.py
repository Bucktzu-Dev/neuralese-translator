"""Rank-boundary probe for the frozen word-sentence SVD.

Labels are a probe. They are not written into the embedding, the codebook,
or ``certify``. The product rank stays ``n_symbols``. Higher-rank k-means
below is a fixture on truncations of the same encoder, not a new default.

Distances are Euclidean in the rank-8 embedding ``learn_pack`` stores.
"""
from __future__ import annotations

import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.clustering import kmeans
from neuralese.contracts import Observation
from neuralese.encode import (
    ENCODER_WORD_SENTENCE_SVD,
    word_sentence_svd,
    word_sentence_tfidf,
)
from neuralese.energy import cosine_similarity
from neuralese.factorization import pin_blas_threads

# Product settings this probe reads. It does not change them.
PRODUCT_RANK = 8
PRODUCT_K = 8
PRODUCT_SEED = 0
PROBE_TOPIC = "weather"
# Same encoder truncations, same k-means. Not a new default rank.
CONFIRMING_RANKS = (9, 12, 16)
# Singular values kept so the cut between component 8 and 9 is visible.
SPECTRUM_START = 5
SPECTRUM_STOP = 11

# Between-centroid distance below this multiple of the within-weather mean
# pairwise distance is small relative to that spread.
OVERLAP_RATIO = 1.0
# Between-centroid distance at or above this multiple is clearly larger
# than the within-weather spread.
SEPARATED_RATIO = 2.0
# A weather row counts as nearer another topic when that distance is strictly
# smaller. A fraction above this line is a majority.
CLOSER_MAJORITY = 0.5
# Same cutoff ``cluster_majority_diagnostic`` uses. Not a certify gate.
MIXED_PURITY = 0.5

FAILURE_MODES = ("RANK_LOSS", "CLUSTERING_LOSS", "UNRESOLVED")


def singular_spectrum(matrix: np.ndarray) -> np.ndarray:
    """Full singular values of a TF-IDF matrix, one BLAS thread.

    ``compute_uv`` stays on so the values match the factorization inside
    ``word_sentence_svd``.
    """
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.zeros((0,), dtype=np.float64)
    pin_blas_threads(1)
    _left, singular, _right = np.linalg.svd(matrix, full_matrices=False)
    return np.asarray(singular, dtype=np.float64)


def mean_pairwise_euclidean(points: np.ndarray) -> float:
    """Mean Euclidean distance over unordered pairs."""
    count = int(points.shape[0])
    if count < 2:
        raise ValueError("mean pairwise distance needs at least two rows")
    total = 0.0
    pairs = 0
    for index in range(count - 1):
        distances = np.linalg.norm(points[index + 1 :] - points[index], axis=1)
        total += float(distances.sum())
        pairs += int(distances.size)
    return total / pairs


def classify_rank_boundary(
    *,
    between_centroid: float,
    within_weather: float,
    closer_fraction: float,
    cluster_mixed: bool,
) -> Dict[str, object]:
    """Apply the rank-loss / clustering-loss rule.

    Overlap (``RANK_LOSS``) when the between-group centroid distance is
    smaller than the within-weather mean pairwise distance, or when a
    majority of weather rows are closer to another topic centroid.

    Separated (``CLUSTERING_LOSS``) when that between-group distance is at
    least twice the within-weather spread and the learned cluster is still
    mixed.

    ``UNRESOLVED`` when those readings conflict, or when the ratio sits
    between the two thresholds. Conflict strings name the numbers. The two
    readings are not averaged.
    """
    if within_weather <= 0.0:
        return {
            "failure_mode": "UNRESOLVED",
            "between_over_within_weather": None,
            "conflicts": ["within-weather mean pairwise distance is zero"],
        }
    ratio = float(between_centroid) / float(within_weather)
    overlap_by_spread = ratio < OVERLAP_RATIO
    overlap_by_assignment = float(closer_fraction) > CLOSER_MAJORITY
    separated = ratio >= SEPARATED_RATIO
    conflicts: List[str] = []
    if separated and overlap_by_assignment:
        conflicts.append(
            "between/within-weather "
            f"{ratio:.6f} >= {SEPARATED_RATIO} (separated) but closer-to-other "
            f"fraction {closer_fraction:.6f} > {CLOSER_MAJORITY} (overlap)"
        )
        mode = "UNRESOLVED"
    elif overlap_by_spread or overlap_by_assignment:
        mode = "RANK_LOSS"
    elif separated and cluster_mixed:
        mode = "CLUSTERING_LOSS"
    else:
        mode = "UNRESOLVED"
        if separated and not cluster_mixed:
            conflicts.append(
                "between/within-weather "
                f"{ratio:.6f} >= {SEPARATED_RATIO}, and the weather-majority "
                "cluster is not mixed"
            )
        elif (not overlap_by_spread) and (not overlap_by_assignment) and not separated:
            conflicts.append(
                "between/within-weather "
                f"{ratio:.6f} is at least {OVERLAP_RATIO} and below "
                f"{SEPARATED_RATIO}, and closer-to-other fraction "
                f"{closer_fraction:.6f} is not above {CLOSER_MAJORITY}"
            )
        else:
            conflicts.append(
                "rank-8 distance readings did not match RANK_LOSS or CLUSTERING_LOSS"
            )
    return {
        "failure_mode": mode,
        "between_over_within_weather": ratio,
        "conflicts": conflicts,
    }


def diagnose_rank_boundary(
    texts: Sequence[str],
    topics: Sequence[str],
    *,
    time_runs: int = 5,
) -> Dict[str, object]:
    """Probe ``texts`` with the product word-sentence SVD and k-means.

    ``topics`` is aligned with ``texts`` and is used only after the embedding
    and the cluster assignment exist. Observation ids passed to ``learn_pack``
    are row indices. Topic strings are not stored on those rows.
    """
    if len(texts) != len(topics):
        raise ValueError(f"texts ({len(texts)}) and topics ({len(topics)}) differ in length")
    if time_runs < 0:
        raise ValueError(f"time_runs must be >= 0, got {time_runs}")
    labels = [str(topic) for topic in topics]
    matrix = word_sentence_tfidf(texts)
    spectrum = singular_spectrum(matrix)
    frozen = np.asarray(word_sentence_svd(texts, rank=PRODUCT_RANK), dtype=np.float64)
    observations = [
        Observation(observation_id=f"row-{index:04d}", text=text or "", metadata={})
        for index, text in enumerate(texts)
    ]
    pack = learn_pack(
        observations,
        config=LearnConfig(
            n_symbols=PRODUCT_K,
            seed=PRODUCT_SEED,
            encoder=ENCODER_WORD_SENTENCE_SVD,
        ),
    )
    stored = np.asarray(
        [pack.observations[obs.observation_id].embedding for obs in observations],
        dtype=np.float64,
    )
    if not np.array_equal(frozen, stored):
        raise RuntimeError("rank-8 embedding diverged from learn_pack")
    for obs in observations:
        if obs.metadata:
            raise RuntimeError("probe labels were written onto learn observations")

    clusters = _live_clusters(pack, labels)
    weather = _select_topic_cluster(clusters, PROBE_TOPIC)
    if weather is None:
        geometry = None
        verdict = {
            "failure_mode": "UNRESOLVED",
            "between_over_within_weather": None,
            "conflicts": [f"no live cluster has majority topic {PROBE_TOPIC!r}"],
        }
    else:
        geometry = _weather_geometry(frozen, labels, weather["indices"])
        verdict = classify_rank_boundary(
            between_centroid=geometry["between_centroid"],
            within_weather=geometry["within_weather"],
            closer_fraction=geometry["closer_fraction"],
            cluster_mixed=bool(weather["mixed"]),
        )

    higher = [
        _kmeans_topic_purity(texts, labels, rank=rank)
        for rank in (PRODUCT_RANK, *CONFIRMING_RANKS)
    ]
    median_ms, samples = _time_learn(observations, time_runs)
    window = []
    for index in range(SPECTRUM_START, min(SPECTRUM_STOP, int(spectrum.shape[0]))):
        window.append(
            {
                "index": index,
                "one_based": index + 1,
                "value": float(spectrum[index]),
            }
        )
    s7 = float(spectrum[7])
    s8 = float(spectrum[8])
    return {
        "encoder": ENCODER_WORD_SENTENCE_SVD,
        "rank": PRODUCT_RANK,
        "k": PRODUCT_K,
        "seed": PRODUCT_SEED,
        "distance": "euclidean",
        "singular_window": window,
        "s7": s7,
        "s8": s8,
        "gap_s7_minus_s8": s7 - s8,
        "ratio_s8_over_s7": s8 / s7,
        "weather_cluster": None if weather is None else _cluster_public(weather, geometry),
        "failure_mode": verdict["failure_mode"],
        "between_over_within_weather": verdict["between_over_within_weather"],
        "conflicts": list(verdict["conflicts"]),
        "higher_rank": higher,
        "cluster_indices": [
            {"code": int(symbol.code), "indices": _member_indices(symbol)}
            for symbol in pack.symbols
        ],
        "learn_ms_median": median_ms,
        "learn_ms_samples": samples,
        "thresholds": {
            "overlap_ratio": OVERLAP_RATIO,
            "separated_ratio": SEPARATED_RATIO,
            "closer_majority": CLOSER_MAJORITY,
            "mixed_purity": MIXED_PURITY,
        },
    }


def _time_learn(observations: Sequence[Observation], runs: int) -> Tuple[Optional[float], List[float]]:
    if runs == 0:
        return None, []
    config = LearnConfig(
        n_symbols=PRODUCT_K,
        seed=PRODUCT_SEED,
        encoder=ENCODER_WORD_SENTENCE_SVD,
    )
    samples: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        learn_pack(observations, config=config)
        samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    return float(median), samples


def _member_indices(symbol) -> List[int]:
    return [int(obs_id.split("-", 1)[1]) for obs_id in symbol.observation_ids]


def _live_clusters(pack, labels: Sequence[str]) -> List[dict]:
    clusters = []
    for symbol in pack.symbols:
        if symbol.quarantined or not symbol.observation_ids:
            continue
        indices = _member_indices(symbol)
        topics = [labels[index] for index in indices]
        majority, count = _majority(topics)
        rows = len(indices)
        purity = float(count / rows) if rows else 0.0
        clusters.append(
            {
                "code": int(symbol.code),
                "indices": indices,
                "rows": rows,
                "majority_topic": majority,
                "majority_count": int(count),
                "purity": purity,
                "mixed": purity < MIXED_PURITY,
            }
        )
    return clusters


def _select_topic_cluster(clusters: Sequence[Mapping], topic: str) -> Optional[dict]:
    chosen = [cluster for cluster in clusters if cluster["majority_topic"] == topic]
    if not chosen:
        return None
    return max(
        chosen,
        key=lambda cluster: (
            cluster["majority_count"],
            cluster["purity"],
            -int(cluster["code"]),
        ),
    )


def _weather_geometry(
    embedding: np.ndarray,
    labels: Sequence[str],
    indices: Sequence[int],
) -> dict:
    weather_idx = [index for index in indices if labels[index] == PROBE_TOPIC]
    other_idx = [index for index in indices if labels[index] != PROBE_TOPIC]
    if len(weather_idx) < 2 or len(other_idx) < 2:
        raise ValueError("weather-majority cluster does not have two groups to compare")
    weather_points = embedding[weather_idx]
    other_points = embedding[other_idx]
    weather_centroid = weather_points.mean(axis=0)
    other_centroid = other_points.mean(axis=0)
    by_topic: Dict[str, List[int]] = {}
    for index in indices:
        by_topic.setdefault(labels[index], []).append(index)
    topic_centroids = {
        topic: embedding[members].mean(axis=0) for topic, members in by_topic.items()
    }
    closer = 0
    for index in weather_idx:
        row = embedding[index]
        own = float(np.linalg.norm(row - weather_centroid))
        other = min(
            float(np.linalg.norm(row - centroid))
            for topic, centroid in topic_centroids.items()
            if topic != PROBE_TOPIC
        )
        if other < own:
            closer += 1
    counts = {topic: len(members) for topic, members in by_topic.items() if topic != PROBE_TOPIC}
    return {
        "n_weather": len(weather_idx),
        "n_other": len(other_idx),
        "within_weather": mean_pairwise_euclidean(weather_points),
        "within_other": mean_pairwise_euclidean(other_points),
        "between_centroid": float(np.linalg.norm(weather_centroid - other_centroid)),
        "closer_to_other_topic": closer,
        "closer_fraction": closer / len(weather_idx),
        "contaminant_counts": dict(sorted(counts.items())),
    }


def _cluster_public(weather: Mapping, geometry: Mapping) -> dict:
    return {
        "code": weather["code"],
        "rows": weather["rows"],
        "majority_topic": weather["majority_topic"],
        "majority_count": weather["majority_count"],
        "purity": weather["purity"],
        "mixed": weather["mixed"],
        "indices": list(weather["indices"]),
        **geometry,
    }


def _kmeans_topic_purity(texts: Sequence[str], labels: Sequence[str], *, rank: int) -> dict:
    """Same k-means as learn (seed, k, quarantine). Width is the SVD truncation."""
    embedding = np.asarray(word_sentence_svd(texts, rank=rank), dtype=np.float64)
    config = LearnConfig()
    rng = np.random.default_rng(PRODUCT_SEED)
    assigned, centroids = kmeans(embedding, PRODUCT_K, rng=rng)
    live = []
    for class_id in range(int(centroids.shape[0])):
        members = np.where(assigned == class_id)[0]
        if members.size:
            proto = centroids[class_id]
            kappa = float(
                sum(cosine_similarity(embedding[int(index)], proto) for index in members)
                / members.size
            )
        else:
            kappa = 0.0
        quarantined = members.size < config.min_cluster_size or kappa < config.tau_kappa
        topics = [labels[int(index)] for index in members]
        majority, count = _majority(topics)
        weather_count = sum(1 for topic in topics if topic == PROBE_TOPIC)
        rows = int(members.size)
        purity = float(count / rows) if rows else 0.0
        record = {
            "code": int(class_id),
            "rows": rows,
            "majority_topic": majority,
            "majority_count": int(count),
            "purity": purity,
            "weather_count": int(weather_count),
            "quarantined": bool(quarantined),
            "mixed": purity < MIXED_PURITY,
        }
        if not quarantined:
            live.append(record)
    weather_live = [record for record in live if record["majority_topic"] == PROBE_TOPIC]
    if weather_live:
        chosen = max(
            weather_live,
            key=lambda record: (record["weather_count"], record["purity"], -record["code"]),
        )
    else:
        chosen = None
    return {
        "rank": rank,
        "n_live": len(live),
        "weather_code": None if chosen is None else chosen["code"],
        "weather_rows": None if chosen is None else chosen["rows"],
        "weather_count": None if chosen is None else chosen["weather_count"],
        "weather_purity": None if chosen is None else chosen["purity"],
        "weather_mixed": None if chosen is None else chosen["mixed"],
        "product_rank": rank == PRODUCT_RANK,
    }


def _majority(topics: Sequence[str]) -> Tuple[Optional[str], int]:
    counts: Dict[str, int] = {}
    for topic in topics:
        counts[topic] = counts.get(topic, 0) + 1
    if not counts:
        return None, 0
    topic, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return topic, int(count)

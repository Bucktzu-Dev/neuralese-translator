"""Clustering isolation on the frozen rank-8 varied-prose matrix.

Labels are a probe. Production k-means stays farthest-first Euclidean.
The admission bar is the constant block in ``neuralese.clustering_isolation``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neuralese import __version__
from neuralese.clustering import kmeans
from neuralese.clustering_isolation import (
    CLUSTER_RUNTIME_FACTOR,
    HOLDOUT_MODULUS,
    HOLDOUT_REMAINDER,
    LEARN_EQUIVALENT_FACTOR,
    LEARN_PACK_BASELINE_MS,
    PRODUCTION_FAMILY,
    PURITY_MOVE,
    RESIDUAL_WORSE_FACTOR,
    SEEDS,
    SIZE_RATIO_IMBALANCE,
    adjusted_rand_index,
    classify_clustering_isolation,
    diagnose_clustering_isolation,
    euclidean_size_contrast,
    holdout_mask,
)
from neuralese.rank_boundary import MIXED_PURITY

VARIED = Path(__file__).resolve().parents[1] / "examples" / "varied_prose" / "observations.jsonl"
ROOT = Path(__file__).resolve().parents[1]

ABS = 1e-9
GAP = 0.021759750122584798
SINGULAR_RATIO = 0.9862086498001754
S7 = 1.577782436621879
S8 = 1.556022686499294
MATRIX_SHA = "7a218f13197a4c9335ee65e4597ad1e3c061f1911843a837290820525ab927e1"


def _load():
    texts = []
    topics = []
    for line in VARIED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts.append(row["text"])
        topics.append(row["metadata"]["topic"])
    return texts, topics


@pytest.fixture(scope="module")
def report():
    texts, topics = _load()
    return diagnose_clustering_isolation(texts, topics)


def test_bar_constants_are_the_predeclared_line():
    assert RESIDUAL_WORSE_FACTOR == 1.10
    assert CLUSTER_RUNTIME_FACTOR == 2.0
    assert LEARN_PACK_BASELINE_MS == 222.369
    assert LEARN_EQUIVALENT_FACTOR == 3.0
    assert PURITY_MOVE == 0.05
    assert SIZE_RATIO_IMBALANCE == 2.0
    assert MIXED_PURITY == 0.5
    assert SEEDS == (0, 1, 2, 3, 4)
    assert len(SEEDS) >= 5
    assert __version__ == "0.7.0"


def test_production_kmeans_source_is_still_farthest_first():
    clustering = (ROOT / "src" / "neuralese" / "clustering.py").read_text(encoding="utf-8")
    alphabet = (ROOT / "src" / "neuralese" / "alphabet.py").read_text(encoding="utf-8")
    assert "def _farthest_first" in clustering
    assert "seeds = _farthest_first(X, k)" in clustering
    assert "labels, centroids = kmeans(X, k, rng=rng)" in alphabet
    assert "0.7.0" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_adjusted_rand_index_matches_the_pair_count_formula():
    same = np.array([0, 0, 1, 1])
    flipped = np.array([1, 1, 0, 0])
    assert adjusted_rand_index(same, same) == pytest.approx(1.0, abs=ABS)
    assert adjusted_rand_index(same, flipped) == pytest.approx(1.0, abs=ABS)
    # 4/7 from the Hubert-Arabie counts on this 4-point pair.
    partial = adjusted_rand_index(np.array([0, 0, 1, 2]), np.array([0, 0, 1, 1]))
    assert partial == pytest.approx(4.0 / 7.0, abs=ABS)


def test_holdout_is_every_fifth_row_before_any_label_is_required():
    mask = holdout_mask(320)
    assert int(mask.sum()) == 64
    assert int((~mask).sum()) == 256
    assert list(np.flatnonzero(mask)[:3]) == [4, 9, 14]
    assert all(int(index) % HOLDOUT_MODULUS == HOLDOUT_REMAINDER for index in np.flatnonzero(mask))


def test_isolation_rule_names_one_cause_or_unresolved():
    quiet = dict(
        probe_ok=True,
        init_purity_span=0.0,
        angular_purity_span=0.01,
        size_contrast=False,
        baseline_size_ratio=1.2,
        best_size_ratio=1.3,
        baseline_weather_purity=0.45,
        best_weather_purity=0.46,
        best_fails_held_out_bar=True,
    )
    assert classify_clustering_isolation(**quiet) == "OBJECTIVE"
    assert (
        classify_clustering_isolation(
            **{**quiet, "init_purity_span": 0.2, "best_fails_held_out_bar": False}
        )
        == "INIT_SENSITIVITY"
    )
    assert (
        classify_clustering_isolation(**{**quiet, "angular_purity_span": 0.2})
        == "ANGULAR_DISTANCE"
    )
    assert classify_clustering_isolation(**{**quiet, "size_contrast": True}) == "SIZE_IMBALANCE"
    assert (
        classify_clustering_isolation(
            **{**quiet, "init_purity_span": 0.2, "angular_purity_span": 0.2}
        )
        == "UNRESOLVED"
    )
    assert classify_clustering_isolation(**{**quiet, "probe_ok": False}) == "UNRESOLVED"
    # A production-sized cluster blocks OBJECTIVE when nothing else explains it.
    assert (
        classify_clustering_isolation(**{**quiet, "baseline_size_ratio": 64.0})
        == "UNRESOLVED"
    )
    assert euclidean_size_contrast(
        (
            {"size_ratio": 4.0, "weather_purity": 0.40},
            {"size_ratio": 1.2, "weather_purity": 0.70},
        )
    )
    assert not euclidean_size_contrast(
        (
            {"size_ratio": 4.0, "weather_purity": 0.70},
            {"size_ratio": 1.2, "weather_purity": 0.72},
        )
    )


def test_clustering_isolation_locks_verdict_purity_and_controls(report):
    assert report["verdict"] == "RESEARCH_FAIL"
    assert report["isolated_cause"] == "UNRESOLVED"
    assert report["admission"] == "ADMISSIBLE"
    assert report["any_admissible"] is True
    assert report["runtime_authority"] == "RUNTIME_NOT_AUTHORIZED"
    assert report["certify_scope"] == "structural_only"

    receipts = report["receipts"]
    assert receipts["gap_s7_minus_s8"] == pytest.approx(GAP, abs=ABS)
    assert receipts["ratio_s8_over_s7"] == pytest.approx(SINGULAR_RATIO, abs=ABS)
    assert receipts["s7"] == pytest.approx(S7, abs=ABS)
    assert receipts["s8"] == pytest.approx(S8, abs=ABS)
    assert receipts["gap_s7_minus_s8"] == pytest.approx(receipts["s7"] - receipts["s8"], abs=ABS)
    assert receipts["embedding_sha256"] == MATRIX_SHA
    assert receipts["n_rows"] == 320
    assert receipts["n_train"] == 256
    assert receipts["n_holdout"] == 64
    assert receipts["holdout_indices_head"] == [4, 9, 14]
    assert receipts["production_ignores_rng"] is True
    assert receipts["production_lloyd_matches"] is True
    assert receipts["production_init"] == "farthest_first"
    assert receipts["production_distance"] == "euclidean"
    assert receipts["rank"] == 8
    assert receipts["k"] == 8

    baseline = report["baseline"]
    assert baseline["weather_count"] == 29
    assert baseline["weather_rows"] == 64
    assert baseline["non_weather"] == 35
    assert baseline["weather_purity"] == 29 / 64
    assert baseline["weather_purity"] == 0.453125
    assert baseline["contamination"] == pytest.approx(1.0 - (29 / 64), abs=ABS)
    assert baseline["weather_contaminants"] == {
        "bakery": 2,
        "harbor": 1,
        "joinery": 1,
        "ledger": 26,
        "orchard": 3,
        "pottery": 2,
    }
    assert baseline["smallest_rows"] == 1
    assert baseline["weather_rows"] == 64
    assert baseline["size_weighted_hits"] == 205
    assert baseline["size_weighted_rows"] == 320
    assert baseline["residual"] == pytest.approx(0.27451605149978975, abs=ABS)
    assert baseline["seed_invariant"] is True
    assert baseline["lloyd_matches"] is True

    identity = report["identity"]
    assert identity["learn_pack_matches"] is True
    assert identity["certify_passed"] is True
    assert identity["certify_tau_residual"] == 0.55
    assert identity["certify_has_cluster_purity"] is False
    assert identity["certify_has_failure_mode"] is False
    assert identity["live_size_weighted_hits"] == 204
    assert identity["live_size_weighted_rows"] == 319
    assert identity["metadata_empty"] is True

    assert report["best_unsupervised"]["family"] == "euclidean_random_subset"
    assert report["best_unsupervised"]["seed"] == 2
    assert report["best_unsupervised"]["residual"] == pytest.approx(0.1960081180406189, abs=ABS)

    contrasts = report["contrasts"]
    assert contrasts["init_explains"] is True
    assert contrasts["angular_explains"] is True
    assert contrasts["size_explains"] is False
    assert contrasts["init_purity_span"] >= PURITY_MOVE
    assert contrasts["angular_purity_span"] >= PURITY_MOVE
    assert contrasts["euclidean_size_contrast"] is False
    assert contrasts["best_fails_held_out_bar"] is False
    assert contrasts["baseline_size_ratio"] == 64.0
    assert contrasts["best_weather_rows"] == 41
    assert contrasts["best_smallest_rows"] == 26

    held = report["held_out"]["production"]
    assert held["weather_count"] == 5
    assert held["weather_rows"] == 14
    assert held["non_weather"] == 9
    assert held["weather_purity"] == pytest.approx(5 / 14, abs=ABS)
    assert held["size_weighted_hits"] == 37
    assert held["size_weighted_rows"] == 64
    assert held["weather_contaminants"] == {
        "harbor": 1,
        "joinery": 1,
        "ledger": 5,
        "orchard": 1,
        "pottery": 1,
    }

    by_family = {row["family"]: row for row in report["candidates"]}
    winner = by_family["euclidean_random_subset"]
    assert winner["seed"] == 2
    assert winner["held_out_weather_count"] == 5
    assert winner["held_out_weather_rows"] == 9
    assert winner["held_out_non_weather"] == 4
    assert winner["held_out_weather_purity"] == pytest.approx(5 / 9, abs=ABS)
    assert winner["held_out_size_weighted_hits"] == 44
    assert winner["held_out_size_weighted_rows"] == 64
    assert winner["held_out_weather_purity"] > held["weather_purity"]
    assert winner["held_out_size_weighted"] > held["size_weighted"]
    assert winner["residual"] <= RESIDUAL_WORSE_FACTOR * baseline["residual"]
    assert winner["admissible"] is True

    for family in (
        "euclidean_kmeanspp",
        "spherical_random_subset",
        "spherical_kmeanspp",
    ):
        row = by_family[family]
        assert row["residual_ok"] is True
        assert row["held_out_weather_ok"] is True
        assert row["held_out_weighted_ok"] is True
        assert row["held_out_weather_count"] == 5
        assert row["held_out_weather_rows"] == 9
        assert row["held_out_size_weighted_hits"] == 44
        assert row["runtime_ok"] is True
        assert row["learn_equivalent_ok"] is True
        assert row["admissible"] is True
        assert row["runtime_ratio"] < CLUSTER_RUNTIME_FACTOR
        assert row["learn_equivalent_ms"] < LEARN_EQUIVALENT_FACTOR * LEARN_PACK_BASELINE_MS
        assert row["peak_bytes"] > 0

    spherical_ff = by_family["spherical_farthest_first"]
    assert spherical_ff["residual_ok"] is False
    assert spherical_ff["held_out_weighted_ok"] is False
    assert spherical_ff["held_out_size_weighted_hits"] == 37
    assert spherical_ff["admissible"] is False
    assert report["failures"] == [
        {
            "family": "spherical_farthest_first",
            "seed": None,
            "failed": ["residual", "held_out_weighted"],
        }
    ]

    controls = report["negative_controls"]
    assert controls["probe_ok"] is True
    label = controls["label_shuffle"]
    assert label["size_weighted_hits"] == 63
    assert label["size_weighted_rows"] == 320
    assert label["size_weighted"] < baseline["size_weighted"]
    assert label["size_weighted"] < baseline["weather_purity"]
    assert label["weather_cluster_majority_count"] == 12
    assert label["weather_cluster_rows"] == 64
    assert label["weather_cluster_purity"] < baseline["weather_purity"]
    row_shuffle = controls["row_shuffle"]
    assert row_shuffle["held_out_size_weighted_hits"] == 19
    assert row_shuffle["held_out_size_weighted_rows"] == 64
    assert row_shuffle["held_out_size_weighted"] < held["size_weighted"]
    assert row_shuffle["below_production_held_out"] is True

    for pair in report["matched_pairs"]:
        assert pair["shared_init"] is True

    # k-means++ seeds 3 and 4 are purer in sample and lose on residual.
    kpp = [run for run in report["runs"] if run["family"] == "euclidean_kmeanspp"]
    chosen = by_family["euclidean_kmeanspp"]
    purer = [
        run
        for run in kpp
        if run["weather_purity"] is not None and run["weather_purity"] > chosen["weather_purity"]
    ]
    # Seeds 0, 3, and 4 are purer in sample. Seed 2 still wins on residual.
    assert [run["seed"] for run in purer] == [0, 3, 4]
    assert all(run["residual"] > chosen["residual"] for run in purer)
    assert chosen["seed"] == min(
        kpp,
        key=lambda run: (run["residual"], run["seed"]),
    )["seed"]


def test_topic_permutation_does_not_move_residuals_or_the_chosen_seed(report):
    texts, topics = _load()
    flipped = diagnose_clustering_isolation(texts, list(reversed(topics)), time_runs=0)
    assert [run["residual"] for run in flipped["runs"]] == [
        run["residual"] for run in report["runs"]
    ]
    assert flipped["best_unsupervised"] == report["best_unsupervised"]
    assert flipped["baseline"]["family"] == PRODUCTION_FAMILY


def test_two_rng_seeds_do_not_move_production_labels():
    texts, _topics = _load()
    from neuralese.encode import word_sentence_svd

    matrix = np.asarray(word_sentence_svd(texts, rank=8), dtype=np.float64)
    first, _ = kmeans(matrix, 8, rng=np.random.default_rng(0))
    second, _ = kmeans(matrix, 8, rng=np.random.default_rng(4))
    assert np.array_equal(first, second)

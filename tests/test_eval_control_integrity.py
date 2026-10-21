"""Evaluation-control integrity. The revised scorer is locked here.

A label shuffle scored with a train-only map must drop. The generalization
trial's admission thresholds, candidate seeds, and original control hits
stay where that trial locked them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from neuralese import __version__
from neuralese.clustering_generalization import (
    CLUSTER_RUNTIME_FACTOR,
    FROZEN_CANDIDATE_IDS,
    LABEL_SHUFFLE_SEED,
    RESIDUAL_WORSE_FACTOR,
    ROW_SHUFFLE_SEED,
    RUNTIME_AUTHORITY,
)
from neuralese.eval_control_integrity import (
    BASELINE_COMMIT,
    BLOB_REAL_MIN_PURITY,
    EMBEDDING_CONTROL,
    GAUSSIAN_ABS_TOLERANCE,
    LABEL_CONTROL,
    MAP_SOURCE,
    REVISED_DEFINITION,
    REVISED_EMBEDDING_SHUFFLE_SEED,
    REVISED_LABEL_SHUFFLE_SEED,
    SHUFFLE_MAX_PURITY,
    SVD_MATERIAL_TRAIN_ARI,
    audit_eval_controls,
    expected_label_permutation_hits,
    integrity_verdict,
    run_analytical_controls,
    score_against_map,
)

ROOT = Path(__file__).resolve().parents[1]
ABS = 1e-9

# Production, then the four frozen candidates, under the train-only map.
CORRECTED_EVAL_HITS = {
    "uneven": (6, 5, 5, 5, 6),
    "overlapping": (4, 5, 5, 4, 5),
    "ambiguous": (7, 4, 3, 4, 3),
    "balanced": (3, 4, 4, 4, 4),
}
# The old self-majority scorer, reproduced so this audit does not rewrite it.
SELF_MAJORITY_HITS = {
    "uneven": (7, 9, 8, 8, 7),
    "overlapping": (6, 7, 7, 6, 7),
    "ambiguous": (8, 7, 6, 7, 6),
    "balanced": (9, 7, 10, 9, 9),
}
PURITY_OK = {
    "uneven": (False, False, False, False),
    "overlapping": (True, True, False, True),
    "ambiguous": (False, False, False, False),
    "balanced": (True, True, True, True),
}
RESIDUAL_OK = {
    "uneven": (True, True, False, False),
    "overlapping": (True, True, True, False),
    "ambiguous": (True, True, True, True),
    "balanced": (False, True, False, True),
}
ADVANTAGE = {
    "uneven": (False, False, False, False),
    "overlapping": (True, True, False, False),
    "ambiguous": (False, False, False, False),
    "balanced": (False, True, False, True),
}
# real, label shuffle, embedding shuffle, both strictly below.
CORRECTED_CONTROLS = {
    "uneven": (6, 5, 5, True),
    "overlapping": (4, 0, 2, True),
    "ambiguous": (7, 2, 1, True),
    "balanced": (3, 3, 5, False),
}
ORIGINAL_CONTROLS = {
    "uneven": {
        "train": (26, 21, 22),
        "eval": (7, 8, 5),
        "probe_ok": False,
        "fixed_points": 1,
        "train_same_topic": 13,
        "train_rows": 40,
        "eval_rows": 10,
    },
    "overlapping": {
        "train": (27, 19, 14),
        "eval": (6, 5, 6),
        "probe_ok": False,
        "fixed_points": 0,
        "train_same_topic": 7,
        "train_rows": 39,
        "eval_rows": 9,
    },
    "ambiguous": {
        "train": (26, 21, 21),
        "eval": (8, 8, 7),
        "probe_ok": False,
        "fixed_points": 0,
        "train_same_topic": 9,
        "train_rows": 39,
        "eval_rows": 9,
    },
    "balanced": {
        "train": (30, 20, 18),
        "eval": (9, 7, 8),
        "probe_ok": True,
        "fixed_points": 1,
        "train_same_topic": 2,
        "train_rows": 64,
        "eval_rows": 16,
    },
}
EVAL_CLASS_COUNTS = {
    "uneven": {
        "bindery": 1,
        "foundry": 1,
        "glasswork": 1,
        "semaphore": 1,
        "tannery": 1,
        "tram": 5,
    },
    "overlapping": {
        "canal": 2,
        "dyehouse": 1,
        "greenhouse": 1,
        "horology": 2,
        "lithography": 1,
        "puppetry": 2,
    },
    "ambiguous": {"archive": 2, "clinic": 2, "kitchen": 3, "radio": 2},
    "balanced": {
        "cartography": 2,
        "chess": 2,
        "luthier": 2,
        "mycology": 2,
        "neon": 2,
        "stenography": 2,
        "subway": 2,
        "volcanology": 2,
    },
}
EVAL_CLUSTER_SIZES = {
    "uneven": [1, 2, 2, 2, 3],
    "overlapping": [1, 1, 2, 2, 3],
    "ambiguous": [1, 1, 1, 1, 2, 3],
    "balanced": [1, 1, 1, 2, 2, 3, 6],
}
LEAKAGE = {
    "uneven": {
        "eval_token_occurrences": 72,
        "eval_only_token_occurrences": 56,
        "eval_only_token_types": 55,
        "train_ari": 0.2971403038427167,
        "eval_ari": 0.011976047904191614,
        "correlation": 0.4082952118911038,
    },
    "overlapping": {
        "eval_token_occurrences": 66,
        "eval_only_token_occurrences": 52,
        "eval_only_token_types": 51,
        # AVX512 np.log measurement of train_ari_full_vs_train_only.
        # The same field under libm np.log is 0.8414648893519251.
        # The locked value stays 0.7222354745271766.
        # See docs/PR1_EVAL_CONTROL_INTEGRITY.md.
        "train_ari": 0.7222354745271766,
        "eval_ari": 0.24999999999999994,
        "correlation": 0.7212291143117482,
    },
    "ambiguous": {
        "eval_token_occurrences": 58,
        "eval_only_token_occurrences": 44,
        "eval_only_token_types": 43,
        "train_ari": 0.3232177014940806,
        "eval_ari": 0.047058823529411764,
        "correlation": 0.38942027980065236,
    },
    "balanced": {
        "eval_token_occurrences": 94,
        "eval_only_token_occurrences": 74,
        "eval_only_token_types": 72,
        "train_ari": 0.30085705404325236,
        "eval_ari": 0.09999999999999999,
        "correlation": 0.41305012619992776,
    },
}


@pytest.fixture(scope="module")
def analytical():
    return run_analytical_controls()


@pytest.fixture(scope="module")
def report():
    return audit_eval_controls()


def test_revised_definition_is_frozen_before_the_candidate_score():
    assert REVISED_DEFINITION == "train_only_majority_map"
    assert MAP_SOURCE == "train_rows_only"
    assert LABEL_CONTROL == "permute_eval_labels"
    assert EMBEDDING_CONTROL == "permute_eval_embeddings_relative_to_eval_labels"
    assert REVISED_LABEL_SHUFFLE_SEED == LABEL_SHUFFLE_SEED == 0
    assert REVISED_EMBEDDING_SHUFFLE_SEED == ROW_SHUFFLE_SEED == 1
    assert BLOB_REAL_MIN_PURITY == 0.99
    assert SHUFFLE_MAX_PURITY == 0.60
    assert GAUSSIAN_ABS_TOLERANCE == 0.02
    assert SVD_MATERIAL_TRAIN_ARI == 0.90
    assert RESIDUAL_WORSE_FACTOR == 1.10
    assert CLUSTER_RUNTIME_FACTOR == 2.0
    assert BASELINE_COMMIT == "75f26cb"
    assert __version__ == "0.7.0"
    source = (ROOT / "src" / "neuralese" / "eval_control_integrity.py").read_text(encoding="utf-8")
    assert source.index("def run_analytical_controls") < source.index("def audit_eval_controls")
    assert "learn_pack(" not in source
    assert "def certify" not in source
    assert "def word_sentence_svd" not in source


def test_two_blobs_are_pure_and_a_train_only_label_shuffle_drops(analytical):
    blobs = analytical["blobs"]
    assert analytical["passed"] is True
    assert blobs["real_hits"] == blobs["n_eval"] == 200
    assert blobs["real_purity"] >= BLOB_REAL_MIN_PURITY
    assert blobs["label_shuffle_hits"] == 92
    assert blobs["label_shuffle_purity"] <= SHUFFLE_MAX_PURITY
    assert blobs["label_shuffle_purity"] < blobs["real_purity"]
    assert analytical["checks"]["blob_label_shuffle_drops"] is True


def test_embedding_permutation_on_the_blobs_drops(analytical):
    blobs = analytical["blobs"]
    assert blobs["embedding_shuffle_hits"] == 110
    assert blobs["embedding_shuffle_purity"] <= SHUFFLE_MAX_PURITY
    assert blobs["embedding_shuffle_purity"] < blobs["real_purity"]
    assert analytical["checks"]["blob_embedding_shuffle_drops"] is True


def test_single_gaussian_purity_matches_the_majority_base_rate(analytical):
    gaussian = analytical["gaussian"]
    assert gaussian["n_eval"] == 2000
    assert gaussian["majority_base_hits"] == 1476
    assert gaussian["hits"] == gaussian["majority_base_hits"]
    assert gaussian["abs_gap"] <= GAUSSIAN_ABS_TOLERANCE
    assert gaussian["map_values"] == ["maj"]
    assert analytical["checks"]["gaussian_near_base_rate"] is True


def test_singleton_shuffle_drops_only_under_the_train_only_map(analytical):
    """Self-majority on singletons stays perfect. The train-only map must fall.

    Routing the shuffle through the old scorer would keep the hit count at
    the row count, and this test would fail.
    """
    single = analytical["singleton"]
    assert single["rows"] == 8
    assert single["real_hits"] == 8
    assert single["self_majority_hits_after_shuffle"] == 8
    assert single["train_map_hits_after_shuffle"] == 2
    assert single["train_map_hits_after_shuffle"] < single["real_hits"]
    assert analytical["checks"]["singleton_train_map_drops"] is True
    assert analytical["checks"]["singleton_self_majority_stays_perfect"] is True


def test_score_against_map_does_not_refit_a_majority():
    import numpy as np

    labels = np.array([0, 0])
    topics = ["alpha", "beta"]
    # The fixed map names a topic that is not even in the pair.
    scored = score_against_map(labels, topics, {0: "gamma"})
    assert scored["hits"] == 0
    assert scored["rows"] == 2
    # A missing cluster id is a miss, not a free hit.
    missing = score_against_map(np.array([3]), ["alpha"], {0: "alpha"})
    assert missing["hits"] == 0


def test_expected_permutation_hits_are_the_class_frequencies():
    predictions = ["alpha", "alpha", "beta", "beta"]
    labels = ["alpha", "alpha", "beta", "beta"]
    assert expected_label_permutation_hits(predictions, labels) == pytest.approx(2.0)
    assert expected_label_permutation_hits([], []) == 0.0


def test_verdict_is_fail_without_the_analytical_gate_and_pass_only_when_clean():
    assert (
        integrity_verdict(
            analytical_passed=False,
            candidates_rerun=False,
            fixture_controls_collapse=False,
            svd_material=False,
        )
        == "FAIL"
    )
    assert (
        integrity_verdict(
            analytical_passed=True,
            candidates_rerun=False,
            fixture_controls_collapse=True,
            svd_material=False,
        )
        == "FAIL"
    )
    assert (
        integrity_verdict(
            analytical_passed=True,
            candidates_rerun=True,
            fixture_controls_collapse=False,
            svd_material=False,
        )
        == "UNRESOLVED"
    )
    assert (
        integrity_verdict(
            analytical_passed=True,
            candidates_rerun=True,
            fixture_controls_collapse=True,
            svd_material=True,
        )
        == "UNRESOLVED"
    )
    assert (
        integrity_verdict(
            analytical_passed=True,
            candidates_rerun=True,
            fixture_controls_collapse=True,
            svd_material=False,
        )
        == "PASS"
    )


def test_failed_analytical_controls_do_not_rerun_candidates(monkeypatch):
    calls = []

    def _fail():
        return {"passed": False, "checks": {"blob_real_near_one": False}}

    def _embed(*_args, **_kwargs):
        calls.append("embed")
        raise AssertionError("fixtures must not be embedded when analytical controls fail")

    monkeypatch.setattr("neuralese.eval_control_integrity.run_analytical_controls", _fail)
    monkeypatch.setattr("neuralese.eval_control_integrity.word_sentence_svd", _embed)
    report = audit_eval_controls(time_runs=0)
    assert report["verdict"] == "FAIL"
    assert report["candidates_rerun"] is False
    assert report["recommendation"] == "NONE"
    assert report["runtime_authority"] == RUNTIME_AUTHORITY == "RUNTIME_NOT_AUTHORIZED"
    assert report["fixtures"] == []
    assert calls == []


def test_audit_reruns_candidates_and_stays_unresolved(report):
    assert report["analytical_passed"] is True
    assert report["candidates_rerun"] is True
    assert report["verdict"] == "UNRESOLVED"
    assert report["recommendation"] == "NONE"
    assert report["runtime_authority"] == "RUNTIME_NOT_AUTHORIZED"
    assert report["fixture_controls_collapse"] is False
    assert report["svd_material"] is True
    assert report["definition"]["admission_thresholds_changed"] is False
    assert report["definition"]["residual_worse_factor"] == 1.10
    assert report["definition"]["cluster_runtime_factor"] == 2.0
    assert report["bar"]["residual_worse_factor"] == 1.10
    assert [fixture["fixture_id"] for fixture in report["fixtures"]] == [
        "uneven",
        "overlapping",
        "ambiguous",
        "balanced",
    ]


def _numeric_environment() -> str:
    """NumPy, BLAS, thread count, and the log kernel this lock reads."""
    import math
    import os

    import numpy as np

    loaded = []
    try:
        for line in open("/proc/self/maps", encoding="utf-8", errors="replace"):
            lower = line.lower()
            if any(token in lower for token in ("libblas", "liblapack", "libopenblas")):
                path = line.split()[-1]
                if path.startswith("/") and path not in loaded:
                    loaded.append(path)
    except OSError:
        loaded = []
    flags = ""
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("flags"):
                flags = line
                break
    except OSError:
        flags = ""
    ratio = 39 / 5
    return (
        f"numpy {np.__version__} file {np.__file__} libraries {loaded} "
        f"cpu_count {os.cpu_count()} "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')} "
        f"avx512f={'avx512f' in flags} "
        f"np.log(39/5)={float(np.log(ratio)).hex()} "
        f"math.log(39/5)={math.log(ratio).hex()}"
    )


def test_corrected_scores_and_original_controls(report):
    for fixture in report["fixtures"]:
        fixture_id = fixture["fixture_id"]
        blocks = [fixture["production"], *fixture["candidates"]]
        assert [block["eval_hits"] for block in blocks] == list(CORRECTED_EVAL_HITS[fixture_id])
        assert [block["self_majority_hits"] for block in blocks] == list(
            SELF_MAJORITY_HITS[fixture_id]
        )
        assert [block["id"] for block in fixture["candidates"]] == list(FROZEN_CANDIDATE_IDS)
        assert [block["seed"] for block in fixture["candidates"]] == [2, 2, 2, 3]
        for candidate, purity_ok, residual_ok, advantage in zip(
            fixture["candidates"],
            PURITY_OK[fixture_id],
            RESIDUAL_OK[fixture_id],
            ADVANTAGE[fixture_id],
        ):
            assert candidate["purity_ok"] is purity_ok
            assert candidate["residual_ok"] is residual_ok
            assert candidate["cluster_median_ms"] > 0
            assert candidate["runtime_ratio"] > 0
            assert candidate["advantage"] is advantage
            # The 2× clause is inside ``advantage``. A slow sample on a
            # candidate that already fails purity or residual does not move
            # the locked bit. A clearing candidate has to stay under 2×.
            assert candidate["advantage"] is (
                purity_ok and residual_ok and candidate["runtime_ok"]
            )
        controls = fixture["corrected_controls"]
        real, label_hits, embed_hits, both = CORRECTED_CONTROLS[fixture_id]
        assert controls["real_hits"] == real
        assert controls["label_shuffle_hits"] == label_hits
        assert controls["embedding_shuffle_hits"] == embed_hits
        assert controls["both_below"] is both
        assert controls["label_shuffle_below"] is (label_hits < real)
        assert controls["embedding_shuffle_below"] is (embed_hits < real)
        original = fixture["original_controls"]
        expected = ORIGINAL_CONTROLS[fixture_id]
        train_prod, train_label, train_row = expected["train"]
        eval_prod, eval_label, eval_row = expected["eval"]
        assert original["production_train_hits"] == train_prod
        assert original["label_shuffle_train_hits"] == train_label
        assert original["row_shuffle_train_hits"] == train_row
        assert original["production_eval_hits"] == eval_prod
        assert original["label_shuffle_eval_hits"] == eval_label
        assert original["row_shuffle_eval_hits"] == eval_row
        assert original["probe_ok"] is expected["probe_ok"]
        assert original["row_permutation_fixed_points"] == expected["fixed_points"]
        assert original["row_permutation_train_same_topic"] == expected["train_same_topic"]
        assert original["row_permutation_train_rows"] == expected["train_rows"]
        assert fixture["n_eval"] == expected["eval_rows"]
        assert fixture["eval_class_counts"] == EVAL_CLASS_COUNTS[fixture_id]
        assert fixture["eval_majority_count"] == max(EVAL_CLASS_COUNTS[fixture_id].values())
        assert fixture["eval_cluster_sizes"] == EVAL_CLUSTER_SIZES[fixture_id]
        leakage = fixture["leakage"]
        assert leakage["shared_observation_ids"] == 0
        assert leakage["shared_texts"] == 0
        assert leakage["duplicate_observation_ids_in_file"] == 0
        assert leakage["duplicate_texts_in_file"] == 0
        locked = LEAKAGE[fixture_id]
        assert leakage["eval_token_occurrences"] == locked["eval_token_occurrences"]
        assert leakage["eval_only_token_occurrences"] == locked["eval_only_token_occurrences"]
        assert leakage["eval_only_token_types"] == locked["eval_only_token_types"]
        got_train_ari = leakage["train_ari_full_vs_train_only"]
        assert got_train_ari == pytest.approx(locked["train_ari"], abs=ABS), (
            f"{fixture_id} train_ari_full_vs_train_only {got_train_ari!r} "
            f"against locked {locked['train_ari']!r}; {_numeric_environment()}"
        )
        assert leakage["eval_ari_full_vs_train_only"] == pytest.approx(locked["eval_ari"], abs=ABS)
        assert leakage["train_pairwise_distance_correlation"] == pytest.approx(
            locked["correlation"],
            abs=ABS,
        )
        assert leakage["material"] is True
        assert leakage["train_ari_full_vs_train_only"] < SVD_MATERIAL_TRAIN_ARI


def test_generalization_locks_are_still_the_original_control_table():
    source = (ROOT / "tests" / "test_clustering_generalization.py").read_text(encoding="utf-8")
    module = (ROOT / "src" / "neuralese" / "clustering_generalization.py").read_text(encoding="utf-8")
    assert '"uneven": {"train": (26, 21, 22), "eval": (7, 8, 5), "rows": (40, 10)}' in source
    assert '"balanced": {"train": (30, 20, 18), "eval": (9, 7, 8), "rows": (64, 16)}' in source
    assert "RESIDUAL_WORSE_FACTOR = 1.10" in module
    assert "CLUSTER_RUNTIME_FACTOR = 2.0" in module
    assert 'label_eval = _score_assignment(production["eval_labels"], eval_topics_shuffled)' in module
    assert 'row_eval = _score_assignment(shuffled_eval_labels, eval_topics)' in module
    clustering = (ROOT / "src" / "neuralese" / "clustering.py").read_text(encoding="utf-8")
    assert "seeds = _farthest_first(X, k)" in clustering
    assert "0.7.0" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_audit_note_and_roadmap_pointer():
    doc = (ROOT / "docs" / "PR1_EVAL_CONTROL_INTEGRITY.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    assert "PR1_EVAL_CONTROL_INTEGRITY.md" in roadmap
    for token in (
        "UNRESOLVED",
        "NONE",
        "RUNTIME_NOT_AUTHORIZED",
        "train_only_majority_map",
        "75f26cb",
        "8/10",
        "6/9",
        "8/9",
        REVISED_DEFINITION,
    ):
        assert token in doc
    assert "learn_pack" in doc
    assert "word_sentence_svd" in doc

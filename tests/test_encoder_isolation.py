"""Train-only encoder isolation. The verdict and the split fingerprint are locked.

Controls are inequalities: a shuffled training map and a permuted eval
embedding must land strictly under the real inductive purity, and the
cross-split of ids and texts must be empty. A tie is not a fall, and a
tie keeps the verdict off ``INDUCTIVE_FAIL``.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from neuralese import __version__
from neuralese.clustering_generalization import LABEL_SHUFFLE_SEED, ROW_SHUFFLE_SEED
from neuralese.encoder_isolation import (
    BASELINE_COMMIT,
    CAUSE_CENTROIDS,
    CAUSE_NOT_APPLICABLE,
    CAUSE_SUBSPACE,
    CAUSE_UNRESOLVED,
    COVERAGE_GATE,
    DEFINED_COVERAGE_REASONS,
    FROZEN_SPLIT_SHA256,
    LABEL_CONTROL_GATE,
    LEAKAGE_GATE,
    PROJECTION_NORM_FLOOR,
    PURITY_GATE,
    RECOMMENDATION,
    ROW_CONTROL_GATE,
    RUNTIME_AUTHORITY,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_UNRESOLVED,
    authority_disposition,
    cross_split_duplicates,
    fit_train_only_encoder,
    inductive_verdict,
    isolate_cause,
    run_analytical_controls,
    run_encoder_isolation,
    split_fingerprint,
    transform_train_only,
)

ROOT = Path(__file__).resolve().parents[1]
ABS = 1e-9

UNEVEN_EVAL_IDS = (
    "uneven-005",
    "uneven-010",
    "uneven-015",
    "uneven-020",
    "uneven-025",
    "uneven-030",
    "uneven-035",
    "uneven-040",
    "uneven-045",
    "uneven-050",
)

# inductive eval, base, transductive eval, label control, row control.
HITS = {
    "uneven": (5, 5, 6, 5, 4),
    "overlapping": (3, 2, 4, 0, 1),
    "ambiguous": (4, 3, 7, 1, 1),
    "balanced": (3, 2, 3, 2, 1),
}
INDUCTIVE_TRAIN_HITS = {
    "uneven": 26,
    "overlapping": 27,
    "ambiguous": 22,
    "balanced": 33,
}
TRANSDUCTIVE_TRAIN_HITS = {
    "uneven": 26,
    "overlapping": 27,
    "ambiguous": 26,
    "balanced": 30,
}
SELF_MAJORITY = {
    "uneven": 7,
    "overlapping": 6,
    "ambiguous": 8,
    "balanced": 9,
}
TRANSDUCTIVE_TRAIN_RESIDUAL = {
    "uneven": 0.22758657335204746,
    "overlapping": 0.1337722038184514,
    "ambiguous": 0.2567684913717868,
    "balanced": 0.21110058070963547,
}
TOKEN_ABSENCE = {
    "uneven": (72, 56, 55),
    "overlapping": (66, 52, 51),
    "ambiguous": (58, 44, 43),
    "balanced": (94, 74, 72),
}
ZERO_ROWS = {"uneven": 3, "overlapping": 3, "ambiguous": 4, "balanced": 8}


@pytest.fixture(scope="module")
def analytical():
    return run_analytical_controls()


@pytest.fixture(scope="module")
def report():
    return run_encoder_isolation(time_runs=0)


def test_gates_are_constants_before_the_inductive_run():
    assert COVERAGE_GATE == "every_eval_row_has_a_finite_train_only_vector"
    assert PURITY_GATE == (
        "eval_size_weighted_purity_strictly_greater_than_eval_majority_base_rate"
    )
    assert "strictly_below" in LABEL_CONTROL_GATE
    assert "strictly_below" in ROW_CONTROL_GATE
    assert LEAKAGE_GATE == "no_shared_ids_or_texts_between_train_and_eval"
    assert BASELINE_COMMIT == "12407d3"
    assert PROJECTION_NORM_FLOOR == 1e-8
    assert __version__ == "0.7.0"
    source = (ROOT / "src" / "neuralese" / "encoder_isolation.py").read_text(encoding="utf-8")
    assert source.index("PURITY_GATE") < source.index("def run_encoder_isolation")
    assert source.index("def inductive_verdict") < source.index("def run_encoder_isolation")
    assert source.index("def run_analytical_controls") < source.index("def _score_fixture")
    assert "0.90" not in source
    assert "SVD_MATERIAL" not in source
    assert "FROZEN_CANDIDATES" not in source
    assert "learn_pack(" not in source
    assert "def word_sentence_svd" not in source
    assert "def kmeans" not in source
    assert "def certify" not in source
    assert "0.7.0" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    clustering = (ROOT / "src" / "neuralese" / "clustering.py").read_text(encoding="utf-8")
    assert "seeds = _farthest_first(X, k)" in clustering
    encode = (ROOT / "src" / "neuralese" / "encode.py").read_text(encoding="utf-8")
    assert "def word_sentence_svd" in encode
    alphabet = (ROOT / "src" / "neuralese" / "alphabet.py").read_text(encoding="utf-8")
    assert "tau_kappa: float = 0.35" in alphabet
    assert "tau_residual: float = 0.55" in alphabet


def test_verdict_stays_unresolved_when_a_control_does_not_fall():
    assert inductive_verdict(
        split_reproduced=True,
        controls_fall=False,
        coverage_complete=True,
        coverage_reason_defined=True,
        every_fixture_passes=False,
        any_base_rate_miss=True,
    ) == VERDICT_UNRESOLVED
    assert inductive_verdict(
        split_reproduced=False,
        controls_fall=True,
        coverage_complete=True,
        coverage_reason_defined=True,
        every_fixture_passes=False,
        any_base_rate_miss=True,
    ) == VERDICT_UNRESOLVED
    assert inductive_verdict(
        split_reproduced=True,
        controls_fall=True,
        coverage_complete=False,
        coverage_reason_defined=False,
        every_fixture_passes=False,
        any_base_rate_miss=True,
    ) == VERDICT_UNRESOLVED
    assert inductive_verdict(
        split_reproduced=True,
        controls_fall=True,
        coverage_complete=True,
        coverage_reason_defined=True,
        every_fixture_passes=False,
        any_base_rate_miss=True,
    ) == VERDICT_FAIL
    assert inductive_verdict(
        split_reproduced=True,
        controls_fall=True,
        coverage_complete=True,
        coverage_reason_defined=True,
        every_fixture_passes=True,
        any_base_rate_miss=False,
    ) == VERDICT_PASS
    with pytest.raises(ValueError):
        inductive_verdict(
            split_reproduced=True,
            controls_fall=False,
            coverage_complete=True,
            coverage_reason_defined=True,
            every_fixture_passes=True,
            any_base_rate_miss=False,
        )


def test_authority_stays_unauthorized_on_a_pass():
    for verdict in (VERDICT_PASS, VERDICT_FAIL, VERDICT_UNRESOLVED):
        disposition = authority_disposition(verdict)
        assert disposition["recommendation"] == RECOMMENDATION == "NONE"
        assert disposition["runtime_authority"] == RUNTIME_AUTHORITY == "RUNTIME_NOT_AUTHORIZED"
        assert disposition["production_replaced"] is False


def test_isolate_cause_names_one_flag_only():
    assert isolate_cause(
        purity_failed=False,
        subspace_moved=True,
        centroids_moved=False,
        eval_far=False,
    ) == CAUSE_NOT_APPLICABLE
    assert isolate_cause(
        purity_failed=True,
        subspace_moved=True,
        centroids_moved=False,
        eval_far=False,
    ) == CAUSE_SUBSPACE
    assert isolate_cause(
        purity_failed=True,
        subspace_moved=False,
        centroids_moved=True,
        eval_far=False,
    ) == CAUSE_CENTROIDS
    assert isolate_cause(
        purity_failed=True,
        subspace_moved=True,
        centroids_moved=True,
        eval_far=False,
    ) == CAUSE_UNRESOLVED
    assert isolate_cause(
        purity_failed=True,
        subspace_moved=False,
        centroids_moved=False,
        eval_far=False,
    ) == CAUSE_UNRESOLVED


def test_unknown_token_adds_no_column_and_idf_uses_the_training_count():
    train = ["kiln brick", "shelf brick"]
    encoder = fit_train_only_encoder(train, rank=2)
    vocabulary = list(encoder["vocabulary"])
    assert "quartz" not in vocabulary
    assert encoder["idf"][vocabulary.index("kiln")] == pytest.approx(math.log(2.0) + 1.0)
    assert encoder["n_train"] == 2
    before = dict(encoder["document_frequency"])
    plain = transform_train_only(encoder, ["kiln"])
    extra = transform_train_only(encoder, ["kiln quartz quartz"])
    assert np.allclose(plain, extra)
    assert encoder["document_frequency"] == before
    assert np.isfinite(extra).all()


def test_duplicate_detector_inequalities():
    empty = cross_split_duplicates(["a"], ["b"], ["text-a"], ["text-b"])
    planted = cross_split_duplicates(["a", "c"], ["b", "a"], ["text-1", "text-2"], ["text-3", "text-1"])
    assert empty["shared_ids"] == 0
    assert empty["shared_texts"] == 0
    assert empty["empty"] is True
    assert planted["shared_ids"] > 0
    assert planted["shared_texts"] > 0
    assert planted["empty"] is False


def test_analytical_controls_are_strict_inequalities(analytical):
    blobs = analytical["blobs"]
    assert analytical["passed"] is True
    assert blobs["real_hits"] == 200
    assert blobs["n_eval"] == 200
    assert blobs["base_hits"] == 100
    assert blobs["real_hits"] > blobs["base_hits"]
    assert blobs["label_hits"] == 95
    assert blobs["label_hits"] < blobs["real_hits"]
    assert blobs["embedding_hits"] == 110
    assert blobs["embedding_hits"] < blobs["real_hits"]
    assert blobs["duplicates_empty"] is True
    duplicates = analytical["duplicates"]
    assert duplicates["empty_ids"] == 0
    assert duplicates["empty_texts"] == 0
    assert duplicates["planted_ids"] > 0
    assert duplicates["planted_texts"] > 0
    assert all(analytical["checks"].values())


def test_failed_analytical_controls_do_not_embed(monkeypatch):
    calls = []

    def _fail():
        return {"passed": False, "checks": {}, "blobs": {}, "duplicates": {}}

    def _svd(*_args, **_kwargs):
        calls.append("svd")
        raise AssertionError("fixtures must not be embedded when analytical controls fail")

    monkeypatch.setattr("neuralese.encoder_isolation.run_analytical_controls", _fail)
    monkeypatch.setattr("neuralese.encoder_isolation.word_sentence_svd", _svd)
    report = run_encoder_isolation(time_runs=0)
    assert report["verdict"] == VERDICT_UNRESOLVED
    assert report["fixtures"] == []
    assert report["recommendation"] == "NONE"
    assert report["runtime_authority"] == RUNTIME_AUTHORITY
    assert report["production_replaced"] is False
    assert calls == []


def test_a_mismatched_fingerprint_does_not_embed(monkeypatch):
    calls = []

    def _svd(*_args, **_kwargs):
        calls.append("svd")
        raise AssertionError("a drifted split must not be embedded")

    monkeypatch.setattr("neuralese.encoder_isolation.FROZEN_SPLIT_SHA256", "0" * 64)
    monkeypatch.setattr("neuralese.encoder_isolation.word_sentence_svd", _svd)
    report = run_encoder_isolation(time_runs=0)
    assert report["verdict"] == VERDICT_UNRESOLVED
    assert report["split_reproduced"] is False
    assert report["fixtures"] == []
    assert calls == []


def test_split_fingerprint_is_the_frozen_eval_ids():
    fingerprint = split_fingerprint()
    assert fingerprint["rule"] == "file-order index % 5 == 4"
    assert fingerprint["modulus"] == 5
    assert fingerprint["remainder"] == 4
    assert fingerprint["reproduced"] is True
    assert fingerprint["sha256"] == FROZEN_SPLIT_SHA256
    assert FROZEN_SPLIT_SHA256 == "55f561c832ef47d453827b3fc9704cf4d98e66d0bda2e4ff4aa9603c5923c20e"
    by_id = {fixture["fixture_id"]: fixture for fixture in fingerprint["fixtures"]}
    assert by_id["uneven"]["eval_ids"] == list(UNEVEN_EVAL_IDS)
    assert by_id["uneven"]["n_train"] == 40 and by_id["uneven"]["n_eval"] == 10
    assert by_id["overlapping"]["n_train"] == 39 and by_id["overlapping"]["n_eval"] == 9
    assert by_id["ambiguous"]["n_train"] == 39 and by_id["ambiguous"]["n_eval"] == 9
    assert by_id["balanced"]["n_train"] == 64 and by_id["balanced"]["n_eval"] == 16
    assert by_id["balanced"]["eval_ids"][0] == "balanced-005"
    assert by_id["balanced"]["eval_ids"][-1] == "balanced-080"
    assert len(by_id["balanced"]["eval_ids"]) == 16


def test_trial_is_unresolved_and_does_not_replace_production(report):
    assert report["verdict"] == VERDICT_UNRESOLVED
    assert report["recommendation"] == "NONE"
    assert report["runtime_authority"] == "RUNTIME_NOT_AUTHORIZED"
    assert report["production_replaced"] is False
    assert report["production_svd_modified"] is False
    assert report["learn_pack_called"] is False
    assert report["candidates_rerun"] is False
    assert report["analytical_passed"] is True
    assert report["split_reproduced"] is True
    assert report["controls_fall"] is False
    assert report["coverage_complete"] is True
    assert report["any_base_rate_miss"] is True
    assert report["every_fixture_passes"] is False
    assert report["diagnostic_cause"] == CAUSE_UNRESOLVED
    assert report["centering"] == "not_a_production_statistic"
    assert report["l2_scope"] == "per_row_not_a_corpus_moment"
    assert report["split_fingerprint"]["sha256"] == FROZEN_SPLIT_SHA256
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert report["git_revision"] == head
    libraries = " ".join(report["numerical_library"]["loaded_libraries"])
    assert "libblas" in libraries
    assert "liblapack" in libraries
    assert report["numerical_library"]["numpy"]
    assert "blas" in report["numerical_library"]["show_config"].lower()
    assert [fixture["fixture_id"] for fixture in report["fixtures"]] == [
        "uneven",
        "overlapping",
        "ambiguous",
        "balanced",
    ]


def test_per_fixture_gates_and_control_inequalities(report):
    assert report["gates"]["purity"] == PURITY_GATE
    saw_label_tie = False
    for fixture in report["fixtures"]:
        fixture_id = fixture["fixture_id"]
        inductive_hits, base_hits, transductive_hits, label_hits, row_hits = HITS[fixture_id]
        inductive = fixture["inductive"]
        controls = fixture["controls"]
        assert inductive["eval_hits"] == inductive_hits
        assert inductive["eval_rows"] == fixture["n_eval"]
        assert fixture["eval_majority_hits"] == base_hits
        assert inductive["train_hits"] == INDUCTIVE_TRAIN_HITS[fixture_id]
        assert inductive["map_fit_rows"] == fixture["n_train"]
        assert inductive["map_source"] == "train_rows_only"
        assert fixture["transductive"]["label"] == "TRANSDUCTIVE"
        assert fixture["transductive"]["is_pass_gate"] is False
        assert fixture["transductive"]["is_inductive_result"] is False
        assert fixture["transductive"]["eval_hits"] == transductive_hits
        assert fixture["transductive"]["train_hits"] == TRANSDUCTIVE_TRAIN_HITS[fixture_id]
        assert fixture["transductive"]["self_majority_eval_hits"] == SELF_MAJORITY[fixture_id]
        assert fixture["transductive"]["train_residual"] == pytest.approx(
            TRANSDUCTIVE_TRAIN_RESIDUAL[fixture_id],
            abs=ABS,
        )
        assert controls["label_shuffle_seed"] == LABEL_SHUFFLE_SEED == 0
        assert controls["row_shuffle_seed"] == ROW_SHUFFLE_SEED == 1
        assert controls["label_hits"] == label_hits
        assert controls["row_hits"] == row_hits
        assert controls["label_below"] is (label_hits < inductive_hits)
        assert controls["row_below"] is (row_hits < inductive_hits)
        assert controls["map_fit_on_eval"] is False
        assert controls["centroids_refit"] is False
        assert fixture["gates"]["above_base_rate"] is (inductive_hits > base_hits)
        assert fixture["gates"]["label_control_below"] is controls["label_below"]
        assert fixture["gates"]["row_control_below"] is controls["row_below"]
        if not controls["label_below"]:
            saw_label_tie = True
            assert label_hits == inductive_hits
        occurrences, absent_occurrences, absent_types = TOKEN_ABSENCE[fixture_id]
        tokens = fixture["tokens"]
        assert tokens["eval_content_token_occurrences"] == occurrences
        assert tokens["absent_occurrences"] == absent_occurrences
        assert tokens["absent_types"] == absent_types
        assert tokens["absence_fraction"] == pytest.approx(absent_occurrences / occurrences)
        coverage = fixture["coverage"]
        assert coverage["complete"] is True
        assert coverage["fraction"] == 1.0
        assert coverage["reason"] == "complete"
        assert coverage["reason"] in DEFINED_COVERAGE_REASONS
        assert coverage["zero_rows"] == ZERO_ROWS[fixture_id]
        assert fixture["duplicates"]["shared_ids"] == 0
        assert fixture["duplicates"]["shared_texts"] == 0
        assert fixture["duplicates"]["duplicate_ids_in_file"] == 0
        assert fixture["duplicates"]["duplicate_texts_in_file"] == 0
        assert fixture["duplicates"]["empty_cross_split"] is True
        assert fixture["train_tfidf_max_abs_diff"] == 0.0
        assert fixture["train_embed_max_abs_diff"] < 1e-12
        assert fixture["costs"]["embed_median_ms"] is None
        assert fixture["costs"]["cluster_median_ms"] is None
    assert saw_label_tie is True


def test_uneven_cause_is_unresolved_because_two_diagnostics_fire(report):
    by_id = {fixture["fixture_id"]: fixture for fixture in report["fixtures"]}
    uneven = by_id["uneven"]["diagnostic"]
    assert uneven["purity_failed"] is True
    assert uneven["subspace_moved"] is True
    assert uneven["centroids_moved"] is True
    assert uneven["eval_far"] is False
    assert uneven["cause"] == CAUSE_UNRESOLVED
    assert uneven["is_gate"] is False
    assert uneven["max_principal_angle_deg"] < uneven["cutoffs"]["max_principal_angle_deg"]
    assert uneven["procrustes_relative_residual"] > uneven["cutoffs"]["procrustes_relative_residual"]
    assert uneven["centroid_mean_l2_displacement"] > uneven["cutoffs"]["centroid_mean_l2_displacement"]
    assert uneven["eval_far_fraction"] < uneven["cutoffs"]["eval_far_fraction"]
    for fixture_id in ("overlapping", "ambiguous", "balanced"):
        diagnostic = by_id[fixture_id]["diagnostic"]
        assert diagnostic["purity_failed"] is False
        assert diagnostic["cause"] == CAUSE_NOT_APPLICABLE
        assert by_id[fixture_id]["gates"]["above_base_rate"] is True
        assert by_id[fixture_id]["controls"]["label_below"] is True
        assert by_id[fixture_id]["controls"]["row_below"] is True


def test_timed_run_reports_positive_cost(report):
    timed = run_encoder_isolation(time_runs=1)
    assert timed["verdict"] == report["verdict"] == VERDICT_UNRESOLVED
    assert timed["costs"]["peak_bytes"] > 0
    for fixture in timed["fixtures"]:
        assert fixture["costs"]["embed_median_ms"] > 0
        assert fixture["costs"]["cluster_median_ms"] > 0
        assert fixture["costs"]["embed_peak_bytes"] > 0
        assert fixture["costs"]["cluster_peak_bytes"] > 0


def test_note_and_roadmap_pointer():
    doc = (ROOT / "docs" / "PR1_ENCODER_ISOLATION.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    assert "PR1_ENCODER_ISOLATION.md" in roadmap
    for token in (
        "UNRESOLVED",
        "NONE",
        "RUNTIME_NOT_AUTHORIZED",
        "TRANSDUCTIVE",
        "55f561c832ef47d453827b3fc9704cf4d98e66d0bda2e4ff4aa9603c5923c20e",
        "12407d3",
        "INDUCTIVE_FAIL",
        "5/10",
        "libblas.so.3.12.0",
    ):
        assert token in doc
    assert "Production was not replaced." in doc

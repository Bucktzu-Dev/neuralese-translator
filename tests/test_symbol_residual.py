"""Symbol-residual distinguishability. The fixture is loaded with numpy only.

The research module is not part of learn, translate, or certify. Admission
constants are the ones written above ``run_symbol_residual``. A missing or
altered fixture stays UNRESOLVED. Torch is not imported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralese import __version__
from neuralese.alphabet import LearnConfig
from neuralese.clustering import kmeans as production_kmeans
from neuralese.symbol_residual import (
    ACCURACY_MARGIN,
    ACTIVATION_SHA256,
    BYTES_PER_ROW,
    DECODER_RULE,
    ENTERING_RECOMMENDATION,
    ENTERING_VERDICT,
    FINAL_LAYER_INDEX,
    FIXTURE_NPZ,
    FULL_VECTOR_BYTES,
    INDEPENDENT_TARGET_STATUS,
    LEARN_SEED,
    N_SYMBOLS,
    PARENT_COMMIT,
    PLAN_STATUS,
    RECOMMENDATION_NONE,
    RESIDUAL_BITS,
    RESIDUAL_DEFINITION,
    RESIDUAL_QMAX,
    RUNTIME_AUTHORITY,
    SCALE_RULE,
    SHUFFLE_ASSIGNMENT_SEED,
    SHUFFLE_FALL,
    STATUS,
    STORAGE_RATIO_MAX,
    VERDICT_GEOMETRY_ONLY,
    VERDICT_RESEARCH_FAIL,
    VERDICT_RESIDUAL,
    VERDICT_UNRESOLVED,
    WITHIN_CLASS_RESIDUAL_SHUFFLE_SEED,
    decide_verdict,
    fit_scale,
    kmeans,
    majority_probe,
    pack_int4,
    pack_rows,
    quantize,
    run_symbol_residual,
    unpack_int4,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "neuralese" / "symbol_residual.py"
NOTE = ROOT / "docs" / "PR1_SYMBOL_RESIDUAL.md"
PACKAGE = ROOT / "src" / "neuralese"


@pytest.fixture(scope="module")
def report():
    return run_symbol_residual()


def test_gates_are_written_above_the_runner():
    text = SOURCE.read_text(encoding="utf-8")
    run_at = text.index("def run_symbol_residual")
    for gate in (
        "RESIDUAL_BITS = 4",
        "RESIDUAL_QMAX = 7",
        "ACCURACY_MARGIN = 0.10",
        "SHUFFLE_FALL = 0.10",
        "STORAGE_RATIO_MAX = 0.25",
        "SHUFFLE_ASSIGNMENT_SEED = 0",
        "WITHIN_CLASS_RESIDUAL_SHUFFLE_SEED = 1",
        "N_SYMBOLS = 8",
        "LEARN_SEED = 0",
        'RESIDUAL_DEFINITION = "activation_minus_assigned_centroid"',
        'SCALE_RULE = "per_coordinate_max_abs_train_residual_zero_maps_to_one"',
        'DECODER_RULE = "within_symbol_nearest_dequantized_residual_then_majority"',
        'INDEPENDENT_TARGET_STATUS = "INDEPENDENT_TARGET_UNAVAILABLE"',
        'ADVANTAGE_KIND = "bytes_per_row"',
    ):
        assert text.index(gate) < run_at
    assert RESIDUAL_BITS == 4
    assert RESIDUAL_QMAX == 7
    assert ACCURACY_MARGIN == 0.10
    assert SHUFFLE_FALL == 0.10
    assert STORAGE_RATIO_MAX == 0.25
    assert SHUFFLE_ASSIGNMENT_SEED == 0
    assert WITHIN_CLASS_RESIDUAL_SHUFFLE_SEED == 1
    assert N_SYMBOLS == 8
    assert LEARN_SEED == 0
    assert BYTES_PER_ROW == 65
    assert FULL_VECTOR_BYTES == 512
    assert RESIDUAL_DEFINITION == "activation_minus_assigned_centroid"
    assert SCALE_RULE.startswith("per_coordinate_max_abs")
    assert "within_symbol" in DECODER_RULE
    assert INDEPENDENT_TARGET_STATUS == "INDEPENDENT_TARGET_UNAVAILABLE"
    assert STATUS == "RESEARCH_ONLY"
    assert PLAN_STATUS == "PLAN_ONLY"
    assert ENTERING_VERDICT == "GEOMETRY_ONLY"
    assert ENTERING_RECOMMENDATION == "NONE"
    assert RUNTIME_AUTHORITY == "RUNTIME_NOT_AUTHORIZED"
    assert PARENT_COMMIT == "81077ec"
    assert FINAL_LAYER_INDEX == 6
    assert "PASS_FOR_LATER_PR" not in text
    assert "LATER_ISOLATED_PR" not in text
    verdict_body = text.split("def decide_verdict", 1)[1].split("\ndef ", 1)[0]
    assert "fidelity" not in verdict_body
    assert "RECOMMENDATION_NONE" in verdict_body
    call = text.split("kmeans(", 1)[1].split(")", 1)[0]
    assert "train_x" in call
    assert "probe" not in call
    assert "held" not in call
    assert "fit_scale(train_residual)" in text
    assert "fit_scale(held" not in text
    assert kmeans is production_kmeans


def test_verdict_table_does_not_read_a_score():
    unresolved = decide_verdict(
        integrity_failure="checksum_mismatch",
        parent_reproduced=True,
        conditions=[{"accuracy_gain": True, "shuffle_fell": True, "storage_ok": True}],
    )
    assert unresolved == (VERDICT_UNRESOLVED, RECOMMENDATION_NONE, "checksum_mismatch")
    parent = decide_verdict(
        integrity_failure=None,
        parent_reproduced=False,
        conditions=[{"accuracy_gain": True, "shuffle_fell": True, "storage_ok": True}],
    )
    assert parent == (
        VERDICT_UNRESOLVED,
        RECOMMENDATION_NONE,
        "parent_symbol_boundary_not_reproduced",
    )
    admitted = decide_verdict(
        integrity_failure=None,
        parent_reproduced=True,
        conditions=[
            {"accuracy_gain": True, "shuffle_fell": True, "storage_ok": True},
            {"accuracy_gain": False, "shuffle_fell": False, "storage_ok": True},
        ],
    )
    assert admitted == (
        VERDICT_RESIDUAL,
        RECOMMENDATION_NONE,
        "admission_gate_met_without_independent_target",
    )
    storage = decide_verdict(
        integrity_failure=None,
        parent_reproduced=True,
        conditions=[{"accuracy_gain": True, "shuffle_fell": True, "storage_ok": False}],
    )
    assert storage == (
        VERDICT_GEOMETRY_ONLY,
        RECOMMENDATION_NONE,
        "accuracy_gain_without_storage_advantage",
    )
    shuffle = decide_verdict(
        integrity_failure=None,
        parent_reproduced=True,
        conditions=[{"accuracy_gain": True, "shuffle_fell": False, "storage_ok": True}],
    )
    assert shuffle == (
        VERDICT_GEOMETRY_ONLY,
        RECOMMENDATION_NONE,
        "accuracy_gain_survives_within_class_residual_shuffle",
    )
    both = decide_verdict(
        integrity_failure=None,
        parent_reproduced=True,
        conditions=[{"accuracy_gain": True, "shuffle_fell": False, "storage_ok": False}],
    )
    assert both == (
        VERDICT_GEOMETRY_ONLY,
        RECOMMENDATION_NONE,
        "accuracy_gain_without_storage_advantage+"
        "accuracy_gain_survives_within_class_residual_shuffle",
    )
    failed = decide_verdict(
        integrity_failure=None,
        parent_reproduced=True,
        conditions=[
            {"accuracy_gain": False, "shuffle_fell": False, "storage_ok": True},
            {"accuracy_gain": False, "shuffle_fell": True, "storage_ok": False},
        ],
    )
    assert failed == (
        VERDICT_RESEARCH_FAIL,
        RECOMMENDATION_NONE,
        "no_accuracy_gain_over_symbol_only",
    )


def test_codec_roundtrip_and_train_scale():
    rng = np.random.default_rng(0)
    codes = rng.integers(-7, 8, size=(6, 128), dtype=np.int8)
    packed = pack_int4(codes)
    assert packed.shape == (6, 64)
    assert packed.dtype == np.uint8
    assert np.array_equal(unpack_int4(packed), codes)
    symbols = np.array([0, 7, 3, 1, 4, 2], dtype=np.int64)
    rows = pack_rows(symbols, codes)
    assert rows.shape == (6, BYTES_PER_ROW)
    assert rows.shape[1] <= STORAGE_RATIO_MAX * FULL_VECTOR_BYTES
    assert np.array_equal(rows[:, 0], symbols.astype(np.uint8))
    assert np.array_equal(unpack_int4(rows[:, 1:]), codes)
    residual = np.zeros((4, 3), dtype=np.float64)
    residual[0, 0] = -2.0
    residual[1, 1] = 0.5
    scale = fit_scale(residual)
    assert scale.tolist() == [2.0, 0.5, 1.0]
    quantized = quantize(np.array([[-2.0, 0.5, 9.0]]), scale)
    assert quantized.tolist() == [[-7, 7, 7]]
    held = np.array([[100.0, -100.0, 0.0]])
    assert fit_scale(residual).tolist() == scale.tolist()
    assert fit_scale(np.vstack([residual, held])).tolist() != scale.tolist()
    assert majority_probe(np.array([1, 0, 1, 0])) == 0
    assert majority_probe(np.array([1, 1, 0])) == 1


def test_research_module_is_not_imported_by_the_package():
    needles = (
        "import neuralese.symbol_residual",
        "from neuralese.symbol_residual",
        "from neuralese import symbol_residual",
    )
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name == "symbol_residual.py":
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in text
    activation_source = (PACKAGE / "activation_source.py").read_text(encoding="utf-8")
    assert "shuffled_symbol_residual" in activation_source
    research = SOURCE.read_text(encoding="utf-8")
    assert "import torch" not in research
    assert "from_pretrained" not in research
    assert "learn_pack(" not in research
    assert "certify(" not in research
    cfg = LearnConfig()
    assert cfg.tau_residual == 0.55
    assert cfg.tau_kappa == 0.35
    assert __version__ == "0.7.0"


def test_missing_or_altered_fixture_stays_unresolved(tmp_path):
    missing = run_symbol_residual(tmp_path / "no.npz")
    assert missing["verdict"] == VERDICT_UNRESOLVED
    assert missing["recommendation"] == RECOMMENDATION_NONE
    assert missing["authority"] == RUNTIME_AUTHORITY
    assert missing["reason"] == "fixture_missing"
    assert missing["independent_target"]["status"] == "INDEPENDENT_TARGET_UNAVAILABLE"
    assert "condition_2" not in missing
    with np.load(FIXTURE_NPZ, allow_pickle=False) as bundle:
        arrays = {key: np.array(bundle[key]) for key in bundle.files}
    broken = {key: np.array(arrays[key]) for key in ACTIVATION_SHA256}
    broken["c2_train_activations"][0, 0] = np.float32(broken["c2_train_activations"][0, 0] + np.float32(1.0))
    bad_path = tmp_path / "bad.npz"
    np.savez(bad_path, **broken)
    mismatched = run_symbol_residual(bad_path)
    assert mismatched["verdict"] == VERDICT_UNRESOLVED
    assert mismatched["reason"] == "checksum_mismatch"
    assert mismatched["recommendation"] == RECOMMENDATION_NONE
    assert "condition_2" not in mismatched
    arrays["c2_train_probe"] = np.array(arrays["c2_train_probe"])
    arrays["c2_train_probe"][0] = np.int32(1)
    probe_path = tmp_path / "probe.npz"
    np.savez(probe_path, **arrays)
    probed = run_symbol_residual(probe_path)
    assert probed["verdict"] == VERDICT_UNRESOLVED
    assert probed["reason"] == "probe_integrity:c2_train_probe"
    assert "condition_2" not in probed


def _assert_fidelity(arm, mean_sq: float, normalized: float) -> None:
    fidelity = arm["fidelity"]
    assert fidelity["distance"] == "mean_per_row_squared_euclidean"
    assert fidelity["mean_squared_euclidean"] == pytest.approx(mean_sq, abs=1e-12)
    assert fidelity["normalized_frobenius"] == pytest.approx(normalized, abs=1e-12)


def test_frozen_slot_counts_and_admission(report):
    assert report["genuine"] is True
    assert report["parent_reproduced"] is True
    assert report["verdict"] == VERDICT_RESIDUAL
    assert report["recommendation"] == RECOMMENDATION_NONE
    assert report["authority"] == RUNTIME_AUTHORITY
    assert report["reason"] == "admission_gate_met_without_independent_target"
    assert report["explicit_failures"] == ["condition_4:no_accuracy_gain_over_symbol_only"]
    assert report["probe_used_in_fit"] is False
    assert report["gloss_used_as_evidence"] is False
    assert report["semantic_interpretation"] is False
    assert report["runtime_integration"] is False
    assert report["production_admission"] is False
    assert report["heldout_refit"] is False
    assert report["min_cluster_size_quarantine"] is False
    assert report["condition_1"] == "omitted_no_controlled_slot"
    assert report["condition_3"] == "omitted_not_in_this_trial"
    assert report["independent_target"]["satisfied"] is False
    assert report["independent_target"]["status"] == "INDEPENDENT_TARGET_UNAVAILABLE"
    assert report["checksums"] == dict(ACTIVATION_SHA256)
    assert report["bytes_per_row"]["symbol_plus_residual"] == 65
    assert report["bytes_per_row"]["full_vector"] == 512
    assert report["bytes_per_row"]["symbol_plus_residual_ratio"] == pytest.approx(65 / 512, abs=0.0)
    counts = {
        "condition_2": {
            "symbol_only": (38, 64, 1, 65536, 0),
            "symbol_plus_residual": (57, 64, 65, 65536, 194560),
            "nearest_train_row": (59, 64, 512, 1048576, 0),
            "full_vector": (59, 64, 512, 1048576, 0),
            "train_mean": (32, 64, 1, 8192, 0),
            "shuffled_assignment": (38, 64, 1, 0, 0),
            "within_class_residual_shuffle": (35, 64, 65, 65536, 194560),
            "residual_without_symbol": (61, 64, 64, 65536, 1056768),
        },
        "condition_4": {
            "symbol_only": (12, 16, 1, 16384, 0),
            "symbol_plus_residual": (12, 16, 65, 16384, 13696),
            "nearest_train_row": (14, 16, 512, 65536, 0),
            "full_vector": (14, 16, 512, 65536, 0),
            "train_mean": (8, 16, 1, 2048, 0),
            "shuffled_assignment": (8, 16, 1, 0, 0),
            "within_class_residual_shuffle": (12, 16, 65, 16384, 13696),
            "residual_without_symbol": (14, 16, 64, 16384, 67584),
        },
    }
    for name, arms in counts.items():
        block = report[name]
        assert block["min_cluster_size_quarantine"] is False
        assert block["heldout_on_empty_train_symbols"] == 0
        assert block["learn_seconds"] >= 0.0
        assert block["learn_seconds_are_a_gate"] is False
        for arm_name, (correct, n, nbytes, assign_ops, decode_ops) in arms.items():
            arm = block["arms"][arm_name]
            assert arm["n_correct"] == correct
            assert arm["n"] == n
            assert arm["accuracy"] == pytest.approx(correct / n, abs=0.0)
            assert arm["bytes_per_row"] == nbytes
            assert arm["assignment_coordinate_ops"] == assign_ops
            assert arm["decode_coordinate_ops"] == decode_ops
            assert arm["decode_row_ops"] == n
            assert arm["seconds_are_a_gate"] is False
            assert arm["assignment_seconds"] >= 0.0
            assert arm["decode_seconds"] >= 0.0
            assert arm["admits_residual"] is False
    c2 = report["condition_2"]
    c4 = report["condition_4"]
    assert c2["train_symbol_counts"] == [32, 17, 2, 9, 32, 15, 14, 7]
    assert c2["held_symbol_counts"] == [16, 9, 0, 3, 16, 7, 8, 5]
    assert c4["train_symbol_counts"] == [6, 2, 6, 3, 2, 2, 7, 4]
    assert c4["held_symbol_counts"] == [3, 0, 7, 2, 0, 0, 3, 1]
    assert c2["train_probe_counts_per_symbol"] == [
        [16, 16],
        [4, 13],
        [1, 1],
        [7, 2],
        [16, 16],
        [12, 3],
        [6, 8],
        [2, 5],
    ]
    assert c4["train_probe_counts_per_symbol"] == [
        [2, 4],
        [1, 1],
        [4, 2],
        [1, 2],
        [1, 1],
        [1, 1],
        [3, 4],
        [3, 1],
    ]
    assert c2["prediction_disagreement_vs_symbol"] == {
        "n_predictions_differ_from_symbol": 19,
        "n_only_residual_correct": 19,
        "n_only_symbol_correct": 0,
        "n_both_correct": 38,
        "n_both_wrong": 7,
    }
    assert c4["prediction_disagreement_vs_symbol"] == {
        "n_predictions_differ_from_symbol": 4,
        "n_only_residual_correct": 2,
        "n_only_symbol_correct": 2,
        "n_both_correct": 10,
        "n_both_wrong": 2,
    }
    for block in (c2, c4):
        quantization = block["quantization"]
        assert quantization["fit_on"] == "train_residual_only"
        assert quantization["train_code_min"] == -7
        assert quantization["train_code_max"] == 7
        assert quantization["held_code_min"] == -7
        assert quantization["held_code_max"] == 7
        assert block["arms"]["symbol_plus_residual"]["queries_with_distance_ties"] == 0
        assert block["arms"]["within_class_residual_shuffle"]["queries_with_distance_ties"] == 0
        assert block["arms"]["residual_without_symbol"]["queries_with_distance_ties"] == 0
        assert block["arms"]["full_vector"]["readout"] == "nearest_training_row"
        assert block["arms"]["residual_without_symbol"]["fidelity"] is None
        assert block["shared_bytes"]["centroids_float64"] == 8192
        assert block["shared_bytes"]["scales_float64"] == 1024
        assert block["shared_bytes"]["included_in_per_row_gate"] is False
    assert c2["quantization"]["held_zero_code_fraction"] == pytest.approx(1176 / 8192, abs=0.0)
    assert c4["quantization"]["held_zero_code_fraction"] == pytest.approx(158 / 2048, abs=0.0)
    assert c2["quantization"]["scale_min"] == pytest.approx(1.6162757312550262, abs=1e-12)
    assert c2["quantization"]["scale_max"] == pytest.approx(9.433844595270998, abs=1e-12)
    assert c4["quantization"]["scale_min"] == pytest.approx(0.8565953118460516, abs=1e-12)
    assert c4["quantization"]["scale_max"] == pytest.approx(5.456004157662392, abs=1e-12)
    assert c2["admission"]["admitted"] is True
    assert c2["admission"]["clause"] == "admission_gate_met"
    assert c2["admission"]["margin_over_symbol"] == pytest.approx(0.296875, abs=0.0)
    assert c2["admission"]["shuffle_fall"] == pytest.approx(0.34375, abs=0.0)
    assert c2["admission"]["storage_ok"] is True
    assert c2["admission"]["residual_without_symbol_carries_slot"] is True
    assert c2["admission"]["references_do_not_admit"] is True
    assert c4["admission"]["admitted"] is False
    assert c4["admission"]["clause"] == "no_accuracy_gain_over_symbol_only"
    assert c4["admission"]["margin_over_symbol"] == pytest.approx(0.0, abs=0.0)
    assert c4["admission"]["shuffle_fall"] == pytest.approx(0.0, abs=0.0)
    assert c4["admission"]["accuracy_gain"] is False
    assert c4["admission"]["residual_without_symbol_carries_slot"] is True
    _assert_fidelity(c2["arms"]["symbol_only"], 500.0086570588038, 0.20856050137538953)
    _assert_fidelity(c2["arms"]["symbol_plus_residual"], 13.72172380594111, 0.03454996701578748)
    _assert_fidelity(c2["arms"]["nearest_train_row"], 467.14654083402917, 0.20159041039987236)
    _assert_fidelity(c2["arms"]["full_vector"], 0.0, 0.0)
    _assert_fidelity(c2["arms"]["train_mean"], 2170.3090957849786, 0.43451432184730054)
    _assert_fidelity(c2["arms"]["shuffled_assignment"], 3876.3199747084736, 0.5807017166937707)
    _assert_fidelity(c2["arms"]["within_class_residual_shuffle"], 650.5428166908108, 0.23789276999537842)
    _assert_fidelity(c4["arms"]["symbol_only"], 838.2705645480764, 0.2672136815462288)
    _assert_fidelity(c4["arms"]["symbol_plus_residual"], 182.00555880726898, 0.12451134043861779)
    _assert_fidelity(c4["arms"]["nearest_train_row"], 809.714322027339, 0.26262283965537503)
    _assert_fidelity(c4["arms"]["full_vector"], 0.0, 0.0)
    _assert_fidelity(c4["arms"]["train_mean"], 982.3489764858955, 0.2892673925478527)
    _assert_fidelity(c4["arms"]["shuffled_assignment"], 1060.7607981003152, 0.3005905430929568)
    _assert_fidelity(c4["arms"]["within_class_residual_shuffle"], 718.5497704888743, 0.24739733092973376)
    arrays = np.load(FIXTURE_NPZ, allow_pickle=False)
    assert arrays["c2_train_activations"].shape == (128, 128)
    assert arrays["c4_heldout_activations"].shape == (16, 128)
    assert not any("logit" in key for key in arrays.files)


def test_note_names_the_frozen_verdict():
    note = NOTE.read_text(encoding="utf-8")
    assert "RESIDUAL_DISTINCTION" in note
    assert "INDEPENDENT_TARGET_UNAVAILABLE" in note
    assert "RUNTIME_NOT_AUTHORIZED" in note
    assert "recommendation is **NONE**" in note
    assert "PASS_FOR_LATER_PR" not in note
    assert "LATER_ISOLATED_PR" not in note
    assert "57/64" in note
    assert "35/64" in note
    assert "61/64" in note
    assert "12/16" in note
    assert "0.296875" in note
    assert "0.34375" in note
    assert "no_accuracy_gain_over_symbol_only" in note
    assert "182.00555880726898" in note
    assert "65/512" in note
    assert "cf967c0a9a04383db6f7b1108d86b2962634b4ac" in note
    assert "as a semantic reading" in note

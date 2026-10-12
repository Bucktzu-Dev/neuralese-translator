"""Symbolic addressing versus quantized numerical memory.

The fixture is loaded with numpy only. The research module is not part of
learn, translate, or certify. Admission constants are the ones written
above ``run_symbol_addressing``. A missing or altered fixture stays
UNRESOLVED. Torch is not imported. Wall-clock samples are not locked.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralese import __version__
from neuralese.alphabet import LearnConfig
from neuralese.symbol_addressing import (
    ADMISSION_STORAGE_FIGURE,
    CENTROID_TABLE_BYTES,
    CLAUSE_BELOW_RESIDUAL,
    CLAUSE_BELOW_STANDALONE,
    CLAUSE_GATE_MET,
    CLAUSE_NO_COST,
    CLAUSE_QUANTIZED,
    CLAUSE_REPRODUCTION,
    COST_WIN_MAX_DENOMINATOR,
    COST_WIN_MAX_NUMERATOR,
    DEQUANT_FORMULA,
    DISTANCE,
    ENTERING_RECOMMENDATION,
    ENTERING_VERDICT,
    FIXTURE_NPZ,
    INDEPENDENT_TARGET_STATUS,
    LATENCY_REPEATS,
    LATENCY_WARMUP,
    PARENT_COMMIT,
    PAYLOAD_BYTES,
    PLAN_STATUS,
    RECOMMENDATION_NONE,
    RESIDUAL_DISTINCTION_REVOKED,
    RUNTIME_AUTHORITY,
    SCALE_TABLE_BYTES,
    SECOND_AMORTIZATION_IS_A_GATE,
    SINGLE_CONDITION_WIN_IS_TRIAL_SUCCESS,
    SLOT_TOLERANCE_VS_RESIDUAL_ONLY,
    STATUS,
    TRIAL_LEVEL_COMBINER,
    VERDICT_QUANTIZED,
    VERDICT_RESEARCH_FAIL,
    VERDICT_SYMBOL,
    VERDICT_UNRESOLVED,
    account_system_bytes,
    cost_win_versus_both,
    decide_verdict,
    median_seconds,
    run_symbol_addressing,
    strictly_below_cost_fraction,
)
from neuralese.symbol_residual import (
    ACCURACY_MARGIN,
    ACTIVATION_SHA256,
    BYTES_PER_ROW,
    RESIDUAL_QMAX,
    SHUFFLE_FALL,
    STORAGE_RATIO_MAX,
    run_symbol_residual,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "neuralese" / "symbol_addressing.py"
NOTE = ROOT / "docs" / "PR1_SYMBOL_ADDRESSING.md"
PARENT_NOTE = ROOT / "docs" / "PR1_SYMBOL_RESIDUAL.md"
PACKAGE = ROOT / "src" / "neuralese"


@pytest.fixture(scope="module")
def report():
    return run_symbol_addressing()


def test_gates_are_written_above_the_runner():
    text = SOURCE.read_text(encoding="utf-8")
    run_at = text.index("def run_symbol_addressing")
    for gate in (
        "SLOT_TOLERANCE_VS_RESIDUAL_ONLY = 0.0",
        "LATENCY_WARMUP = 3",
        "LATENCY_REPEATS = 31",
        'ADMISSION_STORAGE_FIGURE = "system_bytes_divided_by_n_heldout"',
        "SECOND_AMORTIZATION_IS_A_GATE = False",
        'TRIAL_LEVEL_COMBINER = "and_both_conditions"',
        "SINGLE_CONDITION_WIN_IS_TRIAL_SUCCESS = False",
        "RELATIVE_COST_MARGIN_NUMERATOR = 1",
        "RELATIVE_COST_MARGIN_DENOMINATOR = 5",
        "COST_WIN_MAX_NUMERATOR = RELATIVE_COST_MARGIN_DENOMINATOR - RELATIVE_COST_MARGIN_NUMERATOR",
        "COST_WIN_MAX_DENOMINATOR = RELATIVE_COST_MARGIN_DENOMINATOR",
        "RECONSTRUCTION_IS_A_GATE = False",
        "COORDINATE_OPS_ARE_A_GATE = False",
        'INDEPENDENT_TARGET_STATUS = "INDEPENDENT_TARGET_UNAVAILABLE"',
        'ENTERING_VERDICT = "RESIDUAL_DISTINCTION"',
        "RESIDUAL_DISTINCTION_REVOKED = False",
        "PARENT_PER_ROW_CAP_IS_THIS_TRIAL_GATE = False",
        'DEQUANT_FORMULA = "code * scale / RESIDUAL_QMAX"',
        'DISTANCE = "squared_euclidean_on_dequantized_payloads"',
        'SLOT_TASK = "nearest_training_example_retrieval"',
    ):
        assert text.index(gate) < run_at
    assert SLOT_TOLERANCE_VS_RESIDUAL_ONLY == 0.0
    assert LATENCY_WARMUP == 3
    assert LATENCY_REPEATS == 31
    assert ADMISSION_STORAGE_FIGURE == "system_bytes_divided_by_n_heldout"
    assert SECOND_AMORTIZATION_IS_A_GATE is False
    assert TRIAL_LEVEL_COMBINER == "and_both_conditions"
    assert SINGLE_CONDITION_WIN_IS_TRIAL_SUCCESS is False
    assert COST_WIN_MAX_NUMERATOR == 4
    assert COST_WIN_MAX_DENOMINATOR == 5
    assert CENTROID_TABLE_BYTES == 8192
    assert SCALE_TABLE_BYTES == 1024
    assert PAYLOAD_BYTES == 64
    assert DEQUANT_FORMULA == "code * scale / RESIDUAL_QMAX"
    assert DISTANCE == "squared_euclidean_on_dequantized_payloads"
    assert INDEPENDENT_TARGET_STATUS == "INDEPENDENT_TARGET_UNAVAILABLE"
    assert STATUS == "RESEARCH_ONLY"
    assert PLAN_STATUS == "PLAN_ONLY"
    assert ENTERING_VERDICT == "RESIDUAL_DISTINCTION"
    assert ENTERING_RECOMMENDATION == "NONE"
    assert RESIDUAL_DISTINCTION_REVOKED is False
    assert RUNTIME_AUTHORITY == "RUNTIME_NOT_AUTHORIZED"
    assert PARENT_COMMIT == "01b500469f89f3ec7d8292e7e7e3b2f730629baf"
    assert "PASS_FOR_LATER_PR" not in text
    assert "LATER_ISOLATED_PR" not in text
    verdict_body = text.split("def decide_verdict", 1)[1].split("\ndef ", 1)[0]
    assert "fidelity" not in verdict_body
    assert "perf_counter" not in verdict_body
    assert "n_correct" not in verdict_body
    assert "RECOMMENDATION_NONE" in verdict_body
    fit_body = text.split("def _fit_codebooks", 1)[1].split("\ndef ", 1)[0]
    assert "fit_scale(train_x)" in fit_body
    assert "fit_scale(train_residual)" in fit_body
    assert "fit_scale(held" not in fit_body
    assert "train_y" not in fit_body
    assert "probe" not in fit_body
    assert "kmeans(" in fit_body
    assert "import torch" not in text
    assert "from_pretrained" not in text
    assert "learn_pack(" not in text
    assert "certify(" not in text
    for needle in (
        "import neuralese.symbol_residual",
        "from neuralese.symbol_residual",
        "from neuralese import symbol_residual",
    ):
        assert needle not in text
    assert ACCURACY_MARGIN == 0.10
    assert SHUFFLE_FALL == 0.10
    assert STORAGE_RATIO_MAX == 0.25
    assert RESIDUAL_QMAX == 7
    assert BYTES_PER_ROW == 65


def test_byte_predicate_and_verdict_table_do_not_read_a_score():
    standalone = account_system_bytes(
        128,
        64,
        uses_centroids=False,
        uses_scales=True,
        training_symbol_bytes=0,
        query_symbol_bytes=0,
    )
    symbol = account_system_bytes(
        128,
        64,
        uses_centroids=True,
        uses_scales=True,
        training_symbol_bytes=1,
        query_symbol_bytes=1,
    )
    alone = account_system_bytes(
        128,
        64,
        uses_centroids=True,
        uses_scales=True,
        training_symbol_bytes=0,
        query_symbol_bytes=0,
    )
    assert standalone["system_bytes"] == 13440
    assert standalone["admission_storage_bytes"] == 210
    assert symbol["system_bytes"] == 21824
    assert symbol["admission_storage_bytes"] == 341
    assert alone["system_bytes"] == 21632
    assert alone["admission_storage_bytes"] == 338
    assert symbol["training_row_bytes"] == 66
    assert standalone["training_row_bytes"] == 65
    assert symbol["query_row_bytes"] == 65
    assert alone["query_row_bytes"] == 64
    assert symbol["shared_centroid_bytes"] == 8192
    assert standalone["shared_centroid_bytes"] == 0
    assert standalone["shared_scale_bytes"] == 1024
    assert not cost_win_versus_both(341, 210, 338)
    assert strictly_below_cost_fraction(79, 100)
    assert not strictly_below_cost_fraction(80, 100)
    assert not strictly_below_cost_fraction(1.0, 0.0)
    samples = list(range(LATENCY_REPEATS))
    assert median_seconds(samples) == 15
    unresolved = decide_verdict(
        integrity_failure="checksum_mismatch",
        reproduction_ok=True,
        conditions=[
            {
                "admitted": True,
                "arm1_matches_or_beats_arm2": False,
                "arm3_matches_or_beats_arm2": False,
                "failure_clauses": [],
            }
        ],
    )
    assert unresolved == (VERDICT_UNRESOLVED, RECOMMENDATION_NONE, "checksum_mismatch")
    parent = decide_verdict(
        integrity_failure=None,
        reproduction_ok=False,
        conditions=[],
    )
    assert parent == (VERDICT_UNRESOLVED, RECOMMENDATION_NONE, CLAUSE_REPRODUCTION)
    admitted = {
        "admitted": True,
        "arm1_matches_or_beats_arm2": False,
        "arm3_matches_or_beats_arm2": False,
        "failure_clauses": [],
    }
    assert decide_verdict(
        integrity_failure=None,
        reproduction_ok=True,
        conditions=[admitted, admitted],
    ) == (VERDICT_SYMBOL, RECOMMENDATION_NONE, CLAUSE_GATE_MET)
    quantized = {
        "admitted": False,
        "arm1_matches_or_beats_arm2": True,
        "arm3_matches_or_beats_arm2": True,
        "failure_clauses": [CLAUSE_BELOW_STANDALONE, CLAUSE_BELOW_RESIDUAL],
    }
    assert decide_verdict(
        integrity_failure=None,
        reproduction_ok=True,
        conditions=[quantized, quantized],
    ) == (VERDICT_QUANTIZED, RECOMMENDATION_NONE, CLAUSE_QUANTIZED)
    one_condition = decide_verdict(
        integrity_failure=None,
        reproduction_ok=True,
        conditions=[admitted, quantized],
    )
    assert one_condition == (
        VERDICT_RESEARCH_FAIL,
        RECOMMENDATION_NONE,
        "symbol_accuracy_below_standalone_quant+symbol_accuracy_below_residual_only",
    )
    failed = decide_verdict(
        integrity_failure=None,
        reproduction_ok=True,
        conditions=[
            {
                "admitted": False,
                "arm1_matches_or_beats_arm2": False,
                "arm3_matches_or_beats_arm2": False,
                "failure_clauses": [CLAUSE_NO_COST],
            },
            quantized,
        ],
    )
    assert failed[0] == VERDICT_RESEARCH_FAIL
    assert failed[1] == RECOMMENDATION_NONE
    assert CLAUSE_NO_COST in failed[2]
    assert CLAUSE_BELOW_RESIDUAL in failed[2]


def test_research_module_is_not_imported_by_the_package():
    needles = (
        "import neuralese.symbol_addressing",
        "from neuralese.symbol_addressing",
        "from neuralese import symbol_addressing",
    )
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name == "symbol_addressing.py":
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in text
    cfg = LearnConfig()
    assert cfg.tau_residual == 0.55
    assert cfg.tau_kappa == 0.35
    assert __version__ == "0.7.0"


def test_missing_or_altered_fixture_stays_unresolved(tmp_path):
    missing = run_symbol_addressing(tmp_path / "no.npz")
    assert missing["verdict"] == VERDICT_UNRESOLVED
    assert missing["recommendation"] == RECOMMENDATION_NONE
    assert missing["authority"] == RUNTIME_AUTHORITY
    assert missing["reason"] == "fixture_missing"
    assert missing["residual_distinction_revoked"] is False
    assert missing["independent_target"]["status"] == "INDEPENDENT_TARGET_UNAVAILABLE"
    assert "condition_2" not in missing
    with np.load(FIXTURE_NPZ, allow_pickle=False) as bundle:
        arrays = {key: np.array(bundle[key]) for key in bundle.files}
    broken = {key: np.array(arrays[key]) for key in ACTIVATION_SHA256}
    broken["c2_train_activations"][0, 0] = np.float32(broken["c2_train_activations"][0, 0] + np.float32(1.0))
    bad_path = tmp_path / "bad.npz"
    np.savez(bad_path, **broken)
    mismatched = run_symbol_addressing(bad_path)
    assert mismatched["verdict"] == VERDICT_UNRESOLVED
    assert mismatched["reason"] == "checksum_mismatch"
    assert mismatched["recommendation"] == RECOMMENDATION_NONE
    assert "condition_2" not in mismatched


def _assert_fidelity(arm, mean_sq: float, normalized: float) -> None:
    fidelity = arm["reconstruction"]
    assert fidelity["distance"] == "mean_per_row_squared_euclidean"
    assert fidelity["mean_squared_euclidean"] == mean_sq
    assert fidelity["normalized_frobenius"] == normalized


def test_parent_counts_and_three_arm_ledger(report):
    parent = run_symbol_residual()
    assert parent["verdict"] == "RESIDUAL_DISTINCTION"
    assert parent["recommendation"] == "NONE"
    assert parent["condition_2"]["arms"]["symbol_only"]["n_correct"] == 38
    assert parent["condition_2"]["arms"]["symbol_plus_residual"]["n_correct"] == 57
    assert parent["condition_2"]["arms"]["residual_without_symbol"]["n_correct"] == 61
    assert parent["condition_2"]["arms"]["nearest_train_row"]["n_correct"] == 59
    assert parent["condition_4"]["arms"]["symbol_only"]["n_correct"] == 12
    assert parent["condition_4"]["arms"]["symbol_plus_residual"]["n_correct"] == 12
    assert parent["condition_4"]["arms"]["residual_without_symbol"]["n_correct"] == 14
    assert parent["condition_4"]["arms"]["nearest_train_row"]["n_correct"] == 14
    assert report["genuine"] is True
    assert report["reproduction_ok"] is True
    assert report["verdict"] == VERDICT_QUANTIZED
    assert report["recommendation"] == RECOMMENDATION_NONE
    assert report["authority"] == RUNTIME_AUTHORITY
    assert report["reason"] == CLAUSE_QUANTIZED
    assert report["entering_verdict"] == "RESIDUAL_DISTINCTION"
    assert report["residual_distinction_revoked"] is False
    assert report["probe_used_in_fit"] is False
    assert report["gloss_used_as_evidence"] is False
    assert report["semantic_interpretation"] is False
    assert report["runtime_integration"] is False
    assert report["production_admission"] is False
    assert report["heldout_refit"] is False
    assert report["new_model_training"] is False
    assert report["slot_task"] == "nearest_training_example_retrieval"
    assert report["slot_task_is_english_gloss"] is False
    assert report["slot_task_is_independently_decoded_semantic_state"] is False
    assert report["compared_to_semantic_decoder"] is False
    assert report["independent_target"]["satisfied"] is False
    assert report["independent_target"]["status"] == "INDEPENDENT_TARGET_UNAVAILABLE"
    assert report["reconstruction_is_a_gate"] is False
    assert report["coordinate_ops_are_a_gate"] is False
    assert report["checksums"] == dict(ACTIVATION_SHA256)
    assert report["reproduction"]["condition_2"] == {
        "n": 64,
        "symbol_only": 38,
        "symbol_plus_residual": 57,
        "residual_without_symbol": 61,
        "nearest_train_row": 59,
        "matches_published": True,
    }
    assert report["reproduction"]["condition_4"] == {
        "n": 16,
        "symbol_only": 12,
        "symbol_plus_residual": 12,
        "residual_without_symbol": 14,
        "nearest_train_row": 14,
        "matches_published": True,
    }
    assert report["symbol_beats_standalone_on_both"] is False
    assert report["symbol_beats_residual_only_on_both"] is False
    assert report["symbol_matches_or_beats_standalone_on_both"] is False
    assert report["symbol_matches_or_beats_residual_only_on_both"] is False
    expected = {
        "condition_2": {
            "standalone_int4": (58, 64, 0, 1024, 65, 64, 8320, 4096, 13440, 210.0, 70.0, 0, 8192, 1048576),
            "symbol_plus_residual": (57, 64, 8192, 1024, 66, 65, 8448, 4160, 21824, 341.0, 21824 / 192, 65536, 8192, 186368),
            "residual_only": (61, 64, 8192, 1024, 65, 64, 8320, 4096, 21632, 338.0, 21632 / 192, 65536, 8192, 1048576),
        },
        "condition_4": {
            "standalone_int4": (15, 16, 0, 1024, 65, 64, 2080, 1024, 4128, 258.0, 86.0, 0, 2048, 65536),
            "symbol_plus_residual": (12, 16, 8192, 1024, 66, 65, 2112, 1040, 12368, 773.0, 12368 / 48, 16384, 2048, 11648),
            "residual_only": (14, 16, 8192, 1024, 65, 64, 2080, 1024, 12320, 770.0, 12320 / 48, 16384, 2048, 65536),
        },
    }
    accuracy_failures = {
        "condition_2:symbol_accuracy_below_standalone_quant",
        "condition_2:symbol_accuracy_below_residual_only",
        "condition_4:symbol_accuracy_below_standalone_quant",
        "condition_4:symbol_accuracy_below_residual_only",
    }
    assert accuracy_failures <= set(report["explicit_failures"])
    for item in report["explicit_failures"]:
        assert item in accuracy_failures or item.endswith(":no_cost_win_versus_both_arms")
    for name, arms in expected.items():
        block = report[name]
        gate = block["gate"]
        assert gate["admitted"] is False
        assert gate["arm2_at_least_arm1"] is False
        assert gate["arm2_at_least_arm3_minus_tolerance"] is False
        assert gate["tolerance_vs_residual_only"] == 0.0
        assert gate["bytes_win_versus_both"] is False
        assert gate["arm1_matches_or_beats_arm2"] is True
        assert gate["arm3_matches_or_beats_arm2"] is True
        assert gate["symbol_beats_standalone"] is False
        assert gate["symbol_beats_residual_only"] is False
        assert isinstance(gate["latency_win_versus_both"], bool)
        assert isinstance(gate["cost_win"], bool)
        assert CLAUSE_BELOW_STANDALONE in gate["failure_clauses"]
        assert CLAUSE_BELOW_RESIDUAL in gate["failure_clauses"]
        assert block["learn_seconds_are_a_gate"] is False
        for arm_name, spec in arms.items():
            (
                correct,
                n,
                centroids,
                scales,
                train_row,
                query_row,
                train_store,
                query_bytes,
                system,
                admission,
                second,
                assign_ops,
                quant_ops,
                search_ops,
            ) = spec
            arm = block["arms"][arm_name]
            accounting = arm["accounting"]
            assert arm["n_correct"] == correct
            assert arm["n"] == n
            assert arm["accuracy"] == correct / n
            assert accounting["shared_centroid_bytes"] == centroids
            assert accounting["shared_scale_bytes"] == scales
            assert accounting["training_row_bytes"] == train_row
            assert accounting["query_row_bytes"] == query_row
            assert accounting["training_store_bytes"] == train_store
            assert accounting["query_payload_bytes"] == query_bytes
            assert accounting["shared_bytes"] == centroids + scales
            assert arm["system_bytes"] == system
            assert arm["system_bytes"] == centroids + scales + train_store + query_bytes
            assert arm["admission_storage_bytes"] == admission
            assert arm["second_amortization_bytes"] == second
            assert accounting["admission_storage_figure"] == ADMISSION_STORAGE_FIGURE
            assert accounting["second_amortization_is_a_gate"] is False
            assert accounting["parent_per_row_cap_is_this_trial_gate"] is False
            assert arm["assignment_coordinate_ops"] == assign_ops
            assert arm["quantize_coordinate_ops"] == quant_ops
            assert arm["search_coordinate_ops"] == search_ops
            assert arm["coordinate_ops_are_a_gate"] is False
            assert arm["reconstruction_is_a_gate"] is False
            assert arm["quantize_is_in_admission_latency"] is False
            assert arm["queries_with_distance_ties"] == 0
            assert arm["empty_symbol_rows"] == 0
            assert arm["latency_warmup"] == 3
            assert arm["latency_repeats"] == 31
            assert arm["median_assignment_seconds"] >= 0.0
            assert arm["median_quantize_seconds"] > 0.0
            assert arm["median_search_seconds"] > 0.0
            assert arm["admission_latency_seconds"] > 0.0
            assert math_finite(arm["admission_latency_seconds"])
    c2 = report["condition_2"]
    c4 = report["condition_4"]
    assert c2["arms"]["standalone_int4"]["median_assignment_seconds"] == 0.0
    assert c4["arms"]["standalone_int4"]["median_assignment_seconds"] == 0.0
    assert c2["arms"]["symbol_plus_residual"]["symbol_byte_stored"] is True
    assert c2["arms"]["residual_only"]["symbol_byte_stored"] is False
    assert c2["arms"]["residual_only"]["centroid_assignment_is_compute"] is True
    assert c2["arms"]["standalone_int4"]["centroid_assignment_is_compute"] is False
    _assert_fidelity(c2["arms"]["standalone_int4"], 55.89721257565742, 0.06973300427023361)
    _assert_fidelity(c2["arms"]["symbol_plus_residual"], 13.72172380594111, 0.03454996701578748)
    _assert_fidelity(c2["arms"]["residual_only"], 13.72172380594111, 0.03454996701578748)
    _assert_fidelity(c4["arms"]["standalone_int4"], 67.12070775462495, 0.07561271737035702)
    _assert_fidelity(c4["arms"]["symbol_plus_residual"], 182.00555880726898, 0.12451134043861779)
    _assert_fidelity(c4["arms"]["residual_only"], 182.00555880726898, 0.12451134043861779)
    bare2 = c2["arms"]["residual_only"]["reconstruction_without_centroid"]
    bare4 = c4["arms"]["residual_only"]["reconstruction_without_centroid"]
    assert bare2["mean_squared_euclidean"] == 11205.26815787999
    assert bare2["normalized_frobenius"] == 0.9873120271274967
    assert bare4["mean_squared_euclidean"] == 11479.817513712027
    assert bare4["normalized_frobenius"] == 0.9888583261121884
    assert c2["arms"]["residual_only"]["reconstruction_without_centroid_is_faithful"] is False
    assert c2["arms"]["residual_only"]["reconstruction_with_centroid_matches_symbol_arm"] is True
    assert c4["arms"]["residual_only"]["reconstruction_with_centroid_matches_symbol_arm"] is True
    assert c2["quantization_activation"]["train_code_min"] == -7
    assert c2["quantization_activation"]["train_code_max"] == 7
    assert c2["quantization_activation"]["held_code_min"] == -7
    assert c2["quantization_activation"]["held_code_max"] == 7
    assert c4["quantization_activation"]["train_code_min"] == -7
    assert c4["quantization_activation"]["held_code_max"] == 7
    assert c2["quantization_activation"]["fit_on"] == "train_activation_only"
    assert c2["quantization_activation"]["held_zero_code_fraction"] == 594 / 8192
    assert c4["quantization_activation"]["held_zero_code_fraction"] == 87 / 2048
    assert c2["quantization_activation"]["scale_min"] == 4.444333076477051
    assert c2["quantization_activation"]["scale_max"] == 59.3287239074707
    assert c4["quantization_activation"]["scale_min"] == 2.664839506149292
    assert c4["quantization_activation"]["scale_max"] == 53.46266555786133
    again = run_symbol_addressing()
    for name in ("condition_2", "condition_4"):
        for arm_name in ("standalone_int4", "symbol_plus_residual", "residual_only"):
            first = report[name]["arms"][arm_name]
            second = again[name]["arms"][arm_name]
            assert second["n_correct"] == first["n_correct"]
            assert second["system_bytes"] == first["system_bytes"]
            assert second["reconstruction"]["mean_squared_euclidean"] == first["reconstruction"]["mean_squared_euclidean"]
    arrays = np.load(FIXTURE_NPZ, allow_pickle=False)
    assert arrays["c2_train_activations"].shape == (128, 128)
    assert arrays["c2_heldout_activations"].shape == (64, 128)
    assert arrays["c4_train_activations"].shape == (32, 128)
    assert arrays["c4_heldout_activations"].shape == (16, 128)
    assert not any("logit" in key for key in arrays.files)


def math_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def test_note_names_the_frozen_verdict():
    note = NOTE.read_text(encoding="utf-8")
    assert "QUANTIZED_MEMORY_SUFFICIENT" in note
    assert "RESIDUAL_DISTINCTION" in note
    assert "is not revoked" in note
    assert "INDEPENDENT_TARGET_UNAVAILABLE" in note
    assert "RUNTIME_NOT_AUTHORIZED" in note
    assert "recommendation is **NONE**" in note
    assert "nearest-training-example" in note
    assert "not an English gloss" in note
    assert "not an independently decoded semantic state" in note
    assert "Anthropic NLA" in note
    assert "58/64" in note
    assert "57/64" in note
    assert "61/64" in note
    assert "15/16" in note
    assert "12/16" in note
    assert "14/16" in note
    assert "13440" in note
    assert "21824" in note
    assert "21632" in note
    assert "11205.26815787999" in note
    assert "11479.817513712027" in note
    assert "55.89721257565742" in note
    assert "quantized_memory_matches_or_beats_symbol_on_both_conditions" in note
    assert "symbol_accuracy_below_residual_only" in note
    assert "system bytes divided by the held-out count" in note
    assert "PASS_FOR_LATER_PR" not in note
    assert "LATER_ISOLATED_PR" not in note
    parent = PARENT_NOTE.read_text(encoding="utf-8")
    assert "This experimental slice is closed." in parent
    assert "PR1_SYMBOL_ADDRESSING.md" in parent
    assert "RESIDUAL_DISTINCTION stands and is not revoked." in parent
    assert "57/64" in parent
    assert "12/16" in parent
    assert "182.00555880726898" in parent

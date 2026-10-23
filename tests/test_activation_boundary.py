"""Activation-boundary distinguishability. The fixture is loaded with numpy only.

The capture script is not imported. A missing fixture stays UNRESOLVED. The
admission constants are the ones written above ``run_activation_boundary``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from neuralese import __version__
from neuralese.activation_boundary import (
    BASELINE_COMMIT,
    CONDITION_3_TOKEN_COORDINATE,
    CONTROLLED_PROPERTY,
    DISTANCE,
    DISTINCTION_GATE,
    DISTINCTION_MARGIN,
    ENTERING_RECOMMENDATION,
    ENTERING_VERDICT,
    FINAL_LAYER_INDEX,
    FIXTURE_NPZ,
    GEOMETRY_NN_FLOOR,
    GEOMETRY_SEPARATION_GATE,
    HELDOUT_FRAMES,
    INDEPENDENT_TARGET_STATUS,
    INTERMEDIATE_LAYER_INDEX,
    PARENT_COMMIT,
    RECOMMENDATION_NONE,
    RECOMMENDATION_RESEARCH_NOTE,
    RUNTIME_AUTHORITY,
    SHUFFLE_ASSIGNMENT_SEED,
    SLOT_INDEX,
    STATUS,
    TERMINAL_CLAIM_GATE,
    TERMINAL_CONCENTRATION_DROP,
    TRAIN_FRAMES,
    VERDICT_DISTINCTION,
    VERDICT_DISTINCTION_AND_TARGET,
    VERDICT_GEOMETRY_ONLY,
    VERDICT_RESEARCH_FAIL,
    VERDICT_UNRESOLVED,
    decide_verdict,
    design_fingerprint,
    run_activation_boundary,
    validate_design,
)
from neuralese.alphabet import LearnConfig
from neuralese.audit import certify
from neuralese.encode import ENCODER_CHAR_TRIGRAM

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "neuralese" / "activation_boundary.py"
CAPTURE = ROOT / "examples" / "activation_boundary" / "capture_once.py"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

PRODUCTION_SHA256 = {
    "src/neuralese/encode.py": "f28bcfc7ba05a3bdfe598adc9264f4f38a324d63c4fcd51425b076fe92e04a93",
    "src/neuralese/clustering.py": "ab84a98a459ebcd720833731891b05e9e5eaf411d2c964cbde5c8b54dcb8d321",
    "src/neuralese/alphabet.py": "b2bd36b60a96a38f965a22f287fc57137dd7491675d7a4c9b15c3ed89cc405a3",
    # schema 0.2 certificate fields. Encoder, clustering, alphabet, and taus are unchanged.
    "src/neuralese/audit.py": "c066e32fa313db137cee2872cfd0c65c7bb49f98f08800d4f7443408e011174b",
    "src/neuralese/__init__.py": "cda85f31c23360cd03ef5abaa7bdcdb2465d3aed0f04dd97d9f8af71cfef72af",
    "pyproject.toml": "f9ae447c44015663f31485a010162976473d0ad30f3cdb0ddbf22a6271315d73",
}


@pytest.fixture(scope="module")
def report():
    return run_activation_boundary()


def _assert_dist(block, mean: float, std: float, n: int) -> None:
    assert block["distance"] == "squared_euclidean"
    assert block["dispersion"] == "sample_standard_deviation_ddof_1"
    assert block["n"] == n
    assert block["mean"] == pytest.approx(mean, abs=1e-12)
    assert block["std"] == pytest.approx(std, abs=1e-12)


def test_gates_are_written_above_the_runner():
    text = SOURCE.read_text(encoding="utf-8")
    run_at = text.index("def run_activation_boundary")
    for gate in (
        "DISTINCTION_MARGIN = 0.10",
        "GEOMETRY_NN_FLOOR = 0.75",
        "TERMINAL_CONCENTRATION_DROP = 0.25",
        "INTERMEDIATE_LAYER_INDEX = 3",
        "FINAL_LAYER_INDEX = 6",
        "DISTINCTION_GATE",
        "GEOMETRY_SEPARATION_GATE",
        "TERMINAL_CLAIM_GATE",
        'INDEPENDENT_TARGET_STATUS = "INDEPENDENT_TARGET_UNAVAILABLE"',
        "SHUFFLE_ASSIGNMENT_SEED = 0",
        "N_SYMBOLS = 8",
        "CONTROLLED_PROPERTY",
        'DISTANCE = "squared_euclidean"',
        "SLOT_INDEX = 1",
    ):
        assert text.index(gate) < run_at
    assert DISTINCTION_MARGIN == 0.10
    assert GEOMETRY_NN_FLOOR == 0.75
    assert TERMINAL_CONCENTRATION_DROP == 0.25
    assert INTERMEDIATE_LAYER_INDEX == 3
    assert FINAL_LAYER_INDEX == 6
    assert SHUFFLE_ASSIGNMENT_SEED == 0
    assert SLOT_INDEX == 1
    assert DISTANCE == "squared_euclidean"
    assert "nearest_train_row" in DISTINCTION_GATE
    assert "at_least_floor" in GEOMETRY_SEPARATION_GATE
    assert "final_token" in TERMINAL_CLAIM_GATE
    assert INDEPENDENT_TARGET_STATUS == "INDEPENDENT_TARGET_UNAVAILABLE"
    assert CONDITION_3_TOKEN_COORDINATE == "condition_2_final_unpadded_position"
    assert "cat" in CONTROLLED_PROPERTY and "dog" in CONTROLLED_PROPERTY
    assert STATUS == "PLAN_ONLY"
    assert ENTERING_VERDICT == "UNRESOLVED"
    assert ENTERING_RECOMMENDATION == "NONE"
    assert RUNTIME_AUTHORITY == "RUNTIME_NOT_AUTHORIZED"
    assert PARENT_COMMIT == "852b681"
    assert BASELINE_COMMIT == "c6321d5"
    fit = text.split("def _fit_symbols", 1)[1].split("\ndef ", 1)[0]
    fit_body = fit.split('"""', 2)[-1]
    assert "probe" not in fit_body
    assert "token" not in fit_body
    assert text.count("kmeans(") == 1
    assert "independent_target_satisfied = False" in text


def test_verdict_table_does_not_read_a_score():
    missing = decide_verdict(
        genuine=False,
        condition_1_reproduced=True,
        distinction_gate=True,
        geometry_separates=True,
        independent_target_satisfied=True,
    )
    assert missing == (VERDICT_UNRESOLVED, RECOMMENDATION_NONE, "capture_not_genuine_or_not_reproducible")
    unreproduced = decide_verdict(
        genuine=True,
        condition_1_reproduced=False,
        distinction_gate=True,
        geometry_separates=True,
        independent_target_satisfied=False,
    )
    assert unreproduced == (
        VERDICT_UNRESOLVED,
        RECOMMENDATION_NONE,
        "condition_1_parent_boundary_not_reproduced",
    )
    both = decide_verdict(
        genuine=True,
        condition_1_reproduced=True,
        distinction_gate=True,
        geometry_separates=True,
        independent_target_satisfied=True,
    )
    assert both == (
        VERDICT_DISTINCTION_AND_TARGET,
        RECOMMENDATION_RESEARCH_NOTE,
        "distinction_gate_and_independent_target",
    )
    distinction_only = decide_verdict(
        genuine=True,
        condition_1_reproduced=True,
        distinction_gate=True,
        geometry_separates=True,
        independent_target_satisfied=False,
    )
    assert distinction_only == (
        VERDICT_DISTINCTION,
        RECOMMENDATION_NONE,
        "distinction_gate_passed_without_independent_target",
    )
    geometry = decide_verdict(
        genuine=True,
        condition_1_reproduced=True,
        distinction_gate=False,
        geometry_separates=True,
        independent_target_satisfied=False,
    )
    assert geometry == (
        VERDICT_GEOMETRY_ONLY,
        RECOMMENDATION_NONE,
        "geometry_separates_symbols_miss_margin",
    )
    failed = decide_verdict(
        genuine=True,
        condition_1_reproduced=True,
        distinction_gate=False,
        geometry_separates=False,
        independent_target_satisfied=False,
    )
    assert failed == (
        VERDICT_RESEARCH_FAIL,
        RECOMMENDATION_NONE,
        "symbols_and_geometry_missed_controlled_distinction",
    )


def test_production_encoder_and_taus_are_unchanged():
    for relative, digest in PRODUCTION_SHA256.items():
        data = (ROOT / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
    clustering = (ROOT / "src" / "neuralese" / "clustering.py").read_text(encoding="utf-8")
    assert "seeds = _farthest_first(X, k)" in clustering
    cfg = LearnConfig()
    assert cfg.encoder == ENCODER_CHAR_TRIGRAM
    assert cfg.tau_residual == 0.55
    assert cfg.tau_kappa == 0.35
    assert cfg.n_symbols == 8
    assert __version__ == "0.7.0"
    assert certify.__kwdefaults__ == {"require_gloss": True, "tau_residual": 0.55}


def test_tests_do_not_download_weights():
    research = SOURCE.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    capture = CAPTURE.read_text(encoding="utf-8")
    for banned in ("from_pretrained", "import torch", "urlretrieve", "import urllib", "requests.get"):
        assert banned not in research
        assert banned not in workflow
    assert "activation_boundary" not in workflow
    assert "import torch" in capture
    assert "urlretrieve" not in capture
    assert "required=True" in capture


def test_prompts_were_fixed_disjoint():
    validate_design()
    assert len(TRAIN_FRAMES) == 16
    assert len(HELDOUT_FRAMES) == 8
    assert set(TRAIN_FRAMES).isdisjoint(set(HELDOUT_FRAMES))
    fingerprint = design_fingerprint()
    assert fingerprint == "5fcef56e50adf6e5eab2ea643dcebefc02fee26839a8df9e49ddcd1bdfd45db3"


def test_missing_fixture_stays_unresolved(tmp_path):
    report = run_activation_boundary(tmp_path / "no.npz", tmp_path / "no.json")
    assert report["verdict"] == VERDICT_UNRESOLVED
    assert report["recommendation"] == RECOMMENDATION_NONE
    assert report["authority"] == RUNTIME_AUTHORITY
    assert report["blocker"] == "fixture_missing"
    assert report["independent_target"]["status"] == "INDEPENDENT_TARGET_UNAVAILABLE"
    assert report["probe_used_in_fit"] is False
    assert report["gloss_used_as_evidence"] is False


def test_frozen_boundary_numbers(report):
    assert report["genuine"] is True
    assert report["condition_1_reproduced"] is True
    assert report["explicit_failures"] == []
    assert report["verdict"] == VERDICT_GEOMETRY_ONLY
    assert report["recommendation"] == RECOMMENDATION_NONE
    assert report["reason"] == "geometry_separates_symbols_miss_margin"
    assert report["authority"] == RUNTIME_AUTHORITY
    assert report["distinction_gate"] is False
    assert report["geometry_separates"] is True
    assert report["independent_target"]["satisfied"] is False
    assert report["intermediate_layer_index"] == 3
    assert report["final_layer_index"] == 6
    assert report["probe_used_in_fit"] is False
    reproduction = report["condition_1"]["reproduction"]
    assert reproduction["matched"] is True
    assert reproduction["train_counts"] == [1, 2, 5, 1, 3, 3, 16, 1]
    assert reproduction["heldout_counts"] == [0, 0, 4, 0, 0, 1, 11, 0]
    assert reproduction["train_residual"] == pytest.approx(0.22755261731526205, abs=1e-12)
    assert reproduction["heldout_prototype_residual"] == pytest.approx(0.29343442149783966, abs=1e-12)
    assert reproduction["heldout_train_mean_residual"] == pytest.approx(0.30512149677079187, abs=1e-12)
    assert reproduction["heldout_shuffled_assignment_residual"] == pytest.approx(0.33508908512206886, abs=1e-12)
    claim = report["terminal_claim"]
    assert claim["allowed"] is False
    assert claim["heldout_max_fraction_condition_1"] == pytest.approx(0.6875, abs=0.0)
    assert claim["heldout_max_fraction_swap"] == pytest.approx(0.5, abs=0.0)
    assert claim["heldout_fraction_drop"] == pytest.approx(0.1875, abs=0.0)
    assert claim["required_drop"] == 0.25
    assert claim["prefixes_differ_only_at_final_token"] is True
    held = {
        "condition_1": ([0, 0, 4, 0, 0, 1, 11, 0], 11, 16, 3),
        "condition_2": ([16, 9, 0, 3, 16, 7, 8, 5], 16, 64, 7),
        "condition_3": ([14, 0, 16, 6, 2, 10, 14, 2], 16, 64, 7),
        "condition_4": ([3, 0, 7, 2, 0, 0, 3, 1], 7, 16, 5),
        "parent_terminal_swap": ([8, 0, 2, 0, 4, 0, 0, 2], 8, 16, 4),
    }
    for name, (counts, max_count, n, occupied) in held.items():
        block = report[name]["heldout_concentration"]
        assert block["counts"] == counts
        assert block["max_count"] == max_count
        assert block["n"] == n
        assert block["occupied"] == occupied
        assert report[name]["heldout_on_empty_train_clusters"] == 0
    assert report["condition_1"]["token_heldout"]["ids"] == [15]
    assert report["condition_4"]["token_heldout"]["ids"] == [15]
    assert report["condition_2"]["token_heldout"]["ids"] == [2, 27, 28, 32]
    assert report["parent_terminal_swap"]["token_heldout"]["ids"] == [2, 27, 28, 32]
    assert report["condition_1"]["token_heldout"]["entropy_bits"] == 0.0
    assert report["condition_2"]["token_heldout"]["entropy_bits"] == pytest.approx(2.0, abs=0.0)
    distinctions = {
        "condition_2": (38, 64, 7, 32, False, True, 59, 0.09375, 0.0, -0.328125),
        "condition_3": (32, 64, 0, 32, False, False, 44, 0.0, 0.0, -0.1875),
        "condition_4": (12, 16, 5, 8, False, True, 14, 0.25, 0.25, -0.125),
    }
    for name, (correct, n, n_split, n_pairs, beats, floor, nn_correct, mean_margin, shuffle_margin, nn_margin) in distinctions.items():
        distinction = report[name]["distinction"]
        assert distinction["beats_gate"] is beats
        assert distinction["nearest_train_clears_floor"] is floor
        assert distinction["required_margin"] == 0.10
        assert distinction["symbol"]["n_correct"] == correct
        assert distinction["symbol"]["n"] == n
        assert distinction["symbol"]["pair_split"]["n_split"] == n_split
        assert distinction["symbol"]["pair_split"]["n_pairs"] == n_pairs
        assert distinction["symbol"]["unseen_symbol_rows"] == 0
        assert distinction["baselines"]["train_mean"]["n_correct"] == n // 2
        assert distinction["baselines"]["token_identity"]["n_correct"] == n // 2
        assert distinction["baselines"]["token_identity"]["pair_split"]["n_split"] == 0
        assert distinction["baselines"]["nearest_train_row"]["n_correct"] == nn_correct
        assert distinction["margin_over_baseline"]["train_mean"] == pytest.approx(mean_margin, abs=0.0)
        assert distinction["margin_over_baseline"]["shuffled_assignment"] == pytest.approx(shuffle_margin, abs=0.0)
        assert distinction["margin_over_baseline"]["nearest_train_row"] == pytest.approx(nn_margin, abs=0.0)
    assert report["condition_2"]["distinction"]["baselines"]["shuffled_assignment"]["n_correct"] == 38
    assert report["condition_2"]["distinction"]["baselines"]["shuffled_assignment"]["pair_split"]["n_split"] == 26
    assert report["condition_3"]["distinction"]["baselines"]["shuffled_assignment"]["pair_split"]["n_split"] == 27
    assert report["condition_4"]["distinction"]["baselines"]["shuffled_assignment"]["pair_split"]["n_split"] == 6
    assert report["condition_4"]["distinction"]["margin_over_baseline"]["token_identity"] == pytest.approx(0.25, abs=0.0)
    _assert_dist(report["condition_3"]["pair_geometry_heldout"]["within_pair"], 0.0027677179942251794, 0.0019174321423866803, 32)
    _assert_dist(report["condition_3"]["pair_geometry_heldout"]["nonpair"], 0.6437076042869043, 0.25002287694757136, 1984)
    _assert_dist(report["condition_2"]["pair_geometry_heldout"]["within_pair"], 178.10759597811648, 78.21793798242732, 32)
    _assert_dist(report["condition_4"]["pair_geometry_heldout"]["within_pair"], 288.74968795660504, 72.16276942223688, 8)
    _assert_dist(report["between_condition"]["condition_2_vs_condition_3_train"], 11472.302718528139, 811.68635546051, 128)
    _assert_dist(report["between_condition"]["condition_1_vs_parent_terminal_swap_heldout"], 3714.872117149182, 1848.5229244146144, 16)
    _assert_dist(report["condition_1"]["within_heldout"], 2177.8451223707066, 849.7528653780615, 120)
    assert report["condition_1"]["cost"]["capture_seconds"] == pytest.approx(0.24427694100086228, abs=0.0)
    assert report["condition_3"]["cost"]["capture_seconds"] == pytest.approx(0.0030824560271867085, abs=0.0)
    assert report["condition_4"]["cost"]["capture_seconds"] == pytest.approx(0.24731432801127085, abs=0.0)
    assert report["parent_terminal_swap"]["cost"]["capture_seconds"] == pytest.approx(0.28465284099729615, abs=0.0)
    assert report["capture_stack"]["shared_forward_seconds"] == pytest.approx(1.0315953350218479, abs=0.0)
    assert report["capture_stack"]["download_seconds"] == 0.0
    for name in ("condition_1", "condition_2", "condition_3", "condition_4", "parent_terminal_swap"):
        assert report[name]["cost"]["learn_seconds"] > 0
        assert report[name]["cost"]["translate_seconds"] > 0
    assert report["checksums"]["c2_train_activations"] == (
        "f9140300c4308ea67da83fae727b67ce93cd26fb55c396d9f57139fcb3ede999"
    )
    assert report["checksums"]["c3_train_activations"] == (
        "786b78db46f40f8314cd2d9587954c7999f988294ac973ef77548b97bd636b60"
    )
    arrays = np.load(FIXTURE_NPZ, allow_pickle=False)
    assert not any("logit" in key for key in arrays.files)
    assert arrays["c2_train_activations"].shape == (128, 128)
    assert arrays["c3_train_activations"].shape == (128, 128)
    assert not np.array_equal(arrays["c2_train_activations"], arrays["c3_train_activations"])
    assert int(np.asarray(arrays["c3_layer_index"]).reshape(-1)[0]) == 3
    assert int(np.asarray(arrays["c2_layer_index"]).reshape(-1)[0]) == 6
    assert arrays["c2_train_probe"].dtype == np.int32
    assert set(arrays["c2_train_probe"].tolist()) == {0, 1}


def test_note_names_the_frozen_verdict():
    note = (ROOT / "docs" / "PR1_ACTIVATION_BOUNDARY.md").read_text(encoding="utf-8")
    assert "GEOMETRY_ONLY" in note
    assert "INDEPENDENT_TARGET_UNAVAILABLE" in note
    assert "RUNTIME_NOT_AUTHORIZED" in note
    assert "not attributed to tokenizer id 15" in note
    assert "PASS_FOR_LATER_PR" not in note
    assert "0.1875" in note
    assert "0.09375" in note
    assert "cf967c0a9a04383db6f7b1108d86b2962634b4ac" in note

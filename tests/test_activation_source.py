"""Activation-source feasibility. The fixture is loaded with numpy only.

The capture script is not imported. A missing fixture stays UNRESOLVED. The
admission constants are the ones written above ``run_activation_source``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from neuralese import __version__
from neuralese.activation_source import (
    BASELINE_COMMIT,
    DOWNSTREAM_RELATIVE_MARGIN,
    ENTERING_RECOMMENDATION,
    ENTERING_VERDICT,
    FIXTURE_NPZ,
    GEOMETRY_GATE,
    HELDOUT_PROMPTS,
    LAYER_RULE,
    MODEL_ID,
    MODEL_REVISION,
    RECOMMENDATION_LATER_PR,
    RECOMMENDATION_NONE,
    RUNTIME_AUTHORITY,
    SHUFFLED_ASSIGNMENT_GATE,
    STATUS,
    TOKEN_RULE,
    TRAIN_PROMPTS,
    VERDICT_GEOMETRY_ONLY,
    VERDICT_PASS,
    VERDICT_STRUCTURAL_FAILURE,
    VERDICT_UNRESOLVED,
    WEIGHT_SHA256,
    assess_fixture,
    decide_verdict,
    downstream_beats,
    geometry_beats,
    prompt_fingerprint,
    run_activation_source,
    validate_prompts,
)
from neuralese.alphabet import LearnConfig
from neuralese.audit import certify
from neuralese.encode import ENCODER_CHAR_TRIGRAM

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "neuralese" / "activation_source.py"
CAPTURE = ROOT / "examples" / "activation_source" / "capture_once.py"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

PRODUCTION_SHA256 = {
    "src/neuralese/encode.py": "f28bcfc7ba05a3bdfe598adc9264f4f38a324d63c4fcd51425b076fe92e04a93",
    "src/neuralese/clustering.py": "ab84a98a459ebcd720833731891b05e9e5eaf411d2c964cbde5c8b54dcb8d321",
    "src/neuralese/alphabet.py": "b2bd36b60a96a38f965a22f287fc57137dd7491675d7a4c9b15c3ed89cc405a3",
    # schema 0.2 certificate fields. Encoder, clustering, alphabet, and taus are unchanged.
    "src/neuralese/audit.py": "0ce09954c93eff79269854e28024df10abfb394a47f38e205f4d8e288a950467",
    "src/neuralese/__init__.py": "cda85f31c23360cd03ef5abaa7bdcdb2465d3aed0f04dd97d9f8af71cfef72af",
    "pyproject.toml": "f9ae447c44015663f31485a010162976473d0ad30f3cdb0ddbf22a6271315d73",
}


@pytest.fixture(scope="module")
def report():
    return run_activation_source()


def test_gates_are_written_above_the_runner():
    text = SOURCE.read_text(encoding="utf-8")
    run_at = text.index("def run_activation_source")
    for gate in (
        "DOWNSTREAM_RELATIVE_MARGIN = 0.05",
        "GEOMETRY_GATE",
        "SHUFFLED_ASSIGNMENT_GATE",
        "GLOBAL_MEAN_GATE",
        "SHUFFLE_ASSIGNMENT_SEED = 0",
        "SHUFFLE_SYMBOL_SEED = 1",
        "LAYER_RULE",
        "TOKEN_RULE",
        "WEIGHT_SHA256",
    ):
        assert text.index(gate) < run_at
    assert DOWNSTREAM_RELATIVE_MARGIN == 0.05
    assert "strictly_below" in GEOMETRY_GATE
    assert "permuting_" in SHUFFLED_ASSIGNMENT_GATE
    assert BASELINE_COMMIT == "c6321d5"
    assert ENTERING_VERDICT == "UNRESOLVED"
    assert ENTERING_RECOMMENDATION == "NONE"
    assert RUNTIME_AUTHORITY == "RUNTIME_NOT_AUTHORIZED"
    assert STATUS == "RESEARCH_ONLY"
    assert MODEL_ID == "EleutherAI/pythia-14m"
    assert MODEL_REVISION == "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
    assert WEIGHT_SHA256 == "116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1"
    assert LAYER_RULE == "last tensor in outputs.hidden_states"
    assert "add_special_tokens false" in TOKEN_RULE


def test_verdict_table_does_not_read_a_score():
    missing = decide_verdict(
        genuine=False,
        structural_pass=True,
        geometry_ok=True,
        downstream_present=True,
        downstream_ok=True,
        fixture_reproducible=True,
    )
    assert missing == (VERDICT_UNRESOLVED, RECOMMENDATION_NONE, "capture_not_genuine_or_not_reproducible")
    structural = decide_verdict(
        genuine=True,
        structural_pass=False,
        geometry_ok=True,
        downstream_present=True,
        downstream_ok=True,
        fixture_reproducible=True,
    )
    assert structural == (
        VERDICT_STRUCTURAL_FAILURE,
        RECOMMENDATION_NONE,
        "train_pack_failed_structural_certify",
    )
    controls = decide_verdict(
        genuine=True,
        structural_pass=True,
        geometry_ok=False,
        downstream_present=True,
        downstream_ok=True,
        fixture_reproducible=True,
    )
    assert controls[0] == VERDICT_UNRESOLVED
    assert controls[1] == RECOMMENDATION_NONE
    geometry_only = decide_verdict(
        genuine=True,
        structural_pass=True,
        geometry_ok=True,
        downstream_present=False,
        downstream_ok=False,
        fixture_reproducible=True,
    )
    assert geometry_only == (
        VERDICT_GEOMETRY_ONLY,
        RECOMMENDATION_NONE,
        "geometry_without_downstream_quantity",
    )
    downstream = decide_verdict(
        genuine=True,
        structural_pass=True,
        geometry_ok=True,
        downstream_present=True,
        downstream_ok=False,
        fixture_reproducible=True,
    )
    assert downstream[0] == VERDICT_UNRESOLVED
    assert downstream[1] == RECOMMENDATION_NONE
    passed = decide_verdict(
        genuine=True,
        structural_pass=True,
        geometry_ok=True,
        downstream_present=True,
        downstream_ok=True,
        fixture_reproducible=True,
    )
    assert passed == (VERDICT_PASS, RECOMMENDATION_LATER_PR, "all_admission_gates_passed")
    assert geometry_beats(0.2, 0.2, 0.1) is False
    assert geometry_beats(0.2, 0.3, 0.2) is False
    assert geometry_beats(0.2, 0.3, 0.4) is True
    assert downstream_beats(0.96, 1.0) is False
    assert downstream_beats(0.95, 1.0) is True
    assert downstream_beats(0.1, 0.0) is False


def test_production_encoder_and_taus_are_unchanged():
    for relative, digest in PRODUCTION_SHA256.items():
        data = (ROOT / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
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
    assert "capture_once" not in workflow
    assert "pythia" not in workflow
    assert "import torch" in capture
    assert "urlretrieve" in capture


def test_missing_fixture_stays_unresolved(tmp_path):
    missing = assess_fixture(tmp_path / "no.npz", tmp_path / "no.json")
    assert missing["genuine"] is False
    assert missing["blocker"] == "fixture_missing"
    report = run_activation_source(tmp_path / "no.npz", tmp_path / "no.json")
    assert report["verdict"] == VERDICT_UNRESOLVED
    assert report["recommendation"] == RECOMMENDATION_NONE
    assert report["blocker"] == "fixture_missing"
    assert report["authority"] == RUNTIME_AUTHORITY
    assert report["gloss_used_as_evidence"] is False


def test_prompts_were_fixed_disjoint():
    validate_prompts()
    assert len(TRAIN_PROMPTS) == 32
    assert len(HELDOUT_PROMPTS) == 16
    assert set(TRAIN_PROMPTS).isdisjoint(HELDOUT_PROMPTS)
    fingerprint = prompt_fingerprint()
    assert len(fingerprint) == 64


def test_fixture_is_a_real_capture_and_the_verdict_follows_the_gates(report):
    assert FIXTURE_NPZ.is_file()
    assert report["genuine"] is True
    assert report["fixture_reproducible"] is True
    assert report["blocker"] is None
    assert report["model_id"] == MODEL_ID
    assert report["revision"] == MODEL_REVISION
    assert report["layer_rule"] == LAYER_RULE
    assert report["token_rule"] == TOKEN_RULE
    assert report["hidden_state_index"] == report["hidden_states_tuple_length"] - 1
    assert report["activation_shape"] == [32, 128]
    assert report["logit_shape"] == [16, 50304]
    assert report["encoder"] == ENCODER_CHAR_TRIGRAM
    assert report["tau_residual"] == 0.55
    assert report["tau_kappa"] == 0.35
    assert report["weight_sha256"] == WEIGHT_SHA256
    assert report["gloss_used_as_evidence"] is False
    assert report["downstream"]["present"] is True
    assert report["downstream"]["used_in_fit"] is False
    assert report["downstream"]["quantity"] == "next_token_logits_at_the_same_final_position"
    assert report["unknown_state"] == "unknown"
    assert report["receipt_steps"] == ["factorization", "clustering", "finalize"]
    assert report["authority"] == RUNTIME_AUTHORITY
    assert report["verdict"] == VERDICT_PASS
    assert report["recommendation"] == RECOMMENDATION_LATER_PR
    assert report["reason"] == "all_admission_gates_passed"
    assert report["structural_pass"] is True
    assert report["certify_passed"] is True
    assert report["certify_failures"] == []
    assert report["explicit_failures"] == []
    assert report["decision"] == "accept"
    assert report["codebook_size"] == 8
    assert report["n_symbols"] == 8
    assert report["n_live"] == 5
    assert report["n_quarantined"] == 3
    assert report["n_empty_clusters"] == 0
    assert report["hidden_state_index"] == 6
    assert report["hidden_states_tuple_length"] == 7
    assert report["prompt_fingerprint"] == (
        "1780cae1dc80350e1e10c4437f7d8b7d4a2cd5c818a0e0d846febb6a84c7bfba"
    )
    assert report["train_residual"] == pytest.approx(0.22755261731526205, abs=1e-12)
    assert report["cluster_residual"] == pytest.approx(report["train_residual"], abs=1e-12)
    assert report["kappa_avg"] == pytest.approx(0.9832387137375892, abs=1e-12)
    geometry = report["geometry"]
    assert geometry["prototype_residual"] == pytest.approx(0.29343442149783966, abs=1e-12)
    assert geometry["global_mean_residual"] == pytest.approx(0.30512149677079187, abs=1e-12)
    assert geometry["shuffled_assignment_residual"] == pytest.approx(0.33508908512206886, abs=1e-12)
    assert geometry["beats"] is True
    downstream = report["downstream"]
    assert downstream["true_residual"] == pytest.approx(0.2974424885101723, abs=1e-12)
    assert downstream["shuffled_symbol_residual"] == pytest.approx(0.34492301344576626, abs=1e-12)
    assert downstream["relative_drop"] == pytest.approx(0.13765542768882114, abs=1e-12)
    assert downstream["margin"] == 0.05
    assert downstream["beats"] is True
    assert downstream["empty_symbol_fallbacks"] == 0
    assert report["stability"]["nearest_centroid_agreement"] == 1.0
    assert report["stability"]["seed_agreement"] == 1.0
    assert report["stability"]["subsample_agreement"] == 0.8125
    assert report["cost_seconds"]["capture"] == pytest.approx(0.24427694100086228, abs=0.0)
    assert report["peak_memory"]["capture_rss_kb"] == 499580
    assert report["checksums"]["train_activations"] == (
        "487e55dfd8a8712a572fcbeed4226ec50dfcfe40fd19fbf321c118a8de87cc84"
    )
    assert report["checksums"]["heldout_activations"] == (
        "c2fe5cf25f106fea4f64ccc33e67b95b3a5538e3e1824ea2d8aac7b5a4ee00ec"
    )
    geometry = report["geometry"]
    assert geometry["beats"] is geometry_beats(
        geometry["prototype_residual"],
        geometry["global_mean_residual"],
        geometry["shuffled_assignment_residual"],
    )
    downstream = report["downstream"]
    assert downstream["beats"] is downstream_beats(
        downstream["true_residual"],
        downstream["shuffled_symbol_residual"],
    )
    for key in ("capture", "cluster", "lookup", "certify"):
        assert report["cost_seconds"][key] > 0
        assert report["cost_per_observation_seconds"][key] > 0
    assert report["peak_memory"]["capture_rss_kb"] > 0
    assert report["peak_memory"]["trial_tracemalloc_bytes"] > 0
    assert 0.0 <= report["stability"]["nearest_centroid_agreement"] <= 1.0
    assert 0.0 <= report["stability"]["seed_agreement"] <= 1.0
    assert 0.0 <= report["stability"]["subsample_agreement"] <= 1.0
    arrays = np.load(FIXTURE_NPZ, allow_pickle=False)
    assert arrays["train_activations"].dtype == np.float32
    assert arrays["train_logits"].dtype == np.float32
    assert arrays["train_activations"].shape[1] != 32
    assert set(arrays["train_last_token_id"].tolist()) == {15}
    assert set(arrays["heldout_last_token_id"].tolist()) == {15}


def test_note_and_roadmap_name_the_same_verdict():
    note = (ROOT / "docs" / "PR1_ACTIVATION_SOURCE.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    assert "PASS_FOR_LATER_PR" in note
    assert "LATER_ISOLATED_PR" in note
    assert "RUNTIME_NOT_AUTHORIZED" in note
    assert "EleutherAI/pythia-14m" in note
    assert "cf967c0a9a04383db6f7b1108d86b2962634b4ac" in note
    assert "text-encoder trials stay frozen" in roadmap
    assert "PR1_ACTIVATION_SOURCE.md" in roadmap
    assert "PASS_FOR_LATER_PR" in roadmap

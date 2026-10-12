"""OOV coverage trial. Hashed arms stay out of the production encoder.

The synthetic locks are representation properties: one string hashed twice
is one vector, distinct nonsense strings do not all share one vector, and a
planted hash-bucket collision is detected. ``word_sentence_svd`` stays the
function already in ``encode.py``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from neuralese import __version__
from neuralese.encode import word_sentence_svd
from neuralese.oov_coverage import (
    ARM4_RUN_MARKER,
    BASELINE_COMMIT,
    CHAR_NGRAM,
    COLLISION_SEARCH_LIMIT,
    DECISION_NO_ADMISSION,
    DECISION_RECOMMENDED,
    DECISION_STATUSES,
    DECISION_UNRESOLVED,
    DISTINCTION_MIN,
    ENTERING_RECOMMENDATION,
    FROZEN_SPLIT_SHA256,
    HASH_DIM,
    NONSENSE_TEXTS,
    PURITY_ABOVE_ARM1_GATE,
    PURITY_ABOVE_BASE_GATE,
    REPEATED_TEXT,
    RUNTIME_AUTHORITY,
    RUNTIME_FACTOR,
    WORD_TOKEN_PATTERN,
    ZERO_RATE_GATE,
    _buckets,
    arm4_precondition,
    decide_arm,
    decide_trial,
    embed_char,
    embed_combined,
    embed_word,
    hash_collision_fixture,
    justification_in_doc,
    run_oov_coverage,
    score_arm4,
)

ROOT = Path(__file__).resolve().parents[1]
WORD_SENTENCE_SVD_SHA256 = "e4a57d3d0618f2544e0cdf3b221e2bdf500de4c20c4cf9967d255c9302b6ec95"
GOLDEN_SVD = [
    [0.7959177511269515, -0.0, 0.6054047682674922, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.7959177511269524, -0.0, -0.6054047682674913, 0.0, 0.0, 0.0, 0.0, 0.0],
]

# eval hits, base hits, zero rows, label hits, row hits, n_eval
CHAR_HITS = {
    "uneven": (5, 5, 0, 5, 5, 10),
    "overlapping": (3, 2, 0, 1, 2, 9),
    "ambiguous": (2, 3, 0, 2, 2, 9),
    "balanced": (3, 2, 0, 3, 1, 16),
}
WORD_HITS = {
    "uneven": (5, 5, 0, 5, 5, 10),
    "overlapping": (1, 2, 0, 3, 0, 9),
    "ambiguous": (1, 3, 0, 2, 3, 9),
    "balanced": (2, 2, 0, 1, 1, 16),
}


@pytest.fixture(scope="module")
def report():
    return run_oov_coverage(time_runs=0)


def _arm(report, arm_id):
    for arm in report["arms"]:
        if arm["arm_id"] == arm_id:
            return arm
    raise AssertionError(arm_id)


def test_admission_constants_precede_hashed_purity():
    assert HASH_DIM == 32
    assert CHAR_NGRAM == 3
    assert WORD_TOKEN_PATTERN == "[a-z0-9']+"
    assert DISTINCTION_MIN == 0.95
    assert RUNTIME_FACTOR == 5.0
    assert COLLISION_SEARCH_LIMIT == 4096
    assert "strictly_greater_than_arm1" in PURITY_ABOVE_ARM1_GATE
    assert "strictly_greater_than_eval_majority_base_rate" in PURITY_ABOVE_BASE_GATE
    assert "strictly_below" in (ROOT / "src" / "neuralese" / "oov_coverage.py").read_text(encoding="utf-8")
    assert ZERO_RATE_GATE == "zero_vector_rate_strictly_below_arm1"
    assert BASELINE_COMMIT == "9040bb4"
    assert FROZEN_SPLIT_SHA256 == "55f561c832ef47d453827b3fc9704cf4d98e66d0bda2e4ff4aa9603c5923c20e"
    assert ENTERING_RECOMMENDATION == "NONE"
    assert RUNTIME_AUTHORITY == "RUNTIME_NOT_AUTHORIZED"
    assert "INDUCTIVE_PASS" not in DECISION_STATUSES
    assert __version__ == "0.7.0"
    source = (ROOT / "src" / "neuralese" / "oov_coverage.py").read_text(encoding="utf-8")
    assert source.index("DISTINCTION_MIN = 0.95") < source.index("def run_oov_coverage")
    assert source.index("RUNTIME_FACTOR = 5.0") < source.index("def run_oov_coverage")
    assert source.index("def decide_arm") < source.index("def run_oov_coverage")
    assert source.index("def decide_trial") < source.index("def _score_hashed")
    assert "def word_sentence_svd" not in source
    assert "def kmeans" not in source
    assert "def certify" not in source
    assert "learn_pack(" not in source
    assert "0.7.0" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    clustering = (ROOT / "src" / "neuralese" / "clustering.py").read_text(encoding="utf-8")
    assert "seeds = _farthest_first(X, k)" in clustering
    alphabet = (ROOT / "src" / "neuralese" / "alphabet.py").read_text(encoding="utf-8")
    assert "tau_kappa: float = 0.35" in alphabet
    assert "tau_residual: float = 0.55" in alphabet


def test_production_word_sentence_svd_is_unchanged():
    encode = (ROOT / "src" / "neuralese" / "encode.py").read_text(encoding="utf-8")
    start = encode.index("def word_sentence_svd")
    end = encode.index("\ndef embed_texts")
    body = encode[start:end]
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == WORD_SENTENCE_SVD_SHA256
    texts = [
        "The skiff rode the tide beside the wharf.",
        "The kiln held the glaze after the wheel stopped.",
        "The skiff line met the buoy at the wharf.",
    ]
    rows = word_sentence_svd(texts, rank=8)
    assert rows == GOLDEN_SVD


def test_repeated_input_collides_for_the_hash_arms():
    for embed in (embed_char, embed_word, embed_combined):
        matrix = embed([REPEATED_TEXT, REPEATED_TEXT])
        assert matrix.shape == (2, HASH_DIM)
        assert np.array_equal(matrix[0], matrix[1])
        assert not np.all(matrix[0] == 0.0)


def test_distinct_nonsense_strings_do_not_all_collapse():
    assert len(set(NONSENSE_TEXTS)) == len(NONSENSE_TEXTS) >= 2
    for embed in (embed_char, embed_word, embed_combined):
        matrix = embed(NONSENSE_TEXTS)
        first = matrix[0]
        assert not all(np.array_equal(first, row) for row in matrix[1:])
        assert int(np.all(matrix == 0.0, axis=1).sum()) == 0


def test_planted_hash_collision_is_detected_or_impossible():
    fixture = hash_collision_fixture()
    for kind in ("word", "char"):
        hit = fixture[kind]
        assert hit["detected"] or hit["impossible"]
        if hit["detected"]:
            assert hit["left"] != hit["right"]
            assert hit["bucket"] in _buckets(kind, hit["left"])
            assert hit["bucket"] in _buckets(kind, hit["right"])
        else:
            assert hit["impossible"] is True
    disjoint = fixture["disjoint"]
    assert disjoint is not None
    assert disjoint["left"] != disjoint["right"]
    assert _buckets("word", disjoint["left"]).isdisjoint(_buckets("word", disjoint["right"]))
    assert _buckets("char", disjoint["left"]).isdisjoint(_buckets("char", disjoint["right"]))


def test_decide_arm_requires_controls_and_every_gate():
    assert decide_arm(
        fixtures=[{"admitted": True}],
        controls_fall=False,
        repeated_input_identical=True,
    ) == "UNRESOLVED"
    assert decide_arm(
        fixtures=[{"admitted": True}],
        controls_fall=True,
        repeated_input_identical=False,
    ) == "UNRESOLVED"
    assert decide_arm(
        fixtures=[{"admitted": True}, {"admitted": True}],
        controls_fall=True,
        repeated_input_identical=True,
    ) == "RECOMMENDED"
    assert decide_arm(
        fixtures=[{"admitted": False}],
        controls_fall=True,
        repeated_input_identical=True,
    ) == "EXPERIMENTAL"


def test_trial_decision_prefers_no_admission_only_when_controls_fall():
    experimental = {
        "arm_id": "char_hash",
        "candidate": True,
        "status": "EXPERIMENTAL",
        "pattern": "PURITY_MISS",
    }
    other = {
        "arm_id": "word_hash",
        "candidate": True,
        "status": "EXPERIMENTAL",
        "pattern": "PURITY_MISS",
    }
    decision = decide_trial(
        [experimental, other],
        split_reproduced=True,
        baseline_reproduced=True,
    )
    assert decision["decision_status"] == DECISION_NO_ADMISSION
    assert decision["recommendation"] == "NONE"
    assert decision["inductive_pass"] is False
    unresolved = dict(experimental)
    unresolved["status"] = "UNRESOLVED"
    held = decide_trial(
        [unresolved, other],
        split_reproduced=True,
        baseline_reproduced=True,
    )
    assert held["decision_status"] == DECISION_UNRESOLVED
    mixed = dict(other)
    mixed["pattern"] = "COVERAGE_WITHOUT_DISTINCTION"
    mixed_decision = decide_trial(
        [experimental, mixed],
        split_reproduced=True,
        baseline_reproduced=True,
    )
    assert mixed_decision["decision_status"] == DECISION_UNRESOLVED
    assert mixed_decision["pattern"] == "MIXED"
    recommended = dict(experimental)
    recommended["status"] = "RECOMMENDED"
    recommended["pattern"] = "ADMITTED_ARM"
    admitted = decide_trial(
        [recommended],
        split_reproduced=True,
        baseline_reproduced=True,
    )
    assert admitted["decision_status"] == DECISION_RECOMMENDED
    assert admitted["recommendation"] == "char_hash"
    assert admitted["inductive_pass"] is False
    assert admitted["decision_status"] in DECISION_STATUSES


def test_arm4_precondition_needs_a_lost_distinction():
    clear = {
        "repeated_input": {"collapsed": False, "pairwise_distinction_retention": 1.0, "n_distinct": 1},
        "nonsense_text": {"collapsed": False, "pairwise_distinction_retention": 1.0, "n_distinct": 2},
    }
    lost = {
        "repeated_input": {"collapsed": False, "pairwise_distinction_retention": 1.0, "n_distinct": 1},
        "nonsense_text": {"collapsed": True, "pairwise_distinction_retention": 0.0, "n_distinct": 2},
    }
    rows = [{"zero_below_arm1": True}, {"zero_below_arm1": True}]
    both_clear = arm4_precondition(rows, rows, clear, clear)
    assert both_clear["justified"] is False
    kept = arm4_precondition(rows, rows, lost, clear)
    assert kept["justified"] is True
    assert "nonsense_text" in kept["justification_sentence"]
    no_cut = [{"zero_below_arm1": False}]
    assert arm4_precondition(no_cut, rows, lost, clear)["justified"] is False


def test_split_and_arm1_match_the_encoder_isolation_table(report):
    assert report["split_reproduced"] is True
    assert report["split_fingerprint"]["sha256"] == FROZEN_SPLIT_SHA256
    assert report["arm1_matches_published"] is True
    assert report["decision_status"] == DECISION_UNRESOLVED
    assert report["recommendation"] == "NONE"
    assert report["runtime_authority"] == RUNTIME_AUTHORITY
    assert report["production_replaced"] is False
    assert report["inductive_pass"] is False
    assert report["production_svd_modified"] is False
    assert report["learn_pack_called"] is False
    arm1 = _arm(report, "tfidf_svd")
    assert [row["eval_hits"] for row in arm1["fixtures"]] == [5, 3, 4, 3]
    assert [row["base_hits"] for row in arm1["fixtures"]] == [5, 2, 3, 2]
    assert [row["zero_rows"] for row in arm1["fixtures"]] == [3, 3, 4, 8]
    assert [row["controls"]["label_hits"] for row in arm1["fixtures"]] == [5, 0, 1, 2]
    assert [row["controls"]["row_hits"] for row in arm1["fixtures"]] == [4, 1, 1, 1]
    synthetic = arm1["synthetic"]
    assert synthetic["repeated_input"]["projected_identical"] is True
    assert synthetic["unseen_token"]["zero_rows"] == synthetic["unseen_token"]["n_rows"]
    assert synthetic["unseen_token"]["collapsed"] is True
    assert synthetic["unseen_token"]["nonzero_collapse"] is False
    assert synthetic["nonsense_text"]["collapsed"] is True
    assert synthetic["nonsense_text"]["zero_rows"] == len(NONSENSE_TEXTS)


def test_hashed_arms_cut_zeros_and_controls_do_not_all_fall(report):
    assert [arm["arm_id"] for arm in report["arms"]] == ["tfidf_svd", "char_hash", "word_hash"]
    assert report["arm4"]["ran"] is False
    assert report["arm4_precondition"]["justified"] is False
    assert ARM4_RUN_MARKER not in report["arm4_precondition"]["justification_sentence"]
    for arm_id, expected in (("char_hash", CHAR_HITS), ("word_hash", WORD_HITS)):
        arm = _arm(report, arm_id)
        assert arm["status"] == "UNRESOLVED"
        assert arm["vocab_fitted"] is False
        assert arm["fit_on_eval"] is False
        assert arm["hash_dim"] == 32
        for fixture in arm["fixtures"]:
            eval_hits, base_hits, zero_rows, label_hits, row_hits, n_eval = expected[fixture["fixture_id"]]
            assert fixture["n_eval"] == n_eval
            assert fixture["eval_hits"] == eval_hits
            assert fixture["base_hits"] == base_hits
            assert fixture["zero_rows"] == zero_rows
            assert fixture["zero_below_arm1"] is True
            assert fixture["controls"]["label_hits"] == label_hits
            assert fixture["controls"]["row_hits"] == row_hits
            assert fixture["distinction"]["pairwise_distinction_retention"] == 1.0
            assert fixture["oov"]["vocab_fitted"] is False
            assert fixture["oov"]["nonzero_rows"] == n_eval
            assert fixture["costs"]["parameter_pack_nbytes"] == 0
            assert fixture["costs"]["fit_parameters"] is False
            assert fixture["admitted"] is False
        synthetic = arm["synthetic"]
        assert synthetic["repeated_input"]["identical"] is True
        assert synthetic["nonsense_text"]["collapsed"] is False
        assert synthetic["unseen_token"]["collapsed"] is False
        assert synthetic["unseen_token"]["zero_rows"] == 0
        collision = synthetic["hash_collision"]
        assert collision["collision_detected"] or collision["collision_impossible"]
        assert collision["collapsed"] is False


def test_score_arm4_does_not_run_without_the_written_justification(report):
    refused = score_arm4(report, "", time_runs=0)
    assert refused["ran"] is False
    marked = score_arm4(report, ARM4_RUN_MARKER + "\n" + "char_hash uneven zero 0/10", time_runs=0)
    assert marked["ran"] is False
    assert justification_in_doc("", report) is False
    assert justification_in_doc(ARM4_RUN_MARKER, report) is False

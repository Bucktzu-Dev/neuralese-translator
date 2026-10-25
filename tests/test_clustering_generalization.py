"""Generalization trial of the four frozen clustering candidates.

The candidates and the advantage rule are constants in
``neuralese.clustering_generalization``. Production k-means stays
farthest-first Euclidean. Topic labels are a sidecar, not an encoder input.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from neuralese import __version__
from neuralese.clustering_generalization import (
    ANGULAR_PAIR,
    CLUSTER_RUNTIME_FACTOR,
    CLUSTER_TIME_REPEATS,
    ENCODER,
    FAILURE_SAMPLE_CAP,
    FIXTURE_SPECS,
    FROZEN_CANDIDATE_IDS,
    FROZEN_CANDIDATES,
    HOLDOUT_MODULUS,
    HOLDOUT_REMAINDER,
    INIT_PAIR,
    ISOLATION_COMMIT,
    K,
    LABEL_SHUFFLE_SEED,
    MIN_ROWS,
    RANK,
    RESIDUAL_WORSE_FACTOR,
    ROW_SHUFFLE_SEED,
    RUNTIME_AUTHORITY,
    advantage_on_fixture,
    aggregate_cause,
    classify_fixture_cause,
    diagnose_clustering_generalization,
    diagnose_fixture,
    generalization_verdict,
    holdout_mask,
    load_fixture,
    recommend,
)
from neuralese.clustering_isolation import _fit_algorithm as isolation_fit
from neuralese.encode import ENCODER_WORD_SENTENCE_SVD, content_tokens

ROOT = Path(__file__).resolve().parents[1]
GENERAL = ROOT / "examples" / "generalization"
ABS = 1e-9

OLD_CORPORA = (
    ROOT / "examples" / "varied_prose" / "observations.jsonl",
    ROOT / "examples" / "toy_stream" / "observations.jsonl",
    ROOT / "examples" / "public_domain" / "observations.jsonl",
    ROOT / "examples" / "mdl_delta" / "observations.jsonl",
)
OLD_TOPICS = {
    "harbor",
    "orchard",
    "ledger",
    "bakery",
    "pottery",
    "weather",
    "joinery",
    "apiary",
}
SHARED_WORDS = ("water", "pump", "valve", "pipe")

ADVANTAGE = {
    "uneven": {
        "euclidean_random_subset": True,
        "euclidean_kmeanspp": True,
        "spherical_random_subset": False,
        "spherical_kmeanspp": False,
    },
    "overlapping": {
        "euclidean_random_subset": True,
        "euclidean_kmeanspp": True,
        "spherical_random_subset": False,
        "spherical_kmeanspp": False,
    },
    "ambiguous": {
        "euclidean_random_subset": False,
        "euclidean_kmeanspp": False,
        "spherical_random_subset": False,
        "spherical_kmeanspp": False,
    },
    "balanced": {
        "euclidean_random_subset": False,
        "euclidean_kmeanspp": True,
        "spherical_random_subset": False,
        "spherical_kmeanspp": False,
    },
}
PURITY_OK = {
    "uneven": (True, True, True, False),
    "overlapping": (True, True, False, True),
    "ambiguous": (False, False, False, False),
    "balanced": (False, True, False, False),
}
RESIDUAL_OK = {
    "uneven": (True, True, False, False),
    "overlapping": (True, True, True, False),
    "ambiguous": (True, True, True, True),
    "balanced": (False, True, False, True),
}
EVAL_HITS = {
    "uneven": (7, 9, 8, 8, 7),
    "overlapping": (6, 7, 7, 6, 7),
    "ambiguous": (8, 7, 6, 7, 6),
    "balanced": (9, 7, 10, 9, 9),
}
TRAIN_HITS = {
    "uneven": (26, 24, 25, 24, 27),
    "overlapping": (27, 26, 26, 26, 27),
    "ambiguous": (26, 25, 23, 25, 24),
    "balanced": (30, 31, 34, 32, 30),
}
RESIDUALS = {
    "uneven": (
        0.2275865733520475,
        0.2495184284723928,
        0.1780369299396730,
        0.2799813284769040,
        0.2636599753113307,
    ),
    "overlapping": (
        0.1337722038184514,
        0.1374883548826047,
        0.1433576027589774,
        0.1355924793719806,
        0.1564813860878321,
    ),
    "ambiguous": (
        0.2567684913717868,
        0.2465222335555870,
        0.2571746009787331,
        0.2663647183620846,
        0.2750878315033473,
    ),
    "balanced": (
        0.2111005807096355,
        0.2656475310510390,
        0.1926819329759302,
        0.2961975328060867,
        0.2272426492420031,
    ),
}
CAUSES = {
    "uneven": "ANGULAR_DISTANCE",
    "overlapping": "ANGULAR_DISTANCE",
    "ambiguous": "UNRESOLVED",
    "balanced": "INIT_SENSITIVITY",
}
PROBE_OK = {
    "uneven": False,
    "overlapping": False,
    "ambiguous": False,
    "balanced": True,
}
CONTROL_HITS = {
    "uneven": {"train": (26, 21, 22), "eval": (7, 8, 5), "rows": (40, 10)},
    "overlapping": {"train": (27, 19, 14), "eval": (6, 5, 6), "rows": (39, 9)},
    "ambiguous": {"train": (26, 21, 21), "eval": (8, 8, 7), "rows": (39, 9)},
    "balanced": {"train": (30, 20, 18), "eval": (9, 7, 8), "rows": (64, 16)},
}
FAILURE_IDS = {
    "uneven": {
        "euclidean_random_subset": ["uneven-040"],
        "euclidean_kmeanspp": ["uneven-030", "uneven-040"],
        "spherical_random_subset": ["uneven-030", "uneven-040"],
        "spherical_kmeanspp": ["uneven-030", "uneven-040", "uneven-050"],
    },
    "overlapping": {
        "euclidean_random_subset": ["overlapping-005", "overlapping-045"],
        "euclidean_kmeanspp": ["overlapping-005", "overlapping-045"],
        "spherical_random_subset": ["overlapping-005", "overlapping-015", "overlapping-045"],
        "spherical_kmeanspp": ["overlapping-005", "overlapping-045"],
    },
    "ambiguous": {
        "euclidean_random_subset": ["ambiguous-015", "ambiguous-025"],
        "euclidean_kmeanspp": ["ambiguous-015", "ambiguous-020", "ambiguous-030"],
        "spherical_random_subset": ["ambiguous-015", "ambiguous-020"],
        "spherical_kmeanspp": ["ambiguous-015", "ambiguous-020", "ambiguous-030"],
    },
    "balanced": {
        "euclidean_random_subset": [
            "balanced-005",
            "balanced-010",
            "balanced-015",
            "balanced-020",
            "balanced-030",
            "balanced-050",
        ],
        "euclidean_kmeanspp": [
            "balanced-005",
            "balanced-010",
            "balanced-020",
            "balanced-030",
            "balanced-050",
            "balanced-060",
        ],
        "spherical_random_subset": [
            "balanced-005",
            "balanced-010",
            "balanced-020",
            "balanced-030",
            "balanced-045",
            "balanced-050",
        ],
        "spherical_kmeanspp": [
            "balanced-005",
            "balanced-010",
            "balanced-020",
            "balanced-030",
            "balanced-050",
            "balanced-060",
        ],
    },
}
MATRIX_SHA = {
    "uneven": "4cceffeb6047a39615385c6d647f4a5c8e84f7bcaf57816a4003701a313e56f9",
    "overlapping": "682d0a3e03ff04176a09b738989725d74be3737e149ecca2bcbbc4ec44887c51",
    "ambiguous": "a1caeef8574e10c82f10334d932580155d263bbd7f7b0d75480d5db02c8834cb",
    "balanced": "7544590a6a53794f64ce21163427a1e630038437b308c79b163b53f01d3fb893",
}


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def report():
    return diagnose_clustering_generalization()


def test_frozen_candidates_and_bar_constants():
    assert [(row["id"], row["init"], row["distance"], row["seed"]) for row in FROZEN_CANDIDATES] == [
        ("euclidean_random_subset", "random_subset", "euclidean", 2),
        ("euclidean_kmeanspp", "kmeanspp", "euclidean", 2),
        ("spherical_random_subset", "random_subset", "spherical", 2),
        ("spherical_kmeanspp", "kmeanspp", "spherical", 3),
    ]
    assert FROZEN_CANDIDATE_IDS == tuple(row["id"] for row in FROZEN_CANDIDATES)
    assert ANGULAR_PAIR == ("euclidean_random_subset", "spherical_random_subset")
    assert INIT_PAIR == ("euclidean_random_subset", "euclidean_kmeanspp")
    assert RESIDUAL_WORSE_FACTOR == 1.10
    assert CLUSTER_RUNTIME_FACTOR == 2.0
    assert HOLDOUT_MODULUS == 5
    assert HOLDOUT_REMAINDER == 4
    assert MIN_ROWS == 40
    assert K == 8
    assert RANK == 8
    assert ENCODER == ENCODER_WORD_SENTENCE_SVD == "word_sentence_svd"
    assert LABEL_SHUFFLE_SEED == 0
    assert ROW_SHUFFLE_SEED == 1
    assert FAILURE_SAMPLE_CAP == 6
    assert CLUSTER_TIME_REPEATS >= 20
    assert ISOLATION_COMMIT == "c1179c90fb202aefc6979363847e9bfd33e1809c"
    assert __version__ == "0.7.0"
    assert [spec["id"] for spec in FIXTURE_SPECS] == [
        "uneven",
        "overlapping",
        "ambiguous",
        "balanced",
    ]


def test_production_kmeans_source_is_still_farthest_first():
    clustering = (ROOT / "src" / "neuralese" / "clustering.py").read_text(encoding="utf-8")
    alphabet = (ROOT / "src" / "neuralese" / "alphabet.py").read_text(encoding="utf-8")
    module = (ROOT / "src" / "neuralese" / "clustering_generalization.py").read_text(encoding="utf-8")
    assert "def _farthest_first" in clustering
    assert "seeds = _farthest_first(X, k)" in clustering
    assert "labels, centroids = kmeans(X, k, rng=rng)" in alphabet
    assert "0.7.0" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "learn_pack(" not in module
    assert "examples/varied_prose" not in module
    assert isolation_fit.__module__ == "neuralese.clustering_isolation"


def test_advantage_rule_is_strict_on_purity_and_closed_on_the_factors():
    base = dict(
        eval_size_weighted=0.6,
        production_eval_size_weighted=0.5,
        residual=1.10,
        production_residual=1.0,
        median_ms=2.0,
        production_median_ms=1.0,
    )
    assert advantage_on_fixture(**base)["advantage"] is True
    tied = advantage_on_fixture(**{**base, "eval_size_weighted": 0.5})
    assert tied["purity_ok"] is False
    assert tied["advantage"] is False
    worse = advantage_on_fixture(**{**base, "residual": 1.10 + 1e-9})
    assert worse["residual_ok"] is False
    assert worse["advantage"] is False
    slow = advantage_on_fixture(**{**base, "median_ms": 2.0 + 1e-6})
    assert slow["runtime_ok"] is False
    assert slow["advantage"] is False
    untimed = advantage_on_fixture(**{**base, "median_ms": None})
    assert untimed["runtime_ok"] is False
    assert untimed["advantage"] is False


def test_recommendation_requires_every_fixture_and_does_not_drop_a_loser():
    blank = {candidate_id: False for candidate_id in FROZEN_CANDIDATE_IDS}
    assert recommend([]) == "NONE"
    assert recommend([blank, blank]) == "NONE"
    one = dict(blank)
    one["euclidean_kmeanspp"] = True
    assert recommend([one, dict(one)]) == "euclidean_kmeanspp"
    assert recommend([one, blank]) == "NONE"
    both = dict(one)
    both["spherical_kmeanspp"] = True
    assert recommend([both]) == "euclidean_kmeanspp,spherical_kmeanspp"


def test_cause_is_named_only_when_every_fixture_agrees():
    bits = {candidate_id: False for candidate_id in FROZEN_CANDIDATE_IDS}
    assert classify_fixture_cause(bits) == "UNRESOLVED"
    angular = dict(bits)
    angular["euclidean_random_subset"] = True
    angular["euclidean_kmeanspp"] = True
    assert classify_fixture_cause(angular) == "ANGULAR_DISTANCE"
    init = dict(bits)
    init["euclidean_kmeanspp"] = True
    assert classify_fixture_cause(init) == "INIT_SENSITIVITY"
    both = dict(bits)
    both["euclidean_random_subset"] = True
    assert classify_fixture_cause(both) == "UNRESOLVED"
    assert aggregate_cause(["ANGULAR_DISTANCE"] * 4) == "ANGULAR_DISTANCE"
    assert aggregate_cause(["OBJECTIVE"] * 3) == "OBJECTIVE"
    assert aggregate_cause(["UNRESOLVED"] * 4) == "UNRESOLVED"
    assert (
        aggregate_cause(["ANGULAR_DISTANCE", "ANGULAR_DISTANCE", "UNRESOLVED", "INIT_SENSITIVITY"])
        == "UNRESOLVED"
    )


def test_verdict_is_unresolved_when_controls_fail_and_never_a_pass_token():
    assert (
        generalization_verdict(scorable=False, controls_ok=True, recommendation="NONE")
        == "UNRESOLVED"
    )
    assert (
        generalization_verdict(
            scorable=True,
            controls_ok=False,
            recommendation="euclidean_kmeanspp",
        )
        == "UNRESOLVED"
    )
    assert (
        generalization_verdict(scorable=True, controls_ok=True, recommendation="NONE")
        == "RESEARCH_FAIL"
    )
    assert (
        generalization_verdict(
            scorable=True,
            controls_ok=True,
            recommendation="euclidean_kmeanspp",
        )
        == "CONSISTENT_ADVANTAGE"
    )
    assert "RESEARCH_PASS" not in {
        "UNRESOLVED",
        "RESEARCH_FAIL",
        "CONSISTENT_ADVANTAGE",
    }


def test_holdout_is_every_fifth_row_of_the_new_files():
    mask = holdout_mask(50)
    assert int(mask.sum()) == 10
    assert list(np.flatnonzero(mask)[:3]) == [4, 9, 14]
    assert int(holdout_mask(48).sum()) == 9
    assert int(holdout_mask(80).sum()) == 16
    assert all(int(index) % HOLDOUT_MODULUS == HOLDOUT_REMAINDER for index in np.flatnonzero(mask))


def test_fixtures_are_new_sentences_with_sidecar_topics_only():
    old_texts = set()
    for path in OLD_CORPORA:
        for row in _jsonl(path):
            if row.get("text"):
                old_texts.add(row["text"])
    for spec in FIXTURE_SPECS:
        directory = GENERAL / spec["dirname"]
        observations = _jsonl(directory / "observations.jsonl")
        sidecar = _jsonl(directory / "topics.jsonl")
        assert len(observations) >= MIN_ROWS
        assert len(observations) == len(sidecar)
        texts = []
        for obs, side in zip(observations, sidecar):
            assert set(obs) == {"observation_id", "text"}
            assert "topic" not in obs
            assert "embedding" not in obs
            assert obs["observation_id"] == side["observation_id"]
            assert side["topic"]
            assert obs["text"] not in old_texts
            texts.append(obs["text"])
        assert len(texts) == len(set(texts))
        suffix = texts[0]
        for text in texts[1:]:
            while suffix and not text.endswith(suffix):
                suffix = suffix[1:]
        assert len(suffix) <= 1
    loaded = {spec["id"]: load_fixture(GENERAL / spec["dirname"]) for spec in FIXTURE_SPECS}
    uneven_counts = {}
    for topic in loaded["uneven"]["topics"]:
        uneven_counts[topic] = uneven_counts.get(topic, 0) + 1
    assert max(uneven_counts.values()) >= 3 * min(uneven_counts.values())
    assert uneven_counts["tram"] == 25
    assert len(uneven_counts) == 6
    overlap = loaded["overlapping"]
    by_topic = {}
    for topic, text in zip(overlap["topics"], overlap["texts"]):
        by_topic.setdefault(topic, []).append(text)
    for word in SHARED_WORDS:
        for topic in ("greenhouse", "canal"):
            count = sum(word in content_tokens(text) for text in by_topic[topic])
            assert 3 <= count < len(by_topic[topic])
        for topic, texts in by_topic.items():
            if topic in ("greenhouse", "canal"):
                continue
            assert sum(word in content_tokens(text) for text in texts) == 0
    ambiguous = loaded["ambiguous"]
    assert len(ambiguous["secondary"]) == 16
    assert all(
        ambiguous["secondary"][obs_id] != topic
        for obs_id, topic in zip(ambiguous["observation_ids"], ambiguous["topics"])
        if obs_id in ambiguous["secondary"]
    )
    primary = {}
    for topic in ambiguous["topics"]:
        primary[topic] = primary.get(topic, 0) + 1
    assert primary == {"archive": 12, "clinic": 12, "radio": 12, "kitchen": 12}
    balanced = {}
    for topic in loaded["balanced"]["topics"]:
        balanced[topic] = balanced.get(topic, 0) + 1
    assert len(balanced) == 8
    assert set(balanced.values()) == {10}
    assert set(balanced) & OLD_TOPICS == set()


def test_embedder_receives_fixture_text_and_not_the_sidecar(monkeypatch):
    calls = []

    def fake(texts, rank=8):
        calls.append((list(texts), rank))
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(len(texts), rank))
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        return raw.tolist()

    monkeypatch.setattr(
        "neuralese.clustering_generalization.word_sentence_svd",
        fake,
    )
    diagnose_clustering_generalization(time_runs=0)
    assert [rank for _texts, rank in calls] == [8, 8, 8, 8]
    for spec, (texts, _rank) in zip(FIXTURE_SPECS, calls):
        loaded = load_fixture(GENERAL / spec["dirname"])
        assert texts == loaded["texts"]
        assert all("\t" + topic not in text for text, topic in zip(texts, loaded["topics"]))


def test_generalization_locks_recommendation_and_per_fixture_bits(report):
    assert report["verdict"] == "UNRESOLVED"
    assert report["recommendation"] == "NONE"
    assert report["advantage_recommendation"] == "NONE"
    assert report["isolated_cause"] == "UNRESOLVED"
    assert report["runtime_authority"] == RUNTIME_AUTHORITY == "RUNTIME_NOT_AUTHORIZED"
    assert report["scorable"] is True
    assert report["controls_ok"] is False
    assert report["receipts"]["git_revision"] == _git_head()
    assert report["receipts"]["encoder"] == "word_sentence_svd"
    assert report["receipts"]["k"] == 8
    assert report["receipts"]["production_init"] == "farthest_first"
    assert report["receipts"]["production_distance"] == "euclidean"
    assert report["receipts"]["varied_prose_split_reused"] is False
    assert report["receipts"]["learn_pack_called"] is False
    assert report["receipts"]["stability_retuned"] is False
    assert [row["id"] for row in report["receipts"]["candidates"]] == list(FROZEN_CANDIDATE_IDS)

    assert [fixture["fixture_id"] for fixture in report["fixtures"]] == list(ADVANTAGE)
    for fixture in report["fixtures"]:
        fixture_id = fixture["fixture_id"]
        assert fixture["k"] == 8
        assert fixture["scorable"] is True
        assert fixture["stability_retuned"] is False
        assert fixture["cause"] == CAUSES[fixture_id]
        assert fixture["advantages"] == ADVANTAGE[fixture_id]
        assert fixture["negative_controls"]["probe_ok"] is PROBE_OK[fixture_id]
        assert fixture["embedding_sha256"] == MATRIX_SHA[fixture_id]
        assert fixture["max_abs_row_norm_error"] < 1e-12
        if fixture["n_topics"] < 8:
            assert "k stays 8" in fixture["k_note"]
        production = fixture["production"]
        candidates = fixture["candidates"]
        assert [row["id"] for row in candidates] == list(FROZEN_CANDIDATE_IDS)
        assert [row["seed"] for row in candidates] == [2, 2, 2, 3]
        blocks = [production, *candidates]
        residuals = RESIDUALS[fixture_id]
        eval_hits = EVAL_HITS[fixture_id]
        train_hits = TRAIN_HITS[fixture_id]
        assert len(blocks) == 5
        for block, residual, eval_hit, train_hit in zip(blocks, residuals, eval_hits, train_hits):
            assert block["residual"] == pytest.approx(residual, abs=ABS)
            assert block["eval_hits"] == eval_hit
            assert block["train_hits"] == train_hit
        assert production["eval_rows"] == CONTROL_HITS[fixture_id]["rows"][1]
        assert production["train_rows"] == CONTROL_HITS[fixture_id]["rows"][0]
        for candidate, purity_ok, residual_ok in zip(
            candidates,
            PURITY_OK[fixture_id],
            RESIDUAL_OK[fixture_id],
        ):
            assert candidate["purity_ok"] is purity_ok
            assert candidate["residual_ok"] is residual_ok
            assert candidate["advantage"] is ADVANTAGE[fixture_id][candidate["id"]]
            assert candidate["runtime_ratio"] > 0
            # The 2× clause is part of ``advantage``. A single noisy sample can
            # cross that line without moving purity or residual; the locked bit
            # is the conjunction the rule actually stored.
            assert candidate["cluster_peak_bytes"] > 0
            assert candidate["embed_cluster_peak_bytes"] > candidate["cluster_peak_bytes"]
            assert candidate["cluster_median_ms"] > 0
            assert candidate["embed_cluster_median_ms"] > candidate["cluster_median_ms"]
            sample = [row["observation_id"] for row in candidate["failures"]]
            assert sample == FAILURE_IDS[fixture_id][candidate["id"]]
            assert len(sample) == min(candidate["failure_count"], FAILURE_SAMPLE_CAP)
            assert all("secondary_topic" not in row for row in candidate["failures"])
        controls = fixture["negative_controls"]
        train_prod, train_label, train_row = CONTROL_HITS[fixture_id]["train"]
        eval_prod, eval_label, eval_row = CONTROL_HITS[fixture_id]["eval"]
        assert controls["production_train_hits"] == train_prod
        assert controls["label_shuffle"]["train_hits"] == train_label
        assert controls["row_shuffle"]["train_hits"] == train_row
        assert controls["production_eval_hits"] == eval_prod
        assert controls["label_shuffle"]["eval_hits"] == eval_label
        assert controls["row_shuffle"]["eval_hits"] == eval_row
        assert controls["label_shuffle"]["train_below_production"] is True
        assert controls["row_shuffle"]["train_below_production"] is True
        assert controls["eval_informative"] is True

    doc = (ROOT / "docs" / "PR1_CLUSTERING_GENERALIZATION.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    assert "PR1_CLUSTERING_GENERALIZATION.md" in roadmap
    for token in (
        "UNRESOLVED",
        "NONE",
        "RUNTIME_NOT_AUTHORIZED",
        "euclidean random subset",
        "euclidean k-means++",
        "spherical random subset",
        "spherical k-means++",
        "farthest-first",
    ):
        assert token in doc


def test_reversed_topics_do_not_move_residuals_or_ari(report):
    for spec in FIXTURE_SPECS:
        loaded = load_fixture(GENERAL / spec["dirname"])
        flipped = diagnose_fixture(
            spec["id"],
            loaded["texts"],
            list(reversed(loaded["topics"])),
            loaded["observation_ids"],
            time_runs=0,
        )
        original = next(
            fixture for fixture in report["fixtures"] if fixture["fixture_id"] == spec["id"]
        )
        assert flipped["production"]["residual"] == pytest.approx(
            original["production"]["residual"],
            abs=ABS,
        )
        assert flipped["embedding_sha256"] == original["embedding_sha256"]
        for left, right in zip(flipped["candidates"], original["candidates"]):
            assert left["residual"] == pytest.approx(right["residual"], abs=ABS)
            assert left["train_ari_vs_production"] == pytest.approx(
                right["train_ari_vs_production"],
                abs=ABS,
            )

"""Rank-8 weather mixture: singular-value gap and failure mode.

Labels are a probe. The product encoder, k, seed, and certify gates stay
where the varied-prose lock left them. Higher ranks are the same k-means
on wider truncations. They are not a new default.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuralese.adapters import load_observations_jsonl
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify
from neuralese.encode import ENCODER_WORD_SENTENCE_SVD
from neuralese.rank_boundary import (
    CLOSER_MAJORITY,
    MIXED_PURITY,
    OVERLAP_RATIO,
    SEPARATED_RATIO,
    classify_rank_boundary,
    diagnose_rank_boundary,
)

VARIED = Path(__file__).resolve().parents[1] / "examples" / "varied_prose" / "observations.jsonl"

# Reference BLAS/LAPACK 3.12, one BLAS thread. Absolute tolerance is tight
# next to the gap (~2e-2) and the distances (~1e-1).
ABS = 1e-9

# 0-based singular values. Index 7 is component 8. Index 8 is component 9.
S5 = 1.6381173195284298
S6 = 1.6158837658043956
S7 = 1.577782436621879
S8 = 1.556022686499294
S9 = 1.5011858601662842
S10 = 1.494124134967109
GAP = 0.021759750122584798
SINGULAR_RATIO = 0.9862086498001754

WITHIN_WEATHER = 0.3781671594255758
WITHIN_OTHER = 0.7700430881162342
BETWEEN = 0.979159660507253
BETWEEN_OVER_WITHIN = 2.5892244635799844


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


def test_rank_boundary_locks_gap_distances_and_clustering_loss():
    """Thresholds, written here, turn the locked distances into the mode.

    Overlap when ``between < 1.0 * within_weather`` or when the fraction of
    weather rows closer to another topic centroid is above 0.5.
    Separated when ``between >= 2.0 * within_weather``.
    CLUSTERING_LOSS when the groups are separated and the learned cluster
    is still mixed (purity < 0.5). A conflict of those readings is
    UNRESOLVED.
    """
    assert OVERLAP_RATIO == 1.0
    assert SEPARATED_RATIO == 2.0
    assert CLOSER_MAJORITY == 0.5
    assert MIXED_PURITY == 0.5

    texts, topics = _load()
    report = diagnose_rank_boundary(texts, topics, time_runs=0)
    window = {row["index"]: row["value"] for row in report["singular_window"]}
    assert [row["one_based"] for row in report["singular_window"]] == [6, 7, 8, 9, 10, 11]
    assert window[5] == pytest.approx(S5, abs=ABS)
    assert window[6] == pytest.approx(S6, abs=ABS)
    assert window[7] == pytest.approx(S7, abs=ABS)
    assert window[8] == pytest.approx(S8, abs=ABS)
    assert window[9] == pytest.approx(S9, abs=ABS)
    assert window[10] == pytest.approx(S10, abs=ABS)
    assert report["s7"] == pytest.approx(S7, abs=ABS)
    assert report["s8"] == pytest.approx(S8, abs=ABS)
    assert report["gap_s7_minus_s8"] == pytest.approx(GAP, abs=ABS)
    assert report["gap_s7_minus_s8"] == pytest.approx(window[7] - window[8], abs=ABS)
    assert report["ratio_s8_over_s7"] == pytest.approx(SINGULAR_RATIO, abs=ABS)
    # The cut is visible: the discarded step is smaller than either neighbor.
    assert window[7] - window[8] < window[6] - window[7]
    assert window[7] - window[8] < window[8] - window[9]

    weather = report["weather_cluster"]
    assert weather["code"] == 6
    assert weather["rows"] == 64
    assert weather["majority_count"] == 29
    assert weather["purity"] == pytest.approx(29 / 64, abs=ABS)
    assert weather["mixed"] is True
    assert weather["n_weather"] == 29
    assert weather["n_other"] == 35
    assert weather["closer_to_other_topic"] == 8
    assert weather["closer_fraction"] == pytest.approx(8 / 29, abs=ABS)
    assert weather["within_weather"] == pytest.approx(WITHIN_WEATHER, abs=ABS)
    assert weather["within_other"] == pytest.approx(WITHIN_OTHER, abs=ABS)
    assert weather["between_centroid"] == pytest.approx(BETWEEN, abs=ABS)
    assert weather["contaminant_counts"] == {
        "bakery": 2,
        "harbor": 1,
        "joinery": 1,
        "ledger": 26,
        "orchard": 3,
        "pottery": 2,
    }
    assert report["between_over_within_weather"] == pytest.approx(BETWEEN_OVER_WITHIN, abs=ABS)

    between = weather["between_centroid"]
    within = weather["within_weather"]
    fraction = weather["closer_fraction"]
    # Explicit thresholds. The mode is computed from the locked distances.
    assert between >= 2.0 * within
    assert fraction <= 0.5
    assert weather["purity"] < 0.5
    verdict = classify_rank_boundary(
        between_centroid=between,
        within_weather=within,
        closer_fraction=fraction,
        cluster_mixed=True,
    )
    assert verdict["failure_mode"] == "CLUSTERING_LOSS"
    assert verdict["conflicts"] == []
    assert report["failure_mode"] == "CLUSTERING_LOSS"
    assert report["conflicts"] == []

    by_rank = {row["rank"]: row for row in report["higher_rank"]}
    assert by_rank[8]["product_rank"] is True
    assert by_rank[8]["n_live"] == 7
    assert (by_rank[8]["weather_count"], by_rank[8]["weather_rows"]) == (29, 64)
    assert by_rank[9]["n_live"] == 7
    assert (by_rank[9]["weather_count"], by_rank[9]["weather_rows"]) == (31, 107)
    assert by_rank[9]["weather_purity"] == pytest.approx(31 / 107, abs=ABS)
    assert by_rank[9]["weather_mixed"] is True
    assert by_rank[12]["n_live"] == 8
    assert (by_rank[12]["weather_count"], by_rank[12]["weather_rows"]) == (29, 37)
    assert by_rank[12]["weather_purity"] == pytest.approx(29 / 37, abs=ABS)
    assert by_rank[16]["n_live"] == 8
    assert (by_rank[16]["weather_count"], by_rank[16]["weather_rows"]) == (28, 54)
    assert by_rank[16]["weather_purity"] == pytest.approx(28 / 54, abs=ABS)
    # The three wider truncations do not move together, so they are not a
    # second vote on the mode.
    assert by_rank[9]["weather_purity"] < by_rank[8]["weather_purity"]
    assert by_rank[12]["weather_purity"] > 0.5
    assert by_rank[16]["weather_mixed"] is False

    observations = load_observations_jsonl(VARIED)
    pack = learn_pack(
        observations,
        config=LearnConfig(n_symbols=8, seed=0, encoder=ENCODER_WORD_SENTENCE_SVD),
    )
    id_to_index = {obs.observation_id: index for index, obs in enumerate(observations)}
    real = [
        {"code": symbol.code, "indices": sorted(id_to_index[obs_id] for obs_id in symbol.observation_ids)}
        for symbol in pack.symbols
    ]
    got = [
        {"code": cluster["code"], "indices": sorted(cluster["indices"])}
        for cluster in report["cluster_indices"]
    ]
    assert got == real
    for obs in observations:
        assert obs.metadata.get("topic")
    cert = certify(pack)
    assert cert.passed
    assert "failure_mode" not in cert.details
    assert "cluster_purity" not in cert.details
    assert cert.details["tau_residual"] == 0.55


def test_probe_labels_do_not_move_spectrum_or_clusters():
    texts, topics = _load()
    original = diagnose_rank_boundary(texts, topics, time_runs=0)
    flipped = diagnose_rank_boundary(texts, list(reversed(topics)), time_runs=0)
    assert original["singular_window"] == flipped["singular_window"]
    assert original["cluster_indices"] == flipped["cluster_indices"]
    assert original["s7"] == flipped["s7"]
    assert original["gap_s7_minus_s8"] == flipped["gap_s7_minus_s8"]
    # The partition is fixed. The probe label is what moves.
    flipped_weather = flipped["weather_cluster"]
    if flipped_weather is None:
        assert flipped["failure_mode"] == "UNRESOLVED"
        assert original["failure_mode"] == "CLUSTERING_LOSS"
    else:
        assert original["weather_cluster"]["indices"] != flipped_weather["indices"]


def test_failure_mode_rule_on_split_and_overlap_cases():
    overlap = classify_rank_boundary(
        between_centroid=0.2,
        within_weather=1.0,
        closer_fraction=0.1,
        cluster_mixed=True,
    )
    assert overlap["failure_mode"] == "RANK_LOSS"
    assert overlap["conflicts"] == []

    majority = classify_rank_boundary(
        between_centroid=1.2,
        within_weather=1.0,
        closer_fraction=0.75,
        cluster_mixed=True,
    )
    assert majority["failure_mode"] == "RANK_LOSS"
    assert majority["conflicts"] == []

    separated = classify_rank_boundary(
        between_centroid=2.5,
        within_weather=1.0,
        closer_fraction=0.2,
        cluster_mixed=True,
    )
    assert separated["failure_mode"] == "CLUSTERING_LOSS"

    split = classify_rank_boundary(
        between_centroid=2.5,
        within_weather=1.0,
        closer_fraction=0.75,
        cluster_mixed=True,
    )
    assert split["failure_mode"] == "UNRESOLVED"
    assert "separated" in split["conflicts"][0]
    assert "overlap" in split["conflicts"][0]

    gray = classify_rank_boundary(
        between_centroid=1.5,
        within_weather=1.0,
        closer_fraction=0.2,
        cluster_mixed=True,
    )
    assert gray["failure_mode"] == "UNRESOLVED"
    assert "1.500000" in gray["conflicts"][0]

    pure = classify_rank_boundary(
        between_centroid=2.5,
        within_weather=1.0,
        closer_fraction=0.2,
        cluster_mixed=False,
    )
    assert pure["failure_mode"] == "UNRESOLVED"
    assert "not mixed" in pure["conflicts"][0]

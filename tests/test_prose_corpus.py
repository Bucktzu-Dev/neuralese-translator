"""Text path: original prose, hashed n-grams, published thresholds."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuralese.adapters import hashed_ngram_vector, load_observations_jsonl
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify, decodability_report
from neuralese.cli import main
from neuralese.contracts import Observation
from neuralese.translator import translate_stream
from neuralese.unfold import unfold_code

ROOT = Path(__file__).resolve().parents[1] / "examples" / "prose_corpus"
OBSERVATIONS = ROOT / "observations.jsonl"
ANCHORS = ROOT / "anchors.json"

PUBLISHED_TAU_RESIDUAL = 0.55
PUBLISHED_TAU_KAPPA = 0.35
N_SYMBOLS = 8
EMBED_DIM = 32
SEED = 0

# seed 0, default taus, dim 32 trigrams. Locked in examples/prose_corpus/README.md.
ANCHORED_RESIDUAL = 0.4222928756622633
ANCHORED_KAPPA = 0.9062962294721706
UNANCHORED_RESIDUAL = 0.7710345351372891
UNANCHORED_KAPPA = 0.6356755452944026
UNANCHORED_PURITY = 0.30851062907298377
TOLERANCE = 1e-4


def _load_anchors() -> dict[str, str]:
    return json.loads(ANCHORS.read_text(encoding="utf-8"))


def _learn(observations):
    return learn_pack(
        observations,
        config=LearnConfig(n_symbols=N_SYMBOLS, seed=SEED),
    )


def _majority_purity(pack) -> float:
    scores = []
    for symbol in pack.symbols:
        if symbol.quarantined or not symbol.observation_ids:
            continue
        counts: dict[str, int] = {}
        for obs_id in symbol.observation_ids:
            topic = obs_id.rsplit("-", 1)[0]
            counts[topic] = counts.get(topic, 0) + 1
        scores.append(max(counts.values()) / sum(counts.values()))
    return float(sum(scores) / len(scores))


def test_corpus_is_text_only_and_uses_the_published_anchors():
    anchors = _load_anchors()
    rows = []
    for line in OBSERVATIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert "embedding" not in row
        rows.append(row)
    assert len(rows) == 320
    assert len({row["text"] for row in rows}) == 320
    by_topic: dict[str, int] = {}
    for row in rows:
        topic = row["metadata"]["topic"]
        assert row["observation_id"].startswith(topic + "-")
        assert row["text"].endswith(anchors[topic])
        varied = row["text"][: -len(anchors[topic])].strip()
        assert varied
        assert varied != anchors[topic]
        by_topic[topic] = by_topic.get(topic, 0) + 1
    assert by_topic == {topic: 40 for topic in anchors}


def test_anchored_prose_certifies_at_published_thresholds():
    anchors = _load_anchors()
    obs = load_observations_jsonl(OBSERVATIONS)
    pack = _learn(obs)
    assert pack.metadata["config"]["tau_residual"] == PUBLISHED_TAU_RESIDUAL
    assert pack.metadata["config"]["tau_kappa"] == PUBLISHED_TAU_KAPPA
    assert pack.metadata["config"]["n_symbols"] == N_SYMBOLS
    assert pack.metadata["decision"] == "accept"
    assert pack.guards is not None
    assert pack.guards.pass_residual
    assert pack.guards.pass_kappa
    assert pack.reconstruction_error == pytest.approx(ANCHORED_RESIDUAL, abs=TOLERANCE)
    assert pack.guards.kappa_avg == pytest.approx(ANCHORED_KAPPA, abs=TOLERANCE)

    live = [symbol for symbol in pack.symbols if not symbol.quarantined]
    quarantined = [symbol for symbol in pack.symbols if symbol.quarantined]
    assert len(live) == 8
    assert quarantined == []
    assert len(pack.observations) == 320
    topics = set()
    for symbol in live:
        assert symbol.observation_ids
        prefixes = {obs_id.rsplit("-", 1)[0] for obs_id in symbol.observation_ids}
        assert prefixes == {next(iter(prefixes))}
        topics |= prefixes
        assert symbol.metadata["kappa"] >= PUBLISHED_TAU_KAPPA
        report = unfold_code(pack, symbol.code)
        assert report.state == "ok"
        assert report.missing_ids == []
        assert report.observations
        assert report.prototype_l2 is not None
        assert report.prototype_l2 < 1e-5
        assert report.english_bound
    assert topics == set(anchors)

    for stored in pack.observations.values():
        assert len(stored.embedding) == EMBED_DIM
        assert stored.embedding == hashed_ngram_vector(stored.text, dim=EMBED_DIM, n=3)

    unknown = translate_stream(pack, [symbol.code for symbol in live] + [99999])
    assert unknown[-1].state == "unknown"
    assert unknown[-1].english == "[undecodable: no symbol for code 99999]"
    assert all(gloss.state == "ok" for gloss in unknown[:-1])

    cert = certify(pack, tau_residual=PUBLISHED_TAU_RESIDUAL)
    assert cert.passed, cert.failures
    assert cert.fail_closed
    assert cert.details["tau_residual"] == PUBLISHED_TAU_RESIDUAL
    assert cert.details["n_live"] == 8
    assert cert.details["n_quarantined"] == 0

    too_tight = certify(pack, tau_residual=0.40)
    assert too_tight.passed is False
    assert too_tight.residual_ok is False

    summary = decodability_report(pack)
    assert summary["n_live"] == 8
    assert summary["n_quarantined"] == 0
    assert summary["residual"] == pytest.approx(ANCHORED_RESIDUAL, abs=TOLERANCE)
    assert summary["reservoir_size"] == 320
    assert summary["unfold_failures"] == []
    assert summary["gloss_coverage"] == 1.0
    assert summary["decision"] == "accept"
    assert summary["mdl_bits"] == pack.mdl_bits

    mutated = pack.symbols[0].definition
    pack.symbols[0].definition = "rewritten without resealing"
    tampered = certify(pack, tau_residual=PUBLISHED_TAU_RESIDUAL)
    assert tampered.passed is False
    assert tampered.gloss_bound is False
    pack.symbols[0].definition = mutated


def test_varied_sentences_without_anchors_fail_default_residual():
    """The repeated workplace sentence is load-bearing. Do not treat this as a pass."""
    anchors = _load_anchors()
    varied = []
    for obs in load_observations_jsonl(OBSERVATIONS):
        topic = obs.metadata["topic"]
        text = obs.text[: -len(anchors[topic])].strip()
        varied.append(
            Observation(
                observation_id=obs.observation_id,
                text=text,
                metadata={"topic": topic},
            )
        )
    pack = _learn(varied)
    assert pack.reconstruction_error == pytest.approx(UNANCHORED_RESIDUAL, abs=TOLERANCE)
    assert pack.guards is not None
    assert pack.guards.kappa_avg == pytest.approx(UNANCHORED_KAPPA, abs=TOLERANCE)
    assert pack.guards.pass_kappa
    assert pack.guards.pass_residual is False
    assert pack.metadata["decision"] == "reject"
    assert _majority_purity(pack) == pytest.approx(UNANCHORED_PURITY, abs=TOLERANCE)

    cert = certify(pack, tau_residual=PUBLISHED_TAU_RESIDUAL)
    assert cert.passed is False
    assert cert.residual_ok is False
    assert cert.unfoldable
    assert any("tau_residual" in failure for failure in cert.failures)

    unknown = translate_stream(pack, [99999])
    assert unknown[0].state == "unknown"
    assert unknown[0].english.startswith("[undecodable:")


def test_cli_report_and_certify_prose_pack(tmp_path, capsys):
    pack_path = tmp_path / "prose-pack.json"
    rc = main(
        [
            "learn",
            str(OBSERVATIONS),
            "-o",
            str(pack_path),
            "--n-symbols",
            str(N_SYMBOLS),
            "--seed",
            str(SEED),
        ]
    )
    assert rc == 0
    capsys.readouterr()

    rc = main(["report", str(pack_path)])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n_live"] == 8
    assert report["n_quarantined"] == 0
    assert report["reservoir_size"] == 320
    assert report["unfold_failures"] == []
    assert report["gloss_coverage"] == 1.0
    assert report["decision"] == "accept"
    assert report["residual"] == pytest.approx(ANCHORED_RESIDUAL, abs=TOLERANCE)

    rc = main(["certify", str(pack_path), "--fail-on-undecodable"])
    assert rc == 0
    cert = json.loads(capsys.readouterr().out)
    assert cert["passed"] is True
    assert cert["details"]["tau_residual"] == PUBLISHED_TAU_RESIDUAL

"""Varied prose: character trigrams still reject; local SVD certifies."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuralese.adapters import hashed_ngram_vector, load_observations_jsonl
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify, decodability_report
from neuralese.cli import main
from neuralese.encode import ENCODER_CHAR_TRIGRAM, ENCODER_WORD_SENTENCE_SVD, word_sentence_svd
from neuralese.translator import translate_stream
from neuralese.unfold import unfold_code

PROSE = Path(__file__).resolve().parents[1] / "examples" / "prose_corpus"
VARIED = Path(__file__).resolve().parents[1] / "examples" / "varied_prose" / "observations.jsonl"
ANCHORS = PROSE / "anchors.json"
ANCHORED = PROSE / "observations.jsonl"

PUBLISHED_TAU_RESIDUAL = 0.55
PUBLISHED_TAU_KAPPA = 0.35
N_SYMBOLS = 8
SEED = 0

# Same lock as tests/test_prose_corpus.py. Character trigrams on the stripped rows.
UNANCHORED_RESIDUAL = 0.7710345351372891
UNANCHORED_KAPPA = 0.6356755452944026
UNANCHORED_PURITY = 0.30851062907298377

# Sign-canonical word-sentence SVD, n_symbols 8, seed 0, library taus.
SVD_RESIDUAL = 0.5057019350263128
SVD_KAPPA = 0.8713099727751781
SVD_PURITY = 0.7084142910229867
SVD_WEIGHTED_PURITY = 0.659375
SVD_MIN_PURITY = 0.4
# Near-tie members can swap under a 1e-12 coordinate wobble. The certify
# gates stay 0.55 and 0.35. These tolerances cover that swap, not a looser tau.
METRIC_TOLERANCE = 1e-3
PURITY_TOLERANCE = 0.02
TRIGRAM_TOLERANCE = 1e-4


def _learn(observations, encoder):
    return learn_pack(
        observations,
        config=LearnConfig(n_symbols=N_SYMBOLS, seed=SEED, encoder=encoder),
    )


def _cluster_purities(pack):
    scores = []
    majors = []
    hits = 0
    total = 0
    for symbol in pack.symbols:
        if symbol.quarantined or not symbol.observation_ids:
            continue
        counts: dict[str, int] = {}
        for obs_id in symbol.observation_ids:
            topic = obs_id.rsplit("-", 1)[0]
            counts[topic] = counts.get(topic, 0) + 1
        topic, count = max(counts.items(), key=lambda item: item[1])
        scores.append(count / sum(counts.values()))
        majors.append(topic)
        hits += count
        total += sum(counts.values())
    mean = float(sum(scores) / len(scores))
    weighted = float(hits / total)
    return mean, weighted, min(scores), majors


def _stripped_rows():
    anchors = json.loads(ANCHORS.read_text(encoding="utf-8"))
    rows = []
    for line in ANCHORED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        topic = row["metadata"]["topic"]
        row["text"] = row["text"][: -len(anchors[topic])].strip()
        rows.append(row)
    return rows


def test_varied_file_is_the_anchored_corpus_without_the_repeated_line():
    anchors = json.loads(ANCHORS.read_text(encoding="utf-8"))
    expected = _stripped_rows()
    stored = []
    for line in VARIED.read_text(encoding="utf-8").splitlines():
        if line.strip():
            stored.append(json.loads(line))
    assert stored == expected
    assert len(stored) == 320
    assert len({row["text"] for row in stored}) == 320
    by_topic: dict[str, list[str]] = {}
    for row in stored:
        topic = row["metadata"]["topic"]
        text = row["text"]
        assert "embedding" not in row
        assert anchors[topic] not in text
        assert not text.endswith(anchors[topic])
        by_topic.setdefault(topic, []).append(text)
    assert {topic: len(texts) for topic, texts in by_topic.items()} == {topic: 40 for topic in anchors}
    for texts in by_topic.values():
        assert len({text[-40:] for text in texts}) == len(texts)
        suffix = texts[0]
        for text in texts[1:]:
            while suffix and not text.endswith(suffix):
                suffix = suffix[1:]
        assert suffix == "."


def test_character_trigrams_on_the_varied_file_still_fail_the_locked_residual():
    """The hashed default does not certify this file. Keep this failure."""
    observations = load_observations_jsonl(VARIED)
    pack = _learn(observations, ENCODER_CHAR_TRIGRAM)
    assert pack.metadata["config"]["encoder"] == ENCODER_CHAR_TRIGRAM
    assert pack.metadata["config"]["tau_residual"] == PUBLISHED_TAU_RESIDUAL
    assert pack.metadata["config"]["tau_kappa"] == PUBLISHED_TAU_KAPPA
    assert pack.reconstruction_error == pytest.approx(UNANCHORED_RESIDUAL, abs=TRIGRAM_TOLERANCE)
    assert pack.guards is not None
    assert pack.guards.kappa_avg == pytest.approx(UNANCHORED_KAPPA, abs=TRIGRAM_TOLERANCE)
    assert pack.guards.pass_kappa
    assert pack.guards.pass_residual is False
    assert pack.metadata["decision"] == "reject"
    mean, _weighted, _low, _majors = _cluster_purities(pack)
    assert mean == pytest.approx(UNANCHORED_PURITY, abs=TRIGRAM_TOLERANCE)
    for stored in pack.observations.values():
        assert stored.embedding == hashed_ngram_vector(stored.text, dim=32, n=3)
    cert = certify(pack)
    assert cert.passed is False
    assert cert.residual_ok is False
    assert cert.details["tau_residual"] == PUBLISHED_TAU_RESIDUAL


def test_word_sentence_svd_certifies_varied_prose_at_library_defaults():
    observations = load_observations_jsonl(VARIED)
    texts = [obs.text or "" for obs in observations]
    encoded = word_sentence_svd(texts, rank=N_SYMBOLS)
    pack = _learn(observations, ENCODER_WORD_SENTENCE_SVD)
    assert pack.metadata["config"]["encoder"] == ENCODER_WORD_SENTENCE_SVD
    assert pack.metadata["config"]["tau_residual"] == PUBLISHED_TAU_RESIDUAL
    assert pack.metadata["config"]["tau_kappa"] == PUBLISHED_TAU_KAPPA
    assert pack.metadata["config"]["n_symbols"] == N_SYMBOLS
    assert pack.metadata["decision"] == "accept"
    assert pack.guards is not None
    assert pack.guards.pass_residual
    assert pack.guards.pass_kappa
    assert pack.guards.pass_all
    assert pack.reconstruction_error <= PUBLISHED_TAU_RESIDUAL
    assert pack.guards.kappa_avg >= PUBLISHED_TAU_KAPPA
    assert pack.reconstruction_error == pytest.approx(SVD_RESIDUAL, abs=METRIC_TOLERANCE)
    assert pack.guards.kappa_avg == pytest.approx(SVD_KAPPA, abs=METRIC_TOLERANCE)
    assert pack.reconstruction_error > 0.45

    live = [symbol for symbol in pack.symbols if not symbol.quarantined]
    assert len(live) == 8
    assert [symbol for symbol in pack.symbols if symbol.quarantined] == []
    assert len(pack.observations) == 320
    stored_vectors = [pack.observations[obs.observation_id].embedding for obs in observations]
    assert stored_vectors == encoded
    assert all(len(vector) == N_SYMBOLS for vector in stored_vectors)
    for symbol in live:
        assert symbol.metadata["kappa"] >= PUBLISHED_TAU_KAPPA
        report = unfold_code(pack, symbol.code)
        assert report.state == "ok"
        assert report.missing_ids == []
        assert report.observations
        assert report.prototype_l2 is not None
        assert report.prototype_l2 < 1e-5
        assert report.english_bound

    mean, weighted, low, majors = _cluster_purities(pack)
    assert mean == pytest.approx(SVD_PURITY, abs=PURITY_TOLERANCE)
    assert weighted == pytest.approx(SVD_WEIGHTED_PURITY, abs=PURITY_TOLERANCE)
    assert low == pytest.approx(SVD_MIN_PURITY, abs=PURITY_TOLERANCE)
    assert mean > 0.55
    assert low < mean
    assert set(majors) == {
        "harbor",
        "orchard",
        "ledger",
        "bakery",
        "joinery",
        "weather",
        "apiary",
        "pottery",
    }

    unknown = translate_stream(pack, [symbol.code for symbol in live] + [99999])
    assert unknown[-1].state == "unknown"
    assert unknown[-1].english == "[undecodable: no symbol for code 99999]"
    assert all(gloss.state == "ok" for gloss in unknown[:-1])

    cert = certify(pack)
    assert cert.passed, cert.failures
    assert cert.fail_closed
    assert cert.residual_ok
    assert cert.details["tau_residual"] == PUBLISHED_TAU_RESIDUAL
    assert cert.details["n_live"] == 8
    assert cert.details["n_quarantined"] == 0

    too_tight = certify(pack, tau_residual=0.40)
    assert too_tight.passed is False
    assert too_tight.residual_ok is False

    summary = decodability_report(pack)
    assert summary["n_live"] == 8
    assert summary["n_quarantined"] == 0
    assert summary["residual"] == pytest.approx(SVD_RESIDUAL, abs=METRIC_TOLERANCE)
    assert summary["reservoir_size"] == 320
    assert summary["unfold_failures"] == []
    assert summary["gloss_coverage"] == 1.0
    assert summary["decision"] == "accept"
    assert summary["mdl_bits"] == pack.mdl_bits

    mutated = pack.symbols[0].definition
    pack.symbols[0].definition = "rewritten without resealing"
    tampered = certify(pack)
    assert tampered.passed is False
    assert tampered.gloss_bound is False
    pack.symbols[0].definition = mutated


def test_cli_report_and_certify_varied_prose(tmp_path, capsys):
    pack_path = tmp_path / "varied-pack.json"
    rc = main(
        [
            "learn",
            str(VARIED),
            "-o",
            str(pack_path),
            "--encoder",
            ENCODER_WORD_SENTENCE_SVD,
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
    assert report["residual"] <= PUBLISHED_TAU_RESIDUAL
    assert report["residual"] == pytest.approx(SVD_RESIDUAL, abs=METRIC_TOLERANCE)

    rc = main(["certify", str(pack_path), "--fail-on-undecodable"])
    assert rc == 0
    cert = json.loads(capsys.readouterr().out)
    assert cert["passed"] is True
    assert cert["details"]["tau_residual"] == PUBLISHED_TAU_RESIDUAL

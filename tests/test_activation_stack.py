"""Checked-in hf_layers dump derived from the prose corpus, loaded with adapt."""
from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from neuralese.activations import load_activation_dump
from neuralese.adapters import hashed_ngram_vector
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify, decodability_report
from neuralese.cli import main
from neuralese.translator import translate_stream
from neuralese.unfold import unfold_code

ROOT = Path(__file__).resolve().parents[1] / "examples" / "activation_stack"
NPZ = ROOT / "hidden_states.npz"
EXPORTER = ROOT / "export_hidden_states.py"

PUBLISHED_TAU_RESIDUAL = 0.55
PUBLISHED_TAU_KAPPA = 0.35
N_SYMBOLS = 8
SEED = 0
LAYOUT = "hf_layers"
LAYER = 0
POOL = "last"

# seed 0, library defaults, layer 0 of hidden_states.npz. Locked in the example README.
LAYER0_RESIDUAL = 0.4222928756622633
LAYER0_KAPPA = 0.9062962294721706
LAYER1_RESIDUAL = 0.4371939950642517
LAYER1_KAPPA = 0.8952268585663289
LAYER1_PURITY = 0.9344512195121951
LAYER2_RESIDUAL = 0.3792736021895029
LAYER2_KAPPA = 0.9380272106695282
LAYER2_PURITY = 0.9234342892879478
TOLERANCE = 1e-4


def _exporter():
    spec = importlib.util.spec_from_file_location("export_hidden_states", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def _learn_layer(layer: int):
    rows = load_activation_dump(NPZ, layout=LAYOUT, layer=layer, pool=POOL)
    return learn_pack(rows, config=LearnConfig(n_symbols=N_SYMBOLS, seed=SEED))


def test_checked_in_matrix_is_the_prose_hash_stack(tmp_path):
    export = _exporter()
    arrays = export.build_arrays()
    hidden = arrays["hidden_states"]
    assert hidden.shape == (3, 320, 32)
    assert hidden.dtype == np.dtype("<f8")
    assert np.isfinite(hidden).all()
    texts = [str(item) for item in arrays["texts"].tolist()]
    assert len(texts) == 320
    assert len(set(texts)) == 320
    trigram = np.asarray([hashed_ngram_vector(text, dim=32, n=3) for text in texts], dtype=np.float64)
    assert np.allclose(hidden[0], trigram)
    assert len({row.tobytes() for row in hidden[0]}) == 320
    assert not np.allclose(hidden[0], hidden[1])
    assert not np.allclose(hidden[1], hidden[2])

    with np.load(NPZ, allow_pickle=False) as bundle:
        assert np.array_equal(bundle["hidden_states"], hidden)
        assert [str(item) for item in bundle["texts"].tolist()] == texts
        assert [str(item) for item in bundle["observation_ids"].tolist()] == [
            str(item) for item in arrays["observation_ids"].tolist()
        ]
        assert str(np.asarray(bundle["layout"]).reshape(-1)[0]) == LAYOUT
        assert [str(item) for item in bundle["layer_names"].tolist()] == list(export.LAYER_NAMES)

    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    export.save_deterministic_npz(first, arrays)
    export.save_deterministic_npz(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == NPZ.read_bytes()
    with zipfile.ZipFile(NPZ) as archive:
        assert archive.infolist()
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)


def test_layer0_adapts_and_certifies_at_library_defaults():
    rows = load_activation_dump(NPZ, layout=LAYOUT, layer=LAYER, pool=POOL)
    pooled = load_activation_dump(NPZ, layout=LAYOUT, layer=LAYER, pool="mean")
    assert [row.embedding for row in rows] == [row.embedding for row in pooled]
    assert all(row.text for row in rows)

    pack = _learn_layer(LAYER)
    assert pack.metadata["config"]["tau_residual"] == PUBLISHED_TAU_RESIDUAL
    assert pack.metadata["config"]["tau_kappa"] == PUBLISHED_TAU_KAPPA
    assert pack.metadata["config"]["n_symbols"] == N_SYMBOLS
    assert pack.metadata["decision"] == "accept"
    assert pack.guards is not None
    assert pack.guards.pass_residual
    assert pack.guards.pass_kappa
    assert pack.reconstruction_error == pytest.approx(LAYER0_RESIDUAL, abs=TOLERANCE)
    assert pack.guards.kappa_avg == pytest.approx(LAYER0_KAPPA, abs=TOLERANCE)
    assert _majority_purity(pack) == pytest.approx(1.0, abs=TOLERANCE)

    live = [symbol for symbol in pack.symbols if not symbol.quarantined]
    assert len(live) == 8
    assert [symbol for symbol in pack.symbols if symbol.quarantined] == []
    assert len(pack.observations) == 320
    for symbol in live:
        report = unfold_code(pack, symbol.code)
        assert report.state == "ok"
        assert report.missing_ids == []
        assert report.prototype_l2 is not None
        assert report.prototype_l2 < 1e-5

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
    assert summary["residual"] == pytest.approx(LAYER0_RESIDUAL, abs=TOLERANCE)
    assert summary["reservoir_size"] == 320
    assert summary["unfold_failures"] == []
    assert summary["gloss_coverage"] == 1.0
    assert summary["decision"] == "accept"


def test_other_layers_meet_defaults_without_replacing_layer0():
    """Layers 1 and 2 are in the file. They are not the published adapt command."""
    word = _learn_layer(1)
    assert word.reconstruction_error == pytest.approx(LAYER1_RESIDUAL, abs=TOLERANCE)
    assert word.guards is not None
    assert word.guards.kappa_avg == pytest.approx(LAYER1_KAPPA, abs=TOLERANCE)
    assert word.metadata["decision"] == "accept"
    assert _majority_purity(word) == pytest.approx(LAYER1_PURITY, abs=TOLERANCE)
    assert certify(word, tau_residual=PUBLISHED_TAU_RESIDUAL).passed

    mixed = _learn_layer(2)
    assert mixed.reconstruction_error == pytest.approx(LAYER2_RESIDUAL, abs=TOLERANCE)
    assert mixed.guards is not None
    assert mixed.guards.kappa_avg == pytest.approx(LAYER2_KAPPA, abs=TOLERANCE)
    assert mixed.metadata["decision"] == "accept"
    assert sum(1 for symbol in mixed.symbols if symbol.quarantined) == 1
    assert sum(1 for symbol in mixed.symbols if not symbol.quarantined) == 7
    assert _majority_purity(mixed) == pytest.approx(LAYER2_PURITY, abs=TOLERANCE)
    assert certify(mixed, tau_residual=PUBLISHED_TAU_RESIDUAL).passed

    unknown = translate_stream(mixed, [99999])
    assert unknown[0].state == "unknown"
    assert unknown[0].english.startswith("[undecodable:")


def test_cli_adapt_report_and_certify_layer0(tmp_path, capsys):
    observations = tmp_path / "observations.jsonl"
    pack_path = tmp_path / "pack.json"
    rc = main(
        [
            "adapt",
            str(NPZ),
            "-o",
            str(observations),
            "--layout",
            LAYOUT,
            "--layer",
            str(LAYER),
            "--pool",
            POOL,
        ]
    )
    assert rc == 0
    capsys.readouterr()

    rc = main(
        [
            "learn",
            str(observations),
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
    assert report["residual"] == pytest.approx(LAYER0_RESIDUAL, abs=TOLERANCE)

    rc = main(["certify", str(pack_path), "--fail-on-undecodable"])
    assert rc == 0
    cert = json.loads(capsys.readouterr().out)
    assert cert["passed"] is True
    assert cert["details"]["tau_residual"] == PUBLISHED_TAU_RESIDUAL

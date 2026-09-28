import json
from pathlib import Path

import pytest

from neuralese.adapters import load_observations_jsonl, load_pack, save_pack
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify
from neuralese.contracts import Observation, Symbol
from neuralese.unfold import unfold_code

from packutil import make_pack

TOY = Path(__file__).resolve().parents[1] / "examples" / "toy_stream" / "observations.jsonl"


def test_learned_pack_unfolds_to_source_rows():
    obs = load_observations_jsonl(TOY)
    pack = learn_pack(obs, config=LearnConfig(n_symbols=3, min_cluster_size=2, seed=0))
    assert set(pack.observations) == {row.observation_id for row in obs}
    for symbol in pack.symbols:
        if not symbol.observation_ids:
            continue
        report = unfold_code(pack, symbol.code)
        assert report.missing_ids == []
        assert report.observations
        assert report.prototype_l2 is not None
        assert report.prototype_l2 < 1e-5
    cert = certify(pack)
    assert cert.passed, cert.failures
    assert cert.details["reservoir_size"] == len(obs)
    assert cert.details["recomputed_cluster_residual"] <= pack.reconstruction_error + 1e-4


def test_reservoir_survives_json_roundtrip(tmp_path):
    obs = load_observations_jsonl(TOY)
    pack = learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0))
    path = tmp_path / "pack.json"
    save_pack(pack, path)
    loaded = load_pack(path)
    assert loaded.compute_checksum() == pack.checksum
    assert loaded.observations["g-1"].embedding == pytest.approx(obs[0].embedding)
    report = unfold_code(loaded, 0)
    assert report.missing_ids == []


def test_certify_fails_when_reservoir_rows_are_removed():
    pack = make_pack()
    pack.observations = {}
    pack.seal()
    cert = certify(pack)
    assert not cert.passed
    assert not cert.unfoldable
    assert any("reservoir" in failure for failure in cert.failures)


def test_certify_fails_when_prototype_drifts_from_reservoir():
    pack = make_pack()
    pack.symbols[0].proto_embedding = [0.0, 0.0, 0.0]
    pack.seal()
    cert = certify(pack)
    assert not cert.unfoldable
    assert any("prototype" in failure for failure in cert.failures)


def test_mutating_reservoir_text_breaks_the_seal():
    pack = make_pack()
    pack.observations["obs-hello"].text = "rewritten after seal"
    cert = certify(pack)
    assert not cert.gloss_bound
    assert any("checksum" in failure for failure in cert.failures)


def test_sealed_residual_cannot_undercut_reservoir():
    symbols = [
        Symbol(
            class_id=0,
            code=0,
            proto_embedding=[1.0, 0.0],
            observation_ids=["a", "b"],
            definition="spread symbol",
            confidence=0.5,
        )
    ]
    observations = {
        "a": Observation("a", [2.0, 0.0], "left"),
        "b": Observation("b", [0.0, 0.0], "right"),
    }
    pack = make_pack(symbols=symbols, observations=observations, reconstruction_error=0.0)
    cert = certify(pack)
    assert cert.unfoldable
    assert not cert.residual_ok
    assert cert.details["recomputed_cluster_residual"] > 0.5
    assert any("reservoir cluster residual" in failure for failure in cert.failures)


def test_duplicate_observation_id_is_rejected():
    obs = load_observations_jsonl(TOY)
    obs.append(Observation.from_dict(obs[0].to_dict()))
    with pytest.raises(ValueError, match="duplicate observation_id"):
        learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0))


def test_load_pack_rejects_duplicate_reservoir_ids(tmp_path):
    pack = make_pack()
    path = tmp_path / "pack.json"
    save_pack(pack, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["observations"].append(dict(data["observations"][0]))
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate observation_id"):
        load_pack(path)


def test_unfold_unknown_code_has_no_rows():
    pack = make_pack()
    report = unfold_code(pack, 99)
    assert report.state == "unknown"
    assert report.observations == []
    assert report.missing_ids == []

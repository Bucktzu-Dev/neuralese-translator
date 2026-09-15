import json

import pytest

from neuralese.audit import certify
from neuralese.cli import main
from neuralese.contracts import Observation, Receipt, UncertifiedPackError
from neuralese.translator import translate_stream

from packutil import make_pack


def test_nonfinite_metadata_fails_checksum_not_allow_nan():
    pack = make_pack()
    pack.metadata["x"] = float("nan")
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("not JSON-serializable" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])
    pack.metadata["x"] = float("inf")
    cert = certify(pack)
    assert not cert.passed
    assert any("not JSON-serializable" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_supplied_non_string_observation_id_fails_evidence():
    pack = make_pack()
    cert = certify(
        pack,
        observations=[
            Observation(observation_id=1, text="hello there"),
            Observation(observation_id="1", text="different content entirely"),
        ],
    )
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("supplied observation_id 1 is not a string" in f for f in cert.failures)
    assert not any("duplicate observation_id" in f for f in cert.failures)


def test_nonfinite_supplied_embedding_fails_evidence_not_value_error():
    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    obs = Observation(observation_id=obs_id, embedding=[1.0], text="hello there")
    obs.embedding = [float("nan")]
    cert = certify(pack, observations=[obs])
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("evidence content is not verifiable" in f for f in cert.failures)
    obs.embedding = [float("inf")]
    cert = certify(pack, observations=[obs])
    assert not cert.passed
    assert any("evidence content is not verifiable" in f for f in cert.failures)


def test_loaded_null_observation_ids_are_not_normalized_to_empty():
    pack = make_pack()
    data = pack.to_dict()
    data["symbols"][0]["observation_ids"] = None
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].observation_ids is None
    dumped = loaded.to_dict()
    assert dumped["symbols"][0]["observation_ids"] is None
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("observation_ids is not an array" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])
    data["symbols"][0]["quarantined"] = True
    loaded = pack.from_dict(data)
    loaded.seal()
    cert = certify(loaded)
    assert not cert.passed
    assert any("observation_ids is not an array" in f for f in cert.failures)
    del data["symbols"][0]["observation_ids"]
    omitted = pack.from_dict(data)
    assert omitted.symbols[0].observation_ids == []


def test_boolean_embedding_elements_are_rejected_before_float():
    with pytest.raises(TypeError, match="embedding must contain numbers"):
        Observation(observation_id="o1", embedding=[True])
    with pytest.raises(TypeError, match="embedding must contain numbers"):
        Observation.from_dict({"observation_id": "o1", "embedding": [False, 1.0]})
    obs = Observation(observation_id="o1", embedding=[1.0, 0.0])
    assert obs.embedding == [1.0, 0.0]

def test_constructed_nonfinite_embedding_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        Observation(observation_id="o1", embedding=[float("nan")])
    with pytest.raises(ValueError, match="finite"):
        Observation(observation_id="o1", embedding=[float("inf")])
    with pytest.raises(ValueError, match="finite"):
        Observation.from_dict({"observation_id": "o1", "embedding": [float("-inf")]})
    obs = Observation(observation_id="o1", embedding=[1.0, 0.0])
    assert obs.embedding == [1.0, 0.0]

def test_non_observation_supplied_rows_fail_evidence_not_attribute_error():
    pack = make_pack()
    cert = certify(pack, observations=[None])
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("not an Observation" in f for f in cert.failures)
    cert = certify(pack, observations=[{}])
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("not an Observation" in f for f in cert.failures)


def test_symbol_list_metadata_is_not_laundered_by_to_dict():
    pack = make_pack()
    pack.symbols[0].metadata = []
    dumped = pack.to_dict()
    assert dumped["symbols"][0]["metadata"] == []
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("metadata is not an object" in f for f in cert.failures)
    pack.seal()
    dumped = pack.to_dict()
    assert dumped["symbols"][0]["metadata"] == []
    with pytest.raises(TypeError, match="metadata must be an object or null"):
        pack.from_dict(dumped)


def test_pack_to_dict_detaches_metadata():
    pack = make_pack()
    original = dict(pack.metadata)
    dumped = pack.to_dict()
    assert dumped["metadata"] is not pack.metadata
    dumped["metadata"]["x"] = 99
    assert pack.metadata == original
    assert certify(pack).passed
    pack.metadata = []
    dumped = pack.to_dict()
    assert dumped["metadata"] == []
    with pytest.raises(TypeError, match="metadata must be an object or null"):
        pack.from_dict(dumped)



def test_nested_metadata_is_detached_by_to_dict():
    pack = make_pack()
    pack.metadata["config"] = {"tau_residual": 0.55}
    pack.seal()
    dumped = pack.to_dict()
    dumped["metadata"]["config"]["tau_residual"] = 99
    assert pack.metadata["config"]["tau_residual"] == 0.55
    assert certify(pack).passed


def test_infinite_tau_residual_fails_residual_ok():
    pack = make_pack()
    cert = certify(pack, tau_residual=float("inf"))
    assert not cert.residual_ok
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("tau_residual is not a finite non-negative real" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0], tau_residual=float("inf"))
    cert = certify(pack, tau_residual=float("nan"))
    assert not cert.residual_ok
    assert not cert.passed
    cert = certify(pack, tau_residual=-0.1)
    assert not cert.residual_ok
    assert not cert.passed
    assert certify(pack, tau_residual=0.55).passed


def test_cli_tau_residual_inf_is_rejected(tmp_path, capsys):
    pack = make_pack()
    pack.seal()
    pack_path = tmp_path / "pack.json"
    stream = tmp_path / "stream.json"
    pack_path.write_text(json.dumps(pack.to_dict()) + "\n")
    stream.write_text("[0]\n")
    with pytest.raises(SystemExit) as err:
        main(["translate", str(pack_path), str(stream), "--tau-residual", "inf"])
    assert err.value.code == 2
    err_text = capsys.readouterr().err
    assert "finite non-negative real" in err_text


def test_non_string_evidence_key_cannot_be_json_laundered(tmp_path):
    from neuralese.adapters import save_pack

    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    digest = pack.evidence[obs_id]
    pack.evidence = {1: digest}
    cert = certify(pack)
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("not a non-blank string" in f for f in cert.failures)
    with pytest.raises(TypeError, match="evidence keys must be strings"):
        pack.to_dict()
    with pytest.raises(TypeError, match="evidence keys must be strings"):
        save_pack(pack, tmp_path / "pack.json")


def test_numpy_scalar_embedding_elements_are_accepted():
    import numpy as np

    obs = Observation(
        observation_id="obs-2",
        embedding=[np.int64(1), np.float32(0.25)],
        text="hello",
    )
    assert obs.embedding == [1.0, 0.25]


def test_string_codebook_key_cannot_be_json_laundered(tmp_path):
    from neuralese.adapters import save_pack

    pack = make_pack()
    pack.codebook = {"0": 0}
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("codebook entries must be integer-to-integer" in f for f in cert.failures)
    with pytest.raises(TypeError, match="codebook entries must be integer-to-integer"):
        pack.to_dict()
    with pytest.raises(TypeError, match="codebook entries must be integer-to-integer"):
        save_pack(pack, tmp_path / "pack.json")


def test_string_alias_mapping_key_cannot_be_json_laundered(tmp_path):
    from neuralese.adapters import save_pack

    pack = make_pack(aliases={"c" * 64: {"7": 0}}, checksum="unsealed")
    assert pack.aliases["c" * 64] == {"7": 0}
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("alias entries must be integer-to-integer" in f for f in cert.failures)
    with pytest.raises(TypeError, match="alias entries must be integer-to-integer"):
        pack.to_dict()
    with pytest.raises(TypeError, match="alias entries must be integer-to-integer"):
        save_pack(pack, tmp_path / "pack.json")


def test_integer_alias_source_key_cannot_be_json_laundered(tmp_path):
    from neuralese.adapters import save_pack

    pack = make_pack()
    pack.aliases = {1: {7: 0}}
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("alias source key is not a non-empty string" in f for f in cert.failures)
    with pytest.raises(TypeError, match="alias source keys must be non-empty strings"):
        pack.to_dict()
    with pytest.raises(TypeError, match="alias source keys must be non-empty strings"):
        save_pack(pack, tmp_path / "pack.json")


def test_non_checksum_alias_source_is_not_selectable():
    from neuralese.contracts import normalize_aliases, select_alias_table

    with pytest.raises(ValueError, match="legacy or SHA-256"):
        normalize_aliases({"not-a-checksum": {7: 0}})
    with pytest.raises(ValueError, match="legacy or SHA-256"):
        make_pack(aliases={"not-a-checksum": {7: 0}})
    pack = make_pack()
    data = pack.to_dict()
    data["aliases"] = {"not-a-checksum": {"7": 0}}
    with pytest.raises(ValueError, match="legacy or SHA-256"):
        pack.from_dict(data)
    pack.aliases = {"not-a-checksum": {7: 0}}
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("alias source key is not a pack checksum" in f for f in cert.failures)
    assert select_alias_table(pack.aliases, "not-a-checksum") == {}
    glosses = translate_stream(
        pack, [7], source_pack_checksum="not-a-checksum", require_certified=False
    )
    assert glosses[0].state == "unknown"
    with pytest.raises(TypeError, match="legacy or SHA-256"):
        pack.to_dict()



import json

import pytest

from neuralese.audit import certify
from neuralese.cli import main
from neuralese.contracts import Observation, Receipt, UncertifiedPackError
from neuralese.translator import translate_stream

from packutil import make_pack


def test_overflowing_jsonl_embedding_is_clean_cli_failure(tmp_path, capsys):
    from neuralese.adapters import load_observations_jsonl

    path = tmp_path / "obs.jsonl"
    huge = 10**1000
    path.write_text('{"observation_id":"o1","embedding":[' + str(huge) + ']}\n')
    with pytest.raises(ValueError, match="invalid observation record"):
        load_observations_jsonl(path)
    rc = main(["learn", str(path), "-o", str(tmp_path / "pack.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "invalid observation record" in err


def test_overflowing_pack_timestamp_is_clean_cli_failure(tmp_path, capsys):
    from neuralese.adapters import load_pack

    pack = make_pack()
    pack.seal()
    data = pack.to_dict()
    data["timestamp"] = 10**1000
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(data) + "\n")
    loaded = load_pack(path)
    assert loaded.timestamp == 10**1000
    cert = certify(loaded)
    assert not cert.passed
    assert any("timestamp is not finite" in f for f in cert.failures)
    stream = tmp_path / "stream.json"
    stream.write_text("[0]\n")
    rc = main(["translate", str(path), str(stream)])
    assert rc == 1
    err = capsys.readouterr().err
    failed = json.loads(err)
    assert failed["passed"] is False

def test_loaded_float_and_bool_codebook_values_are_not_coerced():
    pack = make_pack()
    data = pack.to_dict()
    key = next(iter(data["codebook"]))
    data["codebook"][key] = 0.9
    loaded = pack.from_dict(data)
    assert loaded.codebook[int(key)] == 0.9
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("codebook entries must be integer-to-integer" in f for f in cert.failures)
    data["codebook"][key] = True
    loaded = pack.from_dict(data)
    assert loaded.codebook[int(key)] is True
    cert = certify(loaded)
    assert not cert.passed
    assert any("codebook entries must be integer-to-integer" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


def test_unhashable_symbol_code_fails_closed():
    pack = make_pack()
    pack.symbols[0].code = []
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any(
        "code is not an integer" in f or "not hashable" in f for f in cert.failures
    )
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])

def test_malformed_receipts_fail_schema_not_attribute_error():
    pack = make_pack()
    pack.receipts = [{}]
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("receipts[0] is not a Receipt" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])
    pack.receipts = None
    cert = certify(pack)
    assert not cert.passed
    assert any("receipts is not an array" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_constructed_non_object_metadata_fails_schema():
    pack = make_pack()
    pack.metadata = []
    pack.seal()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("metadata is not an object" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_null_example_hashes_are_not_normalized_to_empty():
    pack = make_pack()
    data = pack.to_dict()
    data["symbols"][0]["example_hashes"] = None
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].example_hashes is None
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("example_hashes is not an array" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])
    dumped = loaded.to_dict()
    assert dumped["symbols"][0]["example_hashes"] is None


def test_loaded_falsey_receipts_are_not_normalized_to_empty():
    pack = make_pack()
    data = pack.to_dict()
    data["receipts"] = {}
    with pytest.raises(TypeError, match="receipts must be an array"):
        pack.from_dict(data)
    data["receipts"] = False
    with pytest.raises(TypeError, match="receipts must be an array"):
        pack.from_dict(data)
    data["receipts"] = 0
    with pytest.raises(TypeError, match="receipts must be an array"):
        pack.from_dict(data)


def test_overflowing_reconstruction_error_fails_closed_not_overflow():
    pack = make_pack()
    pack.reconstruction_error = 10**1000
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("reconstruction_error is not finite" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_jsonl_bool_and_nonfinite_embedding_elements_are_rejected(tmp_path, capsys):
    from neuralese.adapters import load_observations_jsonl

    path = tmp_path / "obs.jsonl"
    path.write_text('{"observation_id":"o1","embedding":[true]}\n')
    with pytest.raises(ValueError, match="invalid observation record"):
        load_observations_jsonl(path)
    path.write_text('{"observation_id":"o1","embedding":[1e309]}\n')
    with pytest.raises(ValueError, match="invalid observation record"):
        load_observations_jsonl(path)
    rc = main(["learn", str(path), "-o", str(tmp_path / "pack.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "invalid observation record" in err


def test_malformed_symbols_fail_schema_not_attribute_error():
    pack = make_pack()
    pack.symbols = [{}]
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("symbols[0] is not a Symbol" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])
    pack.symbols = [None]
    cert = certify(pack)
    assert not cert.passed
    assert any("symbols[0] is not a Symbol" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])
    pack.symbols = None
    cert = certify(pack)
    assert not cert.passed
    assert any("symbols is not an array" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_null_codebook_is_not_normalized_to_empty():
    pack = make_pack()
    data = pack.to_dict()
    data["codebook"] = None
    loaded = pack.from_dict(data)
    assert loaded.codebook is None
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("codebook is not an object" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])
    del data["codebook"]
    omitted = pack.from_dict(data)
    assert omitted.codebook == {}

def test_nan_receipt_timestamp_fails_schema():
    pack = make_pack(
        guards=None,
        receipts=[Receipt(step="finalize", ok=True, timestamp=float("nan"))],
        checksum="unsealed",
    )
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("receipts[0].timestamp is not finite" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_nonfinite_receipt_metrics_fail_schema():
    pack = make_pack(
        receipts=[Receipt(step="finalize", ok=True, timestamp=1.0, kappa=float("inf"))],
        checksum="unsealed",
    )
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("receipts[0].kappa is not finite" in f for f in cert.failures)
    pack.receipts[0].kappa = None
    pack.receipts[0].reconstruction_error = float("nan")
    cert = certify(pack)
    assert not cert.passed
    assert any("receipts[0].reconstruction_error is not finite" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_scalar_supplied_embedding_fails_evidence_not_type_error():
    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    obs = Observation(observation_id=obs_id, embedding=[1.0], text="hello there")
    obs.embedding = 1.0
    cert = certify(pack, observations=[obs])
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("evidence content is not verifiable" in f for f in cert.failures)


def test_overflowing_supplied_embedding_fails_evidence_not_overflow():
    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    obs = Observation(observation_id=obs_id, embedding=[1.0], text="hello there")
    obs.embedding = [10**1000]
    cert = certify(pack, observations=[obs])
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("evidence content is not verifiable" in f for f in cert.failures)


def test_constructed_symbol_metadata_none_fails_schema():
    pack = make_pack()
    pack.symbols[0].metadata = None
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("metadata is not an object" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_infinite_alias_target_fails_closed_not_overflow():
    pack = make_pack()
    pack.aliases = {"src": {7: float("inf")}}
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_null_codebook_with_aliases_fails_closed_not_type_error():
    pack = make_pack(aliases={"c" * 64: {7: 0}})
    pack.codebook = None
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("codebook is not an object" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_falsey_symbol_metadata_fails_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    for metadata in ([], False, ""):
        data["symbols"][0]["metadata"] = metadata
        with pytest.raises(TypeError, match="metadata must be an object or null"):
            pack.from_dict(data)
    data["symbols"][0]["metadata"] = None
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].metadata == {}



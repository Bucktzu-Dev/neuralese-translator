import json

import pytest

from neuralese.audit import certify
from neuralese.cli import main
from neuralese.contracts import UncertifiedPackError
from neuralese.translator import translate_stream

from packutil import make_pack


def test_none_evidence_key_is_not_stringified():
    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    digest = pack.evidence[obs_id]
    data = pack.to_dict()
    data["evidence"] = {None: digest, obs_id: digest}
    loaded = pack.from_dict(data)
    assert None in loaded.evidence
    assert "None" not in loaded.evidence
    cert = certify(loaded)
    assert not cert.evidence_valid
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("not a non-blank string" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


def test_non_mapping_evidence_survives_to_dict_round_trip():
    pack = make_pack()
    data = pack.to_dict()
    data["evidence"] = []
    loaded = pack.from_dict(data)
    dumped = loaded.to_dict()
    assert dumped["evidence"] == []
    again = pack.from_dict(dumped)
    assert again.evidence == []
    cert = certify(again)
    assert not cert.passed
    assert not cert.evidence_valid
    data["evidence"] = None
    loaded = pack.from_dict(data)
    assert loaded.to_dict()["evidence"] is None
    cert = certify(loaded)
    assert not cert.passed


def test_non_numeric_guard_metric_fails_schema():
    pack = make_pack()
    pack.guards.kappa_avg = "0.8"
    pack.seal()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("guards.kappa_avg is not a number" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_string_guard_metric_fails_schema():
    pack = make_pack()
    data = pack.to_dict()
    data["guards"]["min_survival"] = "1.0"
    loaded = pack.from_dict(data)
    assert loaded.guards.min_survival == "1.0"
    loaded.seal()
    cert = certify(loaded)
    assert not cert.passed
    assert any("guards.min_survival is not a number" in f for f in cert.failures)


def test_translate_uses_caller_tau_residual_not_pack_metadata():
    pack = make_pack(reconstruction_error=0.6)
    pack.metadata["tau_residual"] = 0.9
    pack.seal()
    cert = certify(pack)
    assert not cert.passed
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])
    assert certify(pack, tau_residual=0.9).passed
    glosses = translate_stream(pack, [0], tau_residual=0.9)
    assert glosses[0].state == "ok"


def test_cli_translate_tau_residual_is_operator_gate(tmp_path, capsys):
    from pathlib import Path

    toy = Path(__file__).resolve().parents[1] / "examples" / "toy_stream"
    pack_path = tmp_path / "pack.json"
    stream = tmp_path / "stream.json"
    rc = main(
        [
            "learn",
            str(toy / "observations.jsonl"),
            "-o",
            str(pack_path),
            "--n-symbols",
            "3",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    data = json.loads(pack_path.read_text())
    data["reconstruction_error"] = 0.6
    data["metadata"]["tau_residual"] = 0.9
    if data.get("guards"):
        data["guards"]["reconstruction_error"] = 0.6
    from neuralese.contracts import SymbolPack

    pack = SymbolPack.from_dict(data)
    pack.seal()
    pack_path.write_text(json.dumps(pack.to_dict(), indent=2) + "\n")
    stream.write_text("[0]\n")
    rc = main(["translate", str(pack_path), str(stream)])
    assert rc == 1
    failed = capsys.readouterr()
    assert failed.out == ""
    cert = json.loads(failed.err)
    assert cert["passed"] is False
    rc = main(["translate", str(pack_path), str(stream), "--tau-residual", "0.9"])
    assert rc == 0
    glosses = json.loads(capsys.readouterr().out)
    assert glosses[0]["state"] in {"ok", "unknown", "aliased"}


def test_none_alias_source_key_is_not_stringified():
    from neuralese.contracts import normalize_aliases

    tables = normalize_aliases({None: {7: 0}})
    assert None in tables
    assert "None" not in tables
    pack = make_pack()
    pack.aliases = tables
    pack.seal()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("alias source key is not a non-empty string" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [7])


def test_none_proto_embedding_fails_schema_not_checksum():
    pack = make_pack()
    pack.symbols[0].proto_embedding = None
    pack.seal()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("proto_embedding is not an array" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])
    pack.symbols[0].proto_embedding = [1.0, None]
    pack.seal()
    cert = certify(pack)
    assert not cert.passed
    assert any("proto_embedding[1] is not a number" in f for f in cert.failures)


def test_nan_mdl_bits_fails_schema():
    pack = make_pack()
    pack.mdl_bits = float("nan")
    pack.seal()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("mdl_bits is not finite" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_constructed_private_none_examples_seal_returns_failed_certificate():
    pack = make_pack(include_private=True)
    pack.symbols[0].examples = None
    pack.symbols[0].example_hashes = None
    pack.seal()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("examples is not an array" in f for f in cert.failures)
    assert any("example_hashes is not an array" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_string_mdl_bits_fails_schema():
    pack = make_pack()
    data = pack.to_dict()
    data["mdl_bits"] = str(data["mdl_bits"])
    loaded = pack.from_dict(data)
    assert loaded.mdl_bits == "12.0"
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("mdl_bits is not a number" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


def test_loaded_string_proto_embedding_fails_schema():
    pack = make_pack()
    data = pack.to_dict()
    data["symbols"][0]["proto_embedding"] = [
        str(x) for x in data["symbols"][0]["proto_embedding"]
    ]
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].proto_embedding[0] == "1.0"
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("proto_embedding[0] is not a number" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])
    data["symbols"][0]["proto_embedding"] = None
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].proto_embedding is None
    cert = certify(loaded)
    assert not cert.passed
    assert any("proto_embedding is not an array" in f for f in cert.failures)


def test_none_alias_table_fails_closed():
    pack = make_pack()
    pack.aliases = {"src": None}
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("alias table is not an object" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_non_integer_alias_targets_fail_schema_not_checksum():
    pack = make_pack()
    pack.aliases = {"src": {1: "bad"}}
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("alias entries must be integer-to-integer" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_bool_and_float_alias_targets_are_not_coerced():
    pack = make_pack()
    data = pack.to_dict()
    data["aliases"] = {"src": {"1": False}}
    loaded = pack.from_dict(data)
    assert loaded.aliases["src"][1] is False
    cert = certify(loaded)
    assert not cert.passed
    assert any("alias entries must be integer-to-integer" in f for f in cert.failures)
    data["aliases"] = {"src": {"1": 1.9}}
    loaded = pack.from_dict(data)
    assert loaded.aliases["src"][1] == 1.9
    cert = certify(loaded)
    assert not cert.passed
    assert any("alias entries must be integer-to-integer" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


def test_oversized_confidence_fails_schema_not_overflow():
    pack = make_pack()
    pack.symbols[0].confidence = 10**1000
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("confidence" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_unhashable_class_id_fails_closed():
    pack = make_pack()
    pack.symbols[0].class_id = [0]
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any(
        "class_id is not an integer" in f or "not hashable" in f for f in cert.failures
    )
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_malformed_guards_object_fails_schema_not_attribute_error():
    pack = make_pack()
    pack.guards = {}
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("guards is not a GuardSnapshot" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_float_and_bool_identities_are_not_coerced():
    pack = make_pack()
    data = pack.to_dict()
    data["symbols"][0]["class_id"] = 0.9
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].class_id == 0.9
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("class_id is not an integer" in f for f in cert.failures)
    data = pack.to_dict()
    data["symbols"][0]["code"] = True
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].code is True
    cert = certify(loaded)
    assert not cert.passed
    assert any("code is not an integer" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


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
    with pytest.raises(ValueError, match="invalid pack"):
        load_pack(path)
    stream = tmp_path / "stream.json"
    stream.write_text("[0]\n")
    rc = main(["translate", str(path), str(stream)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "invalid pack" in err

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

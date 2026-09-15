import json

import pytest

from neuralese.audit import certify
from neuralese.cli import main
from neuralese.contracts import Observation, Receipt, UncertifiedPackError
from neuralese.translator import translate_stream

from packutil import make_pack


def test_json_alias_mapping_keys_are_coerced_only_on_load():
    source = "c" * 64
    pack = make_pack(aliases={source: {7: 0}})
    dumped = pack.to_dict()
    assert dumped["aliases"][source]["7"] == 0
    loaded = pack.from_dict(dumped)
    assert loaded.aliases[source][7] == 0
    constructed = make_pack(aliases={source: {"7": 0}}, checksum="unsealed")
    assert constructed.aliases[source] == {"7": 0}
    assert certify(constructed).passed is False


def test_integer_metadata_key_cannot_be_json_laundered(tmp_path):
    from neuralese.adapters import save_pack

    pack = make_pack()
    pack.metadata[1] = "x"
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("metadata keys must be strings" in f for f in cert.failures)
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        pack.to_dict()
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        pack.seal()
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        save_pack(pack, tmp_path / "pack.json")
    nested = make_pack()
    nested.metadata["config"] = {True: 0.55}
    cert = certify(nested)
    assert not cert.passed
    assert any("metadata keys must be strings" in f for f in cert.failures)


def test_symbol_and_receipt_metadata_keys_cannot_be_json_laundered():
    pack = make_pack()
    pack.symbols[0].metadata = {1: "x"}
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("metadata keys must be strings" in f for f in cert.failures)
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        pack.to_dict()
    pack = make_pack(
        guards=None,
        receipts=[Receipt(step="finalize", ok=True, timestamp=1.0, metadata={1: "x"})],
        checksum="unsealed",
    )
    cert = certify(pack)
    assert not cert.passed
    assert any("metadata keys must be strings" in f for f in cert.failures)
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        pack.receipts[0].to_dict()


def test_residual_gate_uses_checksum_canonical_reconstruction_error():
    pack = make_pack(reconstruction_error=0.550000001)
    assert certify(pack, tau_residual=0.55).passed
    original = pack.checksum
    pack.reconstruction_error = 0.55
    assert pack.compute_checksum() == original
    assert certify(pack, tau_residual=0.55).passed


def test_legacy_null_alias_table_is_not_flattened_on_load():
    from neuralese.contracts import normalize_aliases

    pack = make_pack()
    data = pack.to_dict()
    data["aliases"] = {"legacy": None}
    with pytest.raises(ValueError, match="alias source keys must map to an alias table"):
        pack.from_dict(data)
    with pytest.raises(ValueError, match="alias source keys must map to an alias table"):
        normalize_aliases({"legacy": None})
    with pytest.raises(ValueError, match="alias source keys must map to an alias table"):
        normalize_aliases({"c" * 64: None})
    pack.aliases = {"legacy": None}
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("alias table is not an object" in f for f in cert.failures)
    with pytest.raises(TypeError, match="alias table is not an object"):
        pack.to_dict()


def test_flat_non_numeric_alias_key_is_not_legacy_wrapped():
    from neuralese.contracts import normalize_aliases

    with pytest.raises(ValueError, match="legacy or SHA-256"):
        normalize_aliases({"not-a-checksum": 0})
    pack = make_pack()
    data = pack.to_dict()
    data["aliases"] = {"not-a-checksum": 0}
    with pytest.raises(ValueError, match="legacy or SHA-256"):
        pack.from_dict(data)
    loaded = pack.from_dict({**pack.to_dict(), "aliases": {"7": 0}})
    assert loaded.aliases["legacy"][7] == 0


def test_mapping_and_set_embeddings_are_rejected():
    import numpy as np

    with pytest.raises(TypeError, match="embedding must be an array"):
        Observation(observation_id="obs-map", embedding={1: 0.5})
    with pytest.raises(TypeError, match="embedding must be an array"):
        Observation(observation_id="obs-set", embedding={0.5, 1.0})
    obs = Observation(observation_id="obs-np", embedding=np.array([1.0, 0.25]))
    assert obs.embedding == [1.0, 0.25]


def test_private_null_examples_are_not_normalized_to_empty_list():
    pack = make_pack(include_private=True)
    original = pack.checksum
    data = pack.to_dict()
    data["symbols"][0]["examples"] = None
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].examples is None
    assert loaded.compute_checksum() != original
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("examples is not an array" in f for f in cert.failures)
    dumped = loaded.to_dict()
    assert dumped["symbols"][0]["examples"] is None


def test_missing_guards_and_finalize_fail_admission():
    pack = make_pack(guards=None)
    cert = certify(pack)
    assert not cert.admission_valid
    assert not cert.passed
    assert any("admission artifacts are missing" in f for f in cert.failures)
    integrity = certify(pack, policy="integrity")
    assert integrity.passed
    assert not integrity.admission_valid
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])

def test_cyclic_pack_metadata_fails_schema_not_recursion():
    pack = make_pack()
    pack.metadata["self"] = pack.metadata
    cert = certify(pack)
    assert cert.passed is False
    assert cert.integrity_valid is False
    assert any("contains a cycle" in f for f in cert.failures)
    with pytest.raises(TypeError, match="contains a cycle"):
        pack.to_dict()
    with pytest.raises(TypeError, match="contains a cycle"):
        pack.seal()
    with pytest.raises(TypeError, match="contains a cycle"):
        pack.compute_checksum()


def test_cyclic_symbol_and_receipt_metadata_fails_schema_not_recursion():
    pack = make_pack()
    pack.symbols[0].metadata["self"] = pack.symbols[0].metadata
    cert = certify(pack)
    assert cert.passed is False
    assert cert.integrity_valid is False
    assert any("contains a cycle" in f for f in cert.failures)
    with pytest.raises(TypeError, match="contains a cycle"):
        pack.to_dict()

    pack = make_pack(
        guards=None,
        receipts=[Receipt(step="finalize", ok=True, timestamp=1.0)],
    )
    pack.receipts[0].metadata["self"] = pack.receipts[0].metadata
    cert = certify(pack)
    assert cert.passed is False
    assert any("contains a cycle" in f for f in cert.failures)
    with pytest.raises(TypeError, match="contains a cycle"):
        pack.receipts[0].to_dict()
    with pytest.raises(TypeError, match="contains a cycle"):
        pack.to_dict()


def test_nested_list_cycle_in_metadata_fails_schema_not_recursion():
    pack = make_pack()
    nested: list = []
    nested.append(nested)
    pack.metadata["items"] = nested
    cert = certify(pack)
    assert cert.passed is False
    assert cert.integrity_valid is False
    assert any("contains a cycle" in f for f in cert.failures)
    with pytest.raises(TypeError, match="contains a cycle"):
        pack.to_dict()

    a: dict = {}
    b = {"a": a}
    a["b"] = b
    indirect = make_pack()
    indirect.metadata["root"] = a
    cert = certify(indirect)
    assert cert.passed is False
    assert any("contains a cycle" in f for f in cert.failures)


def test_shared_nested_metadata_objects_are_not_treated_as_cycles():
    pack = make_pack()
    shared = {"tau_residual": 0.55}
    pack.metadata["left"] = shared
    pack.metadata["right"] = shared
    pack.seal()
    assert certify(pack).passed
    dumped = pack.to_dict()
    dumped["metadata"]["left"]["tau_residual"] = 99
    assert pack.metadata["left"]["tau_residual"] == 0.55
    assert dumped["metadata"]["right"]["tau_residual"] == 0.55
    assert dumped["metadata"]["left"] is not dumped["metadata"]["right"]

def test_tiny_negative_residual_fails_residual_ok_not_rounding():
    pack = make_pack(reconstruction_error=-1e-9)
    cert = certify(pack)
    assert cert.residual_ok is False
    assert cert.integrity_valid is False
    assert cert.passed is False
    assert any("negative" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_finalize_metadata_decision_must_match_pack_decision():
    pack = make_pack(
        decision="accept",
        receipts=[
            Receipt(
                step="finalize",
                ok=True,
                timestamp=1.0,
                metadata={"decision": "reject"},
            )
        ],
    )
    cert = certify(pack)
    assert cert.admission_valid is False
    assert cert.passed is False
    assert any(
        "finalize receipt decision does not match pack.decision" in f
        for f in cert.failures
    )
    integrity = certify(pack, policy="integrity")
    assert integrity.passed
    assert not integrity.admission_valid
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])
    aligned = make_pack(
        decision="accept",
        receipts=[
            Receipt(
                step="finalize",
                ok=True,
                timestamp=1.0,
                metadata={"decision": "accept"},
            )
        ],
    )
    assert certify(aligned).passed
    missing = make_pack(
        receipts=[Receipt(step="finalize", ok=True, timestamp=1.0)],
    )
    assert certify(missing).passed


def test_non_mapping_aliases_cannot_be_serialized():
    pack = make_pack()
    pack.aliases = None
    cert = certify(pack)
    assert cert.integrity_valid is False
    assert cert.passed is False
    assert any("aliases is not an object" in f for f in cert.failures)
    with pytest.raises(TypeError, match="aliases is not an object"):
        pack.to_dict()
    with pytest.raises(TypeError, match="aliases is not an object"):
        pack.seal()
    pack = make_pack()
    pack.aliases = []
    cert = certify(pack)
    assert not cert.passed
    with pytest.raises(TypeError, match="aliases is not an object"):
        pack.to_dict()

def test_cyclic_supplied_observation_metadata_fails_evidence_not_recursion():
    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    obs = Observation(observation_id=obs_id, embedding=[1.0], text="hello there")
    obs.metadata["self"] = obs.metadata
    cert = certify(pack, observations=[obs])
    assert cert.evidence_valid is False
    assert cert.passed is False
    assert any("evidence content is not verifiable" in f for f in cert.failures)


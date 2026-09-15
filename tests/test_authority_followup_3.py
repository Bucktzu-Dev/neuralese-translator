import json

import pytest

from neuralese.audit import certify
from neuralese.cli import main
from neuralese.contracts import Observation, Receipt, UncertifiedPackError
from neuralese.translator import translate_stream

from packutil import make_pack


def test_nonfinite_pack_timestamp_fails_schema():
    pack = make_pack()
    pack.timestamp = float("nan")
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("timestamp is not finite" in f for f in cert.failures)
    pack.timestamp = float("inf")
    cert = certify(pack)
    assert not cert.passed
    assert any("timestamp is not finite" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_falsey_receipt_metadata_fails_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    data["receipts"] = [
        {"step": "finalize", "ok": True, "timestamp": 1.0, "metadata": False}
    ]
    with pytest.raises(TypeError, match="metadata must be an object or null"):
        pack.from_dict(data)
    data["receipts"] = [
        {"step": "finalize", "ok": True, "timestamp": 1.0, "metadata": []}
    ]
    with pytest.raises(TypeError, match="metadata must be an object or null"):
        pack.from_dict(data)
    data["receipts"] = [
        {"step": "finalize", "ok": True, "timestamp": 1.0, "metadata": ""}
    ]
    with pytest.raises(TypeError, match="metadata must be an object or null"):
        pack.from_dict(data)
    data["receipts"] = [
        {"step": "finalize", "ok": True, "timestamp": 1.0, "metadata": None}
    ]
    loaded = pack.from_dict(data)
    assert loaded.receipts[0].metadata == {}


def test_private_null_example_hashes_are_not_derived_from_examples():
    pack = make_pack(include_private=True)
    data = pack.to_dict()
    data["symbols"][0]["examples"] = ["hello there"]
    data["symbols"][0]["example_hashes"] = None
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].example_hashes is None
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("example_hashes is not an array" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])

def test_unserializable_metadata_fails_checksum_not_type_error():
    pack = make_pack()
    pack.metadata["x"] = object()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("not JSON-serializable" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_falsey_and_string_pack_timestamp_are_not_coerced():
    pack = make_pack()
    data = pack.to_dict()
    data["timestamp"] = None
    loaded = pack.from_dict(data)
    assert loaded.timestamp is None
    cert = certify(loaded)
    assert not cert.passed
    assert any("timestamp is not a number" in f for f in cert.failures)
    data["timestamp"] = False
    loaded = pack.from_dict(data)
    assert loaded.timestamp is False
    cert = certify(loaded)
    assert not cert.passed
    assert any("timestamp is not a number" in f for f in cert.failures)
    data["timestamp"] = "1.0"
    loaded = pack.from_dict(data)
    assert loaded.timestamp == "1.0"
    cert = certify(loaded)
    assert not cert.passed
    assert any("timestamp is not a number" in f for f in cert.failures)
    del data["timestamp"]
    omitted = pack.from_dict(data)
    assert omitted.timestamp == 0.0


def test_loaded_string_receipt_numerics_are_not_coerced():
    pack = make_pack(guards=None)
    data = pack.to_dict()
    data["guards"] = None
    data["receipts"] = [{"step": "finalize", "ok": True, "timestamp": "1.0"}]
    loaded = pack.from_dict(data)
    assert loaded.receipts[0].timestamp == "1.0"
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("receipts[0].timestamp is not a number" in f for f in cert.failures)
    data["receipts"] = [
        {"step": "finalize", "ok": True, "timestamp": 1.0, "kappa": "0.8"}
    ]
    loaded = pack.from_dict(data)
    assert loaded.receipts[0].kappa == "0.8"
    cert = certify(loaded)
    assert not cert.passed
    assert any("receipts[0].kappa is not a number" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


def test_null_codebook_uncertified_translate_does_not_traceback():
    pack = make_pack()
    pack.codebook = None
    glosses = translate_stream(pack, [0], require_certified=False)
    assert glosses[0].state in {"ok", "unknown", "aliased"}
    assert pack.resolve_code(0)[0] == 0

def test_loaded_non_string_receipt_step_is_not_coerced():
    pack = make_pack(guards=None)
    data = pack.to_dict()
    data["guards"] = None
    data["receipts"] = [{"step": None, "ok": True, "timestamp": 1.0}]
    loaded = pack.from_dict(data)
    assert loaded.receipts[0].step is None
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("receipts[0].step is not a string" in f for f in cert.failures)
    data["receipts"] = [{"step": 1, "ok": True, "timestamp": 1.0}]
    loaded = pack.from_dict(data)
    assert loaded.receipts[0].step == 1
    cert = certify(loaded)
    assert not cert.passed
    assert any("receipts[0].step is not a string" in f for f in cert.failures)
    data["receipts"] = [{"step": True, "ok": True, "timestamp": 1.0}]
    loaded = pack.from_dict(data)
    assert loaded.receipts[0].step is True
    cert = certify(loaded)
    assert not cert.passed
    assert any("receipts[0].step is not a string" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


def test_alias_collision_rejects_cycles_with_valid_targets():
    from neuralese.alphabet import _alias_collision, _match_aliases
    from neuralese.contracts import Symbol

    codebook = {0: 0, 1: 1}
    source = "c" * 64
    assert _alias_collision({source: {0: 1, 1: 0}}, codebook) is True
    assert _alias_collision({source: {0: 99}}, codebook) is True
    assert _alias_collision({source: {0: 1}}, codebook) is False
    previous = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0, 0.0],
                observation_ids=["obs-a"],
                definition="a",
                confidence=0.8,
            ),
            Symbol(
                class_id=1,
                code=1,
                proto_embedding=[0.0, 1.0, 0.0],
                observation_ids=["obs-b"],
                definition="b",
                confidence=0.8,
            ),
        ]
    )
    swapped = [
        Symbol(
            class_id=0,
            code=0,
            proto_embedding=[0.0, 1.0, 0.0],
            observation_ids=["obs-c"],
            definition="c",
            confidence=0.8,
        ),
        Symbol(
            class_id=1,
            code=1,
            proto_embedding=[1.0, 0.0, 0.0],
            observation_ids=["obs-d"],
            definition="d",
            confidence=0.8,
        ),
    ]
    remap = _match_aliases(previous, swapped, threshold=0.5)
    assert remap == {0: 1, 1: 0}
    assert _alias_collision({previous.checksum: remap}, codebook) is True


def test_loaded_numeric_checksum_is_not_coerced():
    pack = make_pack()
    data = pack.to_dict()
    numeric = int("1" + "0" * 63)
    data["checksum"] = numeric
    loaded = pack.from_dict(data)
    assert loaded.checksum == numeric
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("checksum is not full SHA-256" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


def test_learn_rejects_blank_observation_ids():
    from neuralese.alphabet import LearnConfig, learn_pack
    from neuralese.contracts import Observation

    with pytest.raises(ValueError, match="blank observation_id"):
        learn_pack(
            [Observation(observation_id="", text="hello there friend")],
            config=LearnConfig(n_symbols=1, min_cluster_size=1, seed=0),
        )
    with pytest.raises(ValueError, match="blank observation_id"):
        learn_pack(
            [Observation(observation_id="   ", text="hello there friend")],
            config=LearnConfig(n_symbols=1, min_cluster_size=1, seed=0),
        )


def test_loaded_numeric_pack_id_is_not_coerced():
    pack = make_pack()
    data = pack.to_dict()
    data["pack_id"] = 123
    loaded = pack.from_dict(data)
    assert loaded.pack_id == 123
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("pack_id is not a string" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])



def test_empty_definition_fails_even_when_unglossed_allowed():
    from neuralese.contracts import Symbol

    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0, 0.0],
                observation_ids=["obs-hello"],
                definition=None,
                confidence=0.8,
            )
        ]
    )
    cert = certify(pack, require_gloss=False)
    assert not cert.gloss_bound
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("bound English" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0], require_gloss=False)
    pack.symbols[0].definition = ""
    pack.seal()
    cert = certify(pack, require_gloss=False)
    assert not cert.passed
    pack.symbols[0].definition = "   "
    pack.seal()
    cert = certify(pack, require_gloss=False)
    assert not cert.passed
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0], require_gloss=False)


def test_loaded_non_string_parent_pack_id_fails_schema_after_reseal():
    pack = make_pack()
    data = pack.to_dict()
    data["parent_pack_id"] = 123
    loaded = pack.from_dict(data)
    assert loaded.parent_pack_id == 123
    loaded.seal()
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("parent_pack_id is not a string" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])
    data["parent_pack_id"] = ["parent"]
    loaded = pack.from_dict(data)
    assert loaded.parent_pack_id == ["parent"]
    loaded.seal()
    cert = certify(loaded)
    assert not cert.passed
    assert any("parent_pack_id is not a string" in f for f in cert.failures)
    data["parent_pack_id"] = None
    loaded = pack.from_dict(data)
    assert loaded.parent_pack_id is None
    loaded.seal()
    assert certify(loaded).passed



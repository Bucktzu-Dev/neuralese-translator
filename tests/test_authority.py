import pytest

from neuralese.audit import certify
from neuralese.contracts import (
    Receipt,
    Symbol,
    UncertifiedPackError,
    normalize_aliases,
)
from neuralese.translator import translate_stream

from packutil import make_pack, passing_guards


def test_translate_refuses_forged_definition_after_seal():
    pack = make_pack()
    pack.symbols[0].definition = "forged after seal"
    with pytest.raises(UncertifiedPackError) as err:
        translate_stream(pack, [0])
    assert err.value.certificate.passed is False
    assert err.value.certificate.integrity_valid is False


def test_seal_covers_prototype_confidence_examples_survival_lineage_guards_mdl_receipts():
    pack = make_pack()
    original = pack.checksum
    assert len(original) == 64

    pack.symbols[0].proto_embedding = [0.5, 0.5, 0.5]
    assert pack.compute_checksum() != original
    pack.symbols[0].proto_embedding = [1.0, 0.0, 0.0]

    pack.symbols[0].confidence = 0.1
    assert pack.compute_checksum() != original
    pack.symbols[0].confidence = 0.8

    pack.symbols[0].example_hashes = ["a" * 64]
    assert pack.compute_checksum() != original
    pack.symbols[0].example_hashes = []

    pack.symbols[0].survival = 0.5
    assert pack.compute_checksum() != original
    pack.symbols[0].survival = 1.0

    pack.parent_checksum = "b" * 64
    assert pack.compute_checksum() != original
    pack.parent_checksum = None

    pack.guards = passing_guards(kappa_avg=0.1)
    assert pack.compute_checksum() != original
    pack.guards = passing_guards()

    pack.mdl_bits = 99.0
    assert pack.compute_checksum() != original
    pack.mdl_bits = 12.0

    pack.receipts = [Receipt(step="finalize", ok=True, timestamp=1.0)]
    assert pack.compute_checksum() != original
    pack.receipts = []

    pack.metadata = {"tampered": True}
    assert pack.compute_checksum() != original


def test_fabricated_observation_id_fails_evidence():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0],
                observation_ids=["not-resolved-anywhere"],
                definition="ghost",
                confidence=0.5,
            )
        ],
        evidence={},
    )
    cert = certify(pack)
    assert not cert.passed
    assert not cert.evidence_valid
    assert any("not-resolved-anywhere" in f for f in cert.failures)


def test_reject_decision_fails_admission_and_default_certify():
    pack = make_pack(
        decision="reject",
        guards=passing_guards(pass_all=False, pass_kappa=False),
    )
    cert = certify(pack)
    assert cert.integrity_valid
    assert not cert.admission_valid
    assert not cert.passed


def test_current_code_is_not_redirected_by_alias():
    pack = make_pack(aliases={0: 1})
    cert = certify(pack)
    assert cert.passed
    glosses = translate_stream(pack, [0])
    assert glosses[0].state == "ok"
    assert glosses[0].class_id == 0
    assert "hello" in glosses[0].english.lower()


def test_negative_residual_fails_schema():
    pack = make_pack(reconstruction_error=-5.0)
    cert = certify(pack)
    assert not cert.passed
    assert not cert.integrity_valid
    assert any("negative" in f for f in cert.failures)


def test_duplicate_class_ids_fail_addressable():
    pack = make_pack(
        symbols=[
            Symbol(0, 0, [1.0], ["obs-a"], definition="a", confidence=0.5),
            Symbol(0, 1, [0.0, 1.0], ["obs-b"], definition="b", confidence=0.5),
        ]
    )
    cert = certify(pack)
    assert not cert.addressable
    assert any("duplicate class_id" in f for f in cert.failures)


def test_public_pack_omits_raw_examples():
    pack = make_pack(
        include_private=False,
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0],
                observation_ids=["obs-1"],
                definition="Symbol for hello.",
                examples=["secret subjective text"],
                example_hashes=["c" * 64],
                confidence=0.5,
            )
        ],
    )
    dumped = pack.to_dict()
    assert "examples" not in dumped["symbols"][0]
    assert dumped["symbols"][0]["example_hashes"] == ["c" * 64]


def test_integrity_policy_can_pass_when_admission_fails():
    pack = make_pack(decision="reject")
    cert = certify(pack, policy="integrity")
    assert cert.integrity_valid
    assert cert.passed
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0], policy="default")
    with pytest.raises(ValueError, match="does not authorize translation"):
        translate_stream(pack, [0], policy="integrity")
    with pytest.raises(ValueError, match="does not authorize translation"):
        translate_stream(pack, [0], policy="integrity", require_certified=False)


def test_blank_observation_id_fails_evidence():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0],
                observation_ids=["   "],
                definition="blank id",
                confidence=0.5,
            )
        ],
        evidence={"   ": "a" * 64},
    )
    cert = certify(pack)
    assert not cert.evidence_valid
    assert any("blank observation_id" in f for f in cert.failures)


def test_fabricated_digest_fails_when_observations_supplied():
    from neuralese.contracts import Observation

    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0],
                observation_ids=["obs-hello"],
                definition="hello",
                confidence=0.5,
            )
        ],
        evidence={"obs-hello": "a" * 64},
    )
    cert = certify(pack)
    assert cert.evidence_valid
    cert = certify(
        pack,
        observations=[Observation(observation_id="obs-hello", text="hello there")],
    )
    assert not cert.evidence_valid
    assert cert.details["observations_checked"] is True
    assert any("does not match content" in f for f in cert.failures)


def test_unknown_source_pack_does_not_merge_unrelated_aliases():
    pack = make_pack(aliases={"cccc" * 16: {7: 0}})
    glosses = translate_stream(pack, [7], source_pack_checksum="dddd" * 16)
    assert glosses[0].state == "unknown"
    assert glosses[0].resolved_code is None


def test_example_hashes_normalized_before_seal():
    symbol = Symbol(
        class_id=0,
        code=0,
        proto_embedding=[1.0],
        observation_ids=["obs-hello"],
        definition="hello",
        examples=["hello there"],
        confidence=0.5,
    )
    assert len(symbol.example_hashes) == 1
    pack = make_pack(symbols=[symbol])
    reloaded = pack.from_dict(pack.to_dict())
    assert reloaded.compute_checksum() == pack.checksum


def test_v010_positional_symbol_confidence_still_binds():
    symbol = Symbol(0, 0, [1.0], ["obs-a"], None, [], 0.5)
    assert symbol.confidence == 0.5
    assert symbol.example_hashes == []


def test_non_sha256_example_hash_fails_integrity():
    pack = make_pack()
    pack.symbols[0].example_hashes = ["raw secret"]
    pack.seal()
    cert = certify(pack, policy="integrity")
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("example_hashes" in f and "not SHA-256" in f for f in cert.failures)


def test_example_hash_with_trailing_newline_fails_integrity():
    pack = make_pack()
    pack.symbols[0].example_hashes = ["c" * 64 + "\n"]
    pack.seal()
    cert = certify(pack, policy="integrity")
    assert not cert.integrity_valid
    assert any("example_hashes" in f and "not SHA-256" in f for f in cert.failures)


def test_evidence_digest_with_trailing_newline_fails():
    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    pack.evidence[obs_id] = pack.evidence[obs_id] + "\n"
    pack.seal()
    cert = certify(pack)
    assert not cert.evidence_valid
    assert any("not SHA-256" in f for f in cert.failures)


def test_empty_source_pack_checksum_is_explicit():
    pack = make_pack(aliases={"cccc" * 16: {7: 0}})
    glosses = translate_stream(pack, [7], source_pack_checksum="")
    assert glosses[0].state == "unknown"
    assert glosses[0].resolved_code is None


def test_versioned_aliases_without_source_do_not_merge():
    pack = make_pack(aliases={"cccc" * 16: {7: 0}})
    glosses = translate_stream(pack, [7])
    assert glosses[0].state == "unknown"


def test_duplicate_supplied_observations_fail_evidence():
    from neuralese.contracts import Observation

    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    cert = certify(
        pack,
        observations=[
            Observation(observation_id=obs_id, text="hello there"),
            Observation(observation_id=obs_id, text="different content entirely"),
        ],
    )
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("duplicate observation_id" in f for f in cert.failures)


def test_observation_without_embedding_or_text_fails_closed():
    from neuralese.contracts import Observation

    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    cert = certify(pack, observations=[Observation(observation_id=obs_id)])
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("neither embedding nor text" in f for f in cert.failures)


def test_foreign_decoder_version_fails_integrity():
    pack = make_pack(decoder_version="0.0.0")
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("decoder_version" in f for f in cert.failures)


def test_numpy_embedding_can_be_hashed():
    import numpy as np

    from neuralese.contracts import Observation, observation_content_hash

    vec = np.array([1.0, 0.0, 0.25])
    digest = observation_content_hash("obs-1", vec, "hello")
    assert len(digest) == 64
    obs = Observation(observation_id="obs-1", embedding=vec, text="hello")
    assert obs.content_hash() == digest
    assert obs.embedding == [1.0, 0.0, 0.25]


def test_unknown_decision_fails_integrity_schema():
    pack = make_pack(decision="garbage")
    cert = certify(pack, policy="integrity")
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("decision" in f for f in cert.failures)


def test_empty_decoder_version_is_preserved_and_fails_integrity():
    pack = make_pack()
    data = pack.to_dict()
    data["decoder_version"] = ""
    loaded = pack.from_dict(data)
    assert loaded.decoder_version == ""
    loaded.seal()
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert any("decoder_version" in f for f in cert.failures)


def test_empty_decision_is_not_coerced_to_accept():
    pack = make_pack()
    data = pack.to_dict()
    data["decision"] = ""
    loaded = pack.from_dict(data)
    assert loaded.decision == ""
    loaded.seal()
    cert = certify(loaded, policy="integrity")
    assert not cert.integrity_valid
    assert any("decision" in f for f in cert.failures)


def test_null_metadata_loads_as_empty_mapping():
    pack = make_pack()
    data = pack.to_dict()
    data["metadata"] = None
    loaded = pack.from_dict(data)
    assert loaded.metadata == {}
    loaded.seal()
    cert = certify(loaded)
    assert cert.passed


def test_int_and_str_observation_ids_are_duplicates():
    from neuralese.contracts import Observation

    pack = make_pack()
    cert = certify(
        pack,
        observations=[
            Observation(observation_id=1, text="hello there"),
            Observation(observation_id="1", text="different content entirely"),
        ],
    )
    assert not cert.evidence_valid
    assert any("duplicate observation_id" in f for f in cert.failures)


def test_inconsistent_pass_all_fails_integrity():
    pack = make_pack(guards=passing_guards(pass_all=True, pass_kappa=False))
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.admission_valid
    assert not cert.passed
    assert any("inconsistent" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0], policy="strict")


def test_shared_live_observation_id_fails_evidence():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0],
                observation_ids=["shared-obs"],
                definition="first",
                confidence=0.5,
            ),
            Symbol(
                class_id=1,
                code=1,
                proto_embedding=[0.0, 1.0],
                observation_ids=["shared-obs"],
                definition="second",
                confidence=0.5,
            ),
        ]
    )
    cert = certify(pack)
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("multiple live symbols" in f for f in cert.failures)


def test_unglossed_sentinel_fails_gloss_bound():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0],
                observation_ids=["obs-1"],
                definition="[unglossed]",
                confidence=0.0,
            )
        ]
    )
    cert = certify(pack)
    assert not cert.gloss_bound
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("bound English" in f for f in cert.failures)
    allowed = certify(pack, require_gloss=False)
    assert allowed.gloss_bound
    assert allowed.passed
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_string_guard_flags_fail_admission():
    pack = make_pack()
    data = pack.to_dict()
    guards = data["guards"]
    for key in ("pass_kappa", "pass_residual", "pass_mdl", "pass_persist", "pass_compat"):
        guards[key] = "false"
    guards["pass_all"] = "true"
    loaded = pack.from_dict(data)
    loaded.seal()
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.admission_valid
    assert not cert.passed
    assert any("not a boolean" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0], policy="strict")


def test_string_false_receipt_ok_fails_admission():
    pack = make_pack(guards=None)
    data = pack.to_dict()
    data["guards"] = None
    data["receipts"] = [{"step": "finalize", "ok": "false", "timestamp": 1.0}]
    loaded = pack.from_dict(data)
    assert loaded.receipts[0].ok == "false"
    loaded.seal()
    cert = certify(loaded)
    assert not cert.admission_valid
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("not a boolean" in f for f in cert.failures)


def test_non_mapping_evidence_loads_and_fails_closed():
    pack = make_pack()
    data = pack.to_dict()
    data["evidence"] = []
    loaded = pack.from_dict(data)
    assert loaded.evidence == {}
    loaded.seal()
    cert = certify(loaded)
    assert not cert.evidence_valid
    assert not cert.passed


def test_non_string_definition_fails_load():
    pack = make_pack()
    data = pack.to_dict()
    data["symbols"][0]["definition"] = 1
    with pytest.raises(TypeError, match="definition"):
        pack.from_dict(data)


def test_string_quarantined_fails_integrity():
    pack = make_pack()
    data = pack.to_dict()
    data["symbols"][0]["quarantined"] = "false"
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].quarantined == "false"
    loaded.seal()
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("not a boolean" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


def test_string_include_private_fails_integrity():
    pack = make_pack()
    data = pack.to_dict()
    data["include_private"] = "false"
    loaded = pack.from_dict(data)
    assert loaded.include_private == "false"
    loaded.seal()
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("include_private" in f and "not a boolean" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


def test_non_object_pack_fails_from_dict():
    with pytest.raises(TypeError, match="JSON object"):
        make_pack().from_dict([])
    with pytest.raises(TypeError, match="JSON object"):
        make_pack().from_dict(None)


def test_non_object_symbol_fails_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    data["symbols"] = [[]]
    with pytest.raises(TypeError, match="symbol record"):
        pack.from_dict(data)


def test_mixed_alias_shapes_fail_load():
    pack = make_pack()
    data = pack.to_dict()
    data["aliases"] = {"": {}, "src": 1}
    with pytest.raises(ValueError, match="uniformly nested"):
        pack.from_dict(data)


def test_legacy_source_pack_checksum_does_not_select_legacy_table():
    pack = make_pack(aliases={7: 0})
    glosses = translate_stream(pack, [7], source_pack_checksum="legacy")
    assert glosses[0].state == "unknown"
    assert pack.alias_table("legacy") == {}
    assert pack.alias_table()[7] == 0


def test_falsey_symbols_fail_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    for symbols in (0, "", {}):
        data["symbols"] = symbols
        with pytest.raises(TypeError, match="symbols must be an array"):
            pack.from_dict(data)


def test_falsey_guards_fail_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    for guards in ([], False, ""):
        data["guards"] = guards
        with pytest.raises(TypeError, match="guards must be an object or null"):
            pack.from_dict(data)
    data["guards"] = {}
    with pytest.raises(KeyError):
        pack.from_dict(data)
    data["guards"] = None
    loaded = pack.from_dict(data)
    assert loaded.guards is None


def test_falsey_aliases_fail_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    for aliases in ([], False, "", 0):
        data["aliases"] = aliases
        with pytest.raises(ValueError, match="aliases must be a mapping"):
            pack.from_dict(data)
        with pytest.raises(ValueError, match="aliases must be a mapping"):
            normalize_aliases(aliases)
    assert normalize_aliases(None) == {}
    assert normalize_aliases({}) == {}
    missing = dict(pack.to_dict())
    del missing["aliases"]
    loaded = pack.from_dict(missing)
    assert loaded.aliases == {}
    data["aliases"] = None
    loaded = pack.from_dict(data)
    assert loaded.aliases == {}


def test_string_include_private_does_not_serialize_examples():
    pack = make_pack(
        include_private=True,
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0],
                observation_ids=["obs-hello"],
                definition="hello",
                examples=["secret subjective text"],
                confidence=0.5,
            )
        ],
    )
    data = pack.to_dict()
    assert data["symbols"][0]["examples"] == ["secret subjective text"]
    data["include_private"] = "false"
    loaded = pack.from_dict(data)
    assert loaded.include_private == "false"
    dumped = loaded.to_dict()
    assert "examples" not in dumped["symbols"][0]
    loaded.seal()
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed


def test_provisional_failed_finalize_fails_admission():
    pack = make_pack(
        decision="accept_provisional",
        guards=passing_guards(pass_all=False, pass_kappa=False),
        receipts=[Receipt(step="finalize", ok=False, timestamp=1.0)],
    )
    cert = certify(pack)
    assert cert.integrity_valid
    assert not cert.admission_valid
    assert not cert.passed
    assert any("finalize receipt is not ok" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_provisional_passing_finalize_allows_failed_guards():
    pack = make_pack(
        decision="accept_provisional",
        guards=passing_guards(pass_all=False, pass_kappa=False),
        receipts=[Receipt(step="finalize", ok=True, timestamp=1.0)],
    )
    cert = certify(pack)
    assert cert.admission_valid
    assert cert.passed


def test_non_string_evidence_digest_fails_closed():
    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    pack.evidence[obs_id] = None
    pack.seal()
    cert = certify(pack)
    assert not cert.evidence_valid
    assert not cert.passed
    pack.evidence[obs_id] = 123
    pack.seal()
    cert = certify(pack)
    assert not cert.evidence_valid
    assert any("not SHA-256" in f for f in cert.failures)


def test_numeric_64_digit_evidence_digest_is_not_coerced():
    pack = make_pack()
    data = pack.to_dict()
    obs_id = pack.symbols[0].observation_ids[0]
    numeric = 10**63
    assert len(str(numeric)) == 64
    data["evidence"][obs_id] = numeric
    loaded = pack.from_dict(data)
    assert loaded.evidence[obs_id] == numeric
    loaded.seal()
    cert = certify(loaded)
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("not SHA-256" in f for f in cert.failures)


def test_numeric_64_digit_example_hash_is_not_coerced():
    pack = make_pack()
    data = pack.to_dict()
    numeric = 10**63
    data["symbols"][0]["example_hashes"] = [numeric]
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].example_hashes == [numeric]
    loaded.seal()
    cert = certify(loaded, policy="integrity")
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("example_hashes" in f and "not SHA-256" in f for f in cert.failures)


def test_empty_alias_source_key_is_not_selectable():
    from neuralese.contracts import select_alias_table

    pack = make_pack()
    data = pack.to_dict()
    data["aliases"] = {"": {"7": 0}}
    with pytest.raises(ValueError, match="non-empty"):
        pack.from_dict(data)
    assert select_alias_table({"": {7: 0}}, "") == {}
    glosses = translate_stream(pack, [7], source_pack_checksum="")
    assert glosses[0].state == "unknown"

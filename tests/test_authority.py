import pytest

from neuralese.audit import certify
from neuralese.contracts import (
    AuditCertificate,
    Observation,
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
    assert cert.addressable is False
    assert any("historical mapping is ambiguous" in f for f in cert.failures)
    assert cert.passed is False
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])
    historical = make_pack(aliases={7: 0})
    cert = certify(historical)
    assert cert.passed, cert.failures
    glosses = translate_stream(historical, [7])
    assert glosses[0].state == "aliased"
    assert glosses[0].resolved_code == 0
    glosses = translate_stream(historical, [0])
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
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("must not retain examples" in f for f in cert.failures)


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

    from neuralese.contracts import observation_content_hash

    vec = np.array([1.0, 0.0, 0.25])
    digest = observation_content_hash("obs-1", vec, "hello")
    assert len(digest) == 64
    obs = Observation(observation_id="obs-1", embedding=vec, text="hello")
    assert obs.content_hash() == digest
    assert obs.embedding == [1.0, 0.0, 0.25]
    scalars = Observation(
        observation_id="obs-2",
        embedding=[np.int64(1), np.float32(0.25)],
        text="hello",
    )
    assert scalars.embedding == [1.0, 0.25]


def test_text_only_content_hash_matches_learn_evidence():
    from neuralese.adapters import ensure_embedding
    from neuralese.alphabet import LearnConfig, learn_pack

    obs = [
        Observation(observation_id="t-1", text="hello there friend"),
        Observation(observation_id="t-2", text="hello there pal"),
        Observation(observation_id="t-3", text="audit the trail please"),
        Observation(observation_id="t-4", text="audit receipts stay bound"),
    ]
    pack = learn_pack(obs, config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0))
    for row in obs:
        filled = ensure_embedding(Observation.from_dict(row.to_dict()))
        assert row.content_hash() == filled.content_hash()
        assert pack.evidence[row.observation_id] == row.content_hash()
        assert row.embedding == []


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


def test_falsey_metadata_fails_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    for metadata in ([], False, ""):
        data["metadata"] = metadata
        with pytest.raises(TypeError, match="metadata must be an object or null"):
            pack.from_dict(data)


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
    assert not cert.passed
    assert any("supplied observation_id 1 is not a string" in f for f in cert.failures)
    assert not any("duplicate observation_id" in f for f in cert.failures)


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
    glosses = translate_stream(pack, [0], require_gloss=False)
    assert glosses[0].english.startswith("[unglossed:")
    assert glosses[0].confidence == 0.0


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
    assert loaded.evidence == []
    cert = certify(loaded)
    assert not cert.evidence_valid
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("evidence is not an object" in f for f in cert.failures)
    data["evidence"] = None
    loaded = pack.from_dict(data)
    assert loaded.evidence is None
    cert = certify(loaded)
    assert not cert.evidence_valid
    assert not cert.passed
    assert any("evidence is not an object" in f for f in cert.failures)


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
    assert loaded.aliases is None
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("aliases is not an object" in f for f in cert.failures)
    with pytest.raises(TypeError, match="aliases is not an object"):
        loaded.to_dict()


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
    assert loaded.symbols[0].examples == []
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


def test_null_observation_ids_in_symbol_fail_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    data["symbols"][0]["observation_ids"] = [None]
    with pytest.raises(TypeError, match="observation_ids must contain strings"):
        pack.from_dict(data)
    data["symbols"][0]["observation_ids"] = "obs-hello"
    with pytest.raises(TypeError, match="observation_ids must be an array"):
        pack.from_dict(data)
    data["symbols"][0]["observation_ids"] = 0
    with pytest.raises(TypeError, match="observation_ids must be an array"):
        pack.from_dict(data)


def test_example_hash_uses_canonical_json_envelope():
    import hashlib

    from neuralese.contracts import example_hash, sha256_hex

    text = "hello there"
    assert example_hash(text) == sha256_hex({"example": text})
    assert example_hash(text) != hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_learn_hashes_original_nonblank_observation_text():
    from neuralese.alphabet import LearnConfig, learn_pack
    from neuralese.contracts import example_hash

    padded = "  hello there friend  "
    obs = [
        Observation(observation_id="t-1", text=padded),
        Observation(observation_id="t-2", text="hello there pal"),
        Observation(observation_id="t-3", text="audit the trail please"),
        Observation(observation_id="t-4", text="audit receipts stay bound"),
    ]
    pack = learn_pack(obs, config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0))
    hashed = example_hash(padded)
    stripped = example_hash(padded.strip())
    assert hashed != stripped
    assert any(hashed in symbol.example_hashes for symbol in pack.symbols)
    assert all(stripped not in symbol.example_hashes for symbol in pack.symbols)


def test_constructed_none_observation_id_fails_certify():
    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    digest = pack.evidence[obs_id]
    pack.symbols[0].observation_ids = [None]
    pack.evidence["None"] = digest
    pack.seal()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.evidence_valid
    assert any("not a string" in f for f in cert.failures)


def test_constructed_string_confidence_fails_certify():
    pack = make_pack()
    pack.symbols[0].confidence = "0.5"
    pack.seal()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("confidence is not a number" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_constructed_bool_survival_fails_certify():
    pack = make_pack()
    pack.symbols[0].survival = True
    pack.seal()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("survival is not a number" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_constructed_string_reconstruction_error_fails_certify():
    pack = make_pack()
    pack.reconstruction_error = "0.1"
    pack.seal()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("reconstruction_error is not a number" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_constructed_non_numeric_confidence_returns_failed_certificate():
    pack = make_pack()
    pack.symbols[0].confidence = "not-a-number"
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("confidence is not a number" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_string_confidence_is_not_coerced():
    pack = make_pack()
    data = pack.to_dict()
    data["symbols"][0]["confidence"] = "0.5"
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].confidence == "0.5"
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert any("confidence is not a number" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


def test_loaded_bool_survival_is_not_coerced():
    pack = make_pack()
    data = pack.to_dict()
    data["symbols"][0]["survival"] = True
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].survival is True
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert any("survival is not a number" in f for f in cert.failures)


def test_private_mismatched_example_hashes_fail_integrity():
    pack = make_pack(
        include_private=True,
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0, 0.0],
                observation_ids=["obs-hello"],
                definition="Symbol for hello.",
                examples=["secret"],
                example_hashes=["c" * 64],
                confidence=0.5,
            )
        ],
    )
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("does not match examples" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_null_confidence_is_not_defaulted():
    pack = make_pack()
    data = pack.to_dict()
    data["symbols"][0]["confidence"] = None
    loaded = pack.from_dict(data)
    assert loaded.symbols[0].confidence is None
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert any("confidence is not a number" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


def test_loaded_null_reconstruction_error_is_not_defaulted():
    pack = make_pack()
    data = pack.to_dict()
    data["reconstruction_error"] = None
    loaded = pack.from_dict(data)
    assert loaded.reconstruction_error is None
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert any("reconstruction_error is not a number" in f for f in cert.failures)


def test_scalar_examples_fail_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    data["include_private"] = True
    data["symbols"][0]["examples"] = "secret"
    with pytest.raises(TypeError, match="examples must be an array"):
        pack.from_dict(data)


def test_constructed_non_string_checksum_returns_failed_certificate():
    pack = make_pack()
    pack.checksum = 123
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("checksum is not full SHA-256" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_constructed_none_example_hashes_returns_failed_certificate():
    pack = make_pack()
    pack.symbols[0].example_hashes = None
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("example_hashes is not an array" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_constructed_non_string_definition_returns_failed_certificate():
    pack = make_pack()
    pack.symbols[0].definition = 1
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("definition is not a string" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_certificate_string_false_passed_is_rejected():
    cert = certify(make_pack())
    data = cert.to_dict()
    data["passed"] = "false"
    with pytest.raises(TypeError, match="passed must be a boolean"):
        AuditCertificate.from_dict(data)
    data = cert.to_dict()
    data["integrity_valid"] = "false"
    with pytest.raises(TypeError, match="integrity_valid must be a boolean"):
        AuditCertificate.from_dict(data)


def test_constructed_none_examples_returns_failed_certificate():
    pack = make_pack()
    pack.symbols[0].examples = None
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("examples is not an array" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_constructed_non_mapping_evidence_returns_failed_certificate():
    pack = make_pack()
    pack.evidence = []
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.evidence_valid
    assert any("evidence is not an object" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])
    pack.evidence = None
    cert = certify(pack)
    assert not cert.evidence_valid
    assert any("evidence is not an object" in f for f in cert.failures)


def test_empty_parent_checksum_fails_schema():
    pack = make_pack()
    pack.parent_checksum = ""
    pack.seal()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("parent_checksum is not full SHA-256" in f for f in cert.failures)


def test_loaded_public_pack_strips_examples():
    pack = make_pack(
        include_private=True,
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0, 0.0],
                observation_ids=["obs-hello"],
                definition="Symbol for hello.",
                examples=["secret subjective text"],
                confidence=0.5,
            )
        ],
    )
    data = pack.to_dict()
    assert data["symbols"][0]["examples"] == ["secret subjective text"]
    data["include_private"] = False
    loaded = pack.from_dict(data)
    assert loaded.include_private is False
    assert loaded.symbols[0].examples == []
    assert "examples" not in loaded.to_dict()["symbols"][0]
    loaded.seal()
    cert = certify(loaded)
    assert cert.passed, cert.failures


def test_orphan_none_evidence_entry_fails_evidence():
    pack = make_pack()
    pack.evidence["orphan"] = None
    pack.seal()
    cert = certify(pack)
    assert not cert.evidence_valid
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("orphan" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_null_evidence_fails_when_all_quarantined():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0],
                observation_ids=["obs-1"],
                definition="quarantined",
                quarantined=True,
            )
        ]
    )
    data = pack.to_dict()
    data["evidence"] = None
    loaded = pack.from_dict(data)
    assert loaded.evidence is None
    cert = certify(loaded)
    assert not cert.evidence_valid
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("evidence is not an object" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


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


def test_none_alias_source_key_is_not_stringified(tmp_path):
    from neuralese.adapters import save_pack

    tables = normalize_aliases({None: {7: 0}})
    assert None in tables
    assert "None" not in tables
    pack = make_pack()
    pack.aliases = tables
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("alias source key is not a non-empty string" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [7])
    with pytest.raises(TypeError, match="alias source keys must be non-empty strings"):
        pack.to_dict()
    with pytest.raises(TypeError, match="alias source keys must be non-empty strings"):
        pack.seal()
    with pytest.raises(TypeError, match="alias source keys must be non-empty strings"):
        save_pack(pack, tmp_path / "pack.json")


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
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("mdl_bits is not finite" in f for f in cert.failures)
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
    data["aliases"] = {"c" * 64: {"1": False}}
    loaded = pack.from_dict(data)
    assert loaded.aliases["c" * 64][1] is False
    cert = certify(loaded)
    assert not cert.passed
    assert any("alias entries must be integer-to-integer" in f for f in cert.failures)
    data["aliases"] = {"c" * 64: {"1": 1.9}}
    loaded = pack.from_dict(data)
    assert loaded.aliases["c" * 64][1] == 1.9
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
    import json

    from neuralese.adapters import load_observations_jsonl
    from neuralese.cli import main

    path = tmp_path / "obs.jsonl"
    path.write_text(
        json.dumps({"observation_id": "o1", "embedding": [10**1000]}) + "\n"
    )
    with pytest.raises(ValueError, match="invalid observation record"):
        load_observations_jsonl(path)
    rc = main(["learn", str(path), "-o", str(tmp_path / "pack.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "invalid observation record" in err


def test_overflowing_pack_timestamp_is_clean_cli_failure(tmp_path, capsys):
    import json

    from neuralese.adapters import load_pack
    from neuralese.cli import main

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
    data["receipts"] = None
    loaded = pack.from_dict(data)
    assert loaded.receipts is None
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("receipts is not an array" in f for f in cert.failures)
    dumped = loaded.to_dict()
    assert dumped["receipts"] is None
    assert loaded.compute_checksum() != pack.checksum
    omitted = dict(pack.to_dict())
    del omitted["receipts"]
    assert pack.from_dict(omitted).receipts == []


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
    from neuralese.cli import main

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
    assert _alias_collision({source: {0: 1}}, codebook) is True
    assert _alias_collision({source: {7: 0}}, codebook) is False
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
    with pytest.raises(ValueError, match="blank observation_id"):
        learn_pack(
            [Observation(observation_id=None, text="hello there friend")],
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
    import json

    from neuralese.cli import main

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
    from neuralese.contracts import select_alias_table

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


def test_alias_resolution_is_linear_and_reuses_shared_suffixes():
    from neuralese.aliases import resolve_alias_table

    class CountingMap(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.lookups = 0

        def __getitem__(self, key):
            self.lookups += 1
            return super().__getitem__(key)

    n = 80
    chain = CountingMap({i: i + 1 for i in range(n)})
    terminals = resolve_alias_table(chain)
    assert terminals[0] == n
    assert terminals[n - 1] == n
    assert chain.lookups <= 2 * n

    shared = CountingMap({0: 2, 1: 2, 2: 4})
    resolved = resolve_alias_table(shared)
    assert resolved == {0: 4, 1: 4, 2: 4}
    assert shared.lookups <= 6

    aliases = {i: i + 1 for i in range(7, n)}
    aliases[n] = 0
    pack = make_pack(aliases={"legacy": aliases})
    cert = certify(pack)
    assert cert.addressable, cert.failures
    live = CountingMap(aliases)
    pack.aliases = {"legacy": live}
    cert = certify(pack)
    assert cert.addressable, cert.failures
    assert live.lookups <= 2 * n


def test_alias_cycle_still_fails_addressability():
    pack = make_pack(aliases={"legacy": {7: 8, 8: 7}}, checksum="unsealed")
    pack.seal()
    cert = certify(pack)
    assert cert.addressable is False
    assert cert.passed is False
    assert any("alias map contains a cycle" in f for f in cert.failures)


def test_historical_alias_refuses_live_source():
    pack = make_pack(aliases={7: 0, 0: 1})
    cert = certify(pack)
    assert cert.addressable is False
    assert any("historical mapping is ambiguous" in f for f in cert.failures)
    assert cert.passed is False
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [7])
    bounce = make_pack(aliases={7: 0, 0: 7})
    assert certify(bounce).addressable is False
    with pytest.raises(ValueError, match="historical mapping is ambiguous"):
        bounce.resolve_code(7)
    historical = make_pack(aliases={7: 0})
    assert certify(historical).addressable, certify(historical).failures
    assert historical.resolve_code(7) == (0, True)


def test_colliding_numeric_codebook_keys_fail_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    data["codebook"] = {"1": 1, "01": 1}
    with pytest.raises(ValueError, match="duplicate keys"):
        pack.from_dict(data)
    data["codebook"] = {"1": 0, "01": 1}
    with pytest.raises(ValueError, match="duplicate keys"):
        pack.from_dict(data)
    loaded = pack.from_dict({**pack.to_dict(), "codebook": {"0": 0, "1": 1}})
    assert loaded.codebook == {0: 0, 1: 1}


def test_colliding_numeric_alias_keys_fail_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    data["aliases"] = {"legacy": {"7": 0, "07": 0}}
    with pytest.raises(ValueError, match="duplicate keys"):
        pack.from_dict(data)
    loaded = pack.from_dict({**pack.to_dict(), "aliases": {"legacy": {"7": 0}}})
    assert loaded.aliases == {"legacy": {7: 0}}


def test_numpy_checksum_returns_failed_certificate():
    import numpy as np

    pack = make_pack()
    pack.checksum = np.array([1])
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("checksum is not full SHA-256" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_certify_alias_targets_use_precomputed_symbol_codes():
    aliases = {i: i + 1 for i in range(7, 40)}
    aliases[40] = 0
    pack = make_pack(aliases={"legacy": aliases})
    calls = {"n": 0}
    original = pack.symbol_by_code

    def counting(code):
        calls["n"] += 1
        return original(code)

    pack.symbol_by_code = counting  # type: ignore[method-assign]
    cert = certify(pack)
    assert cert.addressable, cert.failures
    assert calls["n"] == 0


def test_numpy_decoder_version_and_decision_fail_closed():
    import json

    import numpy as np

    pack = make_pack()
    pack.decoder_version = np.array([1, 2])
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("decoder_version" in f for f in cert.failures)
    json.dumps(cert.to_dict())
    pack = make_pack()
    pack.decision = np.array(["accept", "reject"])
    cert = certify(pack)
    assert not cert.passed
    assert any("decision" in f for f in cert.failures)
    json.dumps(cert.to_dict())


def test_numpy_checksum_certificate_is_json_serializable():
    import json

    import numpy as np

    pack = make_pack()
    pack.checksum = np.array([1])
    cert = certify(pack)
    assert cert.pack_checksum == ""
    json.dumps(cert.to_dict())


def test_certify_codebook_lookups_use_class_index():
    pack = make_pack()
    calls = {"n": 0}
    original = pack.symbol_by_class

    def counting(class_id):
        calls["n"] += 1
        return original(class_id)

    pack.symbol_by_class = counting  # type: ignore[method-assign]
    cert = certify(pack)
    assert cert.addressable, cert.failures
    assert calls["n"] == 0


def test_numpy_quarantined_and_class_id_fail_closed():
    import numpy as np

    pack = make_pack()
    pack.symbols[0].quarantined = np.array([False, True])
    cert = certify(pack)
    assert not cert.passed
    pack = make_pack()
    pack.symbols[0].class_id = np.array([0, 1])
    assert pack.symbol_by_class(0) is None
    cert = certify(pack)
    assert not cert.integrity_valid


def test_certificate_details_are_json_safe_for_numpy_metrics():
    import json

    import numpy as np

    pack = make_pack()
    pack.pack_id = np.array([1, 2])
    pack.reconstruction_error = np.array([0.1, 0.2])
    pack.mdl_bits = np.array([12.0])
    cert = certify(pack, tau_residual=np.array([0.55]))
    dumped = json.dumps(cert.to_dict())
    assert cert.pack_id == ""
    assert '"reconstruction_error": null' in dumped
    assert '"mdl_bits": null' in dumped
    assert '"tau_residual": null' in dumped


def test_numpy_finalize_step_and_example_digest_fail_closed():
    import json

    import numpy as np

    pack = make_pack()
    pack.receipts = [
        Receipt(
            step=np.array(["finalize"]),
            ok=True,
            timestamp=1.0,
            metadata={"decision": pack.decision},
        )
    ]
    cert = certify(pack)
    assert not cert.passed
    json.dumps(cert.to_dict())
    pack = make_pack(include_private=True)
    pack.symbols[0].examples = ["hello"]
    pack.symbols[0].example_hashes = [np.array([1])]
    cert = certify(pack)
    assert not cert.integrity_valid
    json.dumps(cert.to_dict())
    pack = make_pack()
    pack.receipts = [
        Receipt(
            step="finalize",
            ok=True,
            timestamp=1.0,
            metadata={"decision": np.array(["accept"])},
        )
    ]
    cert = certify(pack)
    assert not cert.admission_valid
    json.dumps(cert.to_dict())

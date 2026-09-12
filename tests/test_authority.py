from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuralese.adapters import load_pack, load_stream, save_pack, save_stream
from neuralese.aliases import rewrite_stream
from neuralese.audit import certify, integrity_report
from neuralese.cli import main
from neuralese.contracts import (
    DECODER_VERSION,
    SHA256_HEX,
    Observation,
    Receipt,
    Symbol,
    SymbolPack,
    TranslatedToken,
    TranslatedUtterance,
    TranslationStream,
    load_observations,
)
from neuralese.translator import TRANSLATION_POLICIES, translate_stream
from tests.packutil import certified_pack, dummy_receipt, dummy_symbol, dummy_translated, public_pack, sealed_pack


def test_certified_pack_translates_codes() -> None:
    pack = certified_pack()
    stream = TranslationStream(codes=[0], source_pack_checksum=pack.checksum())
    utterance = translate_stream(stream, pack)
    assert [token.gloss for token in utterance.tokens] == ["hello"]
    assert utterance.certified is True
    assert utterance.unknown_ratio == 0.0


def test_uncertified_pack_refuses_translation() -> None:
    pack = sealed_pack()
    stream = TranslationStream(codes=[0])
    with pytest.raises(PermissionError, match="uncertified"):
        translate_stream(stream, pack)


def test_allow_uncertified_translates_without_promotion() -> None:
    pack = sealed_pack()
    stream = TranslationStream(codes=[0])
    utterance = translate_stream(stream, pack, allow_uncertified=True)
    assert utterance.certified is False
    assert utterance.tokens[0].gloss == "hello"


def test_empty_codes_are_unknown() -> None:
    pack = certified_pack()
    utterance = translate_stream(TranslationStream(codes=[]), pack)
    assert utterance.tokens == []
    assert utterance.unknown_ratio == 1.0
    assert utterance.mean_confidence == 0.0


def test_unknown_code_is_not_hallucinated() -> None:
    pack = certified_pack()
    utterance = translate_stream(TranslationStream(codes=[99]), pack)
    assert utterance.tokens[0].gloss == "[unknown]"
    assert utterance.tokens[0].confidence == 0.0
    assert utterance.unknown_ratio == 1.0


def test_integrity_policy_does_not_authorize_translation() -> None:
    pack = certified_pack(policy="integrity")
    stream = TranslationStream(codes=[0])
    with pytest.raises(PermissionError, match="does not authorize translation"):
        translate_stream(stream, pack)


def test_strict_policy_is_translatable() -> None:
    pack = certified_pack(policy="strict")
    utterance = translate_stream(TranslationStream(codes=[0]), pack)
    assert utterance.certified is True
    assert utterance.policy == "strict"


def test_unknown_translation_policy_is_refused() -> None:
    pack = certified_pack()
    with pytest.raises(PermissionError, match="unknown translation policy"):
        translate_stream(TranslationStream(codes=[0]), pack, policy="integrity")
    assert "integrity" not in TRANSLATION_POLICIES


def test_truncated_checksum_fails_integrity() -> None:
    pack = sealed_pack()
    broken = pack.model_copy(update={"decoder_checksum": pack.decoder_checksum[:16]})
    report = integrity_report(broken)
    assert report.ok is False
    assert any("decoder_checksum must be a full SHA-256" in error for error in report.errors)
    assert certify(broken).ok is False


def test_decoder_version_mismatch_fails_integrity() -> None:
    pack = sealed_pack().model_copy(update={"decoder_version": "0.0.0-other"})
    report = integrity_report(pack)
    assert report.ok is False
    assert any("decoder_version must equal" in error for error in report.errors)


def test_public_pack_omits_raw_examples() -> None:
    pack = public_pack()
    assert pack.include_private is False
    assert pack.symbols[0].examples == []
    assert pack.symbols[0].example_hashes
    assert all(SHA256_HEX.match(item) for item in pack.symbols[0].example_hashes)


def test_public_pack_definition_is_not_raw_observation() -> None:
    pack = public_pack([Observation(id="obs-1", text="secret greeting", embedding=[1.0, 0.0])])
    assert pack.symbols[0].definition == "hello"
    assert "secret greeting" not in json.dumps(pack.to_dict())


def test_alias_hop_bound_is_table_size() -> None:
    pack = certified_pack()
    stream = TranslationStream(codes=[0], aliases={0: 0})
    utterance = translate_stream(stream, pack)
    assert utterance.tokens[0].code == 0
    assert utterance.tokens[0].gloss == "hello"


def test_rewrite_stream_ignores_unrelated_source_table() -> None:
    stream = TranslationStream(
        codes=[7],
        aliases={"other-pack": {7: 0}},
        source_pack_checksum="this-pack",
    )
    rewritten = rewrite_stream(stream)
    assert rewritten.codes == [7]


def test_rewrite_stream_without_source_uses_only_legacy_table() -> None:
    stream = TranslationStream(codes=[7], aliases={"other-pack": {7: 0}})
    rewritten = rewrite_stream(stream)
    assert rewritten.codes == [7]

    legacy = TranslationStream(codes=[7], aliases={"legacy": {7: 0}})
    assert rewrite_stream(legacy).codes == [0]


def test_observation_ids_are_canonicalized() -> None:
    observations = load_observations([{"id": 1, "text": "hello", "embedding": [1.0, 0.0]}])
    assert observations[0].id == "1"


def test_duplicate_observation_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate observation id"):
        load_observations(
            [
                {"id": "obs-1", "text": "hello", "embedding": [1.0, 0.0]},
                {"id": "obs-1", "text": "hello again", "embedding": [1.0, 0.0]},
            ]
        )


def test_cli_rejects_invalid_audit_policy(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.json"
    save_pack(certified_pack(), pack_path)
    with pytest.raises(SystemExit):
        main(["audit", str(pack_path), "--policy", "not-a-policy"])


def test_cli_rejects_invalid_certify_policy(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.json"
    save_pack(sealed_pack(), pack_path)
    with pytest.raises(SystemExit):
        main(["certify", str(pack_path), "--policy", "not-a-policy"])


def test_cli_rejects_invalid_translate_policy(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.json"
    stream_path = tmp_path / "stream.json"
    save_pack(certified_pack(), pack_path)
    save_stream(TranslationStream(codes=[0]), stream_path)
    with pytest.raises(SystemExit):
        main(["translate", str(pack_path), str(stream_path), "--policy", "integrity"])


def test_cli_translate_refuses_uncertified_pack(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pack_path = tmp_path / "pack.json"
    stream_path = tmp_path / "stream.json"
    save_pack(sealed_pack(), pack_path)
    save_stream(TranslationStream(codes=[0]), stream_path)
    with pytest.raises(SystemExit) as exc:
        main(["translate", str(pack_path), str(stream_path)])
    assert exc.value.code == 1
    assert "uncertified" in capsys.readouterr().err


def test_roundtrip_pack_and_stream(tmp_path: Path) -> None:
    pack = certified_pack()
    stream = TranslationStream(codes=[0, 99], source_pack_checksum=pack.checksum())
    pack_path = tmp_path / "pack.json"
    stream_path = tmp_path / "stream.json"
    save_pack(pack, pack_path)
    save_stream(stream, stream_path)
    loaded_pack = load_pack(pack_path)
    loaded_stream = load_stream(stream_path)
    utterance = translate_stream(loaded_stream, loaded_pack)
    assert loaded_pack.checksum() == pack.checksum()
    assert [token.gloss for token in utterance.tokens] == ["hello", "[unknown]"]


def test_translated_models_roundtrip() -> None:
    token = dummy_translated()
    utterance = TranslatedUtterance(tokens=[token], certified=True, policy="default")
    assert TranslatedToken.from_dict(token.to_dict()).gloss == "hello"
    assert TranslatedUtterance.from_dict(utterance.to_dict()).unknown_ratio == 0.0


def test_receipt_from_dict_requires_complete_payload() -> None:
    with pytest.raises((TypeError, ValueError, KeyError)):
        Receipt.from_dict({"ok": True})


def test_empty_decoder_version_fails_from_dict() -> None:
    data = sealed_pack().to_dict()
    data["decoder_version"] = ""
    with pytest.raises((TypeError, ValueError, KeyError)):
        SymbolPack.from_dict(data)


def test_null_metadata_fails_from_dict() -> None:
    data = sealed_pack().to_dict()
    data["metadata"] = None
    with pytest.raises((TypeError, ValueError, KeyError)):
        SymbolPack.from_dict(data)


def test_empty_decision_fails_from_dict() -> None:
    data = dummy_receipt().to_dict()
    data["decision"] = ""
    with pytest.raises((TypeError, ValueError, KeyError)):
        Receipt.from_dict(data)


def test_unknown_decision_fails_integrity() -> None:
    pack = sealed_pack()
    broken = pack.model_copy(
        update={"receipt": dummy_receipt()._replace(decision="maybe")},
    )
    report = integrity_report(broken)
    assert report.ok is False
    assert any("decision must be one of" in error for error in report.errors)


def test_truncated_example_hash_fails_integrity() -> None:
    pack = public_pack()
    broken_symbol = pack.symbols[0].model_copy(update={"example_hashes": ["abc"]})
    broken = pack.model_copy(update={"symbols": [broken_symbol]})
    report = integrity_report(broken)
    assert report.ok is False
    assert any("example_hashes" in error for error in report.errors)


def test_unglossed_sentinel_has_zero_confidence() -> None:
    pack = certified_pack()
    unglossed = pack.symbols[0].model_copy(update={"definition": "[unglossed]", "confidence": 0.0})
    utterance = translate_stream(
        TranslationStream(codes=[0]),
        pack.model_copy(update={"symbols": [unglossed]}),
    )
    assert utterance.tokens[0].gloss == "[unglossed]"
    assert utterance.tokens[0].confidence == 0.0
    assert utterance.mean_confidence == 0.0


def test_unglossed_does_not_satisfy_gloss_bound() -> None:
    pack = sealed_pack()
    unglossed = pack.symbols[0].model_copy(update={"definition": "[unglossed]", "confidence": 0.0})
    broken = pack.model_copy(update={"symbols": [unglossed]})
    receipt = certify(broken, policy="default")
    assert receipt.ok is False
    assert any("gloss_bound" in reason for reason in receipt.reasons)


def test_allow_unglossed_certifies_explicit_gap() -> None:
    pack = sealed_pack()
    unglossed = pack.symbols[0].model_copy(update={"definition": "[unglossed]", "confidence": 0.0})
    opened = pack.model_copy(update={"symbols": [unglossed]})
    receipt = certify(opened, policy="default", require_gloss=False)
    assert receipt.ok is True
    assert receipt.decision == "admit"


def test_cli_rejects_malformed_observations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pack_path = tmp_path / "pack.json"
    observations_path = tmp_path / "obs.json"
    save_pack(sealed_pack(), pack_path)
    observations_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["certify", str(pack_path), "--observations", str(observations_path)])
    assert exc.value.code == 1
    assert "invalid observations" in capsys.readouterr().err


def test_non_object_jsonl_observation_fails_closed() -> None:
    with pytest.raises(ValueError, match="observation records must be objects"):
        load_observations(["not-an-object"])


def test_guards_pass_all_must_match_individual_flags() -> None:
    pack = sealed_pack()
    broken_guards = dict(pack.guards)
    broken_guards["pass_all"] = False
    broken = pack.model_copy(update={"guards": broken_guards})
    receipt = certify(broken, policy="default")
    assert receipt.ok is False
    assert any("pass_all must equal" in reason for reason in receipt.reasons)


def test_string_false_guard_is_not_a_boolean_pass() -> None:
    pack = sealed_pack()
    broken_guards = dict(pack.guards)
    broken_guards["pass_kappa"] = "false"
    broken = pack.model_copy(update={"guards": broken_guards})
    receipt = certify(broken, policy="default")
    assert receipt.ok is False
    assert any("pass_kappa is not a boolean" in reason for reason in receipt.reasons)


def test_same_observation_on_two_classes_fails_evidence() -> None:
    first = dummy_symbol(class_id=0, members=("obs-1",))
    second = dummy_symbol(class_id=1, members=("obs-1",), definition="goodbye")
    pack = sealed_pack().model_copy(update={"symbols": [first, second]})
    observations = [
        Observation(id="obs-1", text="hello", embedding=[1.0, 0.0]),
        Observation(id="obs-2", text="unused", embedding=[0.0, 1.0]),
    ]
    receipt = certify(pack, observations, policy="default")
    assert receipt.ok is False
    assert any("unique class" in reason for reason in receipt.reasons)


def test_string_false_receipt_ok_is_not_boolean_true() -> None:
    receipt = Receipt.from_dict(
        {
            "ok": "false",
            "decision": "admit",
            "policy": "default",
            "reasons": [],
            "metrics": {},
        }
    )
    assert receipt.ok == "false"
    pack = sealed_pack().model_copy(update={"receipt": receipt})
    report = integrity_report(pack)
    assert report.ok is False
    assert any("receipt.ok is not a boolean" in error for error in report.errors)
    certified = certify(pack, policy="default")
    assert certified.ok is False
    assert any("receipt.ok is not a boolean" in reason for reason in certified.reasons)


def test_non_mapping_evidence_fails_certify_not_load() -> None:
    data = sealed_pack().to_dict()
    data["evidence"] = []
    pack = SymbolPack.from_dict(data)
    assert pack.evidence == {}
    receipt = certify(pack, policy="default")
    assert receipt.ok is False
    assert any("evidence_bound" in reason for reason in receipt.reasons)


def test_null_evidence_fails_certify_not_load() -> None:
    data = sealed_pack().to_dict()
    data["evidence"] = None
    pack = SymbolPack.from_dict(data)
    assert pack.evidence == {}
    assert certify(pack, policy="default").ok is False


def test_jsonl_embedding_must_be_array() -> None:
    with pytest.raises(ValueError, match="embedding must be an array of numbers"):
        load_observations([{"id": "obs-1", "text": "hello", "embedding": "1"}])


def test_jsonl_null_embedding_fails_closed() -> None:
    with pytest.raises(ValueError, match="embedding must be an array of numbers"):
        load_observations([{"id": "obs-1", "text": "hello", "embedding": None}])


def test_empty_private_gloss_becomes_unglossed() -> None:
    symbol = dummy_symbol()._replace(definition="", confidence=0.0)
    pack = certified_pack().model_copy(update={"symbols": [symbol], "include_private": True})
    utterance = translate_stream(TranslationStream(codes=[0]), pack)
    assert utterance.tokens[0].gloss == "[unglossed]"
    assert utterance.tokens[0].confidence == 0.0


def test_non_string_definition_fails_from_dict() -> None:
    data = dummy_symbol().to_dict()
    data["definition"] = ["hello"]
    with pytest.raises(TypeError, match="definition must be a string or null"):
        Symbol.from_dict(data)


def test_string_quarantined_fails_integrity() -> None:
    data = dummy_symbol().to_dict()
    data["quarantined"] = "false"
    symbol = Symbol.from_dict(data)
    pack = sealed_pack().model_copy(update={"symbols": [symbol]})
    report = integrity_report(pack)
    assert report.ok is False
    assert any("quarantined is not a boolean" in error for error in report.errors)


def test_string_include_private_fails_integrity() -> None:
    data = sealed_pack().to_dict()
    data["include_private"] = "false"
    pack = SymbolPack.from_dict(data)
    assert pack.include_private == "false"
    report = integrity_report(pack)
    assert report.ok is False
    assert any("include_private is not a boolean" in error for error in report.errors)
    assert certify(pack, policy="integrity").ok is False


def test_non_object_pack_fails_from_dict() -> None:
    with pytest.raises(TypeError, match="pack object"):
        SymbolPack.from_dict([])  # type: ignore[arg-type]


def test_non_object_symbol_fails_from_dict() -> None:
    with pytest.raises(TypeError, match="symbol object"):
        Symbol.from_dict("hello")  # type: ignore[arg-type]


def test_mixed_alias_shapes_fail_load(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.json"
    data = sealed_pack().to_dict()
    data["aliases"] = {"": {}, "src": 1}
    pack_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid pack"):
        load_pack(pack_path)

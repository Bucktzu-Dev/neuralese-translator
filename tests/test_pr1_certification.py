"""Locks from docs/PR1_CERTIFICATION_DISCREPANCY.md section 5, plus schema 0.2.

pack.checksum stays compute_checksum(). proto_embedding is outside that
payload. identity-1 is an opt-in digest_profile. A missing reservoir still
fails unfold. passed is never RESEARCH_PASS. An explicit identity-1 reseal
replaces the digest. A padded or differently cased profile is not identity-1.
"""
import json

import pytest

from neuralese import __version__
from neuralese.adapters import load_pack, save_pack
from neuralese.audit import certify
from neuralese.contracts import (
    CERTIFICATE_AUTHORITY,
    CERTIFICATE_DECISION_RESEARCH_FAIL,
    CERTIFICATE_DECISION_UNRESOLVED,
    CERTIFICATE_SCHEMA_UNVERSIONED,
    CERTIFICATE_SCHEMA_V02,
    DIGEST_DOES_NOT_BIND,
    DIGEST_PAIRING_IDENTITY_1,
    DIGEST_PAIRING_INCONSISTENT,
    DIGEST_PAIRING_UNBOUND,
    DIGEST_PAIRING_UNKNOWN,
    DIGEST_PROFILE_IDENTITY_1,
    AuditCertificate,
    Receipt,
    Symbol,
    SymbolPack,
    sha16,
)
from neuralese.translator import translate_stream
from neuralese.unfold import PROTO_ATOL

from packutil import make_pack


def test_alias_tamper_without_reseal():
    pack = make_pack()
    pack.aliases[7] = 0
    cert = certify(pack)
    assert cert.passed is False
    assert cert.gloss_bound is False
    assert cert.addressable is True
    assert any("checksum" in failure for failure in cert.failures)
    gloss = translate_stream(pack, [7])[0]
    assert gloss.state == "aliased"
    assert gloss.resolved_code == 0


def test_observation_embedding_tamper_without_reseal():
    pack = make_pack()
    pack.observations["obs-hello"].embedding = [0.0, 1.0, 0.0]
    cert = certify(pack)
    assert cert.passed is False
    assert cert.gloss_bound is False
    assert cert.unfoldable is False
    assert cert.residual_ok is False
    assert any("l2=1.414214" in failure for failure in cert.failures)
    assert any("checksum" in failure for failure in cert.failures)
    assert cert.details["recomputed_cluster_residual"] == pytest.approx(1.0)


def test_prototype_tamper_without_reseal():
    pack = make_pack()
    sealed = pack.checksum
    pack.symbols[0].proto_embedding = [0.0, 0.0, 0.0]
    assert pack.compute_checksum() == sealed
    cert = certify(pack)
    assert cert.expected_checksum == sealed
    assert cert.gloss_bound is True
    assert cert.unfoldable is False
    assert cert.residual_ok is False
    assert cert.passed is False
    assert any("l2=1.000000" in failure for failure in cert.failures)
    assert cert.details["recomputed_cluster_residual"] == pytest.approx(
        0.7071067811865476
    )
    assert any("0.707107" in failure for failure in cert.failures)


def test_prototype_drift_inside_atol_passes():
    assert PROTO_ATOL == 1e-5
    pack = make_pack()
    sealed = pack.checksum
    pack.symbols[0].proto_embedding[0] += 1e-6
    assert pack.compute_checksum() == sealed
    cert = certify(pack)
    assert cert.passed is True
    assert cert.unfoldable is True
    assert cert.gloss_bound is True
    assert cert.residual_ok is True


def test_missing_reservoir_id_without_reseal():
    pack = make_pack()
    del pack.observations["obs-hello"]
    cert = certify(pack)
    assert cert.passed is False
    assert cert.gloss_bound is False
    assert cert.unfoldable is False
    assert cert.residual_ok is True
    assert any(
        "missing reservoir observation obs-hello" in failure
        for failure in cert.failures
    )
    gloss = translate_stream(pack, [0])[0]
    assert gloss.state == "ok"


def test_unglossed_ok_is_not_a_certificate():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0],
                observation_ids=["obs-1"],
                definition=None,
            )
        ]
    )
    gloss = translate_stream(pack, [0])[0]
    assert gloss.state == "ok"
    assert gloss.english.startswith("[unglossed:")
    cert = certify(pack)
    assert cert.gloss_bound is False
    assert cert.passed is False
    assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL


def test_confidence_tamper_is_outside_the_digest():
    pack = make_pack()
    sealed = pack.checksum
    pack.symbols[0].confidence = 1.0
    assert pack.compute_checksum() == sealed
    cert = certify(pack)
    assert cert.passed is True
    assert cert.expected_checksum == sealed
    gloss = translate_stream(pack, [0])[0]
    assert gloss.confidence == pytest.approx(0.92)
    assert "confidence" in DIGEST_DOES_NOT_BIND


def test_unversioned_empty_reservoir_fails_unfold():
    pack = make_pack(
        observations={},
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0],
                observation_ids=["obs-1"],
                definition="bound gloss for a missing row",
            )
        ],
    )
    cert = certify(pack)
    assert cert.passed is False
    assert cert.unfoldable is False
    assert cert.gloss_bound is True
    assert any(
        "missing reservoir observation obs-1" in failure for failure in cert.failures
    )
    assert cert.schema_version == CERTIFICATE_SCHEMA_V02
    assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
    assert cert.authority == CERTIFICATE_AUTHORITY
    assert cert.evidence["numerical"]["status"] == "NOT IMPLEMENTED"
    assert cert.evidence["numerical"]["decision"] == CERTIFICATE_DECISION_UNRESOLVED
    assert cert.evidence["semantic"]["status"] == "NOT IMPLEMENTED"
    assert cert.evidence["semantic"]["decision"] == CERTIFICATE_DECISION_UNRESOLVED


def test_legacy_certificate_json_without_schema_version_is_unversioned():
    current = certify(make_pack()).to_dict()
    legacy = {
        key: current[key]
        for key in (
            "pack_id",
            "pack_checksum",
            "expected_checksum",
            "passed",
            "addressable",
            "unfoldable",
            "gloss_bound",
            "residual_ok",
            "fail_closed",
            "failures",
            "timestamp",
            "details",
        )
    }
    loaded = AuditCertificate.from_dict(legacy)
    assert loaded.schema_version == CERTIFICATE_SCHEMA_UNVERSIONED
    assert loaded.package_version is None
    assert loaded.authority == CERTIFICATE_AUTHORITY
    assert loaded.decision == CERTIFICATE_DECISION_UNRESOLVED
    assert loaded.passed is True
    assert loaded.evidence == {}


def test_loader_rejects_research_pass_and_other_authority():
    raw = certify(make_pack()).to_dict()
    raw["decision"] = "RESEARCH_PASS"
    with pytest.raises(ValueError, match="RESEARCH_PASS"):
        AuditCertificate.from_dict(raw)
    raw = certify(make_pack()).to_dict()
    raw["authority"] = "RUNTIME_AUTHORIZED"
    with pytest.raises(ValueError, match="authority"):
        AuditCertificate.from_dict(raw)
    raw = certify(make_pack()).to_dict()
    raw["schema_version"] = "v0.1"
    with pytest.raises(ValueError, match="schema_version"):
        AuditCertificate.from_dict(raw)


def test_structural_pass_is_schema_v02_and_not_a_research_pass():
    pack = make_pack()
    gloss = translate_stream(pack, [0])[0]
    cert = certify(pack)
    assert gloss.state == "ok"
    assert cert.passed is True
    assert cert.schema_version == CERTIFICATE_SCHEMA_V02
    assert cert.package_version == __version__ == "0.7.0"
    assert cert.authority == CERTIFICATE_AUTHORITY
    assert cert.decision == CERTIFICATE_DECISION_UNRESOLVED
    assert cert.evidence["structural"]["addressable"] is True
    assert cert.evidence["structural"]["unfoldable"] is True
    assert cert.evidence["structural"]["gloss_bound"] is True
    assert cert.evidence["structural"]["residual_ok"] is True
    assert cert.evidence["structural"]["failures"] == []
    assert cert.evidence["structural"]["digest_does_not_bind"] == list(
        DIGEST_DOES_NOT_BIND
    )
    assert cert.evidence["numerical"]["decision"] == CERTIFICATE_DECISION_UNRESOLVED
    assert cert.evidence["semantic"]["decision"] == CERTIFICATE_DECISION_UNRESOLVED
    again = AuditCertificate.from_dict(cert.to_dict())
    assert again.schema_version == cert.schema_version
    assert again.decision == cert.decision
    assert again.authority == cert.authority
    assert again.package_version == "0.7.0"
    assert again.evidence == cert.evidence
    assert again.passed is True
    assert again.identity_bound is False
    assert cert.identity_bound is False
    assert cert.evidence["structural"]["identity_bound"] is False
    assert "proto_embedding" in cert.evidence["structural"]["digest_does_not_bind"]


_OMITTED = object()


def _assert_not_admitted(cert):
    assert cert.schema_version == CERTIFICATE_SCHEMA_V02
    assert cert.package_version == __version__ == "0.7.0"
    assert cert.authority == CERTIFICATE_AUTHORITY
    assert cert.decision != "RESEARCH_PASS"
    assert cert.evidence["numerical"]["status"] == "NOT IMPLEMENTED"
    assert cert.evidence["numerical"]["decision"] == CERTIFICATE_DECISION_UNRESOLVED
    assert cert.evidence["semantic"]["status"] == "NOT IMPLEMENTED"
    assert cert.evidence["semantic"]["decision"] == CERTIFICATE_DECISION_UNRESOLVED


def _pack_with_digest_fields(profile, digest) -> SymbolPack:
    raw = make_pack().to_dict()
    if profile is _OMITTED:
        raw.pop("digest_profile", None)
    else:
        raw["digest_profile"] = profile
    if digest is _OMITTED:
        raw.pop("identity_digest", None)
    else:
        raw["identity_digest"] = digest
    return SymbolPack.from_dict(raw)


def _seal_identity(pack):
    checksum = pack.checksum
    pack.seal(digest_profile=DIGEST_PROFILE_IDENTITY_1)
    assert pack.checksum == checksum
    assert pack.compute_checksum() == checksum
    assert pack.digest_profile == DIGEST_PROFILE_IDENTITY_1
    assert pack.identity_digest == pack.compute_identity_digest()
    assert "proto_embedding" not in pack._checksum_payload()["symbols"][0]
    return pack


def test_identity1_1e6_fails_integrity_not_atol():
    pack = _seal_identity(make_pack())
    sealed_checksum = pack.checksum
    sealed_identity = pack.identity_digest
    pack.symbols[0].proto_embedding[0] += 1e-6
    assert round(pack.symbols[0].proto_embedding[0], 8) == 1.000001
    assert pack.identity_payload()["symbols"][0]["proto_embedding"][0] == 1.000001
    assert pack.compute_checksum() == sealed_checksum
    assert pack.compute_identity_digest() != sealed_identity
    cert = certify(pack)
    assert cert.identity_bound is False
    assert cert.evidence["structural"]["identity_bound"] is False
    assert cert.passed is False
    assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
    assert cert.gloss_bound is True
    assert cert.unfoldable is True
    assert cert.addressable is True
    assert cert.residual_ok is True
    assert any("identity digest mismatch" in failure for failure in cert.failures)
    assert not any(
        "prototype does not match reservoir mean" in failure for failure in cert.failures
    )
    assert "proto_embedding" not in cert.evidence["structural"]["digest_does_not_bind"]
    _assert_not_admitted(cert)


def test_identity1_far_prototype_fails_digest_and_unfold():
    pack = _seal_identity(make_pack())
    sealed_checksum = pack.checksum
    pack.symbols[0].proto_embedding = [0.0, 0.0, 0.0]
    assert pack.compute_checksum() == sealed_checksum
    cert = certify(pack)
    assert cert.unfoldable is False
    assert cert.residual_ok is False
    assert cert.identity_bound is False
    assert cert.passed is False
    assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
    assert cert.gloss_bound is True
    assert any("l2=1.000000" in failure for failure in cert.failures)
    assert any("identity digest mismatch" in failure for failure in cert.failures)
    assert cert.details["recomputed_cluster_residual"] == pytest.approx(
        0.7071067811865476
    )
    assert "proto_embedding" not in cert.evidence["structural"]["digest_does_not_bind"]
    _assert_not_admitted(cert)


def test_identity1_sub_ulp_nudge_keeps_the_canonical_value():
    """+1e-9 does not change the identity digest.

    The digest is unchanged because round(float, 8) keeps 1.0, so the
    canonical value is unchanged. That is the rounding rule. It is not a
    claim that the nudge passed PROTO_ATOL.
    """
    pack = _seal_identity(make_pack())
    sealed_identity = pack.identity_digest
    pack.symbols[0].proto_embedding[0] += 1e-9
    assert round(float(pack.symbols[0].proto_embedding[0]), 8) == 1.0
    assert pack.identity_payload()["symbols"][0]["proto_embedding"] == [
        1.0,
        0.0,
        0.0,
    ]
    assert pack.compute_identity_digest() == sealed_identity
    cert = certify(pack)
    assert cert.identity_bound is True
    assert cert.evidence["structural"]["identity_bound"] is True
    assert cert.gloss_bound is True
    assert cert.passed is True
    assert cert.decision == CERTIFICATE_DECISION_UNRESOLVED
    assert "proto_embedding" not in cert.evidence["structural"]["digest_does_not_bind"]
    assert "confidence" in cert.evidence["structural"]["digest_does_not_bind"]
    _assert_not_admitted(cert)


def test_identity1_does_not_bind_excluded_fields():
    pack = _seal_identity(make_pack())
    sealed_identity = pack.identity_digest
    sealed_checksum = pack.checksum
    pack.symbols[0].confidence = 1.0
    pack.symbols[0].metadata["kappa"] = 0.0
    pack.receipts.append(Receipt(step="note", ok=True, timestamp=1.0))
    assert pack.compute_identity_digest() == sealed_identity
    assert pack.identity_digest == sealed_identity
    assert pack.compute_checksum() == sealed_checksum
    cert = certify(pack)
    assert cert.identity_bound is True
    assert cert.passed is True
    assert cert.decision == CERTIFICATE_DECISION_UNRESOLVED
    unbound = cert.evidence["structural"]["digest_does_not_bind"]
    assert "confidence" in unbound
    assert "kappa" in unbound
    assert "receipts" in unbound
    assert "proto_embedding" not in unbound
    gloss = translate_stream(pack, [0])[0]
    assert gloss.confidence == pytest.approx(0.92)
    _assert_not_admitted(cert)


def test_identity1_missing_digest_fails_closed():
    for digest in (_OMITTED, None, "", "   "):
        loaded = _pack_with_digest_fields(DIGEST_PROFILE_IDENTITY_1, digest)
        assert loaded.digest_profile == DIGEST_PROFILE_IDENTITY_1
        assert loaded.identity_digest is None
        cert = certify(loaded)
        assert cert.digest_profile == DIGEST_PROFILE_IDENTITY_1
        assert cert.evidence["structural"]["digest_pairing"] == DIGEST_PAIRING_IDENTITY_1
        assert cert.identity_bound is False
        assert cert.passed is False
        assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
        assert cert.addressable is True
        assert cert.unfoldable is True
        assert cert.gloss_bound is True
        assert cert.residual_ok is True
        assert "proto_embedding" in cert.evidence["structural"]["digest_does_not_bind"]
        assert any(
            "missing identity_digest" in failure for failure in cert.failures
        )
        _assert_not_admitted(cert)


def test_unbound_omission_is_not_an_identity_pass():
    current = certify(make_pack()).to_dict()
    legacy = {
        key: current[key]
        for key in (
            "pack_id",
            "pack_checksum",
            "expected_checksum",
            "passed",
            "addressable",
            "unfoldable",
            "gloss_bound",
            "residual_ok",
            "fail_closed",
            "failures",
            "timestamp",
            "details",
        )
    }
    assert "digest_profile" not in legacy
    assert "identity_digest" not in legacy
    assert "schema_version" not in legacy
    loaded_cert = AuditCertificate.from_dict(legacy)
    assert loaded_cert.schema_version == CERTIFICATE_SCHEMA_UNVERSIONED
    assert loaded_cert.decision == CERTIFICATE_DECISION_UNRESOLVED
    assert loaded_cert.decision != "RESEARCH_PASS"
    assert loaded_cert.identity_bound is False
    assert loaded_cert.digest_profile is None
    assert loaded_cert.identity_digest is None
    assert loaded_cert.authority == CERTIFICATE_AUTHORITY

    for profile, digest in (
        (_OMITTED, _OMITTED),
        (None, None),
        ("", ""),
        ("   ", "   "),
        ("unbound", _OMITTED),
        ("unbound", None),
        ("unbound", ""),
    ):
        loaded = _pack_with_digest_fields(profile, digest)
        cert = certify(loaded)
        assert cert.evidence["structural"]["digest_pairing"] == DIGEST_PAIRING_UNBOUND
        assert cert.identity_bound is False
        assert cert.passed is True
        assert cert.decision == CERTIFICATE_DECISION_UNRESOLVED
        assert "proto_embedding" in cert.evidence["structural"]["digest_does_not_bind"]
        _assert_not_admitted(cert)

    pack = make_pack()
    del pack.observations["obs-hello"]
    cert = certify(pack)
    assert cert.passed is False
    assert cert.unfoldable is False
    assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
    assert cert.identity_bound is False
    assert any(
        "missing reservoir observation obs-hello" in failure for failure in cert.failures
    )
    assert "proto_embedding" in cert.evidence["structural"]["digest_does_not_bind"]
    _assert_not_admitted(cert)


def test_unbound_with_identity_digest_is_not_verified():
    sample = make_pack()
    correct_looking = sample.compute_identity_digest()
    for profile in (_OMITTED, None, "", "   ", "unbound"):
        for digest in ("not-a-verified-digest", correct_looking):
            loaded = _pack_with_digest_fields(profile, digest)
            cert = certify(loaded)
            assert cert.identity_bound is False
            assert cert.evidence["structural"]["identity_bound"] is False
            assert cert.passed is False
            assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
            assert cert.evidence["structural"]["digest_pairing"] == (
                DIGEST_PAIRING_INCONSISTENT
            )
            assert cert.addressable is True
            assert cert.unfoldable is True
            assert cert.gloss_bound is True
            assert cert.residual_ok is True
            blob = " ".join(cert.failures)
            assert "inconsistent" in blob
            assert "unverified" in blob
            assert "identity-1 matched" not in blob
            assert "proto_embedding" in cert.evidence["structural"]["digest_does_not_bind"]
            _assert_not_admitted(cert)


def test_unknown_digest_profile_fails_closed():
    for profile in ("identity-2", "legacy", "v0.1"):
        for digest in (_OMITTED, None, "", "   ", "deadbeefdeadbeef"):
            loaded = _pack_with_digest_fields(profile, digest)
            assert loaded.digest_profile == profile
            assert loaded.digest_profile != "unbound"
            cert = certify(loaded)
            assert cert.passed is False
            assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
            assert cert.identity_bound is False
            assert cert.digest_profile == profile
            assert cert.evidence["structural"]["digest_pairing"] == DIGEST_PAIRING_UNKNOWN
            assert "proto_embedding" in cert.evidence["structural"]["digest_does_not_bind"]
            assert any(
                f"unknown digest_profile {profile!r}" in failure
                for failure in cert.failures
            )
            _assert_not_admitted(cert)
            again = AuditCertificate.from_dict(cert.to_dict())
            assert again.digest_profile == profile
            assert again.passed is False
            assert again.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
            assert again.identity_bound is False

    omitted = SymbolPack.from_dict(make_pack().to_dict())
    assert omitted.digest_profile is None
    assert omitted.identity_digest is None
    unbound = certify(omitted)
    assert unbound.passed is True
    assert unbound.decision == CERTIFICATE_DECISION_UNRESOLVED
    assert unbound.identity_bound is False
    assert unbound.evidence["structural"]["digest_pairing"] == DIGEST_PAIRING_UNBOUND
    assert "proto_embedding" in unbound.evidence["structural"]["digest_does_not_bind"]

    pack = make_pack()
    with pytest.raises(ValueError, match="identity-1"):
        pack.seal(digest_profile="identity-2")
    assert pack.digest_profile is None
    assert pack.identity_digest is None


def test_identity_digest_serialization_roundtrip(tmp_path):
    unbound = make_pack()
    raw = unbound.to_dict()
    assert "digest_profile" not in raw
    assert "identity_digest" not in raw
    loaded = SymbolPack.from_dict(raw)
    assert loaded.checksum == unbound.checksum
    assert loaded.digest_profile is None
    assert loaded.identity_digest is None
    unbound_cert = certify(loaded)
    assert unbound_cert.passed is True
    assert unbound_cert.decision == CERTIFICATE_DECISION_UNRESOLVED
    assert unbound_cert.identity_bound is False

    unbound_path = tmp_path / "unbound.json"
    save_pack(unbound, unbound_path)
    saved = json.loads(unbound_path.read_text(encoding="utf-8"))
    assert "digest_profile" not in saved
    assert "identity_digest" not in saved
    from_disk = load_pack(unbound_path)
    assert from_disk.checksum == unbound.checksum
    assert from_disk.compute_checksum() == unbound.checksum
    assert from_disk.digest_profile is None
    assert from_disk.identity_digest is None

    ident = make_pack()
    checksum = ident.checksum
    _seal_identity(ident)
    assert ident.checksum == checksum
    identity_path = tmp_path / "identity.json"
    save_pack(ident, identity_path)
    loaded_id = load_pack(identity_path)
    assert loaded_id.checksum == checksum
    assert loaded_id.digest_profile == DIGEST_PROFILE_IDENTITY_1
    assert loaded_id.identity_digest == ident.identity_digest
    loaded_id.symbols[0].proto_embedding[0] += 1e-6
    assert loaded_id.compute_checksum() == loaded_id.checksum
    edited = certify(loaded_id)
    assert edited.identity_bound is False
    assert edited.passed is False
    assert edited.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
    assert edited.gloss_bound is True
    assert any("identity digest mismatch" in failure for failure in edited.failures)


def test_default_seal_does_not_opt_into_identity1():
    pack = make_pack()
    assert pack.digest_profile is None
    assert pack.identity_digest is None
    before = pack.to_dict()
    pack.seal()
    assert "digest_profile" not in pack.to_dict()
    assert pack.to_dict().keys() == before.keys()
    assert pack.checksum == pack.compute_checksum()

    _seal_identity(pack)
    sealed_identity = pack.identity_digest
    pack.symbols[0].definition = "rewritten after identity seal"
    pack.seal()
    assert pack.digest_profile == DIGEST_PROFILE_IDENTITY_1
    assert pack.identity_digest == sealed_identity
    assert pack.checksum == pack.compute_checksum()
    cert = certify(pack)
    assert cert.gloss_bound is True
    assert cert.identity_bound is False
    assert cert.passed is False
    assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL


def test_identity1_empty_reservoir_is_not_a_pass():
    pack = make_pack(
        observations={},
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0],
                observation_ids=["obs-1"],
                definition="bound gloss for a missing row",
            )
        ],
    )
    _seal_identity(pack)
    cert = certify(pack)
    assert cert.identity_bound is True
    assert cert.gloss_bound is True
    assert cert.unfoldable is False
    assert cert.passed is False
    assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
    assert any(
        "missing reservoir observation obs-1" in failure for failure in cert.failures
    )
    _assert_not_admitted(cert)


def test_identity1_empty_prototype_is_canonical_empty_list():
    pack = make_pack()
    pack.symbols[0].proto_embedding = []
    _seal_identity(pack)
    row = pack.identity_payload()["symbols"][0]
    assert row["proto_embedding"] == []
    assert "proto_embedding" not in pack._checksum_payload()["symbols"][0]
    cert = certify(pack)
    assert cert.identity_bound is True
    assert cert.passed is False
    assert cert.unfoldable is False
    pack.symbols[0].proto_embedding = [1.0, 0.0, 0.0]
    assert pack.compute_identity_digest() != pack.identity_digest
    assert certify(pack).identity_bound is False


def test_identity1_explicit_reseal_replaces_the_digest():
    """An explicit identity-1 seal recomputes and replaces the digest.

    That seals the object as it exists now. It is not evidence that the
    previous object was unchanged. An auditor must record the original
    identity_digest before any intentional reseal. The reseal is a new
    sealed object, not a proof the old digest still described the mutated
    prototype.
    """
    pack = make_pack()
    pack.seal(digest_profile="identity-1")
    original_digest = pack.identity_digest
    original_checksum = pack.checksum
    original_proto = list(pack.symbols[0].proto_embedding)
    cert = certify(pack)
    assert cert.identity_bound is True
    assert cert.passed is True
    assert cert.decision == CERTIFICATE_DECISION_UNRESOLVED
    assert cert.gloss_bound is True

    pack.symbols[0].proto_embedding[0] += 1e-6
    mutated = certify(pack)
    assert mutated.identity_bound is False
    assert mutated.passed is False
    assert mutated.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
    assert mutated.gloss_bound is True
    assert mutated.unfoldable is True
    assert pack.identity_digest == original_digest
    assert pack.identity_digest != pack.compute_identity_digest()
    assert pack.identity_payload()["symbols"][0]["proto_embedding"][0] == 1.000001

    pack.seal(digest_profile="identity-1")
    assert pack.checksum == original_checksum
    assert pack.identity_digest != original_digest
    assert pack.identity_digest == sha16(pack.identity_payload())
    resealed = certify(pack)
    assert resealed.identity_bound is True
    assert resealed.passed is True
    assert resealed.decision == CERTIFICATE_DECISION_UNRESOLVED
    assert resealed.gloss_bound is True
    new_digest = pack.identity_digest
    assert original_digest != new_digest, (
        "the reseal is a new sealed object, not a proof the old digest "
        "still described the mutated prototype"
    )

    # Assign the saved original digest. Do not seal. The mutated prototype
    # is still the object that was just resealed.
    pack.identity_digest = original_digest
    assert certify(pack).identity_bound is False, (
        "the reseal is a new sealed object, not a proof the old digest "
        "still described the mutated prototype"
    )

    pack.symbols[0].proto_embedding = list(original_proto)
    pack.identity_digest = new_digest
    assert certify(pack).identity_bound is False
    pack.identity_digest = original_digest
    restored = certify(pack)
    assert restored.identity_bound is True
    assert restored.passed is True
    assert restored.decision == CERTIFICATE_DECISION_UNRESOLVED
    assert restored.gloss_bound is True


def test_identity1_default_reseal_does_not_refresh_identity():
    """Default seal() after +1e-6 refreshes checksum only.

    It does not replace identity_digest. A default reseal cannot silently
    legitimize an altered prototype.
    """
    pack = make_pack()
    pack.seal(digest_profile="identity-1")
    original_digest = pack.identity_digest
    original_checksum = pack.checksum
    pack.symbols[0].proto_embedding[0] += 1e-6
    pack.seal()
    assert pack.digest_profile == DIGEST_PROFILE_IDENTITY_1
    assert pack.identity_digest == original_digest
    assert pack.checksum == original_checksum
    assert pack.checksum == pack.compute_checksum()
    assert pack.identity_digest != pack.compute_identity_digest()
    cert = certify(pack)
    assert cert.identity_bound is False
    assert cert.gloss_bound is True
    assert cert.passed is False
    assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
    assert cert.unfoldable is True
    _assert_not_admitted(cert)


def test_inexact_digest_profile_is_not_identity1():
    """Only the exact string identity-1 is an identity profile.

    Case, surrounding whitespace, a blank, a very long string, and a
    non-string are not stripped or folded into identity-1. The only
    legacy word is the exact string unbound, and only with an empty digest.
    """
    sealed = make_pack()
    sealed.seal(digest_profile="identity-1")
    good = sealed.identity_digest
    long_name = "identity-1" + ("x" * 4096)
    rejected = (
        "Identity-1",
        "IDENTITY-1",
        " identity-1",
        "identity-1 ",
        "identity-1\n",
        "\tidentity-1",
        long_name,
        " unbound",
        "unbound ",
        "Unbound",
        "UNBOUND",
    )
    for profile in rejected:
        for digest in (_OMITTED, good):
            loaded = _pack_with_digest_fields(profile, digest)
            assert loaded.digest_profile == profile
            again = SymbolPack.from_dict(json.loads(json.dumps(loaded.to_dict())))
            assert again.digest_profile == profile
            cert = certify(again)
            assert cert.identity_bound is False
            assert cert.passed is False
            assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
            assert cert.digest_profile == profile
            assert cert.evidence["structural"]["digest_pairing"] == DIGEST_PAIRING_UNKNOWN
            assert any(
                f"unknown digest_profile {profile!r}" in failure
                for failure in cert.failures
            )
            _assert_not_admitted(cert)

    legacy = _pack_with_digest_fields("unbound", "")
    legacy_cert = certify(legacy)
    assert legacy_cert.evidence["structural"]["digest_pairing"] == DIGEST_PAIRING_UNBOUND
    assert legacy_cert.identity_bound is False
    assert legacy_cert.passed is True
    assert legacy_cert.decision == CERTIFICATE_DECISION_UNRESOLVED
    not_legacy = _pack_with_digest_fields("unbound", good)
    not_legacy_cert = certify(not_legacy)
    assert not_legacy_cert.identity_bound is False
    assert not_legacy_cert.passed is False
    assert not_legacy_cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
    assert not_legacy_cert.evidence["structural"]["digest_pairing"] == (
        DIGEST_PAIRING_INCONSISTENT
    )

    for profile, digest in (
        ("", ""),
        ("   ", "   "),
        ("\t", "\n"),
        (" \t\n ", " \t "),
        (None, None),
        (_OMITTED, _OMITTED),
    ):
        loaded = _pack_with_digest_fields(profile, digest)
        cert = certify(loaded)
        assert cert.evidence["structural"]["digest_pairing"] == DIGEST_PAIRING_UNBOUND
        assert cert.identity_bound is False
        assert cert.passed is True
        assert cert.decision == CERTIFICATE_DECISION_UNRESOLVED
        assert "digest_profile" not in loaded.to_dict()
        assert "identity_digest" not in loaded.to_dict()

    for profile in ("", "   ", "\t", "\n", " \t\n ", None, _OMITTED):
        loaded = _pack_with_digest_fields(profile, good)
        cert = certify(loaded)
        assert cert.evidence["structural"]["digest_pairing"] == DIGEST_PAIRING_INCONSISTENT
        assert cert.identity_bound is False
        assert cert.passed is False
        assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
        assert "unverified" in " ".join(cert.failures)

    for digest in (f" {good}", f"{good} ", good.upper(), f"\n{good}"):
        loaded = _pack_with_digest_fields("identity-1", digest)
        assert loaded.identity_digest == digest
        cert = certify(loaded)
        assert cert.digest_profile == DIGEST_PROFILE_IDENTITY_1
        assert cert.identity_bound is False
        assert cert.passed is False
        assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
        assert any("identity digest mismatch" in failure for failure in cert.failures)

    for profile in (1, ["identity-1"], {"name": "identity-1"}):
        loaded = _pack_with_digest_fields(profile, good)
        assert isinstance(loaded.digest_profile, str)
        assert loaded.digest_profile not in (
            DIGEST_PROFILE_IDENTITY_1,
            "unbound",
            None,
        )
        blob = json.loads(json.dumps(loaded.to_dict()))
        again = SymbolPack.from_dict(blob)
        cert = certify(again)
        assert cert.identity_bound is False
        assert cert.passed is False
        assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL
        assert cert.digest_profile != DIGEST_PROFILE_IDENTITY_1
        assert cert.evidence["structural"]["digest_pairing"] == DIGEST_PAIRING_UNKNOWN
        _assert_not_admitted(cert)
        direct = make_pack()
        direct.digest_profile = profile
        direct.identity_digest = good
        direct_cert = certify(direct)
        assert direct_cert.identity_bound is False
        assert direct_cert.passed is False
        assert direct_cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL

    for digest in (1, ["identity-1"], {"digest": good}):
        loaded = _pack_with_digest_fields("identity-1", digest)
        cert = certify(loaded)
        assert cert.identity_bound is False
        assert cert.passed is False
        assert cert.decision == CERTIFICATE_DECISION_RESEARCH_FAIL

    control = _pack_with_digest_fields("identity-1", good)
    assert certify(control).identity_bound is True

    raw_cert = certify(make_pack()).to_dict()
    raw_cert["digest_profile"] = " identity-1"
    loaded_cert = AuditCertificate.from_dict(raw_cert)
    assert loaded_cert.digest_profile == " identity-1"
    raw_cert["digest_profile"] = ["identity-1"]
    loaded_cert = AuditCertificate.from_dict(raw_cert)
    assert isinstance(loaded_cert.digest_profile, str)
    assert loaded_cert.digest_profile != DIGEST_PROFILE_IDENTITY_1
    raw_cert["digest_profile"] = "identity-1"
    raw_cert["identity_digest"] = {"digest": good}
    loaded_cert = AuditCertificate.from_dict(raw_cert)
    assert loaded_cert.identity_digest != good

    pack = make_pack()
    pack.seal(digest_profile="identity-1")
    checksum = pack.checksum
    digest = pack.identity_digest
    profile = pack.digest_profile
    bad_seal_values = (
        *rejected,
        "",
        "   ",
        "\t",
        1,
        ["identity-1"],
        {"name": "identity-1"},
        good.upper(),
    )
    for bad in bad_seal_values:
        with pytest.raises(ValueError, match="identity-1"):
            pack.seal(digest_profile=bad)
        assert pack.checksum == checksum
        assert pack.identity_digest == digest
        assert pack.digest_profile == profile
    fresh = make_pack()
    fresh_checksum = fresh.checksum
    with pytest.raises(ValueError, match="identity-1"):
        fresh.seal(digest_profile=" identity-1")
    assert fresh.digest_profile is None
    assert fresh.identity_digest is None
    assert fresh.checksum == fresh_checksum


def _assert_serialized_certificate(
    cert,
    *,
    identity_bound,
    passed,
    decision,
    gloss_bound,
):
    again = AuditCertificate.from_dict(cert.to_dict())
    assert again.identity_bound is identity_bound
    assert again.passed is passed
    assert again.decision == decision
    assert again.gloss_bound is gloss_bound
    assert again.schema_version == CERTIFICATE_SCHEMA_V02 == "0.2"
    assert again.authority == CERTIFICATE_AUTHORITY == "RUNTIME_NOT_AUTHORIZED"
    assert again.evidence["numerical"]["status"] == "NOT IMPLEMENTED"
    assert again.evidence["numerical"]["decision"] == CERTIFICATE_DECISION_UNRESOLVED
    assert again.evidence["semantic"]["status"] == "NOT IMPLEMENTED"
    assert again.evidence["semantic"]["decision"] == CERTIFICATE_DECISION_UNRESOLVED
    assert again.decision != "RESEARCH_PASS"
    return again


def test_identity_certificate_roundtrip_keeps_the_certify_result():
    pack = make_pack()
    pack.seal(digest_profile="identity-1")
    ok = certify(pack)
    again = _assert_serialized_certificate(
        ok,
        identity_bound=True,
        passed=True,
        decision=CERTIFICATE_DECISION_UNRESOLVED,
        gloss_bound=True,
    )
    assert again.digest_profile == DIGEST_PROFILE_IDENTITY_1
    assert again.identity_digest == pack.identity_digest

    pack.symbols[0].proto_embedding[0] += 1e-6
    failed = certify(pack)
    failed_again = _assert_serialized_certificate(
        failed,
        identity_bound=False,
        passed=False,
        decision=CERTIFICATE_DECISION_RESEARCH_FAIL,
        gloss_bound=True,
    )
    assert failed_again.digest_profile == DIGEST_PROFILE_IDENTITY_1
    assert any(
        "identity digest mismatch" in failure for failure in failed_again.failures
    )

    unknown = certify(_pack_with_digest_fields("Identity-1", pack.identity_digest))
    unknown_again = _assert_serialized_certificate(
        unknown,
        identity_bound=False,
        passed=False,
        decision=CERTIFICATE_DECISION_RESEARCH_FAIL,
        gloss_bound=True,
    )
    assert unknown_again.digest_profile == "Identity-1"
    assert unknown_again.evidence["structural"]["digest_pairing"] == (
        DIGEST_PAIRING_UNKNOWN
    )

    stored = ok.to_dict()
    stored["decision"] = "RESEARCH_PASS"
    with pytest.raises(ValueError, match="RESEARCH_PASS"):
        AuditCertificate.from_dict(stored)

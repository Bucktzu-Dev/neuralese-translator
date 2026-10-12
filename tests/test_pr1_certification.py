"""Locks from docs/PR1_CERTIFICATION_DISCREPANCY.md section 5, plus schema 0.2.

The digest is unchanged. proto_embedding, confidence, and the other fields
named in DIGEST_DOES_NOT_BIND stay outside it. A missing reservoir still
fails unfold. passed stays the structural conjunction.
"""
import pytest

from neuralese import __version__
from neuralese.audit import certify
from neuralese.contracts import (
    CERTIFICATE_AUTHORITY,
    CERTIFICATE_DECISION_RESEARCH_FAIL,
    CERTIFICATE_DECISION_UNRESOLVED,
    CERTIFICATE_SCHEMA_UNVERSIONED,
    CERTIFICATE_SCHEMA_V02,
    DIGEST_DOES_NOT_BIND,
    AuditCertificate,
    Symbol,
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

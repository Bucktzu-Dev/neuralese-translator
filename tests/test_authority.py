import pytest

from neuralese.audit import certify
from neuralese.contracts import (
    Receipt,
    Symbol,
    UncertifiedPackError,
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

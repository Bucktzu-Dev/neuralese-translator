from neuralese.audit import certify
from neuralese.contracts import Symbol

from packutil import make_pack


def test_certify_passes_on_sealed_unfoldable_pack():
    pack = make_pack()
    cert = certify(pack)
    assert cert.passed
    assert cert.addressable
    assert cert.unfoldable
    assert cert.gloss_bound
    assert cert.residual_ok
    assert cert.fail_closed
    assert cert.failures == []


def test_certify_fails_without_observation_ids():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0],
                observation_ids=[],
                definition="orphan",
                confidence=0.5,
            )
        ]
    )
    cert = certify(pack)
    assert not cert.passed
    assert not cert.unfoldable
    assert any("observation_ids" in f for f in cert.failures)


def test_certify_fails_without_gloss():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0],
                observation_ids=["obs-1"],
                definition=None,
                confidence=0.5,
            )
        ]
    )
    cert = certify(pack)
    assert not cert.passed
    assert not cert.gloss_bound


def test_certify_fails_unglossed_sentinel():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0],
                observation_ids=["obs-1"],
                definition="[unglossed]",
                confidence=0.0,
            )
        ]
    )
    cert = certify(pack)
    assert not cert.passed
    assert not cert.gloss_bound
    allowed = certify(pack, require_gloss=False)
    assert allowed.gloss_bound
    assert allowed.passed


def test_certify_detects_checksum_tamper():
    pack = make_pack()
    pack.symbols[0].definition = "mutated after seal"
    cert = certify(pack)
    assert not cert.passed
    assert not cert.gloss_bound
    assert any("checksum" in f for f in cert.failures)


def test_certify_fails_alias_cycle():
    pack = make_pack(aliases={7: 8, 8: 7})
    cert = certify(pack)
    assert not cert.addressable
    assert any("cycle" in f for f in cert.failures)


def test_quarantined_without_ids_still_passes_unfold():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0],
                observation_ids=["obs-1"],
                definition="live",
                confidence=0.6,
            ),
            Symbol(
                class_id=1,
                code=1,
                proto_embedding=[0.0, 1.0],
                observation_ids=[],
                definition="[quarantined class 1]",
                quarantined=True,
            ),
        ]
    )
    cert = certify(pack)
    assert cert.unfoldable
    assert cert.passed

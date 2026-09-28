from neuralese.audit import decodability_report
from neuralese.contracts import Symbol

from packutil import make_pack


def test_report_counts_gloss_coverage_and_decision():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0, 0.0],
                observation_ids=["obs-hello"],
                definition="Symbol for hello.",
                examples=["hello"],
                confidence=0.8,
            ),
            Symbol(
                class_id=1,
                code=1,
                proto_embedding=[0.0, 1.0, 0.0],
                observation_ids=["obs-gap"],
                definition="[unglossed: class 1]",
                examples=["gap"],
                confidence=0.2,
            ),
            Symbol(
                class_id=2,
                code=2,
                proto_embedding=[0.0, 0.0, 1.0],
                observation_ids=["obs-held"],
                definition="[quarantined class 2]",
                examples=["held"],
                confidence=0.0,
                quarantined=True,
            ),
        ],
        metadata={"decision": "accept_provisional"},
        mdl_bits=4.5,
    )
    report = decodability_report(pack)
    assert report["n_live"] == 2
    assert report["n_quarantined"] == 1
    assert report["gloss_coverage"] == 0.5
    assert report["decision"] == "accept_provisional"
    assert report["mdl_bits"] == 4.5
    assert report["reservoir_size"] == 3
    assert report["unfold_failures"] == []
    assert report["residual"] == pack.reconstruction_error


def test_report_omits_decision_when_learn_did_not_record_one():
    report = decodability_report(make_pack())
    assert report["decision"] is None


def test_report_lists_prototype_mismatch_as_unfold_failure():
    pack = make_pack()
    pack.observations["obs-hello"].embedding = [0.0, 1.0, 0.0]
    report = decodability_report(pack)
    assert report["unfold_failures"]
    assert any("prototype" in item for item in report["unfold_failures"])

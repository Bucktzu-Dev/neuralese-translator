from neuralese.audit import cluster_majority_diagnostic, decodability_report
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


def test_report_without_topic_labels_omits_cluster_purity():
    report = decodability_report(make_pack())
    assert "cluster_purity" not in report


def test_cluster_purity_under_half_is_mixed_and_not_a_topic_claim():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0],
                observation_ids=["a", "b", "c", "d", "e"],
                definition="Symbol for mixed.",
                examples=["a"],
                confidence=0.5,
            ),
            Symbol(
                class_id=1,
                code=1,
                proto_embedding=[0.0, 1.0],
                observation_ids=["f", "g"],
                definition="Symbol for tied.",
                examples=["f"],
                confidence=0.5,
            ),
        ]
    )
    # 2 bakery, 2 harbor, 1 ledger. Purity 0.4 is mixed. The count tie picks harbor.
    mixed = cluster_majority_diagnostic(
        pack,
        {"a": "bakery", "b": "bakery", "c": "harbor", "d": "harbor", "e": "ledger", "f": "harbor", "g": "bakery"},
    )
    low = mixed["clusters"][0]
    assert low["purity"] == 0.4
    assert low["mixed"] is True
    assert low["label"] == "mixed"
    assert low["majority_topic"] == "harbor"
    assert low["majority_count"] == 2
    # Equal counts: the lexicographically later name wins, and 0.5 is not mixed.
    tied = mixed["clusters"][1]
    assert tied["majority_topic"] == "harbor"
    assert tied["purity"] == 0.5
    assert tied["mixed"] is False
    assert tied["label"] == "harbor"
    # An unlabeled row dilutes purity and is not read off the id.
    diluted = cluster_majority_diagnostic(pack, {"a": "bakery", "b": "bakery"})
    assert diluted["clusters"][0]["rows"] == 5
    assert diluted["clusters"][0]["majority_count"] == 2
    assert diluted["clusters"][0]["purity"] == 0.4
    assert diluted["clusters"][0]["label"] == "mixed"
    assert diluted["n_unlabeled"] == 5
    report = decodability_report(pack, topic_labels={"a": "bakery"})
    assert "cluster_purity" in report


def test_report_lists_prototype_mismatch_as_unfold_failure():
    pack = make_pack()
    pack.observations["obs-hello"].embedding = [0.0, 1.0, 0.0]
    report = decodability_report(pack)
    assert report["unfold_failures"]
    assert any("prototype" in item for item in report["unfold_failures"])

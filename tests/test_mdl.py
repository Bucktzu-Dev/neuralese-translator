from pathlib import Path

from neuralese.adapters import load_observations_jsonl
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.contracts import Observation

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "mdl_delta" / "observations.jsonl"


def _finalize(pack):
    return next(receipt for receipt in pack.receipts if receipt.step == "finalize")


def test_first_pack_delta_is_zero():
    obs = load_observations_jsonl(FIXTURE)
    pack = learn_pack(obs, config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0))
    assert pack.parent_pack_id is None
    assert pack.guards.delta_mdl == 0.0
    assert _finalize(pack).delta_mdl_bits == 0.0
    assert pack.metadata["decision"] == "accept"


def _pair(prefix: str, scale: float) -> list[Observation]:
    """Two centers. `scale` spreads points away from the center."""
    rows = []
    centers = ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
    texts = (
        ["north wind over the ridge", "north slope stayed cold", "a north marker on the trail", "north camp before sunrise"],
        ["south field in full sun", "south road toward the coast", "south garden after noon", "south well beside the oaks"],
    )
    spreads = (0.4, -0.3, 0.2, 0.1)
    for cluster, (center, names) in enumerate(zip(centers, texts)):
        for index, name in enumerate(names):
            embedding = list(center)
            embedding[(cluster + 1) % 4] += scale * spreads[index]
            rows.append(Observation(f"{prefix}-{cluster}-{index}", embedding=embedding, text=name))
    return rows


def test_tighter_child_records_negative_delta():
    parent = learn_pack(_pair("wide", 1.0), config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0))
    child = learn_pack(
        _pair("tight", 0.0),
        config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0),
        previous=parent,
    )
    delta = child.mdl_bits - parent.mdl_bits
    assert delta < -1.0
    assert child.guards.delta_mdl == delta
    assert _finalize(child).delta_mdl_bits == delta
    assert child.guards.pass_mdl
    assert child.parent_pack_id == parent.pack_id
    assert child.metadata["decision"] == "accept"
    assert [symbol for symbol in child.symbols if symbol.quarantined] == []
    assert {symbol.code for symbol in child.symbols} == {symbol.code for symbol in parent.symbols}


def test_larger_child_rejects_unless_exception_is_recorded():
    obs = load_observations_jsonl(FIXTURE)
    parent = learn_pack(obs, config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0))
    child = learn_pack(
        obs,
        config=LearnConfig(n_symbols=4, min_cluster_size=1, seed=0),
        previous=parent,
    )
    delta = child.mdl_bits - parent.mdl_bits
    assert delta > 1.0
    assert child.guards.delta_mdl == delta
    assert child.guards.pass_mdl is False
    assert child.metadata["decision"] == "reject"
    assert "mdl_exception" not in child.metadata

    waived = learn_pack(
        obs,
        config=LearnConfig(
            n_symbols=4,
            min_cluster_size=1,
            seed=0,
            mdl_exception="split for a new domain",
        ),
        previous=parent,
    )
    assert waived.guards.delta_mdl == delta
    assert waived.guards.pass_mdl is False
    assert waived.metadata["decision"] == "accept_provisional"
    assert waived.metadata["mdl_exception"] == "split for a new domain"
    finalize = _finalize(waived)
    assert finalize.metadata["mdl_exception"] == "split for a new domain"
    assert finalize.ok is True

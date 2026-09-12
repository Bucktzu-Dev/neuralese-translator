from pathlib import Path

from neuralese.adapters import load_observations_jsonl
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify
from neuralese.translator import translate_stream

TOY = Path(__file__).resolve().parents[1] / "examples" / "toy_stream" / "observations.jsonl"


def test_learn_toy_pack_certifies():
    obs = load_observations_jsonl(TOY)
    pack = learn_pack(obs, config=LearnConfig(n_symbols=3, min_cluster_size=2, seed=0))
    cert = certify(pack)
    assert cert.passed, cert.failures
    assert len(pack.symbols) == 3
    live = [s for s in pack.symbols if not s.quarantined]
    assert live
    assert all(s.observation_ids for s in live)
    assert all(s.definition for s in live)


def test_learn_then_translate_unknown_code():
    obs = load_observations_jsonl(TOY)
    pack = learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0))
    glosses = translate_stream(pack, [s.code for s in pack.symbols] + [99])
    assert glosses[-1].state == "unknown"
    assert any(g.state == "ok" for g in glosses)


def test_parent_pack_records_parent_id_and_delta():
    obs = load_observations_jsonl(TOY)
    first = learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0))
    second = learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0), previous=first)
    assert second.parent_pack_id == first.pack_id
    finalize = next(r for r in second.receipts if r.step == "finalize")
    assert finalize.delta_mdl_bits is not None

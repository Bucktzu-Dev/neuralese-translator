from pathlib import Path

import pytest

from neuralese.activations import load_activation_dump
from neuralese.adapters import load_observations_jsonl
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify
from neuralese.unfold import unfold_code

ROOT = Path(__file__).resolve().parents[1] / "examples" / "public_domain"
OBSERVATIONS = ROOT / "observations.jsonl"
HIDDEN = ROOT / "hidden_states.jsonl"


def test_public_observations_certify_and_unfold():
    obs = load_observations_jsonl(OBSERVATIONS)
    pack = learn_pack(obs, config=LearnConfig(n_symbols=3, min_cluster_size=2, seed=0))
    cert = certify(pack)
    assert cert.passed, cert.failures
    assert cert.details["reservoir_size"] == 12
    report = unfold_code(pack, 0)
    assert report.missing_ids == []
    assert report.prototype_l2 is not None and report.prototype_l2 < 1e-5


def test_public_hidden_state_dump_matches_observations():
    adapted = load_activation_dump(HIDDEN, layout="vectors")
    direct = load_observations_jsonl(OBSERVATIONS)
    assert [row.observation_id for row in adapted] == [row.observation_id for row in direct]
    for dumped, original in zip(adapted, direct):
        assert dumped.text == original.text
        assert dumped.embedding == pytest.approx(original.embedding)
    pack = learn_pack(adapted, config=LearnConfig(n_symbols=3, min_cluster_size=2, seed=0))
    assert certify(pack).passed

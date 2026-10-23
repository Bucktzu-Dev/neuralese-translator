import json

import numpy as np
import pytest

from neuralese.activations import load_activation_dump, observations_from_hidden_states
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify
from neuralese.cli import main


def test_hf_stack_selects_layer_and_pools_tokens():
    stack = np.zeros((2, 2, 3, 1))
    stack[0, 0, :, 0] = [1, 2, 3]
    stack[0, 1, :, 0] = [4, 5, 6]
    stack[1, 0, :, 0] = [10, 20, 30]
    stack[1, 1, :, 0] = [40, 50, 60]
    last = observations_from_hidden_states(
        stack, layout="hf_stack", layer=-1, pool="last", texts=["a", "b"]
    )
    assert last[0].embedding == pytest.approx([30.0])
    assert last[1].embedding == pytest.approx([60.0])
    mean = observations_from_hidden_states(
        [np.array(stack[0]), np.array(stack[1])], pool="mean", texts=["a", "b"]
    )
    assert mean[0].embedding == pytest.approx([20.0])
    assert mean[1].embedding == pytest.approx([50.0])


def test_layers_layout_picks_one_matrix():
    layers = np.array(
        [
            [[1.0, 0.0], [9.0, 9.0]],
            [[0.0, 1.0], [8.0, 8.0]],
        ]
    )  # (n=2, layers=2, d=2)
    rows = observations_from_hidden_states(layers, layout="layers", layer=0)
    assert rows[0].embedding == pytest.approx([1.0, 0.0])
    assert rows[1].embedding == pytest.approx([0.0, 1.0])


def test_jsonl_tokens_and_cli_roundtrip(tmp_path):
    path = tmp_path / "dump.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "observation_id": "t1",
                        "text": "north marker",
                        "hidden_states": [[0.0, 1.0], [1.0, 0.0]],
                    }
                ),
                json.dumps(
                    {
                        "observation_id": "t2",
                        "text": "north ridge",
                        "hidden_states": [[0.2, 0.2], [0.9, 0.1]],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_activation_dump(path, layout="tokens", pool="last")
    assert rows[0].embedding == pytest.approx([1.0, 0.0])
    assert rows[1].text == "north ridge"
    out = tmp_path / "obs.jsonl"
    rc = main(["adapt", str(path), "-o", str(out), "--layout", "tokens", "--pool", "last"])
    assert rc == 0
    learned = learn_pack(
        load_activation_dump(path, layout="tokens", pool="last"),
        config=LearnConfig(n_symbols=1, min_cluster_size=1, seed=0),
    )
    assert certify(learned).passed


def test_ambiguous_shape_and_bad_layer_fail(tmp_path):
    array = np.zeros((2, 3, 4))
    path = tmp_path / "dump.npy"
    np.save(path, array)
    with pytest.raises(ValueError, match="explicit layout"):
        load_activation_dump(path)
    with pytest.raises(ValueError, match="layer"):
        observations_from_hidden_states(np.zeros((2, 3, 4)), layout="hf_layers", layer=5)


def test_rejects_nonfinite_and_unknown_suffix(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="non-finite"):
        load_activation_dump(path)
    other = tmp_path / "dump.txt"
    other.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_activation_dump(other)

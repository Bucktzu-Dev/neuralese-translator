import json

import numpy as np

from neuralese.adapters import (
    load_activation_matrix,
    load_activations,
    load_alignment_texts,
    observations_from_activations,
)
from neuralese.cli import main


def test_1d_npy_is_one_observation(tmp_path, capsys):
    npy = tmp_path / "state.npy"
    np.save(npy, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    out = tmp_path / "obs.jsonl"
    rc = main(["ingest", str(npy), "-o", str(out)])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_observations"] == 1
    assert summary["dim"] == 3
    loaded = json.loads(out.read_text().splitlines()[0])
    assert loaded["embedding"] == [1.0, 0.0, 0.0]


def test_1d_python_vector_is_one_observation():
    rows = observations_from_activations([1.0, 0.0, 0.5])
    assert len(rows) == 1
    assert rows[0].embedding == [1.0, 0.0, 0.5]


def test_npz_last_hidden_state_ingests(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        last_hidden_state=np.array([[1.0, 0.0], [0.0, 1.0]]),
        prompts=np.array(["hello", "audit"]),
        ids=np.array(["a", "b"]),
    )
    rows = load_activations(path)
    assert [row.observation_id for row in rows] == ["a", "b"]
    assert [row.text for row in rows] == ["hello", "audit"]
    assert rows[0].embedding == [1.0, 0.0]


def test_jsonl_id_prompt_activation_aliases(tmp_path, capsys):
    src = tmp_path / "acts.jsonl"
    src.write_text(
        '{"id":"h1","activation":[1.0,0.0],"prompt":"hello"}\n'
        '{"id":"h2","vector":[0.0,1.0],"prompt":"audit"}\n'
    )
    out = tmp_path / "obs.jsonl"
    rc = main(["ingest", str(src), "-o", str(out)])
    assert rc == 0
    loaded = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert [row["observation_id"] for row in loaded] == ["h1", "h2"]
    assert loaded[0]["text"] == "hello"
    assert loaded[0]["embedding"] == [1.0, 0.0]
    assert "activation" not in loaded[0]
    assert "prompt" not in loaded[0]


def test_json_activation_record_aliases(tmp_path):
    src = tmp_path / "acts.json"
    src.write_text(
        json.dumps(
            {
                "id": "solo",
                "activation": [1.0, 0.0],
                "prompt": "hello there friend",
                "layer": 12,
            }
        )
    )
    rows = load_activations(src)
    assert len(rows) == 1
    assert rows[0].observation_id == "solo"
    assert rows[0].embedding == [1.0, 0.0]
    assert rows[0].text == "hello there friend"
    assert rows[0].metadata["layer"] == 12


def test_json_matrix_prompts_and_ids_aliases(tmp_path):
    src = tmp_path / "acts.json"
    src.write_text(
        json.dumps(
            {
                "hidden_states": [[1.0, 0.0], [0.0, 1.0]],
                "prompts": ["hello", "audit"],
                "ids": ["a", "b"],
            }
        )
    )
    rows = load_activations(src)
    assert [row.observation_id for row in rows] == ["a", "b"]
    assert [row.text for row in rows] == ["hello", "audit"]


def test_alignment_texts_accept_prompt_and_id(tmp_path):
    npy = tmp_path / "states.npy"
    np.save(npy, np.array([[1.0, 0.0], [0.0, 1.0]]))
    texts = tmp_path / "texts.jsonl"
    texts.write_text(
        '{"id":"a","prompt":"hello there friend"}\n'
        '{"id":"b","prompt":"audit the trail"}\n'
    )
    ids, prompts = load_alignment_texts(texts)
    assert ids == ["a", "b"]
    assert prompts == ["hello there friend", "audit the trail"]
    rows = observations_from_activations(
        load_activation_matrix(npy),
        texts=prompts,
        observation_ids=ids,
    )
    assert [row.observation_id for row in rows] == ["a", "b"]
    assert rows[0].text == "hello there friend"

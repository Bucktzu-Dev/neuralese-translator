import math
from unittest.mock import patch

import numpy as np
import pytest

from neuralese.adapters import (
    load_activation_matrix,
    load_activations,
    observations_from_activations,
    save_observations_jsonl,
)
from neuralese.contracts import Observation


def test_padded_observation_id_is_preserved():
    rows = observations_from_activations(
        [[1.0, 0.0], [0.0, 1.0]],
        observation_ids=[" obs-1 ", "obs-1"],
    )
    assert [row.observation_id for row in rows] == [" obs-1 ", "obs-1"]


def test_whitespace_only_observation_id_is_blank():
    with pytest.raises(ValueError, match="is blank"):
        observations_from_activations([[1.0, 0.0]], observation_ids=["  "])


def test_npz_multiple_named_matrices_fail_closed(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        hidden_states=np.array([[1.0, 0.0]]),
        embeddings=np.array([[0.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="only one of"):
        load_activation_matrix(path)


def test_non_finite_metadata_is_not_json_serializable(tmp_path):
    rows = [
        Observation(
            observation_id="obs-1",
            embedding=[1.0, 0.0],
            metadata={"score": math.nan},
        )
    ]
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_observations_jsonl(rows, tmp_path / "obs.jsonl")
    rows[0].metadata["score"] = math.inf
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_observations_jsonl(rows, tmp_path / "obs.jsonl")


def test_recursive_metadata_copy_fails_closed(tmp_path):
    src = tmp_path / "acts.jsonl"
    src.write_text('{"hidden_state":[1.0,0.0],"metadata":{"k":1}}\n')
    with patch(
        "neuralese.adapters._copy_mapping",
        side_effect=RecursionError("nested"),
    ):
        with pytest.raises(ValueError, match="metadata is invalid"):
            load_activations(src)


def test_npz_texts_length_mismatch_fails_closed(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        hidden_states=np.array([[1.0, 0.0], [0.0, 1.0]]),
        texts=np.array(["only-one"]),
    )
    with pytest.raises(ValueError, match="length must match"):
        load_activations(path)


def test_non_finite_npy_fails_closed(tmp_path):
    path = tmp_path / "states.npy"
    np.save(path, np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="finite"):
        load_activation_matrix(path)
    np.save(path, np.array([[1.0, np.inf]]))
    with pytest.raises(ValueError, match="finite"):
        load_activation_matrix(path)


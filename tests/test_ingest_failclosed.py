import json
import math
import zipfile
from unittest.mock import patch

import numpy as np
import pytest

from neuralese.adapters import (
    load_activation_matrix,
    load_activations,
    load_alignment_texts,
    observations_from_activations,
    save_observations_jsonl,
)
from neuralese.cli import main
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


def test_npz_named_matrix_plus_extra_array_fails_closed(tmp_path, capsys):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        hidden_states=np.array([[1.0, 0.0]]),
        arr_0=np.array([[0.0, 1.0]]),
    )
    rc = main(["ingest", str(path), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "exactly one matrix" in err
    assert "Traceback" not in err


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


def test_non_string_metadata_keys_are_not_json_laundered(tmp_path):
    out = tmp_path / "obs.jsonl"
    rows = [
        Observation(
            observation_id="obs-1",
            embedding=[1.0, 0.0],
            metadata={1: "bad"},
        )
    ]
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_observations_jsonl(rows, out)
    assert not out.exists()
    rows[0].metadata = {"nested": {True: 1}}
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_observations_jsonl(rows, out)
    assert not out.exists()


def test_save_does_not_leave_partial_jsonl_on_later_row_failure(tmp_path):
    out = tmp_path / "obs.jsonl"
    out.write_text('{"observation_id":"stale"}\n')
    rows = [
        Observation(
            observation_id="obs-1",
            embedding=[1.0, 0.0],
            metadata={"ok": 1},
        ),
        Observation(
            observation_id="obs-2",
            embedding=[0.0, 1.0],
            metadata={"score": math.nan},
        ),
    ]
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_observations_jsonl(rows, out)
    assert out.read_text() == '{"observation_id":"stale"}\n'


def test_npz_truncated_member_is_clean_cli_failure(tmp_path, capsys):
    path = tmp_path / "states.npz"
    np.savez(path, hidden_states=np.array([[1.0, 0.0]]))
    with patch(
        "neuralese.adapters._array_from_npz",
        side_effect=zipfile.BadZipFile("truncated"),
    ):
        rc = main(["ingest", str(path), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "invalid activation archive" in err
    assert "Traceback" not in err


def test_npz_last_hidden_state_collides_with_hidden_states(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        hidden_states=np.array([[1.0, 0.0]]),
        last_hidden_state=np.array([[0.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="only one of"):
        load_activation_matrix(path)


def test_npz_texts_and_prompts_fail_closed(tmp_path, capsys):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        hidden_states=np.array([[1.0, 0.0]]),
        texts=np.array(["from-texts"]),
        prompts=np.array(["from-prompts"]),
    )
    rc = main(["ingest", str(path), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "texts or prompts" in err
    assert "Traceback" not in err


def test_jsonl_id_and_observation_id_fail_closed(tmp_path, capsys):
    src = tmp_path / "acts.jsonl"
    src.write_text(
        '{"id":"a","observation_id":"b","hidden_state":[1.0,0.0]}\n'
    )
    rc = main(["ingest", str(src), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "observation_id or id" in err
    assert "Traceback" not in err


def test_jsonl_activation_and_hidden_state_fail_closed(tmp_path, capsys):
    src = tmp_path / "acts.jsonl"
    src.write_text(
        '{"hidden_state":[1.0,0.0],"activation":[0.0,1.0]}\n'
    )
    rc = main(["ingest", str(src), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "exactly one of" in err
    assert "Traceback" not in err


def test_jsonl_text_and_prompt_fail_closed(tmp_path, capsys):
    src = tmp_path / "acts.jsonl"
    src.write_text(
        '{"hidden_state":[1.0,0.0],"text":"hello","prompt":"audit"}\n'
    )
    rc = main(["ingest", str(src), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "text or prompt" in err
    assert "Traceback" not in err


def test_mapping_texts_are_not_indexed_as_rows():
    with pytest.raises(TypeError, match="not a mapping"):
        observations_from_activations([[1.0, 0.0]], texts={"label": "hello"})


def test_mapping_observation_ids_are_not_indexed_as_rows():
    with pytest.raises(TypeError, match="not a mapping"):
        observations_from_activations(
            [[1.0, 0.0]],
            observation_ids={"0": "obs-1"},
        )


def test_json_matrix_texts_object_is_clean_cli_failure(tmp_path, capsys):
    src = tmp_path / "acts.json"
    src.write_text(
        '{"hidden_states":[[1.0,0.0]],"texts":{"label":"hello"}}\n'
    )
    rc = main(["ingest", str(src), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not a mapping" in err
    assert "Traceback" not in err


def test_json_matrix_ids_object_is_clean_cli_failure(tmp_path, capsys):
    src = tmp_path / "acts.json"
    src.write_text(
        '{"hidden_states":[[1.0,0.0]],"observation_ids":{"0":"obs-1"}}\n'
    )
    rc = main(["ingest", str(src), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not a mapping" in err
    assert "Traceback" not in err


def test_npz_truncated_member_eoferror_is_clean_cli_failure(tmp_path, capsys):
    path = tmp_path / "states.npz"
    np.savez(path, hidden_states=np.array([[1.0, 0.0]]))
    with patch(
        "neuralese.adapters._array_from_npz",
        side_effect=EOFError("truncated"),
    ):
        rc = main(["ingest", str(path), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "invalid activation archive" in err
    assert "Traceback" not in err


def test_non_object_metadata_is_not_json_serializable(tmp_path):
    out = tmp_path / "obs.jsonl"
    rows = [
        Observation(
            observation_id="obs-1",
            embedding=[1.0, 0.0],
            metadata={"ok": 1},
        )
    ]
    rows[0].metadata = None
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_observations_jsonl(rows, out)
    assert not out.exists()
    rows[0].metadata = []
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_observations_jsonl(rows, out)
    assert not out.exists()


def test_cli_mixed_record_layers_receipt_is_null(tmp_path, capsys):
    src = tmp_path / "acts.jsonl"
    src.write_text(
        '{"hidden_state":[1.0,0.0],"layer":3}\n'
        '{"hidden_state":[0.0,1.0],"layer":9}\n'
    )
    out = tmp_path / "obs.jsonl"
    rc = main(["ingest", str(src), "-o", str(out)])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["layer"] is None
    loaded = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert [row["metadata"]["layer"] for row in loaded] == [3, 9]


def test_jsonl_hidden_state_and_hidden_states_fail_closed(tmp_path, capsys):
    src = tmp_path / "acts.jsonl"
    src.write_text(
        '{"hidden_state":[1.0,0.0],"hidden_states":[0.0,1.0]}\n'
    )
    rc = main(["ingest", str(src), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "hidden_state or hidden_states" in err
    assert "Traceback" not in err


def test_cli_texts_does_not_skip_npz_alias_collision(tmp_path, capsys):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        hidden_states=np.array([[1.0, 0.0]]),
        texts=np.array(["from-texts"]),
        prompts=np.array(["from-prompts"]),
    )
    texts = tmp_path / "texts.jsonl"
    texts.write_text('{"text":"hello there friend"}\n')
    rc = main(
        ["ingest", str(path), "-o", str(tmp_path / "obs.jsonl"), "--texts", str(texts)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "texts or prompts" in err
    assert "Traceback" not in err


def test_ingest_refuses_to_overwrite_texts(tmp_path, capsys):
    src = tmp_path / "states.npy"
    np.save(src, np.array([[1.0, 0.0]]))
    texts = tmp_path / "prompts.jsonl"
    texts.write_text('{"text":"hello there friend"}\n')
    rc = main(["ingest", str(src), "-o", str(texts), "--texts", str(texts)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "differ" in err
    assert "Traceback" not in err
    assert texts.read_text() == '{"text":"hello there friend"}\n'


def test_npz_nan_ids_fail_closed(tmp_path, capsys):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        hidden_states=np.array([[1.0, 0.0]]),
        observation_ids=np.array([np.nan]),
    )
    rc = main(["ingest", str(path), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "must be a string or null" in err
    assert "Traceback" not in err


def test_npz_nan_texts_are_missing(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        hidden_states=np.array([[1.0, 0.0]]),
        texts=np.array([np.nan]),
    )
    rows = load_activations(path)
    assert len(rows) == 1
    assert rows[0].text is None
    assert rows[0].observation_id == "obs-1"


def test_npz_is_loaded_once(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        hidden_states=np.array([[1.0, 0.0]]),
        texts=np.array(["hello"]),
        observation_ids=np.array(["obs-1"]),
    )
    real_load = np.load
    calls = []

    def counting_load(*args, **kwargs):
        calls.append(1)
        return real_load(*args, **kwargs)

    with patch("neuralese.adapters.np.load", side_effect=counting_load):
        rows = load_activations(path)
    assert len(calls) == 1
    assert rows[0].text == "hello"
    assert rows[0].observation_id == "obs-1"


def test_alignment_json_texts_and_prompts_fail_closed(tmp_path, capsys):
    src = tmp_path / "states.npy"
    np.save(src, np.array([[1.0, 0.0]]))
    texts = tmp_path / "texts.json"
    texts.write_text('{"texts":["from-texts"],"prompts":["from-prompts"]}\n')
    rc = main(
        ["ingest", str(src), "-o", str(tmp_path / "obs.jsonl"), "--texts", str(texts)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "texts or prompts" in err
    assert "Traceback" not in err


def test_alignment_json_prompts_object_unwraps(tmp_path):
    texts = tmp_path / "texts.json"
    texts.write_text('{"prompts":["hello there friend"]}\n')
    ids, rows = load_alignment_texts(texts)
    assert ids == [None]
    assert rows == ["hello there friend"]


def test_save_rejects_non_string_observation_id(tmp_path):
    out = tmp_path / "obs.jsonl"
    out.write_text('{"observation_id":"stale"}\n')
    rows = [Observation(observation_id="obs-1", embedding=[1.0, 0.0])]
    rows[0].observation_id = 1
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_observations_jsonl(rows, out)
    assert out.read_text() == '{"observation_id":"stale"}\n'


def test_save_rejects_non_string_text(tmp_path):
    out = tmp_path / "obs.jsonl"
    rows = [Observation(observation_id="obs-1", embedding=[1.0, 0.0], text="ok")]
    rows[0].text = 123
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_observations_jsonl(rows, out)
    assert not out.exists()


def test_save_rejects_bool_embedding(tmp_path):
    out = tmp_path / "obs.jsonl"
    rows = [Observation(observation_id="obs-1", embedding=[1.0, 0.0])]
    rows[0].embedding = [True, False]
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_observations_jsonl(rows, out)
    assert not out.exists()


def test_npz_does_not_rescan_validated_matrix(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez(path, hidden_states=np.array([[1.0, 0.0]]))
    from neuralese import adapters as adapters_mod

    real = adapters_mod._require_2d_finite
    calls = []

    def counting(array, *, label):
        calls.append(label)
        return real(array, label=label)

    with patch.object(adapters_mod, "_require_2d_finite", side_effect=counting):
        rows = load_activations(path)
    assert len(calls) == 1
    assert rows[0].embedding == [1.0, 0.0]

import json
from pathlib import Path
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
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify
from neuralese.cli import main
from neuralese.contracts import Observation


def test_observations_from_activations_round_trip(tmp_path):
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.95, 0.0],
        ],
        dtype=np.float64,
    )
    rows = observations_from_activations(
        matrix,
        texts=["hello there friend", "hello there pal", "audit trail", "audit receipts"],
        layer=12,
        source="states.npy",
    )
    assert [row.observation_id for row in rows] == ["obs-1", "obs-2", "obs-3", "obs-4"]
    assert rows[0].metadata["layer"] == 12
    pack = learn_pack(rows, config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0))
    cert = certify(pack, observations=rows)
    assert cert.passed, cert.failures


def test_npy_ingest_cli(tmp_path, capsys):
    matrix = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float64)
    npy = tmp_path / "states.npy"
    np.save(npy, matrix)
    texts = tmp_path / "texts.jsonl"
    texts.write_text(
        '{"observation_id":"a","text":"hello there friend"}\n'
        '{"observation_id":"b","text":"hello there pal"}\n'
        '{"observation_id":"c","text":"audit the trail"}\n'
    )
    out = tmp_path / "obs.jsonl"
    rc = main(["ingest", str(npy), "-o", str(out), "--texts", str(texts), "--layer", "7"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_observations"] == 3
    assert payload["dim"] == 2
    loaded = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert loaded[0]["observation_id"] == "a"
    assert loaded[0]["metadata"]["layer"] == 7
    assert loaded[0]["text"] == "hello there friend"


def test_jsonl_hidden_state_ingest(tmp_path, capsys):
    src = tmp_path / "acts.jsonl"
    src.write_text(
        '{"observation_id":"h1","hidden_state":[1.0,0.0],"text":"hello"}\n'
        '{"observation_id":"h2","hidden_state":[0.0,1.0],"text":"audit"}\n'
    )
    out = tmp_path / "obs.jsonl"
    rc = main(["ingest", str(src), "-o", str(out)])
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert rows[0]["embedding"] == [1.0, 0.0]
    assert "hidden_state" not in rows[0]


def test_null_jsonl_observation_id_is_treated_as_missing(tmp_path):
    src = tmp_path / "acts.jsonl"
    src.write_text(
        '{"observation_id":null,"hidden_state":[1.0,0.0]}\n'
        '{"hidden_state":[0.0,1.0]}\n'
        '{"observation_id":null,"hidden_state":[0.5,0.5]}\n'
    )
    rows = load_activations(src)
    assert [row.observation_id for row in rows] == ["obs-1", "obs-2", "obs-3"]
    assert "None" not in [row.observation_id for row in rows]


def test_3d_activations_fail_closed(tmp_path, capsys):
    npy = tmp_path / "tokens.npy"
    np.save(npy, np.ones((2, 4, 8)))
    rc = main(["ingest", str(npy), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "2-D" in err
    assert "Traceback" not in err


def test_bool_activation_component_is_rejected():
    with pytest.raises(ValueError, match="real number"):
        observations_from_activations([[1.0, True]])


def test_text_count_mismatch_fails(tmp_path, capsys):
    npy = tmp_path / "states.npy"
    np.save(npy, np.array([[1.0, 0.0], [0.0, 1.0]]))
    texts = tmp_path / "texts.json"
    texts.write_text('["only one"]\n')
    rc = main(["ingest", str(npy), "-o", str(tmp_path / "obs.jsonl"), "--texts", str(texts)])
    assert rc == 1
    assert "length must match" in capsys.readouterr().err


def test_texts_flag_rejected_for_jsonl(tmp_path, capsys):
    src = tmp_path / "acts.jsonl"
    src.write_text('{"hidden_state":[1.0,0.0]}\n')
    texts = tmp_path / "texts.json"
    texts.write_text('["hello"]\n')
    rc = main(["ingest", str(src), "-o", str(tmp_path / "obs.jsonl"), "--texts", str(texts)])
    assert rc == 1
    assert "--texts is only valid" in capsys.readouterr().err


def test_npz_named_hidden_states(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez(path, hidden_states=np.array([[1.0, 0.0], [0.0, 1.0]]))
    matrix = load_activation_matrix(path)
    assert matrix.shape == (2, 2)
    rows = load_activations(path)
    assert len(rows) == 2


def test_duplicate_activation_ids_fail():
    with pytest.raises(ValueError, match="duplicate observation_id"):
        observations_from_activations(
            [[1.0, 0.0], [0.0, 1.0]],
            observation_ids=["same", "same"],
        )


def test_save_observations_jsonl_round_trip(tmp_path):
    rows = observations_from_activations([[1.0, 0.0]], texts=["hello"])
    path = tmp_path / "obs.jsonl"
    save_observations_jsonl(rows, path)
    reloaded = json.loads(path.read_text().splitlines()[0])
    assert reloaded["text"] == "hello"
    assert reloaded["embedding"] == [1.0, 0.0]


def test_string_texts_are_not_indexed_as_rows():
    with pytest.raises(TypeError, match="sequence of strings"):
        observations_from_activations([[1.0, 0.0], [0.0, 1.0]], texts="ab")


def test_string_observation_ids_are_not_indexed_as_rows():
    with pytest.raises(TypeError, match="sequence of ids"):
        observations_from_activations(
            [[1.0, 0.0], [0.0, 1.0]],
            observation_ids="xy",
        )


def test_complex_activations_fail_closed():
    matrix = np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="real numbers"):
        observations_from_activations(matrix)


def test_missing_activation_jsonl_is_clean_cli_failure(tmp_path, capsys):
    missing = tmp_path / "missing.jsonl"
    rc = main(["ingest", str(missing), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert str(missing) in err
    assert "Traceback" not in err


def test_missing_texts_file_is_clean_cli_failure(tmp_path, capsys):
    npy = tmp_path / "states.npy"
    np.save(npy, np.array([[1.0, 0.0], [0.0, 1.0]]))
    missing = tmp_path / "texts.jsonl"
    rc = main(["ingest", str(npy), "-o", str(tmp_path / "obs.jsonl"), "--texts", str(missing)])
    assert rc == 1
    err = capsys.readouterr().err
    assert str(missing) in err
    assert "Traceback" not in err


def test_ingest_helpers_are_exported_from_neuralese():
    import neuralese

    assert neuralese.load_activation_matrix is load_activation_matrix
    assert neuralese.load_activations is load_activations
    assert neuralese.load_alignment_texts is load_alignment_texts
    assert neuralese.observations_from_activations is observations_from_activations
    assert neuralese.save_observations_jsonl is save_observations_jsonl
    for name in (
        "load_activation_matrix",
        "load_activations",
        "load_alignment_texts",
        "observations_from_activations",
        "save_observations_jsonl",
    ):
        assert name in neuralese.__all__


def test_unicode_npy_activations_fail_closed(tmp_path):
    path = tmp_path / "unicode.npy"
    np.save(path, np.array([["1.0"]], dtype="<U8"))
    with pytest.raises(ValueError, match="real numbers"):
        load_activation_matrix(path)


def test_npz_named_as_npy_is_clean_cli_failure(tmp_path, capsys):
    archive = tmp_path / "states.npz"
    np.savez(archive, hidden_states=np.array([[1.0, 0.0]]))
    misnamed = tmp_path / "states.npy"
    misnamed.write_bytes(archive.read_bytes())
    rc = main(["ingest", str(misnamed), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert str(misnamed) in err
    assert "Traceback" not in err


def test_corrupt_npz_is_clean_cli_failure(tmp_path, capsys):
    path = tmp_path / "states.npz"
    path.write_bytes(b"not a zip archive")
    rc = main(["ingest", str(path), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert str(path) in err
    assert "Traceback" not in err


def test_corrupt_npy_zip_is_clean_cli_failure(tmp_path, capsys):
    path = tmp_path / "states.npy"
    path.write_bytes(b"PK\x03\x04truncated-zip")
    rc = main(["ingest", str(path), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert str(path) in err
    assert "Traceback" not in err


def test_numpy_scalar_activation_rows_are_accepted():
    rows = observations_from_activations(
        [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.int64)]
    )
    assert len(rows) == 2
    assert rows[0].embedding[0] == 1.0
    assert observations_from_activations([[np.float32(1.0), np.int64(0)]])[0].embedding == [
        1.0,
        0.0,
    ]


def test_overflowing_jsonl_activation_is_clean_cli_failure(tmp_path, capsys):
    path = tmp_path / "states.jsonl"
    path.write_text(
        json.dumps({"observation_id": "obs-1", "hidden_state": [10**1000, 0]}) + "\n",
        encoding="utf-8",
    )
    rc = main(["ingest", str(path), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "real number" in err or "hidden_state" in err


def test_non_string_jsonl_observation_id_fails_closed(tmp_path, capsys):
    src = tmp_path / "acts.jsonl"
    src.write_text('{"observation_id":1,"hidden_state":[1.0,0.0]}\n')
    rc = main(["ingest", str(src), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "string or null" in err


def test_bool_jsonl_observation_id_fails_closed(tmp_path):
    src = tmp_path / "acts.jsonl"
    src.write_text('{"observation_id":true,"hidden_state":[1.0,0.0]}\n')
    with pytest.raises(ValueError, match="string or null"):
        load_activations(src)


def test_numeric_library_observation_ids_fail_closed():
    with pytest.raises(ValueError, match="string or null"):
        observations_from_activations([[1.0, 0.0]], observation_ids=[1])
    with pytest.raises(ValueError, match="string or null"):
        observations_from_activations([[1.0, 0.0]], observation_ids=[np.int64(1)])


def test_numpy_integer_layer_is_stored_as_python_int():
    rows = observations_from_activations([[1.0, 0.0]], layer=np.int64(12))
    assert rows[0].metadata["layer"] == 12
    assert type(rows[0].metadata["layer"]) is int


def test_bool_layer_is_rejected():
    with pytest.raises(ValueError, match="layer must be an integer"):
        observations_from_activations([[1.0, 0.0]], layer=True)


def test_float_jsonl_layer_fails_closed(tmp_path):
    src = tmp_path / "acts.jsonl"
    src.write_text('{"hidden_state":[1.0,0.0],"layer":12.0}\n')
    with pytest.raises(ValueError, match="layer must be an integer"):
        load_activations(src)


def test_json_loaders_reject_recursive_activation_payloads(tmp_path, capsys):
    acts = tmp_path / "acts.jsonl"
    acts.write_text("{}\n")
    with patch("neuralese.adapters.json.loads", side_effect=RecursionError("nested")):
        with pytest.raises(ValueError, match="invalid JSON"):
            load_activations(acts)
    texts = tmp_path / "texts.json"
    texts.write_text("[]\n")
    with patch("neuralese.adapters.json.loads", side_effect=RecursionError("nested")):
        with pytest.raises(ValueError, match="invalid JSON"):
            load_alignment_texts(texts)
    deep = tmp_path / "deep.jsonl"
    deep.write_text('{"a":' * 10000 + "1" + "}" * 10000 + "\n")
    rc = main(["ingest", str(deep), "-o", str(tmp_path / "obs.jsonl")])
    assert rc == 1
    assert "Traceback" not in capsys.readouterr().err


def test_cyclic_metadata_is_not_json_serializable(tmp_path):
    metadata = {}
    metadata["self"] = metadata
    rows = [
        Observation(
            observation_id="obs-1",
            embedding=[1.0, 0.0],
            metadata=metadata,
        )
    ]
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_observations_jsonl(rows, tmp_path / "obs.jsonl")


def test_npz_texts_and_ids_are_aligned(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        hidden_states=np.array([[1.0, 0.0], [0.0, 1.0]]),
        texts=np.array(["hello there friend", "audit the trail"]),
        observation_ids=np.array(["a", "b"]),
    )
    rows = load_activations(path)
    assert [row.observation_id for row in rows] == ["a", "b"]
    assert [row.text for row in rows] == ["hello there friend", "audit the trail"]


def test_cli_texts_replace_npz_alignment(tmp_path, capsys):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        hidden_states=np.array([[1.0, 0.0], [0.0, 1.0]]),
        texts=np.array(["from-archive", "also-archive"]),
        observation_ids=np.array(["old-a", "old-b"]),
    )
    texts = tmp_path / "texts.jsonl"
    texts.write_text(
        '{"observation_id":"new-a","text":"hello there friend"}\n'
        '{"observation_id":"new-b","text":"audit the trail"}\n'
    )
    out = tmp_path / "obs.jsonl"
    rc = main(["ingest", str(path), "-o", str(out), "--texts", str(texts)])
    assert rc == 0
    loaded = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert [row["observation_id"] for row in loaded] == ["new-a", "new-b"]
    assert loaded[0]["text"] == "hello there friend"


def test_npz_integer_ids_fail_closed(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez(
        path,
        hidden_states=np.array([[1.0, 0.0], [0.0, 1.0]]),
        observation_ids=np.array([1, 2]),
    )
    with pytest.raises(ValueError, match="string or null"):
        load_activations(path)


def test_non_string_source_fails_closed():
    with pytest.raises(ValueError, match="source must be a string"):
        observations_from_activations([[1.0, 0.0]], source=12)


def test_jsonl_cli_layer_overwrites_record_layer(tmp_path, capsys):
    src = tmp_path / "acts.jsonl"
    src.write_text('{"observation_id":"h1","hidden_state":[1.0,0.0],"layer":3}\n')
    out = tmp_path / "obs.jsonl"
    rc = main(["ingest", str(src), "-o", str(out), "--layer", "9"])
    assert rc == 0
    loaded = json.loads(out.read_text().splitlines()[0])
    assert loaded["metadata"]["layer"] == 9


def test_null_alignment_id_is_generated(tmp_path):
    npy = tmp_path / "states.npy"
    np.save(npy, np.array([[1.0, 0.0], [0.0, 1.0]]))
    texts = tmp_path / "texts.jsonl"
    texts.write_text(
        '{"observation_id":null,"text":"hello there friend"}\n'
        '{"text":"audit the trail"}\n'
    )
    rows_ids, rows_texts = load_alignment_texts(texts)
    assert rows_ids == [None, None]
    assert rows_texts == ["hello there friend", "audit the trail"]
    out = tmp_path / "obs.jsonl"
    rc = main(["ingest", str(npy), "-o", str(out), "--texts", str(texts)])
    assert rc == 0
    loaded = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert [row["observation_id"] for row in loaded] == ["obs-1", "obs-2"]

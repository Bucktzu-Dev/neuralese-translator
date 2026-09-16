"""Observation and pack I/O. No event bus; JSONL and JSON only."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from neuralese.contracts import (
    Observation,
    SymbolPack,
    _copy_mapping,
    _require_json_object_keys,
)

PathLike = Union[str, Path]


def hashed_ngram_vector(text: str, dim: int = 32, n: int = 3) -> List[float]:
    """Deterministic hashed character n-gram embedding for text-only rows."""
    vec = np.zeros(dim, dtype=np.float64)
    blob = f"^{text.lower()}$"
    if len(blob) < n:
        blob = blob.ljust(n, "_")
    for i in range(len(blob) - n + 1):
        gram = blob[i : i + n].encode("utf-8")
        digest = hashlib.sha256(gram).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def ensure_embedding(obs: Observation, *, dim: int = 32) -> Observation:
    if obs.embedding:
        return obs
    if obs.text:
        obs.embedding = hashed_ngram_vector(obs.text, dim=dim)
        return obs
    raise ValueError(f"observation {obs.observation_id!r} has neither embedding nor text")


def load_observations_jsonl(path: PathLike) -> List[Observation]:
    rows: List[Observation] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, RecursionError) as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON") from exc
            if not isinstance(data, dict):
                raise ValueError(
                    f"{path}:{line_no} observation record must be a JSON object"
                )
            if "observation_id" not in data:
                data["observation_id"] = f"obs-{line_no}"
            elif not isinstance(data["observation_id"], str):
                raise ValueError(f"{path}:{line_no} invalid observation record")
            text = data.get("text")
            if text is not None and not isinstance(text, str):
                raise ValueError(f"{path}:{line_no} invalid observation record")
            embedding = data.get("embedding")
            if embedding is not None:
                if not isinstance(embedding, list):
                    raise ValueError(f"{path}:{line_no} invalid observation record")
                for item in embedding:
                    if isinstance(item, bool) or not isinstance(item, (int, float)):
                        raise ValueError(f"{path}:{line_no} invalid observation record")
                    try:
                        number = float(item)
                    except OverflowError as exc:
                        raise ValueError(
                            f"{path}:{line_no} invalid observation record"
                        ) from exc
                    if not math.isfinite(number):
                        raise ValueError(f"{path}:{line_no} invalid observation record")
            try:
                rows.append(Observation.from_dict(data))
            except (TypeError, ValueError, KeyError, OverflowError) as exc:
                raise ValueError(
                    f"{path}:{line_no} invalid observation record"
                ) from exc
    if not rows:
        raise ValueError(f"no observations in {path}")
    return rows


def load_stream(path: PathLike) -> List[int]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(
            "stream file must be a JSON list or an object with a codes array"
        ) from exc
    if isinstance(payload, dict) and "codes" in payload:
        codes = payload["codes"]
    elif isinstance(payload, list):
        codes = payload
    else:
        raise ValueError("stream file must be a JSON list or an object with a codes array")
    if not isinstance(codes, (list, tuple)):
        raise ValueError("stream codes must be an array")
    try:
        return [_as_stream_code(c) for c in codes]
    except (TypeError, ValueError) as exc:
        raise ValueError("stream codes must be integers") from exc


def _as_stream_code(value: object) -> int:
    # bool is a subclass of int; JSON true/false must not become 1/0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("stream codes must be integers")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("stream codes must be integers")
        return int(value)
    return int(value)


def _loads_json(raw: str, *, label: str) -> object:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{label} invalid JSON") from exc


def _is_real_number(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _as_layer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _as_optional_id(value: object, *, label: str, generated: str) -> str:
    if value is None:
        return generated
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string or null")
    if not value.strip():
        raise ValueError(f"{label} is blank")
    return value


def _as_source(value: object, *, label: str = "source") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} is blank")
    return value


def _activation_source_label(path: PathLike, source: Optional[str] = None) -> str:
    if source is None:
        return _as_source(Path(path).name)
    return _as_source(source)


def _detach_metadata(value: object, *, label: str) -> Dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object or null")
    try:
        copied = _copy_mapping(value)
    except (TypeError, RecursionError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not isinstance(copied, dict):
        raise ValueError(f"{label} must be an object or null")
    return copied


def _as_activation_vector(value: object, *, label: str) -> List[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{label} must be a numeric array")
    items = value.tolist() if isinstance(value, np.ndarray) else list(value)
    if not isinstance(items, list):
        raise ValueError(f"{label} must be a 1-D vector")
    if items and isinstance(items[0], list):
        raise ValueError(f"{label} must be a 1-D vector")
    out: List[float] = []
    for index, item in enumerate(items):
        if not _is_real_number(item):
            raise ValueError(f"{label}[{index}] is not a real number")
        try:
            number = float(item)
        except OverflowError as exc:
            raise ValueError(f"{label}[{index}] is not a real number") from exc
        if not np.isfinite(number):
            raise ValueError(f"{label}[{index}] is not finite")
        out.append(number)
    if not out:
        raise ValueError(f"{label} must be nonempty")
    return out


def _require_2d_finite(array: np.ndarray, *, label: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(
            f"{label} must be a 2-D array of shape (n_observations, hidden_dim); "
            "pool token/layer axes before ingest "
            "(see examples/activations/README.md)"
        )
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{label} must be nonempty")
    if (
        array.dtype == object
        or np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
        or not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        )
    ):
        raise ValueError(f"{label} must contain real numbers")
    try:
        matrix = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain real numbers") from exc
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} must contain finite real numbers")
    return matrix


def _array_from_npz(bundle: np.lib.npyio.NpzFile) -> np.ndarray:
    names = list(bundle.files)
    if not names:
        raise ValueError("npz archive contains no arrays")
    recognized = [
        key for key in ("hidden_states", "activations", "embeddings") if key in bundle
    ]
    if len(recognized) > 1:
        raise ValueError(
            "npz archive must contain only one of hidden_states, activations, embeddings"
        )
    if recognized:
        return np.asarray(bundle[recognized[0]])
    extras = {"texts", "observation_ids"}
    matrices = [name for name in names if name not in extras]
    if len(matrices) != 1:
        raise ValueError(
            "npz archive must contain hidden_states, activations, embeddings, "
            "or exactly one matrix array"
        )
    return np.asarray(bundle[matrices[0]])


def _as_optional_str_vector(array: np.ndarray, *, label: str) -> List[object]:
    if array.ndim != 1:
        raise ValueError(f"{label} must be a 1-D array")
    if array.shape[0] < 1:
        raise ValueError(f"{label} must be nonempty")
    out: List[object] = []
    for index, item in enumerate(array.tolist()):
        if item is None:
            out.append(None)
            continue
        if isinstance(item, (float, np.floating)) and np.isnan(item):
            out.append(None)
            continue
        if not isinstance(item, str):
            raise ValueError(f"{label}[{index}] must be a string or null")
        out.append(item)
    return out


def _optional_npz_alignment(path: PathLike):
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as bundle:
            texts_arr = np.asarray(bundle["texts"]) if "texts" in bundle else None
            ids_arr = (
                np.asarray(bundle["observation_ids"])
                if "observation_ids" in bundle
                else None
            )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid activation archive in {source}") from exc
    texts = (
        _as_optional_str_vector(texts_arr, label=f"{source} texts")
        if texts_arr is not None
        else None
    )
    ids = (
        _as_optional_str_vector(ids_arr, label=f"{source} observation_ids")
        if ids_arr is not None
        else None
    )
    return ids, texts


def load_activation_matrix(path: PathLike) -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        try:
            array = np.load(source, allow_pickle=False)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise ValueError(f"invalid activation matrix in {source}") from exc
        if not isinstance(array, np.ndarray):
            closer = getattr(array, "close", None)
            if callable(closer):
                closer()
            raise ValueError(f"invalid activation matrix in {source}")
    elif suffix == ".npz":
        try:
            loaded = np.load(source, allow_pickle=False)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise ValueError(f"invalid activation archive in {source}") from exc
        closer = getattr(loaded, "close", None)
        try:
            if not hasattr(loaded, "files"):
                raise ValueError(f"invalid activation archive in {source}")
            try:
                array = _array_from_npz(loaded)
            except (OSError, zipfile.BadZipFile) as exc:
                raise ValueError(f"invalid activation archive in {source}") from exc
        finally:
            if callable(closer):
                closer()
    else:
        raise ValueError("activation matrix must be a .npy or .npz file")
    return _require_2d_finite(array, label=str(source))


def observations_from_activations(
    hidden_states,
    *,
    texts=None,
    observation_ids=None,
    layer: Optional[int] = None,
    source: Optional[str] = None,
) -> List[Observation]:
    if isinstance(hidden_states, np.ndarray):
        matrix = _require_2d_finite(hidden_states, label="hidden_states")
        vectors = [
            _as_activation_vector(row, label=f"hidden_states[{index}]")
            for index, row in enumerate(matrix)
        ]
    else:
        if isinstance(hidden_states, (str, bytes)) or not isinstance(
            hidden_states, (list, tuple)
        ):
            raise ValueError("hidden_states must be a 2-D array")
        vectors = [
            _as_activation_vector(row, label=f"hidden_states[{index}]")
            for index, row in enumerate(hidden_states)
        ]
        if not vectors:
            raise ValueError("hidden_states must be nonempty")
        dim = len(vectors[0])
        if any(len(row) != dim for row in vectors):
            raise ValueError("hidden_states rows must share one hidden_dim")
    n_rows = len(vectors)
    if texts is not None:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not a string")
        if len(texts) != n_rows:
            raise ValueError("texts length must match hidden_states rows")
    if observation_ids is not None:
        if isinstance(observation_ids, (str, bytes)):
            raise TypeError("observation_ids must be a sequence of ids, not a string")
        if len(observation_ids) != n_rows:
            raise ValueError("observation_ids length must match hidden_states rows")
    parsed_layer = None if layer is None else _as_layer(layer, label="layer")
    parsed_source = None if source is None else _as_source(source)
    rows: List[Observation] = []
    seen: set[str] = set()
    for index in range(n_rows):
        raw_id = None if observation_ids is None else observation_ids[index]
        obs_id = _as_optional_id(
            raw_id,
            label=f"observation_ids[{index}]",
            generated=f"obs-{index + 1}",
        )
        if obs_id in seen:
            raise ValueError(f"duplicate observation_id {obs_id!r}")
        seen.add(obs_id)
        text = None if texts is None else texts[index]
        if text is not None and not isinstance(text, str):
            raise TypeError("texts must contain strings or null")
        metadata: Dict[str, object] = {}
        if parsed_layer is not None:
            metadata["layer"] = parsed_layer
        if parsed_source is not None:
            metadata["source"] = parsed_source
        rows.append(
            Observation(
                observation_id=obs_id,
                embedding=list(vectors[index]),
                text=text,
                metadata=metadata,
            )
        )
    return rows


def _text_row(data: object, *, path: PathLike, line_no: int):
    if isinstance(data, str):
        return None, data
    if not isinstance(data, dict):
        raise ValueError(f"{path}:{line_no} text record must be a JSON object or string")
    text = data.get("text")
    if text is not None and not isinstance(text, str):
        raise ValueError(f"{path}:{line_no} text must be a string or null")
    raw_id = data.get("observation_id")
    if raw_id is None:
        return None, text
    return (
        _as_optional_id(
            raw_id,
            label=f"{path}:{line_no} observation_id",
            generated="obs-unused",
        ),
        text,
    )


def _read_utf8(path: PathLike) -> str:
    return Path(path).read_text(encoding="utf-8-sig")


def load_alignment_texts(path: PathLike):
    source = Path(path)
    raw = _read_utf8(source).strip()
    if not raw:
        raise ValueError(f"no texts in {source}")
    ids: List[Optional[str]] = []
    texts: List[Optional[str]] = []
    if source.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(raw.splitlines(), start=1):
            blob = line.strip()
            if not blob or blob.startswith("#"):
                continue
            data = _loads_json(blob, label=f"{source}:{line_no}")
            oid, text = _text_row(data, path=source, line_no=line_no)
            ids.append(oid)
            texts.append(text)
    else:
        payload = _loads_json(raw, label=str(source))
        if isinstance(payload, dict) and "texts" in payload:
            payload = payload["texts"]
        if not isinstance(payload, list):
            raise ValueError(f"{source} texts must be a JSON array or JSONL")
        for index, row in enumerate(payload, start=1):
            oid, text = _text_row(row, path=source, line_no=index)
            ids.append(oid)
            texts.append(text)
    if not texts:
        raise ValueError(f"no texts in {source}")
    return ids, texts


def _activation_from_record(
    data: Dict[str, object], *, path: PathLike, line_no: int
) -> List[float]:
    if "hidden_state" in data and "embedding" in data:
        raise ValueError(f"{path}:{line_no} provide hidden_state or embedding, not both")
    if "hidden_state" in data:
        return _as_activation_vector(
            data["hidden_state"], label=f"{path}:{line_no} hidden_state"
        )
    if "embedding" in data:
        return _as_activation_vector(
            data["embedding"], label=f"{path}:{line_no} embedding"
        )
    if "hidden_states" in data:
        raise ValueError(
            f"{path}:{line_no} missing hidden_state or embedding "
            "(JSON rows use hidden_state; hidden_states is the npz matrix name)"
        )
    raise ValueError(f"{path}:{line_no} missing hidden_state or embedding")


def _observation_from_activation_payload(
    data: object,
    *,
    path: PathLike,
    line_no: int,
    seen: set[str],
    dim: Optional[int],
) -> tuple[Observation, int]:
    if isinstance(data, list):
        data = {"hidden_state": data}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}:{line_no} activation record must be a JSON object or array"
        )
    vector = _activation_from_record(data, path=path, line_no=line_no)
    if dim is None:
        dim = len(vector)
    elif len(vector) != dim:
        raise ValueError(
            f"{path}:{line_no} embedding dim mismatch: {len(vector)}, expected {dim}"
        )
    obs_id = _as_optional_id(
        data.get("observation_id"),
        label=f"{path}:{line_no} observation_id",
        generated=f"obs-{line_no}",
    )
    if obs_id in seen:
        raise ValueError(f"duplicate observation_id {obs_id!r}")
    seen.add(obs_id)
    text = data.get("text")
    if text is not None and not isinstance(text, str):
        raise ValueError(f"{path}:{line_no} text must be a string or null")
    metadata = _detach_metadata(
        data.get("metadata"), label=f"{path}:{line_no} metadata"
    )
    record_layer = data.get("layer")
    if record_layer is not None:
        metadata["layer"] = _as_layer(
            record_layer, label=f"{path}:{line_no} layer"
        )
    return (
        Observation(
            observation_id=obs_id,
            embedding=vector,
            text=text,
            metadata=metadata,
        ),
        dim,
    )


def _stamp_ingest_metadata(
    rows: List[Observation],
    *,
    layer: Optional[int] = None,
    source: Optional[str] = None,
) -> List[Observation]:
    parsed_layer = None if layer is None else _as_layer(layer, label="layer")
    parsed_source = None if source is None else _as_source(source)
    if parsed_layer is None and parsed_source is None:
        return rows
    for row in rows:
        metadata = dict(row.metadata)
        if parsed_layer is not None:
            metadata["layer"] = parsed_layer
        if parsed_source is not None:
            metadata["source"] = parsed_source
        row.metadata = metadata
    return rows


def load_activation_jsonl(
    path: PathLike,
    *,
    layer: Optional[int] = None,
    source: Optional[str] = None,
) -> List[Observation]:
    origin = Path(path)
    rows: List[Observation] = []
    seen: set[str] = set()
    dim: Optional[int] = None
    with origin.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            data = _loads_json(raw, label=f"{origin}:{line_no}")
            obs, dim = _observation_from_activation_payload(
                data, path=origin, line_no=line_no, seen=seen, dim=dim
            )
            rows.append(obs)
    if not rows:
        raise ValueError(f"no activations in {origin}")
    return _stamp_ingest_metadata(rows, layer=layer, source=source)


def _json_activation_conflict(payload: dict, *, origin: Path) -> None:
    collections = [name for name in ("activations", "rows") if name in payload]
    has_record = "hidden_state" in payload or "embedding" in payload
    has_matrix = "hidden_states" in payload
    if len(collections) > 1:
        raise ValueError(f"{origin} provide activations or rows, not both")
    if sum((bool(collections), has_record, has_matrix)) > 1:
        raise ValueError(
            f"{origin} provide one of activations, hidden_state, or hidden_states"
        )


def _as_json_row_list(payload: object, *, origin: Path) -> List[object]:
    if isinstance(payload, dict) and (
        "hidden_state" in payload or "embedding" in payload
    ):
        return [payload]
    if not isinstance(payload, list):
        raise ValueError(
            f"{origin} must be a JSON array, a hidden_state record, "
            "an object with activations, or a 2-D hidden_states matrix"
        )
    if payload and not isinstance(payload[0], (dict, list)):
        return [payload]
    return payload


def _normalize_json_activation_payload(
    payload: object, *, origin: Path
) -> tuple[str, object]:
    if isinstance(payload, list):
        return ("rows", _as_json_row_list(payload, origin=origin))
    if not isinstance(payload, dict):
        raise ValueError(
            f"{origin} must be a JSON array, a hidden_state record, "
            "an object with activations, or a 2-D hidden_states matrix"
        )
    _json_activation_conflict(payload, origin=origin)
    if "activations" in payload:
        return ("rows", _as_json_row_list(payload["activations"], origin=origin))
    if "rows" in payload:
        return ("rows", _as_json_row_list(payload["rows"], origin=origin))
    if "hidden_state" in payload or "embedding" in payload:
        return ("rows", [payload])
    if "hidden_states" in payload:
        return ("matrix", payload)
    raise ValueError(
        f"{origin} must be a JSON array, a hidden_state record, "
        "an object with activations, or a 2-D hidden_states matrix"
    )


def _observations_from_json_matrix(
    payload: dict,
    *,
    origin: Path,
    layer: Optional[int] = None,
    source: Optional[str] = None,
) -> List[Observation]:
    matrix = payload["hidden_states"]
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(f"no activations in {origin}")
    if not isinstance(matrix[0], list):
        raise ValueError(
            f"{origin} missing hidden_state or embedding "
            "(JSON rows use hidden_state; hidden_states is a 2-D matrix or the npz name)"
        )
    overlay_layer = layer if layer is not None else payload.get("layer")
    return observations_from_activations(
        matrix,
        texts=payload.get("texts"),
        observation_ids=payload.get("observation_ids"),
        layer=overlay_layer,
        source=source,
    )


def load_activation_json(
    path: PathLike,
    *,
    layer: Optional[int] = None,
    source: Optional[str] = None,
) -> List[Observation]:
    origin = Path(path)
    payload = _loads_json(_read_utf8(origin), label=str(origin))
    kind, body = _normalize_json_activation_payload(payload, origin=origin)
    if kind == "matrix":
        if not isinstance(body, dict):
            raise ValueError(
                f"{origin} must be a JSON array, a hidden_state record, "
                "an object with activations, or a 2-D hidden_states matrix"
            )
        return _observations_from_json_matrix(
            body, origin=origin, layer=layer, source=source
        )
    rows: List[Observation] = []
    seen: set[str] = set()
    dim: Optional[int] = None
    for index, item in enumerate(body, start=1):
        obs, dim = _observation_from_activation_payload(
            item, path=origin, line_no=index, seen=seen, dim=dim
        )
        rows.append(obs)
    if not rows:
        raise ValueError(f"no activations in {origin}")
    return _stamp_ingest_metadata(rows, layer=layer, source=source)


def load_activations(
    path: PathLike,
    *,
    layer: Optional[int] = None,
    source: Optional[str] = None,
) -> List[Observation]:
    origin = Path(path)
    suffix = origin.suffix.lower()
    source_label = _activation_source_label(origin, source)
    if suffix in {".npy", ".npz"}:
        ids = None
        texts = None
        if suffix == ".npz":
            ids, texts = _optional_npz_alignment(origin)
        return observations_from_activations(
            load_activation_matrix(origin),
            texts=texts,
            observation_ids=ids,
            layer=layer,
            source=source_label,
        )
    if suffix == ".jsonl":
        return load_activation_jsonl(origin, layer=layer, source=source_label)
    if suffix == ".json":
        return load_activation_json(origin, layer=layer, source=source_label)
    raise ValueError("activations must be a .npy, .npz, .jsonl, or .json file")


def save_observations_jsonl(observations: Sequence[Observation], path: PathLike) -> None:
    if not observations:
        raise ValueError("no observations to save")
    lines: List[str] = []
    for obs in observations:
        try:
            _require_json_object_keys(obs.metadata)
            payload = json.dumps(
                obs.to_dict(),
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError, OverflowError) as exc:
            raise ValueError(
                f"observation {obs.observation_id!r} is not JSON-serializable"
            ) from exc
        lines.append(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for payload in lines:
                handle.write(payload)
                handle.write("\n")
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_pack(pack: SymbolPack, path: PathLike) -> None:
    Path(path).write_text(
        json.dumps(pack.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_pack(path: PathLike) -> SymbolPack:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid pack in {path}")
        return SymbolPack.from_dict(data)
    except (
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        OverflowError,
        RecursionError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(f"invalid pack in {path}") from exc


def stack_embeddings(observations: Sequence[Observation]) -> np.ndarray:
    ensured = [ensure_embedding(o) for o in observations]
    dim = len(ensured[0].embedding)
    for obs in ensured:
        if len(obs.embedding) != dim:
            raise ValueError(
                f"embedding dim mismatch: {obs.observation_id} has {len(obs.embedding)}, expected {dim}"
            )
    return np.asarray([obs.embedding for obs in ensured], dtype=np.float64)


def iter_jsonl(path: PathLike) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                yield json.loads(raw)

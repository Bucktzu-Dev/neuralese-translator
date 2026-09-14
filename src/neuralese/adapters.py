"""Observation and pack I/O. No event bus; JSONL and JSON only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from neuralese.contracts import Observation, SymbolPack

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
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON") from exc
            if not isinstance(data, dict):
                raise ValueError(
                    f"{path}:{line_no} observation record must be a JSON object"
                )
            if "observation_id" not in data:
                data["observation_id"] = f"obs-{line_no}"
            text = data.get("text")
            if text is not None and not isinstance(text, str):
                raise ValueError(f"{path}:{line_no} invalid observation record")
            embedding = data.get("embedding")
            if embedding is not None and not isinstance(embedding, list):
                raise ValueError(f"{path}:{line_no} invalid observation record")
            try:
                rows.append(Observation.from_dict(data))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"{path}:{line_no} invalid observation record"
                ) from exc
    if not rows:
        raise ValueError(f"no observations in {path}")
    return rows


def load_stream(path: PathLike) -> List[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "codes" in payload:
        codes = payload["codes"]
    elif isinstance(payload, list):
        codes = payload
    else:
        raise ValueError("stream file must be a JSON list or {\"codes\": [...]}")
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


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_activation_vector(value: object, *, label: str) -> List[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{label} must be a numeric array")
    items = np.asarray(value, dtype=object).tolist()
    if not isinstance(items, list):
        raise ValueError(f"{label} must be a 1-D vector")
    if items and isinstance(items[0], list):
        raise ValueError(f"{label} must be a 1-D vector")
    out: List[float] = []
    for index, item in enumerate(items):
        if not _is_real_number(item):
            raise ValueError(f"{label}[{index}] is not a real number")
        number = float(item)
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
            "pool token/layer axes before ingest"
        )
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{label} must be nonempty")
    if array.dtype == object or np.issubdtype(array.dtype, np.bool_):
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
    for key in ("hidden_states", "activations", "embeddings"):
        if key in bundle:
            return np.asarray(bundle[key])
    if len(names) != 1:
        raise ValueError(
            "npz archive must contain hidden_states, activations, embeddings, "
            "or exactly one array"
        )
    return np.asarray(bundle[names[0]])


def load_activation_matrix(path: PathLike) -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        try:
            array = np.load(source, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid activation matrix in {source}") from exc
    elif suffix == ".npz":
        try:
            with np.load(source, allow_pickle=False) as bundle:
                array = _array_from_npz(bundle)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid activation archive in {source}") from exc
    else:
        raise ValueError("activation matrix must be a .npy or .npz file")
    return _require_2d_finite(array, label=str(source))


def observations_from_activations(
    hidden_states: Union[np.ndarray, Sequence[Sequence[float]]],
    *,
    texts: Optional[Sequence[Optional[str]]] = None,
    observation_ids: Optional[Sequence[object]] = None,
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
        if isinstance(hidden_states, (str, bytes)) or not isinstance(hidden_states, (list, tuple)):
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
    if texts is not None and len(texts) != n_rows:
        raise ValueError("texts length must match hidden_states rows")
    if observation_ids is not None and len(observation_ids) != n_rows:
        raise ValueError("observation_ids length must match hidden_states rows")
    if layer is not None and (isinstance(layer, bool) or not isinstance(layer, int)):
        raise ValueError("layer must be an integer")
    rows: List[Observation] = []
    seen: set[str] = set()
    for index in range(n_rows):
        raw_id = None if observation_ids is None else observation_ids[index]
        if raw_id is None:
            obs_id = f"obs-{index + 1}"
        else:
            obs_id = str(raw_id).strip()
            if not obs_id:
                raise ValueError(f"observation_ids[{index}] is blank")
        if obs_id in seen:
            raise ValueError(f"duplicate observation_id {obs_id!r}")
        seen.add(obs_id)
        text = None if texts is None else texts[index]
        if text is not None and not isinstance(text, str):
            raise TypeError("texts must contain strings or null")
        metadata: Dict[str, object] = {}
        if layer is not None:
            metadata["layer"] = layer
        if source:
            metadata["source"] = source
        rows.append(
            Observation(
                observation_id=obs_id,
                embedding=list(vectors[index]),
                text=text,
                metadata=metadata,
            )
        )
    return rows


def _text_row(data: object, *, path: PathLike, line_no: int) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(data, str):
        return None, data
    if not isinstance(data, dict):
        raise ValueError(f"{path}:{line_no} text record must be a JSON object or string")
    text = data.get("text")
    if text is not None and not isinstance(text, str):
        raise ValueError(f"{path}:{line_no} text must be a string or null")
    obs_id = data.get("observation_id")
    if obs_id is None:
        return None, text
    oid = str(obs_id).strip()
    if not oid:
        raise ValueError(f"{path}:{line_no} observation_id is blank")
    return oid, text


def load_alignment_texts(
    path: PathLike,
) -> Tuple[List[Optional[str]], List[Optional[str]]:
    source = Path(path)
    raw = source.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"no texts in {source}")
    ids: List[Optional[str]] = []
    texts: List[Optional[str]] = []
    if source.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(raw.splitlines(), start=1):
            blob = line.strip()
            if not blob or blob.startswith("#"):
                continue
            try:
                data = json.loads(blob)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no} invalid JSON") from exc
            oid, text = _text_row(data, path=source, line_no=line_no)
            ids.append(oid)
            texts.append(text)
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} invalid JSON") from exc
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


def _activation_from_record(data: Dict[str, object], *, path: PathLike, line_no: int) -> List[float]:
    if "hidden_state" in data and "embedding" in data:
        raise ValueError(f"{path}:{line_no} provide hidden_state or embedding, not both")
    if "hidden_state" in data:
        return _as_activation_vector(data["hidden_state"], label=f"{path}:{line_no} hidden_state")
    if "embedding" in data:
        return _as_activation_vector(data["embedding"], label=f"{path}:{line_no} embedding")
    raise ValueError(f"{path}:{line_no} missing hidden_state or embedding")


def load_activation_jsonl(path: PathLike) -> List[Observation]:
    source = Path(path)
    rows: List[Observation] = []
    seen: set[str] = set()
    dim: Optional[int] = None
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no} invalid JSON") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{source}:{line_no} activation record must be a JSON object")
            vector = _activation_from_record(data, path=source, line_no=line_no)
            if dim is None:
                dim = len(vector)
            elif len(vector) != dim:
                raise ValueError(
                    f"{source}:{line_no} embedding dim mismatch: {len(vector)}, expected {dim}"
                )
            if "observation_id" in data:
                obs_id = str(data["observation_id"]).strip()
                if not obs_id:
                    raise ValueError(f"{source}:{line_no} observation_id is blank")
            else:
                obs_id = f"obs-{line_no}"
            if obs_id in seen:
                raise ValueError(f"duplicate observation_id {obs_id!r}")
            seen.add(obs_id)
            text = data.get("text")
            if text is not None and not isinstance(text, str):
                raise ValueError(f"{source}:{line_no} text must be a string or null")
            metadata = data.get("metadata")
            if metadata is None:
                metadata = {}
            elif not isinstance(metadata, dict):
                raise ValueError(f"{source}:{line_no} metadata must be an object or null")
            layer = data.get("layer")
            if layer is not None:
                if isinstance(layer, bool) or not isinstance(layer, int):
                    raise ValueError(f"{source}:{line_no} layer must be an integer")
                metadata = dict(metadata)
                metadata["layer"] = layer
            rows.append(
                Observation(
                    observation_id=obs_id,
                    embedding=vector,
                    text=text,
                    metadata=dict(metadata),
                )
            )
    if not rows:
        raise ValueError(f"no activations in {source}")
    return rows


def load_activations(path: PathLike) -> List[Observation]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return observations_from_activations(
            load_activation_matrix(source),
            source=str(source),
        )
    if suffix == ".jsonl":
        return load_activation_jsonl(source)
    raise ValueError("activations must be a .npy, .npz, or .jsonl file")


def save_observations_jsonl(observations: Sequence[Observation], path: PathLike) -> None:
    target = Path(path)
    lines = [json.dumps(obs.to_dict(), sort_keys=True, ensure_ascii=True) for obs in observations]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_pack(pack: SymbolPack, path: PathLike) -> None:
    Path(path).write_text(
        json.dumps(pack.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_pack(path: PathLike) -> SymbolPack:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid pack in {path}")
    try:
        return SymbolPack.from_dict(data)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
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

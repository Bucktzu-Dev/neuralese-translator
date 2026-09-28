"""Load hidden-state dumps into observations.

This module reads arrays that a model exporter already wrote. It does not
download weights, run a HuggingFace model, or invent English from activations.
Text on a row is stored as given; glosses still come from ``learn_pack``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from neuralese.contracts import Observation

PathLike = Union[str, Path]

LAYOUTS = ("vectors", "tokens", "layers", "hf_layers", "hf_stack")
POOLS = ("last", "mean")


def reduce_hidden_states(
    hidden: np.ndarray,
    *,
    layout: str,
    layer: int = -1,
    pool: str = "last",
) -> np.ndarray:
    """Collapse a hidden-state array to one vector per observation, shape (n, d)."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}")
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {', '.join(POOLS)}")
    array = np.asarray(hidden, dtype=np.float64)
    if layout == "vectors":
        if array.ndim != 2:
            raise ValueError(f"vectors layout expects (n, d), got shape {array.shape}")
        vectors = array
    elif layout == "tokens":
        if array.ndim != 3:
            raise ValueError(f"tokens layout expects (n, seq, d), got shape {array.shape}")
        vectors = _pool(array, axis=1, pool=pool)
    elif layout == "layers":
        if array.ndim != 3:
            raise ValueError(f"layers layout expects (n, layers, d), got shape {array.shape}")
        vectors = _take(array, axis=1, index=layer)
    elif layout == "hf_layers":
        if array.ndim != 3:
            raise ValueError(f"hf_layers layout expects (layers, n, d), got shape {array.shape}")
        vectors = _take(array, axis=0, index=layer)
    else:
        if array.ndim != 4:
            raise ValueError(
                f"hf_stack layout expects (layers, n, seq, d), got shape {array.shape}"
            )
        chosen = _take(array, axis=0, index=layer)
        vectors = _pool(chosen, axis=1, pool=pool)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError(f"activation dump produced an empty matrix from shape {array.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("hidden states contain non-finite values")
    return vectors


def observations_from_hidden_states(
    hidden: object,
    texts: Optional[Sequence[Optional[str]]] = None,
    observation_ids: Optional[Sequence[str]] = None,
    *,
    layout: Optional[str] = None,
    layer: int = -1,
    pool: str = "last",
    id_prefix: str = "act",
) -> List[Observation]:
    """Build observations from an array or a HuggingFace ``hidden_states`` tuple.

    A tuple or list of arrays, one per layer, is stacked. Each layer may be
    ``(n, d)`` (layout ``hf_layers``) or ``(n, seq, d)`` (layout ``hf_stack``).
    That matches ``numpy.stack(outputs.hidden_states)`` after the tensors are
    converted to numpy. A plain ``(n, d)`` matrix uses layout ``vectors``.
    """
    array, resolved = _coerce_hidden(hidden, layout)
    if resolved is None:
        if array.ndim == 2:
            resolved = "vectors"
        elif array.ndim == 4:
            resolved = "hf_stack"
        else:
            raise ValueError(
                f"shape {array.shape} needs an explicit layout "
                "(tokens, layers, or hf_layers)"
            )
    vectors = reduce_hidden_states(array, layout=resolved, layer=layer, pool=pool)
    ids = _align_ids(observation_ids, count=vectors.shape[0], prefix=id_prefix)
    glosses = _align_texts(texts, count=vectors.shape[0])
    rows: List[Observation] = []
    seen = set()
    for index, (obs_id, text) in enumerate(zip(ids, glosses)):
        if obs_id in seen:
            raise ValueError(f"duplicate observation_id {obs_id!r}")
        seen.add(obs_id)
        rows.append(
            Observation(
                observation_id=obs_id,
                embedding=vectors[index].tolist(),
                text=text,
            )
        )
    return rows


def load_activation_dump(
    path: PathLike,
    *,
    layout: Optional[str] = None,
    layer: int = -1,
    pool: str = "last",
) -> List[Observation]:
    """Load ``.npy``, ``.npz``, or ``.jsonl`` hidden states."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        array = np.load(source)
        return observations_from_hidden_states(array, layout=layout, layer=layer, pool=pool)
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as bundle:
            array = _npz_array(bundle)
            file_layout = layout or _optional_layout(bundle)
            return observations_from_hidden_states(
                array,
                texts=_optional_strings(bundle, ("texts", "text")),
                observation_ids=_optional_strings(bundle, ("observation_ids", "ids")),
                layout=file_layout,
                layer=layer,
                pool=pool,
            )
    if suffix == ".jsonl":
        return _load_jsonl_dump(source, layout=layout, layer=layer, pool=pool)
    raise ValueError(f"unsupported activation dump {source.name}; use .npy, .npz, or .jsonl")


def _coerce_hidden(hidden: object, layout: Optional[str]) -> Tuple[np.ndarray, Optional[str]]:
    if isinstance(hidden, (list, tuple)) and hidden and _item_is_matrix(hidden[0]):
        stacked = np.stack([np.asarray(item, dtype=np.float64) for item in hidden])
        first = np.asarray(hidden[0])
        if first.ndim == 2:
            return stacked, layout or "hf_layers"
        if first.ndim == 3:
            return stacked, layout or "hf_stack"
        raise ValueError(f"cannot stack hidden states with item shape {first.shape}")
    array = np.asarray(hidden, dtype=np.float64)
    return array, layout


def _item_is_matrix(item: object) -> bool:
    # Only already-built arrays stack as layers. A nested list is one tensor.
    return isinstance(item, np.ndarray) and item.ndim >= 2


def _pool(array: np.ndarray, *, axis: int, pool: str) -> np.ndarray:
    if array.shape[axis] == 0:
        raise ValueError("hidden-state sequence axis is empty")
    if pool == "last":
        return np.take(array, -1, axis=axis)
    return array.mean(axis=axis)


def _take(array: np.ndarray, *, axis: int, index: int) -> np.ndarray:
    size = int(array.shape[axis])
    resolved = index if index >= 0 else size + index
    if resolved < 0 or resolved >= size:
        raise ValueError(f"layer {index} is outside 0..{size - 1}")
    return np.take(array, resolved, axis=axis)


def _align_ids(observation_ids: Optional[Sequence[str]], *, count: int, prefix: str) -> List[str]:
    if observation_ids is None:
        return [f"{prefix}-{index}" for index in range(1, count + 1)]
    ids = [str(item) for item in observation_ids]
    if len(ids) != count:
        raise ValueError(f"observation_ids length {len(ids)} != {count} hidden states")
    return ids


def _align_texts(
    texts: Optional[Sequence[Optional[str]]],
    *,
    count: int,
) -> List[Optional[str]]:
    if texts is None:
        return [None] * count
    rows = [None if item is None or str(item) == "" else str(item) for item in texts]
    if len(rows) != count:
        raise ValueError(f"texts length {len(rows)} != {count} hidden states")
    return rows


def _npz_array(bundle: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("hidden_states", "activations", "embeddings"):
        if key in bundle:
            return np.asarray(bundle[key], dtype=np.float64)
    raise ValueError("npz dump needs hidden_states, activations, or embeddings")


def _optional_layout(bundle: np.lib.npyio.NpzFile) -> Optional[str]:
    if "layout" not in bundle:
        return None
    value = np.asarray(bundle["layout"])
    if value.shape != () and value.size != 1:
        raise ValueError("layout must be a single string")
    layout = str(value.reshape(-1)[0])
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}")
    return layout


def _optional_strings(bundle: np.lib.npyio.NpzFile, keys: Sequence[str]) -> Optional[List[str]]:
    for key in keys:
        if key in bundle:
            return [str(item) for item in np.asarray(bundle[key]).reshape(-1).tolist()]
    return None


def _load_jsonl_dump(
    path: Path,
    *,
    layout: Optional[str],
    layer: int,
    pool: str,
) -> List[Observation]:
    rows: List[Observation] = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON") from exc
            vector = _jsonl_vector(data, layout=layout, layer=layer, pool=pool, line_no=line_no)
            obs_id = str(data.get("observation_id") or data.get("id") or f"act-{line_no}")
            if obs_id in seen:
                raise ValueError(f"duplicate observation_id {obs_id!r}")
            seen.add(obs_id)
            text = data.get("text")
            rows.append(
                Observation(
                    observation_id=obs_id,
                    embedding=vector,
                    text=None if text is None or str(text) == "" else str(text),
                    metadata=dict(data.get("metadata") or {}),
                )
            )
    if not rows:
        raise ValueError(f"no observations in {path}")
    width = len(rows[0].embedding)
    for obs in rows:
        if len(obs.embedding) != width:
            raise ValueError(
                f"embedding dim mismatch: {obs.observation_id} has {len(obs.embedding)}, expected {width}"
            )
    return rows


def _jsonl_vector(
    data: dict,
    *,
    layout: Optional[str],
    layer: int,
    pool: str,
    line_no: int,
) -> List[float]:
    payload = None
    for key in ("hidden_states", "hidden_state", "activations", "embedding"):
        if key in data and data[key] is not None:
            payload = data[key]
            break
    if payload is None:
        raise ValueError(f"line {line_no} has no hidden_states or embedding")
    array = np.asarray(payload, dtype=np.float64)
    if array.ndim == 1:
        if layout not in (None, "vectors"):
            raise ValueError(f"line {line_no} is a vector but layout is {layout}")
        if not np.isfinite(array).all():
            raise ValueError(f"line {line_no} contains non-finite values")
        return array.tolist()
    if array.ndim != 2:
        raise ValueError(
            f"line {line_no} hidden state ndim {array.ndim} is unsupported; "
            "use an npz for hf_stack"
        )
    if layout == "tokens":
        vector = _pool(array, axis=0, pool=pool)
    elif layout == "layers":
        vector = _take(array, axis=0, index=layer)
    else:
        raise ValueError(f"line {line_no} is 2D and needs --layout tokens or layers")
    if not np.isfinite(vector).all():
        raise ValueError(f"line {line_no} contains non-finite values")
    return vector.astype(np.float64).tolist()

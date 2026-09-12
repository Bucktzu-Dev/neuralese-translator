"""Observation and pack I/O. No event bus; JSONL and JSON only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Union

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
    return [int(c) for c in codes]


def save_pack(pack: SymbolPack, path: PathLike) -> None:
    Path(path).write_text(
        json.dumps(pack.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_pack(path: PathLike) -> SymbolPack:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SymbolPack.from_dict(data)


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

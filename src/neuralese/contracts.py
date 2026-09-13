"""Public data contracts for the Neuralese to English Translator."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterable, List, Optional, Tuple

GLOSS_STATES = ("ok", "aliased", "quarantined", "unknown")
DECODER_VERSION = "0.1.1"
CERT_POLICIES = ("default", "strict", "integrity")
TRANSLATION_POLICIES = ("default", "strict")
DECISIONS = ("accept", "accept_provisional", "reject")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}\Z")
LEGACY_ALIAS_KEY = "legacy"
AliasTables = Dict[str, Dict[int, int]]


def canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_dumps(obj).encode("utf-8")).hexdigest()


def round_vec(values: Iterable[float]) -> List[float]:
    return [round(float(x), 8) for x in values]


def _int_keyed(d: Dict[Any, Any]) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_object(data: Dict[str, Any], key: str, label: str) -> Dict[str, Any]:
    if key not in data or data[key] is None:
        return {}
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object or null")
    return dict(value)


def _string_id_list(data: Dict[str, Any], key: str) -> List[str]:
    if key not in data or data[key] is None:
        return []
    raw = data[key]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise TypeError(f"{key} must be an array")
    ids: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise TypeError(f"{key} must contain strings")
        ids.append(item)
    return ids


def _json_bool(data: Dict[str, Any], key: str, *, default: Any = None, required: bool = False) -> Any:
    if key not in data:
        if required:
            raise KeyError(key)
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _present_str(data: Dict[str, Any], key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    return "" if value is None else str(value)


def _load_decision(data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    if "decision" in data:
        return _present_str(data, "decision", "")
    if "decision" in metadata:
        value = metadata.get("decision")
        return "" if value is None else str(value)
    return "accept"


def normalize_aliases(raw: Any) -> AliasTables:
    """Accept legacy {code: code} or versioned {source_checksum: {old: new}}."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("aliases must be a mapping")
    if not raw:
        return {}
    values = list(raw.values())
    nested = [isinstance(value, dict) for value in values]
    if any(nested):
        if not all(nested):
            raise ValueError("aliases must be uniformly nested mappings")
        tables: AliasTables = {}
        for src, mapping in raw.items():
            key = str(src)
            if not key:
                raise ValueError("alias source keys must be non-empty")
            tables[key] = _int_keyed(mapping)
        return tables
    return {LEGACY_ALIAS_KEY: _int_keyed(raw)}


def select_alias_table(
    aliases: AliasTables,
    source_pack_checksum: Optional[str] = None,
    *,
    parent_checksum: Optional[str] = None,
) -> Dict[int, int]:
    """Pick one alias table. The reserved legacy key is never an explicit source."""
    if source_pack_checksum is not None:
        if not source_pack_checksum or source_pack_checksum == LEGACY_ALIAS_KEY:
            return {}
        if source_pack_checksum in aliases:
            return dict(aliases[source_pack_checksum])
        if parent_checksum is not None and source_pack_checksum == parent_checksum:
            return dict(aliases.get(parent_checksum, {}))
        return {}
    if LEGACY_ALIAS_KEY in aliases:
        return dict(aliases[LEGACY_ALIAS_KEY])
    return {}


def aliases_to_dict(aliases: AliasTables) -> Dict[str, Dict[str, int]]:
    return {
        str(src): {str(k): int(v) for k, v in sorted(mapping.items())}
        for src, mapping in sorted(aliases.items())
    }


def observation_content_hash(
    observation_id: str,
    embedding: Optional[Iterable[float]] = None,
    text: Optional[str] = None,
) -> str:
    # NumPy arrays are truthy-ambiguous; never use `embedding or []`.
    values = [] if embedding is None else embedding
    return sha256_hex(
        {
            "observation_id": str(observation_id),
            "embedding": round_vec(values),
            "text": text,
        }
    )


def example_hash(text: str) -> str:
    """Unsalted SHA-256 of canonical JSON ``{\"example\": text}``, not of UTF-8 bytes alone."""
    return sha256_hex({"example": text})


@dataclass
class Observation:
    observation_id: str
    embedding: List[float] = field(default_factory=list)
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observation_id = str(self.observation_id)
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("text must be a string or null")
        if self.embedding is None:
            self.embedding = []
        elif isinstance(self.embedding, (str, bytes)):
            raise TypeError("embedding must be an array or null")
        else:
            self.embedding = [float(x) for x in self.embedding]

    def content_hash(self) -> str:
        values: Optional[Iterable[float]] = self.embedding
        if not self.embedding and self.text:
            from neuralese.adapters import hashed_ngram_vector

            values = hashed_ngram_vector(self.text)
        return observation_content_hash(self.observation_id, values, self.text)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        raw_embedding = data.get("embedding")
        if raw_embedding is None:
            raw_embedding = []
        elif not isinstance(raw_embedding, (list, tuple)):
            raise TypeError("embedding must be an array or null")
        text = data.get("text")
        if text is not None and not isinstance(text, str):
            raise TypeError("text must be a string or null")
        return cls(
            observation_id=str(data["observation_id"]),
            embedding=[float(x) for x in raw_embedding],
            text=text,
            metadata=dict(data.get("metadata") or {}),
        )

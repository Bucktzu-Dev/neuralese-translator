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
AliasTables = Dict[str, Dict[int, int]]


def canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_dumps(obj).encode("utf-8")).hexdigest()


def round_vec(values: Iterable[float]) -> List[float]:
    return [round(float(x), 8) for x in values]


def _int_keyed(d: Dict[Any, Any]) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def normalize_aliases(raw: Any) -> AliasTables:
    """Accept legacy {code: code} or versioned {source_checksum: {old: new}}."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("aliases must be a mapping")
    values = list(raw.values())
    if values and isinstance(values[0], dict):
        return {str(src): _int_keyed(mapping) for src, mapping in raw.items()}
    return {"legacy": _int_keyed(raw)}


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
    return sha256_hex(
        {
            "observation_id": str(observation_id),
            "embedding": round_vec(embedding or []),
            "text": text,
        }
    )


def example_hash(text: str) -> str:
    return sha256_hex({"example": text})


@dataclass
class Symbol:
    class_id: int
    code: int
    proto_embedding: List[float]
    observation_ids: List[str]
    definition: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    confidence: float = 0.0
    quarantined: bool = False
    survival: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    example_hashes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.examples and not self.example_hashes:
            self.example_hashes = [example_hash(x) for x in self.examples]

    def to_dict(self, *, include_private: bool = False) -> Dict[str, Any]:
        payload = {
            "class_id": self.class_id,
            "code": self.code,
            "proto_embedding": list(self.proto_embedding),
            "observation_ids": list(self.observation_ids),
            "definition": self.definition,
            "example_hashes": list(self.example_hashes),
            "confidence": self.confidence,
            "quarantined": self.quarantined,
            "survival": self.survival,
            "metadata": dict(self.metadata),
        }
        if include_private:
            payload["examples"] = list(self.examples)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Symbol":
        examples = [str(x) for x in data.get("examples") or []]
        hashes = [str(x) for x in data.get("example_hashes") or []]
        if examples and not hashes:
            hashes = [example_hash(x) for x in examples]
        return cls(
            class_id=int(data["class_id"]),
            code=int(data["code"]),
            proto_embedding=[float(x) for x in data.get("proto_embedding") or []],
            observation_ids=[str(x) for x in data.get("observation_ids") or []],
            definition=data.get("definition"),
            examples=examples,
            example_hashes=hashes,
            confidence=float(data.get("confidence") or 0.0),
            quarantined=bool(data.get("quarantined") or False),
            survival=float(data.get("survival") if data.get("survival") is not None else 1.0),
            metadata=dict(data.get("metadata") or {}),
        )

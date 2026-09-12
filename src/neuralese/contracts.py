from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from neuralese.audit import SCHEMA_VERSION, check_pack

Decision = Literal["certified", "needs_review", "rejected"]
TranslationPolicy = Literal["default", "strict"]
CertificationPolicy = Literal["default", "strict", "integrity"]


def _mapping(value: Any) -> dict[str, Any]:
    if value is None or not isinstance(value, Mapping):
        return {}
    return dict(value)


def _present_str(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    value = payload[key]
    if value is None:
        return ""
    return str(value)


def _load_decision(payload: Mapping[str, Any]) -> Decision | str:
    if "decision" not in payload:
        return "needs_review"
    value = payload["decision"]
    if value is None:
        return ""
    return str(value)


@dataclass(frozen=True)
class Observation:
    observation_id: str
    text: str
    tokens: tuple[str, ...] = ()
    embeddings: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", str(self.observation_id))
        if not str(self.observation_id).strip():
            raise ValueError("observation_id must be a non-empty string")
        if not str(self.text).strip():
            raise ValueError("observation text must be a non-empty string")
        if self.embeddings is not None and len(self.embeddings) == 0:
            raise ValueError("embeddings must be omitted or non-empty")


@dataclass(frozen=True)
class Atom:
    atom_id: str
    form: str
    definition: str
    aliases: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    source_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationPack:
    pack_id: str
    schema_version: str = SCHEMA_VERSION
    decoder_version: str = ""
    atoms: tuple[Atom, ...] = ()
    observations: tuple[Observation, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)
    parent_checksum: str = ""
    checksum: str = ""
    decision: Decision | str = "needs_review"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "schema_version": self.schema_version,
            "decoder_version": self.decoder_version,
            "atoms": [
                {
                    "atom_id": atom.atom_id,
                    "form": atom.form,
                    "definition": atom.definition,
                    "aliases": list(atom.aliases),
                    "evidence": list(atom.evidence),
                    "keywords": list(atom.keywords),
                    "examples": list(atom.examples),
                    "source_count": atom.source_count,
                    "metadata": atom.metadata,
                }
                for atom in self.atoms
            ],
            "observations": [
                {
                    "observation_id": observation.observation_id,
                    "text": observation.text,
                    "tokens": list(observation.tokens),
                    "embeddings": list(observation.embeddings) if observation.embeddings is not None else None,
                    "metadata": observation.metadata,
                }
                for observation in self.observations
            ],
            "aliases": dict(self.aliases),
            "parent_checksum": self.parent_checksum,
            "checksum": self.checksum,
            "decision": self.decision,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TranslationPack":
        atoms = tuple(
            Atom(
                atom_id=str(item.get("atom_id", "")),
                form=str(item.get("form", "")),
                definition=str(item.get("definition", "")),
                aliases=tuple(item.get("aliases", ()) or ()),
                evidence=tuple(item.get("evidence", ()) or ()),
                keywords=tuple(item.get("keywords", ()) or ()),
                examples=tuple(item.get("examples", ()) or ()),
                source_count=int(item.get("source_count", 0) or 0),
                metadata=_mapping(item.get("metadata")),
            )
            for item in payload.get("atoms", ()) or ()
        )
        observations = tuple(
            Observation(
                observation_id=str(item.get("observation_id", "")),
                text=str(item.get("text", "")),
                tokens=tuple(item.get("tokens", ()) or ()),
                embeddings=(
                    tuple(item.get("embeddings"))
                    if item.get("embeddings") is not None
                    else None
                ),
                metadata=_mapping(item.get("metadata")),
            )
            for item in payload.get("observations", ()) or ()
        )
        return cls(
            pack_id=str(payload.get("pack_id", "")),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            decoder_version=_present_str(payload, "decoder_version"),
            atoms=atoms,
            observations=observations,
            aliases=dict(payload.get("aliases", {}) or {}),
            parent_checksum=str(payload.get("parent_checksum", "") or ""),
            checksum=str(payload.get("checksum", "") or ""),
            decision=_load_decision(payload),
            metadata=_mapping(payload.get("metadata")),
        )

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def loads(cls, raw: str) -> "TranslationPack":
        return cls.from_dict(json.loads(raw))

    def audit(self, policy: CertificationPolicy = "default") -> tuple[bool, tuple[str, ...]]:
        return check_pack(self, policy=policy)

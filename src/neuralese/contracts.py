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


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


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
    return sha256_hex({"example": text})


@dataclass
class Observation:
    observation_id: str
    embedding: List[float] = field(default_factory=list)
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observation_id = str(self.observation_id)
        if self.embedding is None:
            self.embedding = []
        else:
            self.embedding = [float(x) for x in self.embedding]

    def content_hash(self) -> str:
        return observation_content_hash(self.observation_id, self.embedding, self.text)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        raw_embedding = data.get("embedding")
        if raw_embedding is None:
            raw_embedding = []
        return cls(
            observation_id=str(data["observation_id"]),
            embedding=[float(x) for x in raw_embedding],
            text=data.get("text"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Receipt:
    step: str
    ok: bool
    timestamp: float
    kappa: Optional[float] = None
    reconstruction_error: Optional[float] = None
    delta_mdl_bits: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Receipt":
        return cls(
            step=str(data["step"]),
            ok=bool(data["ok"]),
            timestamp=float(data["timestamp"]),
            kappa=_opt_float(data.get("kappa")),
            reconstruction_error=_opt_float(data.get("reconstruction_error")),
            delta_mdl_bits=_opt_float(data.get("delta_mdl_bits")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class GuardSnapshot:
    kappa_avg: float
    reconstruction_error: float
    delta_mdl: float
    min_survival: float
    pass_kappa: bool
    pass_residual: bool
    pass_mdl: bool
    pass_persist: bool
    pass_compat: bool
    pass_all: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuardSnapshot":
        return cls(**{f.name: data[f.name] for f in fields(cls)})


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


@dataclass
class SymbolPack:
    pack_id: str
    symbols: List[Symbol]
    codebook: Dict[int, int]
    aliases: AliasTables
    reconstruction_error: float
    checksum: str = ""
    parent_pack_id: Optional[str] = None
    receipts: List[Receipt] = field(default_factory=list)
    guards: Optional[GuardSnapshot] = None
    mdl_bits: float = 0.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_checksum: Optional[str] = None
    evidence: Dict[str, str] = field(default_factory=dict)
    decoder_version: str = DECODER_VERSION
    decision: str = "accept"
    include_private: bool = False

    def __post_init__(self) -> None:
        self.aliases = normalize_aliases(self.aliases)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "decoder_version": self.decoder_version,
            "decision": self.decision,
            "symbols": [s.to_dict(include_private=self.include_private) for s in self.symbols],
            "codebook": {str(k): int(v) for k, v in self.codebook.items()},
            "aliases": aliases_to_dict(self.aliases),
            "reconstruction_error": self.reconstruction_error,
            "checksum": self.checksum,
            "parent_pack_id": self.parent_pack_id,
            "parent_checksum": self.parent_checksum,
            "receipts": [r.to_dict() for r in self.receipts],
            "guards": None if self.guards is None else self.guards.to_dict(),
            "mdl_bits": self.mdl_bits,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "evidence": dict(self.evidence),
            "include_private": self.include_private,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolPack":
        guards_raw = data.get("guards")
        metadata = _mapping(data.get("metadata"))
        return cls(
            pack_id=str(data["pack_id"]),
            symbols=[Symbol.from_dict(s) for s in data.get("symbols") or []],
            codebook=_int_keyed(data.get("codebook") or {}),
            aliases=normalize_aliases(data.get("aliases") or {}),
            reconstruction_error=float(data.get("reconstruction_error") or 0.0),
            checksum=str(data.get("checksum") or ""),
            parent_pack_id=data.get("parent_pack_id"),
            parent_checksum=data.get("parent_checksum"),
            receipts=[Receipt.from_dict(r) for r in data.get("receipts") or []],
            guards=None if not guards_raw else GuardSnapshot.from_dict(guards_raw),
            mdl_bits=float(data.get("mdl_bits") or 0.0),
            timestamp=float(data.get("timestamp") or 0.0),
            metadata=metadata,
            evidence={str(k): str(v) for k, v in (data.get("evidence") or {}).items()},
            decoder_version=_present_str(data, "decoder_version", DECODER_VERSION),
            decision=_load_decision(data, metadata),
            include_private=bool(data.get("include_private") or False),
        )

    def seal(self) -> "SymbolPack":
        self.checksum = self.compute_checksum()
        return self

    def compute_checksum(self) -> str:
        payload = {
            "pack_id": self.pack_id,
            "decoder_version": self.decoder_version,
            "decision": self.decision,
            "parent_pack_id": self.parent_pack_id,
            "parent_checksum": self.parent_checksum,
            "codebook": {str(k): int(v) for k, v in sorted(self.codebook.items())},
            "aliases": aliases_to_dict(self.aliases),
            "reconstruction_error": round(float(self.reconstruction_error), 8),
            "mdl_bits": round(float(self.mdl_bits), 8),
            "guards": None if self.guards is None else self.guards.to_dict(),
            "receipts": [r.to_dict() for r in self.receipts],
            "evidence": {k: self.evidence[k] for k in sorted(self.evidence)},
            "metadata": self.metadata,
            "include_private": self.include_private,
            "symbols": [
                {
                    "class_id": s.class_id,
                    "code": s.code,
                    "proto_embedding": round_vec(s.proto_embedding),
                    "observation_ids": list(s.observation_ids),
                    "definition": self._normalize_definition(s.definition),
                    "example_hashes": list(s.example_hashes),
                    "examples": list(s.examples) if self.include_private else [],
                    "confidence": round(float(s.confidence), 8),
                    "quarantined": bool(s.quarantined),
                    "survival": round(float(s.survival), 8),
                    "metadata": dict(s.metadata),
                }
                for s in sorted(self.symbols, key=lambda x: x.class_id)
            ],
        }
        return sha256_hex(payload)

    @staticmethod
    def _normalize_definition(definition: Optional[str]) -> Optional[str]:
        if definition is None:
            return None
        stripped = definition.strip()
        return stripped if stripped else None

    def symbol_by_class(self, class_id: int) -> Optional[Symbol]:
        for symbol in self.symbols:
            if symbol.class_id == class_id:
                return symbol
        return None

    def symbol_by_code(self, code: int) -> Optional[Symbol]:
        class_id = self.codebook.get(code)
        if class_id is not None:
            found = self.symbol_by_class(class_id)
            if found is not None:
                return found
        for symbol in self.symbols:
            if symbol.code == code:
                return symbol
        return None

    def alias_table(self, source_pack_checksum: Optional[str] = None) -> Dict[int, int]:
        if source_pack_checksum is not None:
            if source_pack_checksum in self.aliases:
                return dict(self.aliases[source_pack_checksum])
            if self.parent_checksum is not None and source_pack_checksum == self.parent_checksum:
                return dict(self.aliases.get(self.parent_checksum, {}))
            return {}
        if "legacy" in self.aliases:
            return dict(self.aliases["legacy"])
        return {}

    def resolve_code(
        self,
        code: int,
        source_pack_checksum: Optional[str] = None,
    ) -> Tuple[int, bool]:
        from neuralese.aliases import follow_aliases

        code = int(code)
        if code in self.codebook:
            return code, False
        table = self.alias_table(source_pack_checksum)
        resolved = follow_aliases(code, table)
        return resolved, resolved != code


@dataclass
class Gloss:
    code: int
    english: str
    confidence: float
    state: str
    pack_checksum: str
    class_id: Optional[int] = None
    resolved_code: Optional[int] = None
    observation_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state not in GLOSS_STATES:
            raise ValueError(f"invalid gloss state {self.state!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditCertificate:
    pack_id: str
    pack_checksum: str
    expected_checksum: str
    passed: bool
    addressable: bool
    unfoldable: bool
    gloss_bound: bool
    residual_ok: bool
    fail_closed: bool
    failures: List[str]
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    integrity_valid: bool = False
    evidence_valid: bool = False
    admission_valid: bool = False
    policy: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditCertificate":
        return cls(
            pack_id=str(data["pack_id"]),
            pack_checksum=str(data["pack_checksum"]),
            expected_checksum=str(data["expected_checksum"]),
            passed=bool(data["passed"]),
            integrity_valid=bool(data.get("integrity_valid", data.get("passed"))),
            evidence_valid=bool(data.get("evidence_valid", data.get("unfoldable"))),
            admission_valid=bool(data.get("admission_valid", data.get("passed"))),
            addressable=bool(data["addressable"]),
            unfoldable=bool(data["unfoldable"]),
            gloss_bound=bool(data["gloss_bound"]),
            residual_ok=bool(data["residual_ok"]),
            fail_closed=bool(data.get("fail_closed", True)),
            failures=[str(x) for x in data.get("failures") or []],
            timestamp=float(data["timestamp"]),
            policy=str(data.get("policy") or "default"),
            details=dict(data.get("details") or {}),
        )


class UncertifiedPackError(ValueError):
    def __init__(self, certificate: AuditCertificate):
        self.certificate = certificate
        super().__init__(
            "pack failed certification policy "
            f"{certificate.policy!r}: {'; '.join(certificate.failures) or 'not passed'}"
        )


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def iter_live_symbols(pack: SymbolPack) -> Iterable[Symbol]:
    for symbol in pack.symbols:
        if not symbol.quarantined:
            yield symbol

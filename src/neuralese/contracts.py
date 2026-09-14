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


def _load_evidence(data: Dict[str, Any]) -> Any:
    if "evidence" not in data:
        return {}
    value = data["evidence"]
    if not isinstance(value, dict):
        return value
    return dict(value)


def _serialize_evidence(evidence: Any, *, sort_keys: bool = False) -> Any:
    if not isinstance(evidence, dict):
        return evidence
    if sort_keys:
        return {k: evidence[k] for k in sorted(evidence)}
    return dict(evidence)


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


def _present_value(data: Dict[str, Any], key: str, default: Any) -> Any:
    if key not in data:
        return default
    return data[key]


def _round_real(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(float(value), 8)


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
    """Unsalted SHA-256 of canonical JSON {'example': text}, not of UTF-8 bytes alone."""
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
            ok=data["ok"],
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
        if self.definition is not None and not isinstance(self.definition, str):
            raise TypeError("definition must be a string or null")
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
        if include_private is True:
            payload["examples"] = list(self.examples)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, include_private: bool = False) -> "Symbol":
        if not isinstance(data, dict):
            raise TypeError("symbol record must be an object")
        examples = _string_id_list(data, "examples")
        if include_private is not True:
            examples = []
        if "example_hashes" not in data or data["example_hashes"] is None:
            hashes: List[Any] = []
        else:
            raw_hashes = data["example_hashes"]
            if not isinstance(raw_hashes, (list, tuple)):
                raise TypeError("example_hashes must be an array")
            hashes = list(raw_hashes)
        if examples and not hashes:
            hashes = [example_hash(x) for x in examples]
        definition = data.get("definition")
        if definition is not None and not isinstance(definition, str):
            raise TypeError("definition must be a string or null")
        return cls(
            class_id=int(data["class_id"]),
            code=int(data["code"]),
            proto_embedding=[float(x) for x in data.get("proto_embedding") or []],
            observation_ids=_string_id_list(data, "observation_ids"),
            definition=definition,
            examples=examples,
            example_hashes=hashes,
            confidence=_present_value(data, "confidence", 0.0),
            quarantined=data["quarantined"] if "quarantined" in data else False,
            survival=_present_value(data, "survival", 1.0),
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
            "symbols": [s.to_dict(include_private=self.include_private is True) for s in self.symbols],
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
            "evidence": _serialize_evidence(self.evidence),
            "include_private": self.include_private,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolPack":
        if not isinstance(data, dict):
            raise TypeError("pack must be a JSON object")
        metadata = _optional_object(data, "metadata", "metadata")
        if "symbols" not in data:
            symbols_raw: Any = []
        else:
            symbols_raw = data["symbols"]
        if not isinstance(symbols_raw, (list, tuple)):
            raise TypeError("symbols must be an array")
        if "guards" not in data or data["guards"] is None:
            guards = None
        else:
            guards_raw = data["guards"]
            if not isinstance(guards_raw, dict):
                raise TypeError("guards must be an object or null")
            guards = GuardSnapshot.from_dict(guards_raw)
        if "aliases" not in data or data["aliases"] is None:
            aliases_raw: Any = {}
        else:
            aliases_raw = data["aliases"]
        include_private = data["include_private"] if "include_private" in data else False
        return cls(
            pack_id=str(data["pack_id"]),
            symbols=[
                Symbol.from_dict(s, include_private=include_private is True)
                for s in symbols_raw
            ],
            codebook=_int_keyed(data.get("codebook") or {}),
            aliases=normalize_aliases(aliases_raw),
            reconstruction_error=_present_value(data, "reconstruction_error", 0.0),
            checksum=str(data.get("checksum") or ""),
            parent_pack_id=data.get("parent_pack_id"),
            parent_checksum=data.get("parent_checksum"),
            receipts=[Receipt.from_dict(r) for r in data.get("receipts") or []],
            guards=guards,
            mdl_bits=float(data.get("mdl_bits") or 0.0),
            timestamp=float(data.get("timestamp") or 0.0),
            metadata=metadata,
            evidence=_load_evidence(data),
            decoder_version=_present_str(data, "decoder_version", DECODER_VERSION),
            decision=_load_decision(data, metadata),
            include_private=include_private,
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
            "reconstruction_error": _round_real(self.reconstruction_error),
            "mdl_bits": _round_real(self.mdl_bits),
            "guards": None if self.guards is None else self.guards.to_dict(),
            "receipts": [r.to_dict() for r in self.receipts],
            "evidence": _serialize_evidence(self.evidence, sort_keys=True),
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
                    "examples": list(s.examples) if self.include_private is True else [],
                    "confidence": _round_real(s.confidence),
                    "quarantined": bool(s.quarantined),
                    "survival": _round_real(s.survival),
                    "metadata": dict(s.metadata),
                }
                for s in sorted(self.symbols, key=lambda x: x.class_id)
            ],
        }
        return sha256_hex(payload)

    @staticmethod
    def _normalize_definition(definition: Optional[str]) -> Optional[str]:
        if definition is None or not isinstance(definition, str):
            return definition
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
        return select_alias_table(
            self.aliases,
            source_pack_checksum,
            parent_checksum=self.parent_checksum,
        )

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
        passed = _json_bool(data, "passed", required=True)
        unfoldable = _json_bool(data, "unfoldable", required=True)
        if "integrity_valid" in data:
            integrity_valid = _json_bool(data, "integrity_valid")
        else:
            integrity_valid = passed
        if "evidence_valid" in data:
            evidence_valid = _json_bool(data, "evidence_valid")
        else:
            evidence_valid = unfoldable
        if "admission_valid" in data:
            admission_valid = _json_bool(data, "admission_valid")
        else:
            admission_valid = passed
        return cls(
            pack_id=str(data["pack_id"]),
            pack_checksum=str(data["pack_checksum"]),
            expected_checksum=str(data["expected_checksum"]),
            passed=passed,
            integrity_valid=integrity_valid,
            evidence_valid=evidence_valid,
            admission_valid=admission_valid,
            addressable=_json_bool(data, "addressable", required=True),
            unfoldable=unfoldable,
            gloss_bound=_json_bool(data, "gloss_bound", required=True),
            residual_ok=_json_bool(data, "residual_ok", required=True),
            fail_closed=_json_bool(data, "fail_closed", default=True),
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

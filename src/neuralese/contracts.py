"""Public data contracts for the Neuralese to English Translator."""
from __future__ import annotations

import hashlib
import json
import math
import numbers
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
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_dumps(obj).encode("utf-8")).hexdigest()


def round_vec(values: Iterable[float]) -> List[float]:
    return [round(float(x), 8) for x in values]


def _integral_code(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _alias_key(value: Any) -> Any:
    if _integral_code(value):
        return value
    if isinstance(value, str):
        body = value[1:] if value.startswith("-") else value
        if body.isdigit() and body:
            return int(value)
    return value


def _copy_alias_table(mapping: Dict[Any, Any]) -> Dict[Any, Any]:
    return {_alias_key(key): value for key, value in mapping.items()}


def _copy_codebook(raw: Any) -> Any:
    if not isinstance(raw, dict):
        return raw
    return {_alias_key(key): value for key, value in raw.items()}


def _codebook_to_dict(codebook: Any) -> Any:
    if not isinstance(codebook, dict):
        return codebook
    try:
        entries = sorted(codebook.items())
    except TypeError:
        entries = list(codebook.items())
    serialized: Dict[str, int] = {}
    for key, value in entries:
        if not _integral_code(key) or not _integral_code(value):
            raise TypeError("codebook entries must be integer-to-integer")
        serialized[str(key)] = int(value)
    return serialized


def _int_keyed(d: Dict[Any, Any]) -> Dict[Any, Any]:
    return _copy_codebook(d)


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
    for key in evidence:
        if not isinstance(key, str):
            raise TypeError("evidence keys must be strings")
    if sort_keys:
        return {k: evidence[k] for k in sorted(evidence)}
    return dict(evidence)


def _copy_seq(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _copy_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_mapping(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_mapping(item) for item in value]
    return value


def _checksum_vec(values: Any) -> Any:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        return values
    try:
        return [_round_real(x) for x in values]
    except (TypeError, ValueError, OverflowError):
        return list(values)


def _optional_object(data: Dict[str, Any], key: str, label: str) -> Dict[str, Any]:
    if key not in data or data[key] is None:
        return {}
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object or null")
    return dict(value)


def _string_id_list(
    data: Dict[str, Any], key: str, *, preserve_null: bool = False
) -> Any:
    if key not in data:
        return []
    raw = data[key]
    if raw is None:
        return None if preserve_null else []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise TypeError(f"{key} must be an array")
    ids: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise TypeError(f"{key} must contain strings")
        ids.append(item)
    return ids


def _receipts_list(data: Dict[str, Any]) -> List[Any]:
    if "receipts" not in data:
        return []
    raw = data["receipts"]
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise TypeError("receipts must be an array")
    return list(raw)


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


def _embedding_values(values: Any) -> List[float]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise TypeError("embedding must be an array or null")
    try:
        items = list(values)
    except TypeError as exc:
        raise TypeError("embedding must be an array or null") from exc
    out: List[float] = []
    for x in items:
        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            raise TypeError("embedding must contain numbers")
        try:
            value = float(x)
        except OverflowError as exc:
            raise ValueError("embedding must contain finite numbers") from exc
        if not math.isfinite(value):
            raise ValueError("embedding must contain finite numbers")
        out.append(value)
    return out


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
            if not isinstance(src, str):
                tables[src] = _copy_alias_table(mapping)
                continue
            if not src:
                raise ValueError("alias source keys must be non-empty")
            tables[src] = _copy_alias_table(mapping)
        return tables
    return {LEGACY_ALIAS_KEY: _copy_alias_table(raw)}


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


def aliases_to_dict(aliases: AliasTables) -> Any:
    if not isinstance(aliases, dict):
        return aliases
    items = list(aliases.items())
    serialized: Dict[str, Any] = {}
    for src, mapping in items:
        if not isinstance(src, str) or not src:
            raise TypeError("alias source keys must be non-empty strings")
        if not isinstance(mapping, dict):
            serialized[src] = mapping
            continue
        try:
            entries = sorted(mapping.items())
        except TypeError:
            entries = list(mapping.items())
        table: Dict[str, int] = {}
        for key, value in entries:
            if not _integral_code(key) or not _integral_code(value):
                raise TypeError("alias entries must be integer-to-integer")
            table[str(key)] = int(value)
        serialized[src] = table
    return {src: serialized[src] for src, _ in sorted(serialized.items())}


def observation_content_hash(
    observation_id: str,
    embedding: Optional[Iterable[float]] = None,
    text: Optional[str] = None,
) -> str:
    # NumPy arrays are truthy-ambiguous; never use `embedding or []`.
    values = [] if embedding is None else embedding
    return sha256_hex(
        {
            "observation_id": observation_id,
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
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("text must be a string or null")
        if self.embedding is None:
            self.embedding = []
        else:
            self.embedding = _embedding_values(self.embedding)

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
            observation_id=data["observation_id"],
            embedding=list(raw_embedding),
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
            step=data["step"],
            ok=data["ok"],
            timestamp=data["timestamp"],
            kappa=data.get("kappa"),
            reconstruction_error=data.get("reconstruction_error"),
            delta_mdl_bits=data.get("delta_mdl_bits"),
            metadata=_optional_object(data, "metadata", "metadata"),
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
        if (
            self.examples
            and isinstance(self.example_hashes, (list, tuple))
            and not self.example_hashes
        ):
            self.example_hashes = [example_hash(x) for x in self.examples]

    def to_dict(self, *, include_private: bool = False) -> Dict[str, Any]:
        payload = {
            "class_id": self.class_id,
            "code": self.code,
            "proto_embedding": _copy_seq(self.proto_embedding),
            "observation_ids": _copy_seq(self.observation_ids),
            "definition": self.definition,
            "example_hashes": _copy_seq(self.example_hashes),
            "confidence": self.confidence,
            "quarantined": self.quarantined,
            "survival": self.survival,
            "metadata": _copy_mapping(self.metadata),
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
        if "example_hashes" not in data:
            hashes: Any = []
        else:
            hashes = data["example_hashes"]
            if hashes is not None:
                if isinstance(hashes, (str, bytes)) or not isinstance(hashes, (list, tuple)):
                    raise TypeError("example_hashes must be an array")
                hashes = list(hashes)
        if examples and hashes is not None and not hashes:
            hashes = [example_hash(x) for x in examples]
        definition = data.get("definition")
        if definition is not None and not isinstance(definition, str):
            raise TypeError("definition must be a string or null")
        return cls(
            class_id=data["class_id"],
            code=data["code"],
            proto_embedding=_copy_seq(_present_value(data, "proto_embedding", [])),
            observation_ids=_string_id_list(
                data, "observation_ids", preserve_null=True
            ),
            definition=definition,
            examples=examples,
            example_hashes=hashes,
            confidence=_present_value(data, "confidence", 0.0),
            quarantined=data["quarantined"] if "quarantined" in data else False,
            survival=_present_value(data, "survival", 1.0),
            metadata=_optional_object(data, "metadata", "metadata"),
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
            "codebook": _codebook_to_dict(self.codebook),
            "aliases": aliases_to_dict(self.aliases),
            "reconstruction_error": self.reconstruction_error,
            "checksum": self.checksum,
            "parent_pack_id": self.parent_pack_id,
            "parent_checksum": self.parent_checksum,
            "receipts": [r.to_dict() for r in self.receipts],
            "guards": None if self.guards is None else self.guards.to_dict(),
            "mdl_bits": self.mdl_bits,
            "timestamp": self.timestamp,
            "metadata": _copy_mapping(self.metadata),
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
            pack_id=data["pack_id"],
            symbols=[
                Symbol.from_dict(s, include_private=include_private is True)
                for s in symbols_raw
            ],
            codebook=_copy_codebook(data["codebook"] if "codebook" in data else {}),
            aliases=normalize_aliases(aliases_raw),
            reconstruction_error=_present_value(data, "reconstruction_error", 0.0),
            checksum=_present_value(data, "checksum", ""),
            parent_pack_id=data.get("parent_pack_id"),
            parent_checksum=data.get("parent_checksum"),
            receipts=[Receipt.from_dict(r) for r in _receipts_list(data)],
            guards=guards,
            mdl_bits=_present_value(data, "mdl_bits", 0.0),
            timestamp=_present_value(data, "timestamp", 0.0),
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
            "codebook": _codebook_to_dict(self.codebook),
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
                    "proto_embedding": _checksum_vec(s.proto_embedding),
                    "observation_ids": list(s.observation_ids)
                    if isinstance(s.observation_ids, (list, tuple))
                    else s.observation_ids,
                    "definition": self._normalize_definition(s.definition),
                    "example_hashes": _copy_seq(s.example_hashes),
                    "examples": _copy_seq(s.examples) if self.include_private is True else [],
                    "confidence": _round_real(s.confidence),
                    "quarantined": bool(s.quarantined),
                    "survival": _round_real(s.survival),
                    "metadata": _copy_mapping(s.metadata),
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
        for symbol in _iter_symbol_objects(self.symbols):
            if symbol.class_id == class_id:
                return symbol
        return None

    def symbol_by_code(self, code: int) -> Optional[Symbol]:
        try:
            class_id = self.codebook.get(code) if isinstance(self.codebook, dict) else None
        except TypeError:
            class_id = None
        if class_id is not None:
            found = self.symbol_by_class(class_id)
            if found is not None:
                return found
        for symbol in _iter_symbol_objects(self.symbols):
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
        if isinstance(self.codebook, dict) and code in self.codebook:
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


def _is_array(value: Any) -> bool:
    return not isinstance(value, (str, bytes)) and isinstance(value, (list, tuple))


def _iter_symbol_objects(symbols: Any) -> Iterable["Symbol"]:
    if not _is_array(symbols):
        return
    for symbol in symbols:
        if isinstance(symbol, Symbol):
            yield symbol


def iter_live_symbols(pack: SymbolPack) -> Iterable[Symbol]:
    for symbol in _iter_symbol_objects(pack.symbols):
        if not symbol.quarantined:
            yield symbol

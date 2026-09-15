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
    serialized: Dict[Any, Any] = {}
    for key, value in entries:
        serialized[str(key) if _integral_code(key) else key] = (
            int(value) if _integral_code(value) else value
        )
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
    if sort_keys:
        return {k: evidence[k] for k in sorted(evidence)}
    return dict(evidence)


def _copy_seq(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return list(value)
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
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError("embedding must contain numbers")
        out.append(float(x))
    return out


def _load_decision(data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    if "decision" in data:
        return _present_str(data, "decision", "")
    if "decision" in metadata:
        value = metadata.get("decision")
        return "" if value is None else str(value)
    return "accept"

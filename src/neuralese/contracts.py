"""Public data contracts for the Neuralese to English Translator."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterable, List, Optional, Tuple

GLOSS_STATES = ("ok", "aliased", "quarantined", "unknown")

# Certificate schema is not the package version. 0.2 is the certificate
# object certify() writes. A certificate dict that omits the field loads
# as unversioned. There is no v0.1 mode that turns a missing reservoir
# into passed=true.
CERTIFICATE_SCHEMA_V02 = "0.2"
CERTIFICATE_SCHEMA_UNVERSIONED = "unversioned"
CERTIFICATE_AUTHORITY = "RUNTIME_NOT_AUTHORIZED"
CERTIFICATE_DECISION_RESEARCH_FAIL = "RESEARCH_FAIL"
CERTIFICATE_DECISION_UNRESOLVED = "UNRESOLVED"
_CERTIFICATE_SCHEMAS = frozenset(
    {CERTIFICATE_SCHEMA_V02, CERTIFICATE_SCHEMA_UNVERSIONED}
)
_CERTIFICATE_DECISIONS = frozenset(
    {CERTIFICATE_DECISION_RESEARCH_FAIL, CERTIFICATE_DECISION_UNRESOLVED}
)

# Named by the v0.2 schema as outside today's compute_checksum. certify()
# records this list for unbound packs and for every pairing row that does
# not carry a non-empty identity-1 digest. identity-1 drops only
# proto_embedding from the certificate list, and only once identity_digest
# is present. This tuple is not rewritten, and proto_embedding is not added
# to compute_checksum.
DIGEST_DOES_NOT_BIND = (
    "proto_embedding",
    "confidence",
    "examples",
    "survival",
    "symbol.metadata",
    "kappa",
    "guards",
    "receipts",
    "parent_pack_id",
    "mdl_bits",
    "pack.metadata",
)

# Digest profile is not schema_version and not the package version.
# Omitted, null, and blank are absence. The word "unbound" is the legacy
# profile. identity-1 is opt-in. Any other string is an unknown profile.
DIGEST_PROFILE_UNBOUND = "unbound"
DIGEST_PROFILE_IDENTITY_1 = "identity-1"
DIGEST_PAIRING_UNBOUND = "unbound"
DIGEST_PAIRING_INCONSISTENT = "INCONSISTENT"
DIGEST_PAIRING_UNKNOWN = "UNKNOWN PROFILE"
DIGEST_PAIRING_IDENTITY_1 = "identity-1"


def canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha16(obj: Any) -> str:
    return hashlib.sha256(canonical_dumps(obj).encode("utf-8")).hexdigest()[:16]


def digest_field_absent(value: Any) -> bool:
    """True for omitted, null, or blank. A blank string is whitespace only."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def canonical_digest_token(value: Any) -> Optional[str]:
    """Return an exact token, or None when the field is absent.

    None and a blank string (empty or whitespace-only) are absence.
    Any other string is returned unchanged. Case is not folded and
    surrounding whitespace is not removed, so ``" identity-1"`` and
    ``"identity-1 "`` are not ``identity-1``, and ``" unbound"`` is not
    the legacy word ``unbound``.

    A non-string is rendered as a representable token that is neither
    ``identity-1`` nor ``unbound`` and cannot equal a sha16 digest.
    certify then fails closed instead of raising or accepting it.
    """
    if digest_field_absent(value):
        return None
    if isinstance(value, str):
        return value
    return f"non-string:{type(value).__name__}:{value}"


@dataclass(frozen=True)
class DigestPairing:
    """Fail-closed reading of digest_profile against identity_digest.

    ``profile`` and ``identity_digest`` are the normalized tokens. ``profile``
    is None when the profile was omitted, null, or blank, and is the word
    ``unbound`` when that word was stored. Unknown strings stay on ``profile``
    and are not relabeled unbound.
    """

    pairing: str
    profile: Optional[str]
    identity_digest: Optional[str]

    @property
    def identity_digest_present(self) -> bool:
        return self.identity_digest is not None


def classify_digest_pairing(profile: Any, identity_digest: Any) -> DigestPairing:
    """Apply the identity-1 pairing table.

    Does not coerce unknown profiles. Does not strip or case-fold a
    string into ``identity-1`` or ``unbound``.
    """
    token = canonical_digest_token(profile)
    digest = canonical_digest_token(identity_digest)
    if token is None or token == DIGEST_PROFILE_UNBOUND:
        if digest is None:
            return DigestPairing(DIGEST_PAIRING_UNBOUND, token, None)
        return DigestPairing(DIGEST_PAIRING_INCONSISTENT, token, digest)
    if token == DIGEST_PROFILE_IDENTITY_1:
        return DigestPairing(DIGEST_PAIRING_IDENTITY_1, token, digest)
    return DigestPairing(DIGEST_PAIRING_UNKNOWN, token, digest)


def digest_does_not_bind_for(pairing: DigestPairing) -> Tuple[str, ...]:
    """Certificate list for this pairing.

    proto_embedding stays on the list for unbound, inconsistent, unknown,
    and identity-1 with no digest. A present identity-1 digest drops only
    that name, including when the digest does not match.
    """
    if (
        pairing.pairing == DIGEST_PAIRING_IDENTITY_1
        and pairing.identity_digest_present
    ):
        return tuple(name for name in DIGEST_DOES_NOT_BIND if name != "proto_embedding")
    return DIGEST_DOES_NOT_BIND


def _int_keyed(d: Dict[Any, Any]) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


@dataclass
class Observation:
    observation_id: str
    embedding: List[float] = field(default_factory=list)
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(
            observation_id=str(data["observation_id"]),
            embedding=[float(x) for x in data.get("embedding") or []],
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Symbol":
        return cls(
            class_id=int(data["class_id"]),
            code=int(data["code"]),
            proto_embedding=[float(x) for x in data.get("proto_embedding") or []],
            observation_ids=[str(x) for x in data.get("observation_ids") or []],
            definition=data.get("definition"),
            examples=[str(x) for x in data.get("examples") or []],
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
    aliases: Dict[int, int]
    reconstruction_error: float
    checksum: str = ""
    parent_pack_id: Optional[str] = None
    receipts: List[Receipt] = field(default_factory=list)
    guards: Optional[GuardSnapshot] = None
    mdl_bits: float = 0.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    observations: Dict[str, Observation] = field(default_factory=dict)
    # Optional. Both absent means unbound. Default seal() does not set them.
    digest_profile: Optional[str] = None
    identity_digest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "pack_id": self.pack_id,
            "symbols": [s.to_dict() for s in self.symbols],
            "codebook": {str(k): int(v) for k, v in self.codebook.items()},
            "aliases": {str(k): int(v) for k, v in self.aliases.items()},
            "observations": [
                self.observations[key].to_dict() for key in sorted(self.observations)
            ],
            "reconstruction_error": self.reconstruction_error,
            "checksum": self.checksum,
            "parent_pack_id": self.parent_pack_id,
            "receipts": [r.to_dict() for r in self.receipts],
            "guards": None if self.guards is None else self.guards.to_dict(),
            "mdl_bits": self.mdl_bits,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        # Unbound packs omit both keys so legacy JSON keeps today's shape.
        # A present profile or digest writes both keys. An absent side is
        # JSON null, and from_dict reads null as absence. Null plus null is
        # unbound. One present side is not unbound.
        profile = canonical_digest_token(self.digest_profile)
        digest = canonical_digest_token(self.identity_digest)
        if profile is not None or digest is not None:
            payload["digest_profile"] = profile
            payload["identity_digest"] = digest
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolPack":
        """Load a pack. Omitted digest fields are absence, not identity-1.

        Unknown digest_profile strings are stored as given. They are not
        relabeled unbound and this loader does not raise for them. certify
        fail-closes that pack with passed false. Blank and null are absence.
        Any other string is stored exactly. A non-string is kept as a
        representable token and is not coerced into identity-1 or unbound.
        """
        guards_raw = data.get("guards")
        if "digest_profile" in data:
            digest_profile = canonical_digest_token(data.get("digest_profile"))
        else:
            digest_profile = None
        if "identity_digest" in data:
            identity_digest = canonical_digest_token(data.get("identity_digest"))
        else:
            identity_digest = None
        pack = cls(
            pack_id=str(data["pack_id"]),
            symbols=[Symbol.from_dict(s) for s in data.get("symbols") or []],
            codebook=_int_keyed(data.get("codebook") or {}),
            aliases=_int_keyed(data.get("aliases") or {}),
            reconstruction_error=float(data.get("reconstruction_error") or 0.0),
            checksum=str(data.get("checksum") or ""),
            parent_pack_id=data.get("parent_pack_id"),
            receipts=[Receipt.from_dict(r) for r in data.get("receipts") or []],
            guards=None if not guards_raw else GuardSnapshot.from_dict(guards_raw),
            mdl_bits=float(data.get("mdl_bits") or 0.0),
            timestamp=float(data.get("timestamp") or 0.0),
            metadata=dict(data.get("metadata") or {}),
            observations=_load_observations(data.get("observations")),
            digest_profile=digest_profile,
            identity_digest=identity_digest,
        )
        return pack

    def seal(self, *, digest_profile: Optional[str] = None) -> "SymbolPack":
        """Assign pack.checksum from compute_checksum().

        The default call does not write digest_profile or identity_digest
        and does not refresh an identity_digest already stored. Pass the
        exact string ``digest_profile="identity-1"`` to store that profile
        and identity_digest = sha16(identity payload) for the object as it
        is now. The opt-in still assigns pack.checksum from
        compute_checksum(), so checksum bytes stay the legacy digest. Any
        other value, including a different case or surrounding whitespace,
        raises ValueError and writes nothing.
        """
        if digest_profile is not None and digest_profile != DIGEST_PROFILE_IDENTITY_1:
            raise ValueError(
                "seal opt-in accepts only digest_profile 'identity-1'; "
                f"got {digest_profile!r}"
            )
        self.checksum = self.compute_checksum()
        if digest_profile is None:
            return self
        self.digest_profile = DIGEST_PROFILE_IDENTITY_1
        self.identity_digest = self.compute_identity_digest()
        return self

    def _checksum_payload(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "codebook": {str(k): int(v) for k, v in sorted(self.codebook.items())},
            "aliases": {str(k): int(v) for k, v in sorted(self.aliases.items())},
            "reconstruction_error": round(float(self.reconstruction_error), 8),
            "reservoir": _reservoir_rows(self.observations),
            "symbols": [
                {
                    "class_id": s.class_id,
                    "code": s.code,
                    "observation_ids": list(s.observation_ids),
                    "definition": self._normalize_definition(s.definition),
                    "quarantined": bool(s.quarantined),
                }
                for s in sorted(self.symbols, key=lambda x: x.class_id)
            ],
        }

    def compute_checksum(self) -> str:
        return sha16(self._checksum_payload())

    def identity_payload(self) -> Dict[str, Any]:
        """Checksum object plus proto_embedding rounded to 8 decimals.

        Symbols stay sorted by class_id. An empty prototype is [].
        Confidence, examples, survival, symbol metadata, guards, receipts,
        parent_pack_id, mdl_bits, pack metadata, and timestamp are not keys.
        """
        payload = self._checksum_payload()
        ordered = sorted(self.symbols, key=lambda item: item.class_id)
        enriched: List[Dict[str, Any]] = []
        for row, symbol in zip(payload["symbols"], ordered):
            proto = symbol.proto_embedding or []
            enriched.append(
                {
                    **row,
                    "proto_embedding": [round(float(x), 8) for x in proto],
                }
            )
        payload["symbols"] = enriched
        return payload

    def compute_identity_digest(self) -> str:
        return sha16(self.identity_payload())

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

    def resolve_code(self, code: int) -> Tuple[int, bool]:
        seen: set[int] = set()
        current = int(code)
        aliased = False
        while current in self.aliases:
            if current in seen:
                return current, True
            seen.add(current)
            current = int(self.aliases[current])
            aliased = True
        return current, aliased


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
    schema_version: str = CERTIFICATE_SCHEMA_UNVERSIONED
    package_version: Optional[str] = None
    authority: str = CERTIFICATE_AUTHORITY
    decision: str = CERTIFICATE_DECISION_UNRESOLVED
    evidence: Dict[str, Any] = field(default_factory=dict)
    # Independent of gloss_bound. True only when identity-1 matches.
    # Omitted on legacy JSON loads as False, which is not an identity pass.
    identity_bound: bool = False
    digest_profile: Optional[str] = None
    identity_digest: Optional[str] = None

    def __post_init__(self) -> None:
        if self.schema_version not in _CERTIFICATE_SCHEMAS:
            raise ValueError(
                f"unsupported certificate schema_version {self.schema_version!r}"
            )
        if self.authority != CERTIFICATE_AUTHORITY:
            raise ValueError(
                f"unsupported certificate authority {self.authority!r}"
            )
        if self.decision not in _CERTIFICATE_DECISIONS:
            raise ValueError(
                "unsupported certificate decision "
                f"{self.decision!r}; RESEARCH_PASS is not defined"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditCertificate":
        """Load a certificate. A missing schema_version is unversioned.

        Today's twelve-field JSON still loads. schema_version is optional
        and defaults to unversioned. An omitted decision stays UNRESOLVED
        rather than being recomputed into a pass. authority defaults to
        RUNTIME_NOT_AUTHORIZED. package_version is informational and is
        not a gate; a missing copy stays None.

        digest_profile and identity_digest are optional. Omitted, null, and
        blank are absence. Unknown profile strings are stored exactly and
        are not relabeled unbound, identity-1, or a stripped form of either.
        This loader does not raise for them and does not re-run certify.
        identity_bound defaults to false when omitted, so a legacy
        certificate does not become an identity pass. A non-string profile
        is kept representable and is not coerced into an accepting word.
        """
        raw_evidence = data.get("evidence")
        if raw_evidence is None:
            evidence: Dict[str, Any] = {}
        elif isinstance(raw_evidence, dict):
            evidence = dict(raw_evidence)
        else:
            raise ValueError("certificate evidence must be an object")
        schema_raw = data.get("schema_version")
        if schema_raw is None or str(schema_raw).strip() == "":
            schema_version = CERTIFICATE_SCHEMA_UNVERSIONED
        else:
            schema_version = str(schema_raw)
        package_raw = data.get("package_version") if "package_version" in data else None
        authority_raw = data.get("authority")
        decision_raw = data.get("decision")
        if "digest_profile" in data:
            digest_profile = canonical_digest_token(data.get("digest_profile"))
        else:
            digest_profile = None
        if "identity_digest" in data:
            identity_digest = canonical_digest_token(data.get("identity_digest"))
        else:
            identity_digest = None
        if "identity_bound" in data and data.get("identity_bound") is not None:
            identity_bound = bool(data.get("identity_bound"))
        else:
            identity_bound = False
        return cls(
            pack_id=str(data["pack_id"]),
            pack_checksum=str(data["pack_checksum"]),
            expected_checksum=str(data["expected_checksum"]),
            passed=bool(data["passed"]),
            addressable=bool(data["addressable"]),
            unfoldable=bool(data["unfoldable"]),
            gloss_bound=bool(data["gloss_bound"]),
            residual_ok=bool(data["residual_ok"]),
            fail_closed=bool(data.get("fail_closed", True)),
            failures=[str(x) for x in data.get("failures") or []],
            timestamp=float(data["timestamp"]),
            details=dict(data.get("details") or {}),
            schema_version=schema_version,
            package_version=None if package_raw is None else str(package_raw),
            authority=(
                CERTIFICATE_AUTHORITY
                if authority_raw is None
                else str(authority_raw)
            ),
            decision=(
                CERTIFICATE_DECISION_UNRESOLVED
                if decision_raw is None
                else str(decision_raw)
            ),
            evidence=evidence,
            identity_bound=identity_bound,
            digest_profile=digest_profile,
            identity_digest=identity_digest,
        )


def _reservoir_rows(observations: Dict[str, Observation]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in sorted(observations):
        obs = observations[key]
        rows.append(
            {
                "observation_id": obs.observation_id,
                "text": obs.text,
                "embedding": [round(float(x), 8) for x in obs.embedding],
                "metadata": obs.metadata,
            }
        )
    return rows


def _load_observations(raw: Any) -> Dict[str, Observation]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        records = []
        for key, value in raw.items():
            if not isinstance(value, dict):
                raise ValueError(f"observation {key!r} must be an object")
            record = dict(value)
            record.setdefault("observation_id", str(key))
            records.append(record)
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("observations must be a list or object")
    loaded: Dict[str, Observation] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each observation must be an object")
        obs = Observation.from_dict(record)
        if obs.observation_id in loaded:
            raise ValueError(f"duplicate observation_id {obs.observation_id!r}")
        loaded[obs.observation_id] = obs
    return loaded


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def iter_live_symbols(pack: SymbolPack) -> Iterable[Symbol]:
    for symbol in pack.symbols:
        if not symbol.quarantined:
            yield symbol

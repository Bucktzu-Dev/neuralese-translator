"""Fail-closed certification of a SymbolPack."""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Set

from neuralese.adapters import ensure_embedding
from neuralese.aliases import follow_aliases, has_alias_cycle
from neuralese.contracts import (
    CERT_POLICIES,
    DECISIONS,
    DECODER_VERSION,
    SHA256_HEX,
    AuditCertificate,
    Observation,
    SymbolPack,
    example_hash,
    iter_live_symbols,
)
from neuralese.gloss import UNGLOSSED


def certify(
    pack: SymbolPack,
    *,
    require_gloss: bool = True,
    tau_residual: float = 0.55,
    policy: str = "default",
    observations: Optional[Sequence[Observation]] = None,
) -> AuditCertificate:
    if policy not in CERT_POLICIES:
        raise ValueError(f"unknown certification policy {policy!r}")

    failures: List[str] = []

    schema_ok, schema_failures = _schema_errors(pack, tau_residual)
    failures.extend(schema_failures)

    addressable = True
    for code, class_id in pack.codebook.items():
        symbol = pack.symbol_by_class(int(class_id))
        if symbol is None:
            addressable = False
            failures.append(f"code {code} maps to missing class {class_id}")
    for symbol in pack.symbols:
        mapped = pack.codebook.get(symbol.code)
        if mapped is None:
            addressable = False
            failures.append(
                f"symbol class {symbol.class_id} code {symbol.code} missing from codebook"
            )
        elif mapped != symbol.class_id:
            addressable = False
            failures.append(
                f"code {symbol.code} codebook class {mapped} != symbol class {symbol.class_id}"
            )

    class_ids = [s.class_id for s in pack.symbols]
    codes = [s.code for s in pack.symbols]
    if len(class_ids) != len(set(class_ids)):
        addressable = False
        failures.append("duplicate class_id identities")
    if len(codes) != len(set(codes)):
        addressable = False
        failures.append("duplicate code identities")

    if has_alias_cycle(pack.aliases):
        addressable = False
        failures.append("alias map contains a cycle")
    for source, mapping in pack.aliases.items():
        for old, new in mapping.items():
            resolved = follow_aliases(int(old), mapping)
            if resolved not in pack.codebook and pack.symbol_by_code(resolved) is None:
                addressable = False
                failures.append(
                    f"alias {source}:{old}->{new} does not resolve to a current symbol"
                )

    expected = pack.checksum or ""
    checksum_ok = False
    if schema_ok:
        expected = pack.compute_checksum()
        checksum_ok = bool(pack.checksum) and expected == pack.checksum
        if not pack.checksum:
            failures.append("pack is unsealed (empty checksum)")
        elif not isinstance(pack.checksum, str) or not SHA256_HEX.match(pack.checksum):
            checksum_ok = False
            failures.append("checksum is not full SHA-256")
        elif expected != pack.checksum:
            failures.append("checksum mismatch: pack mutated after sealing")

    if not _real_number(pack.reconstruction_error):
        residual_ok = False
    else:
        residual_ok = (
            _finite(pack.reconstruction_error)
            and pack.reconstruction_error >= 0.0
            and pack.reconstruction_error <= tau_residual
        )
        if pack.reconstruction_error < 0:
            residual_ok = False
            failures.append(f"reconstruction_error {pack.reconstruction_error} is negative")
        elif not residual_ok:
            failures.append(
                f"reconstruction_error {pack.reconstruction_error:.6f} exceeds tau_residual {tau_residual}"
            )

    gloss_bound = checksum_ok
    if require_gloss:
        for symbol in iter_live_symbols(pack):
            if symbol.definition is not None and not isinstance(symbol.definition, str):
                gloss_bound = False
                failures.append(f"class {symbol.class_id} definition is not a string")
                continue
            definition = (symbol.definition or "").strip()
            if not definition or definition == UNGLOSSED:
                gloss_bound = False
                failures.append(f"class {symbol.class_id} has no bound English gloss")

    integrity_valid = schema_ok and addressable and checksum_ok and residual_ok and gloss_bound

    unfoldable = True
    evidence_valid = True
    map_errors = _evidence_map_errors(pack.evidence)
    if map_errors:
        evidence_valid = False
        unfoldable = False
    live_ids: Set[str] = set()
    provided = None
    if observations is not None:
        provided = {}
        duplicate_ids: Set[str] = set()
        for obs in observations:
            oid = str(obs.observation_id)
            if oid in provided:
                duplicate_ids.add(oid)
            provided[oid] = obs
        if duplicate_ids:
            evidence_valid = False
            unfoldable = False
            for oid in sorted(duplicate_ids):
                failures.append(
                    f"duplicate observation_id {oid!r} in supplied observations"
                )
    for symbol in iter_live_symbols(pack):
        if not isinstance(symbol.observation_ids, (list, tuple)):
            unfoldable = False
            evidence_valid = False
            failures.append(f"class {symbol.class_id} observation_ids is not an array")
            continue
        if not symbol.observation_ids:
            unfoldable = False
            evidence_valid = False
            failures.append(f"class {symbol.class_id} has no observation_ids")
            continue
        for obs_id in symbol.observation_ids:
            if not isinstance(obs_id, str):
                unfoldable = False
                evidence_valid = False
                failures.append(
                    f"class {symbol.class_id} observation_id is not a string"
                )
                continue
            if not obs_id.strip():
                unfoldable = False
                evidence_valid = False
                failures.append(f"class {symbol.class_id} has a blank observation_id")
                continue
            if obs_id in live_ids:
                unfoldable = False
                evidence_valid = False
                failures.append(
                    f"observation {obs_id!r} is attached to multiple live symbols"
                )
            live_ids.add(obs_id)
            if not isinstance(pack.evidence, dict):
                evidence_valid = False
                unfoldable = False
                continue
            digest = pack.evidence.get(obs_id)
            if digest is None:
                evidence_valid = False
                failures.append(f"observation {obs_id!r} is not in the evidence manifest")
            elif not isinstance(digest, str) or not SHA256_HEX.match(digest):
                evidence_valid = False
                failures.append(f"observation {obs_id!r} evidence hash is not SHA-256")
            elif provided is not None:
                obs = provided.get(obs_id)
                if obs is None:
                    evidence_valid = False
                    failures.append(
                        f"observation {obs_id!r} was not supplied for content verification"
                    )
                else:
                    try:
                        normalized = ensure_embedding(Observation.from_dict(obs.to_dict()))
                    except ValueError:
                        evidence_valid = False
                        failures.append(
                            f"observation {obs_id!r} has neither embedding nor text"
                        )
                        continue
                    if normalized.content_hash() != digest:
                        evidence_valid = False
                        failures.append(
                            f"observation {obs_id!r} evidence hash does not match content"
                        )
    unfoldable = unfoldable and evidence_valid

    admission_valid = _admission_valid(pack, policy, failures)

    if policy == "integrity":
        passed = integrity_valid
    elif policy == "strict":
        passed = (
            integrity_valid
            and evidence_valid
            and admission_valid
            and pack.decision == "accept"
            and pack.guards is not None
            and _effective_pass_all(pack.guards)
        )
        if pack.decision != "accept":
            failures.append("strict policy requires decision=accept")
        if pack.guards is None or not _effective_pass_all(pack.guards):
            failures.append("strict policy requires guards.pass_all")
    else:
        passed = integrity_valid and evidence_valid and admission_valid

    live = list(iter_live_symbols(pack))
    return AuditCertificate(
        pack_id=pack.pack_id,
        pack_checksum=pack.checksum,
        expected_checksum=expected,
        passed=passed,
        integrity_valid=integrity_valid,
        evidence_valid=evidence_valid,
        admission_valid=admission_valid,
        addressable=addressable,
        unfoldable=unfoldable,
        gloss_bound=gloss_bound,
        residual_ok=residual_ok,
        fail_closed=True,
        failures=_unique(failures),
        timestamp=time.time(),
        policy=policy,
        details={
            "n_symbols": len(pack.symbols),
            "n_live": len(live),
            "n_quarantined": sum(1 for s in pack.symbols if s.quarantined),
            "n_aliases": (
                sum(len(m) for m in pack.aliases.values())
                if isinstance(pack.aliases, dict)
                else 0
            ),
            "n_evidence": len(pack.evidence) if isinstance(pack.evidence, dict) else 0,
            "reconstruction_error": pack.reconstruction_error,
            "tau_residual": tau_residual,
            "require_gloss": require_gloss,
            "mdl_bits": pack.mdl_bits,
            "decision": pack.decision,
            "decoder_version": pack.decoder_version,
            "expected_decoder_version": DECODER_VERSION,
            "observations_checked": observations is not None,
        },
    )


def _evidence_map_errors(evidence: object) -> List[str]:
    failures: List[str] = []
    if not isinstance(evidence, dict):
        failures.append("evidence is not an object")
        return failures
    for key, digest in evidence.items():
        if not isinstance(key, str) or not key.strip():
            failures.append("evidence key is not a non-blank string")
            continue
        if not isinstance(digest, str) or not SHA256_HEX.match(digest):
            failures.append(f"observation {key!r} evidence hash is not SHA-256")
    return failures


def _admission_valid(pack: SymbolPack, policy: str, failures: List[str]) -> bool:
    decision = pack.decision
    if decision not in ("accept", "accept_provisional", "reject"):
        failures.append(f"unknown decision {decision!r}")
        return False
    if decision == "reject":
        failures.append("decision=reject is a draft, not an admitted lexicon")
        return False
    finalize = [r for r in pack.receipts if r.step == "finalize"]
    if finalize:
        ok = finalize[-1].ok
        if not isinstance(ok, bool):
            failures.append("finalize receipt ok is not a boolean")
            return False
        if ok is not True:
            failures.append("finalize receipt is not ok")
            return False
    if pack.guards is not None and not _effective_pass_all(pack.guards):
        if decision == "accept_provisional" and policy != "strict":
            return True
        failures.append("guards.pass_all is false")
        return False
    return True


def _schema_errors(pack: SymbolPack, tau_residual: float) -> tuple[bool, List[str]]:
    failures: List[str] = []
    if pack.decoder_version != DECODER_VERSION:
        failures.append(
            f"decoder_version {pack.decoder_version!r} is not this decoder "
            f"({DECODER_VERSION})"
        )
    if pack.decision not in DECISIONS:
        failures.append(f"unknown decision {pack.decision!r}")
    if not isinstance(pack.include_private, bool):
        failures.append("include_private is not a boolean")
    if pack.guards is not None:
        for name in (*_GUARD_PASS_FLAGS, "pass_all"):
            if not isinstance(getattr(pack.guards, name), bool):
                failures.append(f"guards.{name} is not a boolean")
        if isinstance(pack.guards.pass_all, bool) and pack.guards.pass_all != _effective_pass_all(
            pack.guards
        ):
            failures.append("guards.pass_all is inconsistent with component flags")
        for name in _GUARD_METRIC_FIELDS:
            value = getattr(pack.guards, name)
            if not _real_number(value):
                failures.append(f"guards.{name} is not a number")
            elif not _finite(value):
                failures.append(f"guards.{name} is not finite")
    for index, receipt in enumerate(pack.receipts):
        if not isinstance(receipt.ok, bool):
            failures.append(f"receipts[{index}].ok is not a boolean")
    failures.extend(_evidence_map_errors(pack.evidence))
    if pack.parent_checksum is not None and (
        not isinstance(pack.parent_checksum, str) or not SHA256_HEX.match(pack.parent_checksum)
    ):
        failures.append("parent_checksum is not full SHA-256")
    if not _real_number(pack.reconstruction_error):
        failures.append("reconstruction_error is not a number")
    elif not _finite(pack.reconstruction_error):
        failures.append("reconstruction_error is not finite")
    elif pack.reconstruction_error < 0:
        failures.append("reconstruction_error must be >= 0")
    for symbol in pack.symbols:
        if not isinstance(symbol.quarantined, bool):
            failures.append(f"class {symbol.class_id} quarantined is not a boolean")
        if not _real_number(symbol.confidence):
            failures.append(f"class {symbol.class_id} confidence is not a number")
        elif not _finite(symbol.confidence) or not (0.0 <= float(symbol.confidence) <= 1.0):
            failures.append(f"class {symbol.class_id} confidence out of [0, 1]")
        if not _real_number(symbol.survival):
            failures.append(f"class {symbol.class_id} survival is not a number")
        elif not _finite(symbol.survival) or not (0.0 <= float(symbol.survival) <= 1.0):
            failures.append(f"class {symbol.class_id} survival out of [0, 1]")
        if symbol.definition is not None and not isinstance(symbol.definition, str):
            failures.append(f"class {symbol.class_id} definition is not a string")
        if not isinstance(symbol.observation_ids, (list, tuple)):
            failures.append(f"class {symbol.class_id} observation_ids is not an array")
        else:
            for index, obs_id in enumerate(symbol.observation_ids):
                if not isinstance(obs_id, str):
                    failures.append(
                        f"class {symbol.class_id} observation_ids[{index}] is not a string"
                    )
        hashes = symbol.example_hashes
        if not isinstance(hashes, (list, tuple)):
            failures.append(f"class {symbol.class_id} example_hashes is not an array")
        else:
            for index, digest in enumerate(hashes):
                if not isinstance(digest, str) or not SHA256_HEX.match(digest):
                    failures.append(
                        f"class {symbol.class_id} example_hashes[{index}] is not SHA-256"
                    )
        examples = symbol.examples
        if isinstance(examples, (str, bytes)) or not isinstance(examples, (list, tuple)):
            failures.append(f"class {symbol.class_id} examples is not an array")
        elif pack.include_private is not True and examples:
            failures.append(f"class {symbol.class_id} public pack must not retain examples")
        elif pack.include_private is True and examples and isinstance(hashes, (list, tuple)):
            if len(examples) != len(hashes):
                failures.append(
                    f"class {symbol.class_id} example_hashes length does not match examples"
                )
            else:
                for index, example in enumerate(examples):
                    digest = hashes[index]
                    if not isinstance(example, str) or example_hash(example) != digest:
                        failures.append(
                            f"class {symbol.class_id} example_hashes[{index}] does not match examples"
                        )
    return (len(failures) == 0, failures)


_GUARD_PASS_FLAGS = (
    "pass_kappa",
    "pass_residual",
    "pass_mdl",
    "pass_persist",
    "pass_compat",
)
_GUARD_METRIC_FIELDS = (
    "kappa_avg",
    "reconstruction_error",
    "delta_mdl",
    "min_survival",
)


def _effective_pass_all(guards) -> bool:
    return all(getattr(guards, name) is True for name in _GUARD_PASS_FLAGS)


def _real_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _unique(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out

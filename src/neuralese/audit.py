"""Fail-closed certification of a SymbolPack."""
from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Set

from neuralese.adapters import ensure_embedding
from neuralese.aliases import follow_aliases, has_alias_cycle
from neuralese.contracts import (
    CERT_POLICIES,
    DECISIONS,
    DECODER_VERSION,
    LEGACY_ALIAS_KEY,
    SHA256_HEX,
    AuditCertificate,
    GuardSnapshot,
    Observation,
    Receipt,
    Symbol,
    SymbolPack,
    _json_object_key_errors,
    _round_real,
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
    if not isinstance(pack.codebook, dict):
        addressable = False
        failures.append("codebook is not an object")
    else:
        for code, class_id in pack.codebook.items():
            if not _integral_code(class_id):
                addressable = False
                failures.append(f"code {code} maps to non-integer class {class_id}")
                continue
            symbol = pack.symbol_by_class(class_id)
            if symbol is None:
                addressable = False
                failures.append(f"code {code} maps to missing class {class_id}")
        if not _is_array(pack.symbols):
            addressable = False
        else:
            for symbol in pack.symbols:
                if not isinstance(symbol, Symbol):
                    addressable = False
                    continue
                if not _integral_code(symbol.code):
                    addressable = False
                    continue
                try:
                    mapped = pack.codebook.get(symbol.code)
                except TypeError:
                    addressable = False
                    failures.append(
                        f"symbol class {symbol.class_id} code {symbol.code} is not a codebook key"
                    )
                    continue
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

    if not _is_array(pack.symbols):
        addressable = False
        class_ids: List[object] = []
        codes: List[object] = []
    else:
        if any(not isinstance(symbol, Symbol) for symbol in pack.symbols):
            addressable = False
        class_ids = [s.class_id for s in pack.symbols if isinstance(s, Symbol)]
        codes = [s.code for s in pack.symbols if isinstance(s, Symbol)]
    try:
        unique_class_ids = set(class_ids)
        unique_codes = set(codes)
    except TypeError:
        addressable = False
        failures.append("class_id or code identities are not hashable")
    else:
        if len(class_ids) != len(unique_class_ids):
            addressable = False
            failures.append("duplicate class_id identities")
        if len(codes) != len(unique_codes):
            addressable = False
            failures.append("duplicate code identities")

    if isinstance(pack.aliases, dict):
        if not all(isinstance(mapping, dict) for mapping in pack.aliases.values()):
            addressable = False
            failures.append("alias table is not an object")
        else:
            try:
                cyclic = has_alias_cycle(pack.aliases)
            except (TypeError, ValueError, OverflowError):
                addressable = False
                failures.append("aliases is not an object")
            else:
                if cyclic:
                    addressable = False
                    failures.append("alias map contains a cycle")
                codebook = pack.codebook if isinstance(pack.codebook, dict) else {}
                for source, mapping in pack.aliases.items():
                    for old, new in mapping.items():
                        try:
                            resolved = follow_aliases(int(old), mapping)
                        except (TypeError, ValueError, OverflowError):
                            addressable = False
                            failures.append(
                                f"alias {source}:{old}->{new} does not resolve to a current symbol"
                            )
                            continue
                        if resolved not in codebook and pack.symbol_by_code(resolved) is None:
                            addressable = False
                            failures.append(
                                f"alias {source}:{old}->{new} does not resolve to a current symbol"
                            )
    else:
        addressable = False
        failures.append("aliases is not an object")

    expected = pack.checksum or ""
    checksum_ok = False
    if schema_ok:
        try:
            expected = pack.compute_checksum()
        except (TypeError, ValueError, OverflowError):
            checksum_ok = False
            failures.append("checksum payload is not JSON-serializable")
        else:
            checksum_ok = bool(pack.checksum) and expected == pack.checksum
            if not pack.checksum:
                failures.append("pack is unsealed (empty checksum)")
            elif not isinstance(pack.checksum, str) or not SHA256_HEX.match(pack.checksum):
                checksum_ok = False
                failures.append("checksum is not full SHA-256")
            elif expected != pack.checksum:
                failures.append("checksum mismatch: pack mutated after sealing")

    if not _real_number(tau_residual) or not _finite(tau_residual) or tau_residual < 0:
        residual_ok = False
        failures.append("tau_residual is not a finite non-negative real")
    elif not _real_number(pack.reconstruction_error):
        residual_ok = False
    else:
        try:
            residual = _round_real(pack.reconstruction_error)
        except (TypeError, ValueError, OverflowError):
            residual_ok = False
        else:
            if pack.reconstruction_error < 0:
                residual_ok = False
                failures.append(
                    f"reconstruction_error {pack.reconstruction_error} is negative"
                )
            else:
                residual_ok = _finite(residual) and residual <= tau_residual
                if not residual_ok and _finite(residual):
                    failures.append(
                        f"reconstruction_error {residual:.6f} exceeds tau_residual {tau_residual}"
                    )

    gloss_bound = checksum_ok
    for symbol in iter_live_symbols(pack):
        if symbol.definition is not None and not isinstance(symbol.definition, str):
            gloss_bound = False
            failures.append(f"class {symbol.class_id} definition is not a string")
            continue
        definition = (symbol.definition or "").strip()
        if not definition:
            gloss_bound = False
            failures.append(f"class {symbol.class_id} has no bound English gloss")
        elif require_gloss and definition == UNGLOSSED:
            gloss_bound = False
            failures.append(f"class {symbol.class_id} has no bound English gloss")

    integrity_valid = schema_ok and addressable and checksum_ok and residual_ok and gloss_bound

    unfoldable = True
    evidence_valid = True
    map_errors = _evidence_map_errors(pack.evidence)
    if map_errors:
        evidence_valid = False
        unfoldable = False
        failures.extend(map_errors)
    live_ids: Set[str] = set()
    provided = None
    if observations is not None:
        provided = {}
        duplicate_ids: Set[str] = set()
        for obs in observations:
            if not isinstance(obs, Observation):
                evidence_valid = False
                unfoldable = False
                failures.append("supplied observation is not an Observation")
                continue
            oid = obs.observation_id
            if not isinstance(oid, str):
                evidence_valid = False
                unfoldable = False
                failures.append(
                    f"supplied observation_id {oid!r} is not a string"
                )
                continue
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
                        reconstructed = Observation.from_dict(obs.to_dict())
                    except (TypeError, ValueError, OverflowError, RecursionError):
                        evidence_valid = False
                        failures.append(
                            f"observation {obs_id!r} evidence content is not verifiable"
                        )
                        continue
                    try:
                        normalized = ensure_embedding(reconstructed)
                    except ValueError:
                        evidence_valid = False
                        failures.append(
                            f"observation {obs_id!r} has neither embedding nor text"
                        )
                        continue
                    except (TypeError, OverflowError, RecursionError):
                        evidence_valid = False
                        failures.append(
                            f"observation {obs_id!r} evidence content is not verifiable"
                        )
                        continue
                    try:
                        matched = normalized.content_hash() == digest
                    except (TypeError, ValueError, OverflowError, RecursionError):
                        evidence_valid = False
                        failures.append(
                            f"observation {obs_id!r} evidence content is not verifiable"
                        )
                        continue
                    if not matched:
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
            "n_symbols": len(pack.symbols) if _is_array(pack.symbols) else 0,
            "n_live": len(live),
            "n_quarantined": (
                sum(1 for s in pack.symbols if isinstance(s, Symbol) and s.quarantined)
                if _is_array(pack.symbols)
                else 0
            ),
            "n_aliases": (
                sum(len(m) for m in pack.aliases.values() if isinstance(m, dict))
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
    if isinstance(pack.receipts, (str, bytes)) or not isinstance(pack.receipts, (list, tuple)):
        failures.append("receipts is not an array")
        return False
    finalize = [r for r in pack.receipts if isinstance(r, Receipt) and r.step == "finalize"]
    if not finalize and not isinstance(pack.guards, GuardSnapshot):
        failures.append("admission artifacts are missing")
        return False
    if finalize:
        ok = finalize[-1].ok
        if not isinstance(ok, bool):
            failures.append("finalize receipt ok is not a boolean")
            return False
        if ok is not True:
            failures.append("finalize receipt is not ok")
            return False
        recorded = finalize[-1].metadata
        if isinstance(recorded, dict) and "decision" in recorded:
            if recorded.get("decision") != decision:
                failures.append(
                    "finalize receipt decision does not match pack.decision"
                )
                return False
    if pack.guards is not None:
        if not isinstance(pack.guards, GuardSnapshot):
            failures.append("guards is not a GuardSnapshot")
            return False
        if not _effective_pass_all(pack.guards):
            if decision == "accept_provisional" and policy != "strict":
                return True
            failures.append("guards.pass_all is false")
            return False
    return True


def _schema_errors(pack: SymbolPack, tau_residual: float) -> tuple[bool, List[str]]:
    failures: List[str] = []
    if not isinstance(pack.pack_id, str):
        failures.append("pack_id is not a string")
    if pack.parent_pack_id is not None and not isinstance(pack.parent_pack_id, str):
        failures.append("parent_pack_id is not a string")
    if pack.decoder_version != DECODER_VERSION:
        failures.append(
            f"decoder_version {pack.decoder_version!r} is not this decoder "
            f"({DECODER_VERSION})"
        )
    if pack.decision not in DECISIONS:
        failures.append(f"unknown decision {pack.decision!r}")
    if not isinstance(pack.include_private, bool):
        failures.append("include_private is not a boolean")
    if not isinstance(pack.metadata, dict):
        failures.append("metadata is not an object")
    else:
        failures.extend(_json_object_key_errors(pack.metadata, "metadata"))
    if pack.guards is not None:
        if not isinstance(pack.guards, GuardSnapshot):
            failures.append("guards is not a GuardSnapshot")
        else:
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
    if isinstance(pack.receipts, (str, bytes)) or not isinstance(pack.receipts, (list, tuple)):
        failures.append("receipts is not an array")
    else:
        for index, receipt in enumerate(pack.receipts):
            if not isinstance(receipt, Receipt):
                failures.append(f"receipts[{index}] is not a Receipt")
                continue
            if not isinstance(receipt.step, str):
                failures.append(f"receipts[{index}].step is not a string")
            if not isinstance(receipt.ok, bool):
                failures.append(f"receipts[{index}].ok is not a boolean")
            if not _real_number(receipt.timestamp):
                failures.append(f"receipts[{index}].timestamp is not a number")
            elif not _finite(receipt.timestamp):
                failures.append(f"receipts[{index}].timestamp is not finite")
            for field_name in ("kappa", "reconstruction_error", "delta_mdl_bits"):
                value = getattr(receipt, field_name)
                if value is None:
                    continue
                if not _real_number(value):
                    failures.append(f"receipts[{index}].{field_name} is not a number")
                elif not _finite(value):
                    failures.append(f"receipts[{index}].{field_name} is not finite")
            if not isinstance(receipt.metadata, dict) or isinstance(
                receipt.metadata, (str, bytes)
            ):
                failures.append(f"receipts[{index}].metadata is not an object")
            else:
                failures.extend(
                    _json_object_key_errors(
                        receipt.metadata, f"receipts[{index}].metadata"
                    )
                )
    failures.extend(_evidence_map_errors(pack.evidence))
    if not isinstance(pack.codebook, dict):
        failures.append("codebook is not an object")
    else:
        for key, value in pack.codebook.items():
            if not _integral_code(key) or not _integral_code(value):
                failures.append("codebook entries must be integer-to-integer")
                break
    if not isinstance(pack.aliases, dict):
        failures.append("aliases is not an object")
    else:
        for source, mapping in pack.aliases.items():
            if not isinstance(source, str) or not source.strip():
                failures.append("alias source key is not a non-empty string")
            elif source != LEGACY_ALIAS_KEY and not SHA256_HEX.match(source):
                failures.append("alias source key is not a pack checksum")
            if not isinstance(mapping, dict):
                failures.append("alias table is not an object")
                continue
            for key, value in mapping.items():
                if not _integral_code(key) or not _integral_code(value):
                    failures.append("alias entries must be integer-to-integer")
                    break
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
    if not _real_number(pack.mdl_bits):
        failures.append("mdl_bits is not a number")
    elif not _finite(pack.mdl_bits):
        failures.append("mdl_bits is not finite")
    if not _real_number(pack.timestamp):
        failures.append("timestamp is not a number")
    elif not _finite(pack.timestamp):
        failures.append("timestamp is not finite")
    if not _is_array(pack.symbols):
        failures.append("symbols is not an array")
        return (len(failures) == 0, failures)
    for index, symbol in enumerate(pack.symbols):
        if not isinstance(symbol, Symbol):
            failures.append(f"symbols[{index}] is not a Symbol")
            continue
        if not _integral_code(symbol.class_id):
            failures.append(f"class {symbol.class_id} class_id is not an integer")
        if not _integral_code(symbol.code):
            failures.append(f"class {symbol.class_id} code is not an integer")
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
        if not isinstance(symbol.metadata, dict) or isinstance(symbol.metadata, (str, bytes)):
            failures.append(f"class {symbol.class_id} metadata is not an object")
        else:
            failures.extend(
                _json_object_key_errors(
                    symbol.metadata, f"class {symbol.class_id} metadata"
                )
            )
        proto = symbol.proto_embedding
        if isinstance(proto, (str, bytes)) or not isinstance(proto, (list, tuple)):
            failures.append(f"class {symbol.class_id} proto_embedding is not an array")
        else:
            for index, value in enumerate(proto):
                if not _real_number(value):
                    failures.append(
                        f"class {symbol.class_id} proto_embedding[{index}] is not a number"
                    )
                elif not _finite(value):
                    failures.append(
                        f"class {symbol.class_id} proto_embedding[{index}] is not finite"
                    )
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
    if not isinstance(guards, GuardSnapshot):
        return False
    return all(getattr(guards, name) is True for name in _GUARD_PASS_FLAGS)


def _is_array(value: object) -> bool:
    return not isinstance(value, (str, bytes)) and isinstance(value, (list, tuple))


def _real_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _integral_code(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _unique(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out

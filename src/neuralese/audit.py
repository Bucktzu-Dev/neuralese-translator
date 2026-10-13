"""Fail-closed certification of a SymbolPack."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional

from neuralese.aliases import has_alias_cycle
from neuralese.contracts import (
    CERTIFICATE_AUTHORITY,
    CERTIFICATE_DECISION_RESEARCH_FAIL,
    CERTIFICATE_DECISION_UNRESOLVED,
    CERTIFICATE_SCHEMA_V02,
    DIGEST_PAIRING_IDENTITY_1,
    DIGEST_PAIRING_INCONSISTENT,
    DIGEST_PAIRING_UNBOUND,
    DIGEST_PAIRING_UNKNOWN,
    AuditCertificate,
    SymbolPack,
    classify_digest_pairing,
    digest_does_not_bind_for,
    iter_live_symbols,
)
from neuralese.unfold import unfold_failures

_UNGLOSSED_PREFIX = "[unglossed:"


def certify(
    pack: SymbolPack,
    *,
    require_gloss: bool = True,
    tau_residual: float = 0.55,
) -> AuditCertificate:
    failures: List[str] = []

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
            failures.append(f"symbol class {symbol.class_id} code {symbol.code} missing from codebook")
        elif mapped != symbol.class_id:
            addressable = False
            failures.append(
                f"code {symbol.code} codebook class {mapped} != symbol class {symbol.class_id}"
            )

    if has_alias_cycle(pack.aliases):
        addressable = False
        failures.append("alias map contains a cycle")
    for old, new in pack.aliases.items():
        resolved, _ = pack.resolve_code(int(old))
        if pack.symbol_by_code(resolved) is None:
            addressable = False
            failures.append(f"alias {old}->{new} does not resolve to a symbol")

    unfoldable = True
    for symbol in iter_live_symbols(pack):
        if not symbol.observation_ids:
            unfoldable = False
            failures.append(f"class {symbol.class_id} has no observation_ids")

    expected = pack.compute_checksum()
    gloss_bound = expected == pack.checksum and bool(pack.checksum)
    if not pack.checksum:
        gloss_bound = False
        failures.append("pack is unsealed (empty checksum)")
    elif expected != pack.checksum:
        gloss_bound = False
        failures.append("checksum mismatch: gloss/pack mutated after sealing")
    if require_gloss:
        for symbol in iter_live_symbols(pack):
            definition = (symbol.definition or "").strip()
            if not definition:
                gloss_bound = False
                failures.append(f"class {symbol.class_id} has no bound English gloss")

    residual_ok = _finite(pack.reconstruction_error) and pack.reconstruction_error <= tau_residual
    if not residual_ok:
        failures.append(
            f"reconstruction_error {pack.reconstruction_error:.6f} exceeds tau_residual {tau_residual}"
        )

    reservoir_unfold, reservoir_residual, reservoir_details = unfold_failures(pack)
    if reservoir_unfold:
        unfoldable = False
        failures.extend(reservoir_unfold)
    if reservoir_residual:
        residual_ok = False
        failures.extend(reservoir_residual)

    pairing = classify_digest_pairing(pack.digest_profile, pack.identity_digest)
    identity_bound = False
    if pairing.pairing == DIGEST_PAIRING_INCONSISTENT:
        failures.append(
            "inconsistent digest_profile and identity_digest; "
            "identity digest is unverified"
        )
    elif pairing.pairing == DIGEST_PAIRING_UNKNOWN:
        failures.append(f"unknown digest_profile {pairing.profile!r}")
    elif pairing.pairing == DIGEST_PAIRING_IDENTITY_1:
        if not pairing.identity_digest_present:
            failures.append(
                "identity-1 digest_profile is missing identity_digest"
            )
        else:
            identity_bound = pairing.identity_digest == pack.compute_identity_digest()
            if not identity_bound:
                failures.append("identity digest mismatch")

    structural_passed = addressable and unfoldable and gloss_bound and residual_ok
    # Unbound: passed is the four structural gates. identity_bound is not
    # a pass condition and is not reported true. identity-1: passed is
    # those four gates AND identity_bound. Inconsistent, unknown, and
    # identity-1 without a matching digest cannot be made true by the
    # four gates. Numerical and semantic gates are not implemented, so
    # they cannot promote this to RESEARCH_PASS. A pass stays UNRESOLVED.
    # A miss is RESEARCH_FAIL.
    if pairing.pairing == DIGEST_PAIRING_UNBOUND:
        passed = structural_passed
    elif pairing.pairing == DIGEST_PAIRING_IDENTITY_1 and identity_bound:
        passed = structural_passed
    else:
        passed = False
    if passed:
        decision = CERTIFICATE_DECISION_UNRESOLVED
    else:
        decision = CERTIFICATE_DECISION_RESEARCH_FAIL
    live = list(iter_live_symbols(pack))
    return AuditCertificate(
        pack_id=pack.pack_id,
        pack_checksum=pack.checksum,
        expected_checksum=expected,
        passed=passed,
        addressable=addressable,
        unfoldable=unfoldable,
        gloss_bound=gloss_bound,
        residual_ok=residual_ok,
        fail_closed=True,
        failures=failures,
        timestamp=time.time(),
        details={
            "n_symbols": len(pack.symbols),
            "n_live": len(live),
            "n_quarantined": sum(1 for s in pack.symbols if s.quarantined),
            "n_aliases": len(pack.aliases),
            "reconstruction_error": pack.reconstruction_error,
            "tau_residual": tau_residual,
            "require_gloss": require_gloss,
            "mdl_bits": pack.mdl_bits,
            **reservoir_details,
        },
        schema_version=CERTIFICATE_SCHEMA_V02,
        package_version=_package_version(),
        authority=CERTIFICATE_AUTHORITY,
        decision=decision,
        evidence=_evidence(
            addressable=addressable,
            unfoldable=unfoldable,
            gloss_bound=gloss_bound,
            residual_ok=residual_ok,
            failures=failures,
            identity_bound=identity_bound,
            digest_does_not_bind=digest_does_not_bind_for(pairing),
            digest_profile=pairing.profile,
            identity_digest=pairing.identity_digest,
            digest_pairing=pairing.pairing,
        ),
        identity_bound=identity_bound,
        digest_profile=pairing.profile,
        identity_digest=pairing.identity_digest,
    )


def decodability_report(
    pack: SymbolPack,
    topic_labels: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Summarize a sealed pack without applying a pass/fail threshold.

    ``gloss_coverage`` is the fraction of live symbols whose definition is
    non-empty and is not an ``[unglossed: ...]`` marker. Quarantined symbols
    are left out of that fraction. ``unfold_failures`` lists reservoir
    mismatches from ``unfold_failures``; residual-threshold failures stay on
    ``certify``. ``decision`` is the learn-time metadata value, or ``None``.

    ``topic_labels`` is an optional diagnostic map from observation id to a
    corpus topic. When it is passed, the report gains ``cluster_purity``.
    Those labels are not a certify gate and were not an input to the embedding.
    """
    unfold, _residual_failures, _details = unfold_failures(pack)
    live = list(iter_live_symbols(pack))
    covered = sum(1 for symbol in live if _gloss_covered(symbol.definition))
    coverage = float(covered / len(live)) if live else 0.0
    decision = _learn_decision(pack)
    report: Dict[str, Any] = {
        "n_live": len(live),
        "n_quarantined": sum(1 for symbol in pack.symbols if symbol.quarantined),
        "residual": float(pack.reconstruction_error),
        "reservoir_size": len(pack.observations),
        "unfold_failures": list(unfold),
        "gloss_coverage": coverage,
        "mdl_bits": float(pack.mdl_bits),
        "decision": decision,
    }
    if topic_labels is not None:
        report["cluster_purity"] = cluster_majority_diagnostic(pack, topic_labels)
    return report


def cluster_majority_diagnostic(pack: SymbolPack, labels: Mapping[str, str]) -> Dict[str, Any]:
    """Majority-topic purity for live symbols, using an external label map.

    Purity is ``majority_count / rows``. Rows with no label dilute the purity.
    On a count tie, the lexicographically later topic name is the majority.
    A purity under 0.5 sets ``mixed`` and sets ``label`` to ``mixed``. The
    majority topic string stays on ``majority_topic`` so the count can be
    checked, and it is not a claim that the cluster is that topic.
    """
    clusters: List[Dict[str, Any]] = []
    labeled_rows = 0
    unlabeled_rows = 0
    for symbol in pack.symbols:
        if symbol.quarantined or not symbol.observation_ids:
            continue
        counts: Dict[str, int] = {}
        unlabeled = 0
        for obs_id in symbol.observation_ids:
            topic = labels.get(obs_id)
            if not isinstance(topic, str) or not topic.strip():
                unlabeled += 1
                continue
            name = topic.strip()
            counts[name] = counts.get(name, 0) + 1
        rows = len(symbol.observation_ids)
        labeled_rows += rows - unlabeled
        unlabeled_rows += unlabeled
        if counts:
            topic, count = max(counts.items(), key=lambda item: (item[1], item[0]))
        else:
            topic, count = None, 0
        purity = float(count / rows) if rows else 0.0
        mixed = purity < 0.5
        clusters.append(
            {
                "code": int(symbol.code),
                "rows": rows,
                "majority_topic": topic,
                "majority_count": int(count),
                "purity": purity,
                "mixed": mixed,
                "label": "mixed" if mixed else topic,
            }
        )
    clusters.sort(key=lambda item: item["code"])
    if clusters:
        unweighted = float(sum(item["purity"] for item in clusters) / len(clusters))
        weight_rows = sum(item["rows"] for item in clusters)
        weight_hits = sum(item["majority_count"] for item in clusters)
        weighted = float(weight_hits / weight_rows) if weight_rows else 0.0
        worst = min(item["purity"] for item in clusters)
    else:
        unweighted = 0.0
        weighted = 0.0
        worst = 0.0
    return {
        "n_labeled": labeled_rows,
        "n_unlabeled": unlabeled_rows,
        "unweighted_mean": unweighted,
        "size_weighted": weighted,
        "worst_purity": worst,
        "clusters": clusters,
    }


def _gloss_covered(definition: Optional[str]) -> bool:
    text = (definition or "").strip()
    if not text:
        return False
    return not text.startswith(_UNGLOSSED_PREFIX)


def _learn_decision(pack: SymbolPack) -> Optional[str]:
    raw = pack.metadata.get("decision") if pack.metadata else None
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _package_version() -> str:
    """Informational copy. Imported lazily so audit can load during package init."""
    from neuralese import __version__

    return __version__


def _evidence(
    *,
    addressable: bool,
    unfoldable: bool,
    gloss_bound: bool,
    residual_ok: bool,
    failures: List[str],
    identity_bound: bool,
    digest_does_not_bind: tuple,
    digest_profile: Optional[str],
    identity_digest: Optional[str],
    digest_pairing: str,
) -> Dict[str, Any]:
    """Schema 0.2 evidence classes. Numerical and semantic stay unimplemented.

    identity_bound is reported beside gloss_bound. It is true only for a
    matching identity-1 digest. Unbound leaves it false.
    """
    return {
        "structural": {
            "addressable": addressable,
            "unfoldable": unfoldable,
            "gloss_bound": gloss_bound,
            "residual_ok": residual_ok,
            "identity_bound": identity_bound,
            "failures": list(failures),
            "digest_does_not_bind": list(digest_does_not_bind),
            "digest_profile": digest_profile,
            "identity_digest": identity_digest,
            "digest_pairing": digest_pairing,
        },
        "numerical": {
            "status": "NOT IMPLEMENTED",
            "decision": CERTIFICATE_DECISION_UNRESOLVED,
            "forbidden_reading": (
                "reservoir centroid residual, or gloss text, "
                "scored as a reconstructed activation"
            ),
        },
        "semantic": {
            "status": "NOT IMPLEMENTED",
            "decision": CERTIFICATE_DECISION_UNRESOLVED,
            "forbidden_inputs": [
                "fluency",
                "the confidence heuristic",
                "kappa cosine",
                "cluster purity",
                "cosine similarity",
            ],
        },
    }

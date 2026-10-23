"""Unfold a sealed pack back to the observations that minted each code."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from neuralese.contracts import Observation, Symbol, SymbolPack

PROTO_ATOL = 1e-5
RESIDUAL_ATOL = 1e-4


@dataclass
class UnfoldReport:
    code: int
    state: str
    english_bound: bool
    observations: List[Observation] = field(default_factory=list)
    missing_ids: List[str] = field(default_factory=list)
    class_id: Optional[int] = None
    resolved_code: Optional[int] = None
    prototype_l2: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "code": self.code,
            "resolved_code": self.resolved_code,
            "class_id": self.class_id,
            "state": self.state,
            "english_bound": self.english_bound,
            "prototype_l2": self.prototype_l2,
            "missing_ids": list(self.missing_ids),
            "observations": [obs.to_dict() for obs in self.observations],
        }


def unfold_code(pack: SymbolPack, code: int) -> UnfoldReport:
    """Resolve one code and return the reservoir rows that produced it."""
    resolved, aliased = pack.resolve_code(int(code))
    symbol = pack.symbol_by_code(resolved)
    if symbol is None:
        return UnfoldReport(
            code=int(code),
            state="unknown",
            english_bound=False,
        )

    found, missing = _collect(pack, symbol)
    if symbol.quarantined:
        state = "quarantined"
    elif aliased:
        state = "aliased"
    else:
        state = "ok"
    distance = _prototype_l2(symbol, found) if found and not missing else None
    definition = (symbol.definition or "").strip()
    return UnfoldReport(
        code=int(code),
        state=state,
        english_bound=bool(definition) and not symbol.quarantined,
        observations=found,
        missing_ids=missing,
        class_id=symbol.class_id,
        resolved_code=resolved,
        prototype_l2=distance,
    )


def recomputed_cluster_residual(pack: SymbolPack) -> Optional[float]:
    """Centroid residual from stored embeddings. None when the reservoir is incomplete.

    Symbols may use different embedding widths (a retired parent class can). The
    score is the Frobenius ratio over each symbol's own rows, which matches a
    single stacked residual when every row has the same width.
    """
    numerator = 0.0
    denominator = 0.0
    saw_row = False
    for symbol in pack.symbols:
        if not symbol.observation_ids:
            continue
        if not symbol.proto_embedding:
            return None
        proto = np.asarray(symbol.proto_embedding, dtype=np.float64)
        for obs_id in symbol.observation_ids:
            obs = pack.observations.get(obs_id)
            if obs is None or not obs.embedding:
                return None
            if len(obs.embedding) != proto.shape[0]:
                return None
            vector = np.asarray(obs.embedding, dtype=np.float64)
            delta = vector - proto
            numerator += float(np.dot(delta, delta))
            denominator += float(np.dot(vector, vector))
            saw_row = True
    if not saw_row or denominator <= 1e-12:
        return 0.0
    return float(np.sqrt(numerator / denominator))


def unfold_failures(pack: SymbolPack) -> tuple[List[str], List[str], Dict[str, object]]:
    """Return (unfold failures, residual-honesty failures, certificate details)."""
    unfold: List[str] = []
    residual_failures: List[str] = []
    for symbol in pack.symbols:
        if not symbol.observation_ids:
            continue
        found, missing = _collect(pack, symbol)
        for obs_id in missing:
            unfold.append(
                f"class {symbol.class_id} missing reservoir observation {obs_id}"
            )
        if missing:
            continue
        blocked = False
        for obs in found:
            if not obs.embedding:
                unfold.append(
                    f"class {symbol.class_id} observation {obs.observation_id} has no embedding"
                )
                blocked = True
                continue
            if symbol.proto_embedding and len(obs.embedding) != len(symbol.proto_embedding):
                unfold.append(
                    f"class {symbol.class_id} observation {obs.observation_id} "
                    f"embedding dim {len(obs.embedding)} != prototype dim {len(symbol.proto_embedding)}"
                )
                blocked = True
        if blocked:
            continue
        if not symbol.proto_embedding:
            unfold.append(f"class {symbol.class_id} has observation ids but no prototype")
            continue
        distance = _prototype_l2(symbol, found)
        if distance is None or distance > PROTO_ATOL:
            shown = float("inf") if distance is None else distance
            unfold.append(
                f"class {symbol.class_id} prototype does not match reservoir mean (l2={shown:.6f})"
            )

    recomputed = recomputed_cluster_residual(pack)
    if recomputed is not None and recomputed > float(pack.reconstruction_error) + RESIDUAL_ATOL:
        residual_failures.append(
            "reservoir cluster residual "
            f"{recomputed:.6f} exceeds sealed reconstruction_error {pack.reconstruction_error:.6f}"
        )
    details: Dict[str, object] = {
        "reservoir_size": len(pack.observations),
        "recomputed_cluster_residual": recomputed,
    }
    return unfold, residual_failures, details


def _collect(pack: SymbolPack, symbol: Symbol) -> tuple[List[Observation], List[str]]:
    found: List[Observation] = []
    missing: List[str] = []
    for obs_id in symbol.observation_ids:
        obs = pack.observations.get(obs_id)
        if obs is None:
            missing.append(obs_id)
        else:
            found.append(obs)
    return found, missing


def _prototype_l2(symbol: Symbol, observations: Sequence[Observation]) -> Optional[float]:
    if not observations or not symbol.proto_embedding:
        return None
    width = len(symbol.proto_embedding)
    rows = []
    for obs in observations:
        if len(obs.embedding) != width:
            return None
        rows.append(obs.embedding)
    if not rows:
        return None
    mean = np.mean(np.asarray(rows, dtype=np.float64), axis=0)
    proto = np.asarray(symbol.proto_embedding, dtype=np.float64)
    return float(np.linalg.norm(mean - proto))

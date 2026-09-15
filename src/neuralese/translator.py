"""Translate a neuralese code stream into English glosses."""
from __future__ import annotations

from typing import List, Optional, Sequence

from neuralese.audit import certify
from neuralese.contracts import (
    TRANSLATION_POLICIES,
    Gloss,
    SymbolPack,
    UncertifiedPackError,
)
from neuralese.energy import confidence_cap
from neuralese.gloss import UNGLOSSED


def translate_stream(
    pack: SymbolPack,
    codes: Sequence[int],
    *,
    policy: str = "default",
    require_certified: bool = True,
    require_gloss: bool = True,
    source_pack_checksum: Optional[str] = None,
    tau_residual: float = 0.55,
) -> List[Gloss]:
    """Map each code to a gloss. Never invent English for a missing symbol.

    By default translation is refused unless `certify(pack, policy=policy)` passes.
    Pass `require_gloss=False` to opt into translating packs that record
    `[unglossed]`. Empty or missing definitions still fail certification;
    the opt-in relaxes only that explicit sentinel. That still requires
    evidence and admission when `require_certified` is true. `integrity`
    may inspect a seal; it does not authorize emitting English, including
    when `require_certified` is false.
    Residual admission uses the caller `tau_residual` (default 0.55), never a
    threshold persisted in pack metadata.
    """
    if policy == "integrity" or policy not in TRANSLATION_POLICIES:
        raise ValueError(
            f"policy {policy!r} does not authorize translation; "
            f"use one of {TRANSLATION_POLICIES}"
        )
    if require_certified:
        certificate = certify(
            pack,
            policy=policy,
            require_gloss=require_gloss,
            tau_residual=tau_residual,
        )
        if not certificate.passed:
            raise UncertifiedPackError(certificate)

    glosses: List[Gloss] = []
    for raw in codes:
        code = int(raw)
        resolved, aliased = pack.resolve_code(code, source_pack_checksum=source_pack_checksum)
        symbol = pack.symbol_by_code(resolved)
        if symbol is None:
            glosses.append(
                Gloss(
                    code=code,
                    english=f"[undecodable: no symbol for code {code}]",
                    confidence=0.0,
                    state="unknown",
                    pack_checksum=pack.checksum,
                    class_id=None,
                    resolved_code=None,
                    observation_ids=[],
                )
            )
            continue

        conf = confidence_cap(symbol.confidence, pack.reconstruction_error)
        if symbol.quarantined:
            glosses.append(
                Gloss(
                    code=code,
                    english=(
                        f"[quarantined: symbol {resolved} is not certified unfoldable]"
                    ),
                    confidence=0.0,
                    state="quarantined",
                    pack_checksum=pack.checksum,
                    class_id=symbol.class_id,
                    resolved_code=resolved,
                    observation_ids=(
                        list(symbol.observation_ids)
                        if isinstance(symbol.observation_ids, (list, tuple))
                        else []
                    ),
                )
            )
            continue

        definition = (symbol.definition or "").strip()
        if not definition or definition == UNGLOSSED:
            english = f"[unglossed: symbol {resolved} has no bound English]"
            conf = 0.0
        else:
            english = definition

        state = "aliased" if aliased else "ok"
        glosses.append(
            Gloss(
                code=code,
                english=english,
                confidence=conf,
                state=state,
                pack_checksum=pack.checksum,
                class_id=symbol.class_id,
                resolved_code=resolved,
                observation_ids=(
                    list(symbol.observation_ids)
                    if isinstance(symbol.observation_ids, (list, tuple))
                    else []
                ),
            )
        )
    return glosses

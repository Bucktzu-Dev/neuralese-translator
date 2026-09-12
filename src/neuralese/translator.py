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


def translate_stream(
    pack: SymbolPack,
    codes: Sequence[int],
    *,
    policy: str = "default",
    require_certified: bool = True,
    source_pack_checksum: Optional[str] = None,
) -> List[Gloss]:
    """Map each code to a gloss. Never invent English for a missing symbol.

    By default translation is refused unless `certify(pack, policy=policy)` passes.
    `integrity` may inspect a seal; it does not authorize emitting English.
    """
    if require_certified:
        if policy not in TRANSLATION_POLICIES:
            raise ValueError(
                f"policy {policy!r} does not authorize translation; "
                f"use one of {TRANSLATION_POLICIES}"
            )
        certificate = certify(pack, policy=policy)
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
                    observation_ids=list(symbol.observation_ids),
                )
            )
            continue

        definition = (symbol.definition or "").strip()
        if not definition:
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
                observation_ids=list(symbol.observation_ids),
            )
        )
    return glosses

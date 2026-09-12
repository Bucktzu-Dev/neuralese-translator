"""Translate a neuralese code stream into English glosses."""
from __future__ import annotations

from typing import List, Sequence

from neuralese.contracts import Gloss, SymbolPack
from neuralese.energy import confidence_cap


def translate_stream(pack: SymbolPack, codes: Sequence[int]) -> List[Gloss]:
    """Map each code to a gloss. Never invent English for a missing symbol."""
    glosses: List[Gloss] = []
    for raw in codes:
        code = int(raw)
        resolved, aliased = pack.resolve_code(code)
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

"""Lossless integer remaps when a codebook evolves."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

from neuralese.contracts import (
    LEGACY_ALIAS_KEY,
    AliasTables,
    normalize_aliases,
    select_alias_table,
)


def follow_aliases(
    code: int,
    aliases: Dict[int, int],
    *,
    max_hops: Optional[int] = None,
) -> int:
    # A finite table's longest acyclic chain is len(aliases). A fixed 64-hop
    # cap would truncate a valid longer chain to an intermediate code.
    bound = len(aliases) if max_hops is None else max_hops
    seen: set[int] = set()
    current = int(code)
    hops = 0
    while current in aliases and hops < bound:
        if current in seen:
            return current
        seen.add(current)
        current = int(aliases[current])
        hops += 1
    return current


def rewrite_stream(
    tokens: Sequence[int],
    aliases: Union[Dict[int, int], AliasTables],
    *,
    source_pack_checksum: Optional[str] = None,
) -> List[int]:
    """Rewrite a token stream, following alias hops until a current code."""
    tables = normalize_aliases(aliases)
    if not tables:
        return [int(token) for token in tokens]
    if source_pack_checksum is None and LEGACY_ALIAS_KEY not in tables:
        raise ValueError(
            "versioned alias tables require source_pack_checksum; "
            "refusing to merge unrelated sources"
        )
    table = select_alias_table(tables, source_pack_checksum)
    return [follow_aliases(int(token), table) for token in tokens]


def has_alias_cycle(aliases: Union[Dict[int, int], AliasTables]) -> bool:
    tables = normalize_aliases(aliases)
    for mapping in tables.values():
        for start in mapping:
            seen: set[int] = set()
            current = int(start)
            while current in mapping:
                if current in seen:
                    return True
                seen.add(current)
                current = int(mapping[current])
    return False

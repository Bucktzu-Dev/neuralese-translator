"""Lossless integer remaps when a codebook evolves."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

from neuralese.contracts import AliasTables, normalize_aliases


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
    if not aliases:
        return [int(token) for token in tokens]
    nested = isinstance(next(iter(aliases.values())), dict)
    if nested:
        tables = normalize_aliases(aliases)
        if source_pack_checksum is not None:
            table = dict(tables.get(source_pack_checksum, {}))
        elif "legacy" in tables:
            # Match SymbolPack.alias_table(): no source selects only legacy
            # and ignores versioned parent tables.
            table = dict(tables.get("legacy", {}))
        else:
            raise ValueError(
                "versioned alias tables require source_pack_checksum; "
                "refusing to merge unrelated sources"
            )
    else:
        table = {int(k): int(v) for k, v in aliases.items()}
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

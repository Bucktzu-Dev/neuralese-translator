"""Lossless integer remaps when a codebook evolves."""
from __future__ import annotations

from typing import Dict, List, Sequence, Union

from neuralese.contracts import AliasTables, normalize_aliases


def follow_aliases(code: int, aliases: Dict[int, int], *, max_hops: int = 64) -> int:
    seen: set[int] = set()
    current = int(code)
    hops = 0
    while current in aliases and hops < max_hops:
        if current in seen:
            return current
        seen.add(current)
        current = int(aliases[current])
        hops += 1
    return current


def rewrite_stream(
    tokens: Sequence[int],
    aliases: Union[Dict[int, int], AliasTables],
) -> List[int]:
    """Rewrite a token stream, following alias hops until a current code."""
    if aliases and isinstance(next(iter(aliases.values())), dict):
        table: Dict[int, int] = {}
        for mapping in normalize_aliases(aliases).values():
            table.update(mapping)
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

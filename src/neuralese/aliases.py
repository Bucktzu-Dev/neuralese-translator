"""Lossless integer remaps when a codebook evolves."""
from __future__ import annotations

from typing import Dict, List, Sequence


def rewrite_stream(tokens: Sequence[int], aliases: Dict[int, int]) -> List[int]:
    """Rewrite a token stream using old_code -> new_code aliases."""
    return [int(aliases.get(int(token), token)) for token in tokens]


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


def has_alias_cycle(aliases: Dict[int, int]) -> bool:
    for start in aliases:
        seen: set[int] = set()
        current = int(start)
        while current in aliases:
            if current in seen:
                return True
            seen.add(current)
            current = int(aliases[current])
    return False

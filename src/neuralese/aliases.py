"""Lossless integer remaps when a codebook evolves."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Union

from neuralese.contracts import (
    LEGACY_ALIAS_KEY,
    AliasTables,
    normalize_aliases,
    select_alias_table,
)

_CYCLE = "alias map contains a cycle"
AMBIGUOUS_HISTORICAL_MAPPING = "historical mapping is ambiguous"


def resolve_alias_table(
    aliases: Dict[int, int],
    *,
    current_codes: Optional[Set[int]] = None,
) -> Dict[int, int]:
    """Map every alias start to its terminal code in linear time.

    Each start walking the remainder of its chain independently is quadratic
    on a valid acyclic table. Certification and translation consume untrusted
    packs, so a long chain would otherwise become a denial of service.
    Shared suffixes are reused; a cycle raises ValueError.
    When `current_codes` is provided (pack codebook / live symbol codes), an
    alias source that is also a current code is ambiguous without
    `(pack_checksum, code)` identity: `{0: 1}` is refused instead of silently
    keeping live `0`. `{7: 0}` still rewrites historical `7`. Standalone
    resolution with `current_codes=None` still follows every table key.
    """
    live = current_codes
    if live is not None:
        for start in aliases:
            current = int(start)
            if current in live:
                raise ValueError(AMBIGUOUS_HISTORICAL_MAPPING)
    terminals: Dict[int, int] = {}
    for start in aliases:
        current = int(start)
        if current in terminals:
            continue
        path: List[int] = []
        seen: Dict[int, int] = {}
        while current in aliases:
            if current in terminals:
                break
            if current in seen:
                raise ValueError(_CYCLE)
            seen[current] = len(path)
            path.append(current)
            current = int(aliases[current])
        terminal = terminals.get(current, current)
        for node in path:
            terminals[node] = terminal
    return terminals


def follow_aliases(
    code: int,
    aliases: Dict[int, int],
    *,
    max_hops: Optional[int] = None,
    terminals: Optional[Dict[int, int]] = None,
    current_codes: Optional[Set[int]] = None,
) -> int:
    # A finite table's longest acyclic chain is len(aliases). The previous
    # per-start walk floored max_hops at len(aliases)+1 so a caller cap could
    # not hide a cycle or truncate a valid longer chain. Table-wide
    # memoization preserves that: every start is resolved once, cycles still
    # raise, and max_hops is only type-checked for API compatibility.
    if max_hops is not None:
        int(max_hops)
    if terminals is None:
        terminals = resolve_alias_table(aliases, current_codes=current_codes)
    return terminals.get(int(code), int(code))


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
    terminals = resolve_alias_table(table)
    return [terminals.get(int(token), int(token)) for token in tokens]


def has_alias_cycle(
    aliases: Union[Dict[int, int], AliasTables],
    *,
    current_codes: Optional[Set[int]] = None,
) -> bool:
    tables = normalize_aliases(aliases)
    for mapping in tables.values():
        try:
            resolve_alias_table(mapping, current_codes=current_codes)
        except ValueError as copilot_exc:
            if str(copilot_exc) == _CYCLE:
                return True
            raise
    return False

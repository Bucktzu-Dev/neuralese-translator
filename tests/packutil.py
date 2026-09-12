"""Shared pack builders for tests."""
from __future__ import annotations

from neuralese.contracts import Symbol, SymbolPack


def make_pack(**overrides) -> SymbolPack:
    symbols = overrides.pop("symbols", None)
    if symbols is None:
        symbols = [
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0, 0.0, 0.0],
                observation_ids=["obs-hello"],
                definition="Symbol for hello.",
                examples=["hello"],
                confidence=0.8,
            ),
            Symbol(
                class_id=1,
                code=1,
                proto_embedding=[0.0, 1.0, 0.0],
                observation_ids=["obs-audit"],
                definition="Symbol for audit.",
                examples=["audit"],
                confidence=0.7,
            ),
        ]
    aliases = overrides.pop("aliases", {})
    reconstruction_error = overrides.pop("reconstruction_error", 0.08)
    pack_id = overrides.pop("pack_id", "pack-test")
    parent = overrides.pop("parent_pack_id", None)
    checksum = overrides.pop("checksum", "")
    pack = SymbolPack(
        pack_id=pack_id,
        symbols=symbols,
        codebook=overrides.pop("codebook", {s.code: s.class_id for s in symbols}),
        aliases=aliases,
        reconstruction_error=reconstruction_error,
        checksum=checksum,
        parent_pack_id=parent,
        mdl_bits=overrides.pop("mdl_bits", 12.0),
        metadata=overrides.pop("metadata", {}),
    )
    if overrides:
        raise AssertionError(f"unexpected overrides {overrides}")
    if not pack.checksum:
        pack.seal()
    return pack

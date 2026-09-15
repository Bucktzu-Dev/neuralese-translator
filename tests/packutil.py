"""Shared pack builders for tests."""
from __future__ import annotations

from neuralese.contracts import (
    GuardSnapshot,
    Symbol,
    SymbolPack,
    observation_content_hash,
)


def passing_guards(**overrides) -> GuardSnapshot:
    data = dict(
        kappa_avg=0.8,
        reconstruction_error=0.08,
        delta_mdl=0.0,
        min_survival=1.0,
        pass_kappa=True,
        pass_residual=True,
        pass_mdl=True,
        pass_persist=True,
        pass_compat=True,
        pass_all=True,
    )
    data.update(overrides)
    return GuardSnapshot(**data)


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
                example_hashes=[],
                confidence=0.8,
            ),
            Symbol(
                class_id=1,
                code=1,
                proto_embedding=[0.0, 1.0, 0.0],
                observation_ids=["obs-audit"],
                definition="Symbol for audit.",
                example_hashes=[],
                confidence=0.7,
            ),
        ]
    evidence = overrides.pop("evidence", None)
    if evidence is None:
        evidence = {}
        for symbol in symbols:
            for obs_id in symbol.observation_ids:
                evidence.setdefault(obs_id, observation_content_hash(obs_id))
    aliases = overrides.pop("aliases", {})
    reconstruction_error = overrides.pop("reconstruction_error", 0.08)
    pack_id = overrides.pop("pack_id", "pack-test")
    parent = overrides.pop("parent_pack_id", None)
    parent_checksum = overrides.pop("parent_checksum", None)
    checksum = overrides.pop("checksum", "")
    decision = overrides.pop("decision", "accept")
    guards = overrides.pop("guards", passing_guards(reconstruction_error=reconstruction_error))
    pack = SymbolPack(
        pack_id=pack_id,
        symbols=symbols,
        codebook=overrides.pop("codebook", {s.code: s.class_id for s in symbols}),
        aliases=aliases,
        reconstruction_error=reconstruction_error,
        checksum=checksum,
        parent_pack_id=parent,
        parent_checksum=parent_checksum,
        mdl_bits=overrides.pop("mdl_bits", 12.0),
        metadata=overrides.pop("metadata", {"decision": decision}),
        evidence=evidence,
        decision=decision,
        guards=guards,
        receipts=overrides.pop("receipts", []),
        decoder_version=overrides.pop("decoder_version", None) or "0.1.1",
        include_private=overrides.pop("include_private", False),
    )
    if overrides:
        raise AssertionError(f"unexpected overrides {overrides}")
    if not pack.checksum:
        pack.seal()
    return pack

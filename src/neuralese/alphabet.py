"""Alphabet reconstruction: observations → sealed SymbolPack."""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from neuralese.adapters import ensure_embedding, stack_embeddings
from neuralese.clustering import cluster_survival, kmeans
from neuralese.contracts import (
    DECODER_VERSION,
    GuardSnapshot,
    Observation,
    Receipt,
    Symbol,
    SymbolPack,
    example_hash,
)
from neuralese.energy import cosine_similarity
from neuralese.factorization import reconstruction_error, svd_factors
from neuralese.gloss import UNGLOSSED, learn_definition
from neuralese.receipts import create_receipt


@dataclass
class LearnConfig:
    n_symbols: int = 8
    min_cluster_size: int = 2
    tau_kappa: float = 0.35
    tau_residual: float = 0.55
    tau_persist: float = 0.80
    svd_rank: Optional[int] = None
    seed: int = 0
    match_threshold: float = 0.55
    include_private: bool = False


def mdl_bits(n_symbols: int, residual: float, dim: int, n_obs: int, gloss_chars: int) -> float:
    vocab = max(n_symbols, 1) * math.log2(max(n_symbols, 2))
    residual_bits = residual * dim * n_obs
    gloss_bits = gloss_chars * 8.0 / 32.0
    return float(vocab + residual_bits + gloss_bits)


def learn_pack(
    observations: Sequence[Observation],
    *,
    config: Optional[LearnConfig] = None,
    previous: Optional[SymbolPack] = None,
    llm_client: object = None,
) -> SymbolPack:
    cfg = config or LearnConfig()
    if not observations:
        raise ValueError("learn_pack requires at least one observation")
    seen_ids: set[str] = set()
    duplicates: List[str] = []
    for obs in observations:
        oid = str(obs.observation_id)
        if oid in seen_ids:
            if oid not in duplicates:
                duplicates.append(oid)
        else:
            seen_ids.add(oid)
    if duplicates:
        raise ValueError(
            "duplicate observation_id values are not a fold path: "
            + ", ".join(repr(x) for x in duplicates)
        )
    rows = [ensure_embedding(Observation.from_dict(o.to_dict())) for o in observations]
    X = stack_embeddings(rows)

    rank = cfg.svd_rank or min(max(cfg.n_symbols, 1), X.shape[0], X.shape[1])
    _, _, svd_residual = svd_factors(X, rank)

    rng = np.random.default_rng(cfg.seed)
    k = max(1, min(cfg.n_symbols, X.shape[0]))
    labels, centroids = kmeans(X, k, rng=rng)
    k = int(centroids.shape[0])

    old_protos = None
    if previous and previous.symbols:
        old_protos = np.asarray([s.proto_embedding for s in previous.symbols], dtype=np.float64)
    survivals = cluster_survival(
        old_protos if old_protos is not None else np.zeros((0, X.shape[1])),
        centroids,
        threshold=cfg.match_threshold,
    )

    symbols: List[Symbol] = []
    codebook: Dict[int, int] = {}
    for class_id in range(k):
        member_idx = np.where(labels == class_id)[0]
        member_obs = [rows[i] for i in member_idx]
        proto = centroids[class_id].tolist() if member_idx.size else [0.0] * X.shape[1]
        if member_idx.size:
            kappas = [cosine_similarity(rows[i].embedding, proto) for i in member_idx]
            kappa = float(sum(kappas) / len(kappas))
        else:
            kappa = 0.0
        survival = float(survivals[class_id]) if class_id < len(survivals) else 1.0
        if previous is None:
            survival = 1.0
        gloss = learn_definition(
            member_obs,
            llm_client=llm_client,
            include_private=cfg.include_private,
        )
        quarantined = member_idx.size < cfg.min_cluster_size or kappa < cfg.tau_kappa
        definition = gloss["definition"] or None
        if not (definition or "").strip():
            definition = UNGLOSSED
        if quarantined and (not definition or definition.strip() == UNGLOSSED):
            definition = f"[quarantined class {class_id}]"
        unglossed = (definition or "").strip() == UNGLOSSED
        examples = list(gloss["examples"])
        hash_source = examples or [
            o.text for o in member_obs if o.text and o.text.strip()
        ][:8]
        symbol = Symbol(
            class_id=class_id,
            code=class_id,
            proto_embedding=proto,
            observation_ids=[o.observation_id for o in member_obs],
            definition=definition if definition else None,
            examples=examples if cfg.include_private else [],
            example_hashes=[example_hash(x) for x in hash_source],
            confidence=float(0.0 if quarantined or unglossed else gloss["confidence"]),
            quarantined=bool(quarantined),
            survival=survival,
            metadata={"kappa": kappa, "size": int(member_idx.size)},
        )
        symbols.append(symbol)
        codebook[class_id] = class_id

    # Prototype reconstruction (centroids) is the operational residual.
    X_hat = np.zeros_like(X)
    for i, label in enumerate(labels):
        X_hat[i] = centroids[int(label)]
    cluster_residual = reconstruction_error(X, X_hat)
    residual = max(float(cluster_residual), float(svd_residual) * 0.25)

    evidence = {row.observation_id: row.content_hash() for row in rows}

    aliases: Dict[str, Dict[int, int]] = {}
    if previous is not None:
        remap = _match_aliases(previous, symbols, threshold=cfg.match_threshold)
        if remap:
            aliases[previous.checksum] = remap

    gloss_chars = sum(len(s.definition or "") for s in symbols)
    bits = mdl_bits(len(symbols), residual, X.shape[1], X.shape[0], gloss_chars)
    if previous is None:
        delta_mdl = 0.0
    else:
        delta_mdl = bits - float(previous.mdl_bits)

    kappa_avg = float(np.mean([s.metadata.get("kappa", 0.0) for s in symbols])) if symbols else 0.0
    min_survival = float(min((s.survival for s in symbols), default=1.0))
    pass_kappa = kappa_avg >= cfg.tau_kappa
    pass_residual = residual <= cfg.tau_residual
    pass_mdl = delta_mdl <= 0.0
    pass_persist = min_survival >= cfg.tau_persist or previous is None
    pass_compat = not _alias_collision(aliases, codebook)
    pass_all = pass_kappa and pass_residual and pass_mdl and pass_persist and pass_compat

    if pass_all:
        decision = "accept"
    elif pass_residual and pass_compat:
        decision = "accept_provisional"
    else:
        decision = "reject"

    receipts: List[Receipt] = [
        create_receipt(
            "factorization",
            ok=svd_residual <= cfg.tau_residual,
            reconstruction_error=float(svd_residual),
            metadata={"rank": rank},
        ),
        create_receipt(
            "clustering",
            ok=min_survival >= cfg.tau_persist or previous is None,
            kappa=kappa_avg,
            metadata={"k": k, "min_survival": min_survival},
        ),
        create_receipt(
            "finalize",
            ok=decision != "reject",
            kappa=kappa_avg,
            reconstruction_error=residual,
            delta_mdl_bits=float(delta_mdl),
            metadata={"decision": decision},
        ),
    ]

    guards = GuardSnapshot(
        kappa_avg=kappa_avg,
        reconstruction_error=residual,
        delta_mdl=float(delta_mdl),
        min_survival=min_survival,
        pass_kappa=pass_kappa,
        pass_residual=pass_residual,
        pass_mdl=pass_mdl,
        pass_persist=pass_persist,
        pass_compat=pass_compat,
        pass_all=pass_all,
    )

    pack = SymbolPack(
        pack_id=str(uuid.uuid4()),
        symbols=symbols,
        codebook=codebook,
        aliases=aliases,
        reconstruction_error=residual,
        parent_pack_id=None if previous is None else previous.pack_id,
        parent_checksum=None if previous is None else previous.checksum,
        receipts=receipts,
        guards=guards,
        mdl_bits=bits,
        timestamp=time.time(),
        evidence=evidence,
        decoder_version=DECODER_VERSION,
        decision=decision,
        include_private=cfg.include_private,
        metadata={
            "n_observations": len(rows),
            "decision": decision,
            "status": "draft" if decision == "reject" else "admitted",
            "config": {
                "n_symbols": cfg.n_symbols,
                "tau_kappa": cfg.tau_kappa,
                "tau_residual": cfg.tau_residual,
                "tau_persist": cfg.tau_persist,
            },
        },
    )
    return pack.seal()


def _match_aliases(
    previous: SymbolPack,
    symbols: Sequence[Symbol],
    *,
    threshold: float,
) -> Dict[int, int]:
    aliases: Dict[int, int] = {}
    used: set[int] = set()
    for old in previous.symbols:
        best_code = None
        best_sim = threshold
        for new in symbols:
            if new.class_id in used:
                continue
            sim = cosine_similarity(old.proto_embedding, new.proto_embedding)
            if sim > best_sim:
                best_sim = sim
                best_code = new.code
        if best_code is not None:
            used.add(next(s.class_id for s in symbols if s.code == best_code))
            if old.code != best_code:
                aliases[int(old.code)] = int(best_code)
    return aliases


def _alias_collision(aliases: Dict[str, Dict[int, int]], codebook: Dict[int, int]) -> bool:
    for mapping in aliases.values():
        if any(t not in codebook for t in mapping.values()):
            return True
    return False

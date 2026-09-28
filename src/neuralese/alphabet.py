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
from neuralese.contracts import GuardSnapshot, Observation, Receipt, Symbol, SymbolPack
from neuralese.encode import ENCODER_CHAR_TRIGRAM, ENCODER_WORD_SENTENCE_SVD, word_sentence_svd
from neuralese.energy import cosine_similarity
from neuralese.factorization import reconstruction_error, svd_factors
from neuralese.gloss import learn_definition
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
    mdl_exception: Optional[str] = None
    # char_trigram keeps the hashed default. word_sentence_svd is local TF-IDF SVD.
    encoder: str = ENCODER_CHAR_TRIGRAM


def _embed_rows(observations: Sequence[Observation], cfg: LearnConfig) -> List[Observation]:
    """Copy observations and fill missing embeddings.

    ``char_trigram`` is the per-row hash. ``word_sentence_svd`` is fit only to
    the texts that still need a vector, in file order. Rows that already carry
    an embedding are left unchanged, so toy fixtures keep their vectors.
    """
    rows = [Observation.from_dict(obs.to_dict()) for obs in observations]
    if cfg.encoder == ENCODER_CHAR_TRIGRAM:
        return [ensure_embedding(obs) for obs in rows]
    if cfg.encoder != ENCODER_WORD_SENTENCE_SVD:
        raise ValueError(
            f"unknown encoder {cfg.encoder!r}; expected {ENCODER_CHAR_TRIGRAM!r} "
            f"or {ENCODER_WORD_SENTENCE_SVD!r}"
        )
    missing = [index for index, obs in enumerate(rows) if not obs.embedding]
    if not missing:
        return rows
    texts: List[str] = []
    for index in missing:
        text = rows[index].text
        if not text:
            raise ValueError(
                f"observation {rows[index].observation_id!r} has neither embedding nor text"
            )
        texts.append(text)
    vectors = word_sentence_svd(texts, rank=max(1, cfg.n_symbols))
    for index, vector in zip(missing, vectors):
        rows[index].embedding = vector
    return rows


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
    rows = _embed_rows(observations, cfg)
    seen_ids = set()
    for obs in rows:
        if obs.observation_id in seen_ids:
            raise ValueError(f"duplicate observation_id {obs.observation_id!r}")
        seen_ids.add(obs.observation_id)
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
        gloss = learn_definition(member_obs, llm_client=llm_client)
        quarantined = member_idx.size < cfg.min_cluster_size or kappa < cfg.tau_kappa
        definition = gloss["definition"] or None
        if not definition and not quarantined:
            definition = f"[unglossed: class {class_id}]"
        if quarantined and not definition:
            definition = f"[quarantined class {class_id}]"
        symbol = Symbol(
            class_id=class_id,
            code=class_id,
            proto_embedding=proto,
            observation_ids=[o.observation_id for o in member_obs],
            definition=definition if definition else None,
            examples=list(gloss["examples"]),
            confidence=float(gloss["confidence"] if not quarantined else 0.0),
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

    learned = list(symbols)
    reservoir = {obs.observation_id: Observation.from_dict(obs.to_dict()) for obs in rows}
    aliases: Dict[int, int] = {}
    if previous is not None:
        aliases, retired = _rebind_parent(
            previous,
            symbols,
            reservoir,
            threshold=cfg.match_threshold,
        )
        symbols.extend(retired)
    codebook = {symbol.code: symbol.class_id for symbol in symbols}

    gloss_chars = sum(len(s.definition or "") for s in symbols)
    bits = mdl_bits(len(symbols), residual, X.shape[1], X.shape[0], gloss_chars)
    if previous is None:
        delta_mdl = 0.0
    else:
        delta_mdl = bits - float(previous.mdl_bits)

    kappa_avg = float(np.mean([s.metadata.get("kappa", 0.0) for s in learned])) if learned else 0.0
    min_survival = float(min((s.survival for s in learned), default=1.0))
    pass_kappa = kappa_avg >= cfg.tau_kappa
    pass_residual = residual <= cfg.tau_residual
    pass_mdl = delta_mdl <= 0.0
    pass_persist = min_survival >= cfg.tau_persist or previous is None
    pass_compat = not _alias_collision(aliases, codebook) and _parent_codes_kept(
        previous, aliases, codebook
    )
    pass_all = pass_kappa and pass_residual and pass_mdl and pass_persist and pass_compat
    mdl_exception = (cfg.mdl_exception or "").strip() or None
    # An increase stays visible on the guard. A recorded exception waives only the reject.
    mdl_blocked = delta_mdl > 0.0 and mdl_exception is None

    if pass_all:
        decision = "accept"
    elif mdl_blocked or not pass_residual or not pass_compat:
        decision = "reject"
    else:
        decision = "accept_provisional"

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
            metadata={"decision": decision, **({"mdl_exception": mdl_exception} if mdl_exception else {})},
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
        receipts=receipts,
        guards=guards,
        mdl_bits=bits,
        observations=reservoir,
        timestamp=time.time(),
        metadata={
            "n_observations": len(rows),
            "decision": decision,
            **({"mdl_exception": mdl_exception} if mdl_exception else {}),
            "config": {
                "n_symbols": cfg.n_symbols,
                "tau_kappa": cfg.tau_kappa,
                "tau_residual": cfg.tau_residual,
                "tau_persist": cfg.tau_persist,
                "encoder": cfg.encoder,
            },
        },
    )
    return pack.seal()


def _match_successors(
    previous: SymbolPack,
    symbols: Sequence[Symbol],
    threshold: float,
) -> Dict[int, Symbol]:
    """Greedy one-to-one match from parent symbol identity to a child symbol."""
    used: set[int] = set()
    matched: Dict[int, Symbol] = {}
    for old in previous.symbols:
        best: Optional[Symbol] = None
        best_sim = threshold
        for new in symbols:
            if id(new) in used:
                continue
            if len(old.proto_embedding) != len(new.proto_embedding) or not old.proto_embedding:
                continue
            sim = cosine_similarity(old.proto_embedding, new.proto_embedding)
            if sim > best_sim:
                best_sim = sim
                best = new
        if best is not None:
            used.add(id(best))
            matched[id(old)] = best
    return matched


def _rebind_parent(
    previous: SymbolPack,
    symbols: List[Symbol],
    reservoir: Dict[str, Observation],
    *,
    threshold: float,
) -> tuple[Dict[int, int], List[Symbol]]:
    """Keep every parent code. Matched prototypes inherit it; the rest stay quarantined.

    New clusters take a fresh integer rather than a code that already names a
    parent symbol or an alias. Deleting the integer is not an inverse.
    """
    matched = _match_successors(previous, symbols, threshold)
    reserved = {int(code) for code in previous.codebook}
    reserved.update(int(key) for key in previous.aliases)
    reserved.update(int(value) for value in previous.aliases.values())

    inherited: Dict[int, int] = {}
    for old in previous.symbols:
        successor = matched.get(id(old))
        if successor is not None:
            inherited[id(successor)] = int(old.code)

    used = set(inherited.values())

    def fresh_code() -> int:
        code = 0
        while code in used or code in reserved:
            code += 1
        used.add(code)
        return code

    for symbol in symbols:
        if id(symbol) in inherited:
            symbol.code = inherited[id(symbol)]
        else:
            symbol.code = fresh_code()

    retired: List[Symbol] = []
    next_class = max((symbol.class_id for symbol in symbols), default=-1) + 1
    live_codes = {symbol.code for symbol in symbols}
    for old in previous.symbols:
        if id(old) in matched:
            continue
        if int(old.code) in live_codes:
            continue
        for obs_id in old.observation_ids:
            if obs_id in reservoir:
                continue
            parent_obs = previous.observations.get(obs_id)
            if parent_obs is not None:
                reservoir[obs_id] = Observation.from_dict(parent_obs.to_dict())
        retired.append(
            Symbol(
                class_id=next_class,
                code=int(old.code),
                proto_embedding=list(old.proto_embedding),
                observation_ids=list(old.observation_ids),
                definition=old.definition,
                examples=list(old.examples),
                confidence=0.0,
                quarantined=True,
                survival=0.0,
                metadata={
                    "retired_from": previous.pack_id,
                    "prior_class_id": old.class_id,
                },
            )
        )
        live_codes.add(int(old.code))
        next_class += 1

    aliases: Dict[int, int] = {}
    for src in previous.aliases:
        src_code = int(src)
        if src_code in live_codes:
            continue
        final, _ = previous.resolve_code(src_code)
        if int(final) not in live_codes or int(final) == src_code:
            continue
        aliases[src_code] = int(final)
    return aliases, retired


def _alias_collision(aliases: Dict[int, int], codebook: Dict[int, int]) -> bool:
    if any(int(key) in codebook for key in aliases):
        return True
    return any(int(target) not in codebook for target in aliases.values())


def _parent_codes_kept(
    previous: Optional[SymbolPack],
    aliases: Dict[int, int],
    codebook: Dict[int, int],
) -> bool:
    if previous is None:
        return True

    def resolve(code: int) -> int:
        seen: set[int] = set()
        current = int(code)
        while current in aliases:
            if current in seen:
                return current
            seen.add(current)
            current = int(aliases[current])
        return current

    codes = set(codebook)
    for old in previous.symbols:
        if resolve(old.code) not in codes:
            return False
    for src in previous.aliases:
        if resolve(int(src)) not in codes:
            return False
    return True

"""Deterministic clustering of embeddings into symbol classes."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from neuralese.energy import cosine_similarity


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    max_iter: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=int), np.zeros((0, X.shape[1] if X.ndim == 2 else 0))
    k = max(1, min(k, n))
    seeds = _farthest_first(X, k)
    centroids = X[seeds].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        new_centroids = centroids.copy()
        for i in range(k):
            members = X[labels == i]
            if members.shape[0]:
                new_centroids[i] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids


def _farthest_first(X: np.ndarray, k: int) -> List[int]:
    norms = np.linalg.norm(X, axis=1)
    start = int(np.argmax(norms))
    chosen = [start]
    dmin = ((X - X[start]) ** 2).sum(axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dnew = ((X - X[nxt]) ** 2).sum(axis=1)
        dmin = np.minimum(dmin, dnew)
    return chosen


def cluster_survival(
    old_protos: np.ndarray,
    new_protos: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """For each new prototype, max cosine vs old; used as a persistence proxy."""
    if new_protos.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if old_protos.size == 0:
        return np.ones(new_protos.shape[0], dtype=np.float64)
    scores = []
    for proto in new_protos:
        best = max(cosine_similarity(proto, old) for old in old_protos)
        scores.append(float(best if best >= 0 else 0.0))
    return np.asarray(scores, dtype=np.float64)

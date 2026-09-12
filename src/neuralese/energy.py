"""Energy and distance helpers for symbol geometry."""
from __future__ import annotations

from typing import Sequence

import numpy as np

EPS = 1e-12


def as_vector(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va = as_vector(a)
    vb = as_vector(b)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return 1.0 - cosine_similarity(a, b)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(as_vector(a) - as_vector(b)))


def compute_energy(
    embedding: Sequence[float],
    prototype: Sequence[float],
    *,
    kappa: float,
    residual: float,
    lambda_sem: float = 0.40,
    lambda_h: float = 0.20,
    lambda_r: float = 0.10,
) -> float:
    """E = λ_sem·d_sem + λ_h·(1-κ) + λ_r·residual."""
    d_sem = cosine_distance(embedding, prototype)
    return float(lambda_sem * d_sem + lambda_h * (1.0 - kappa) + lambda_r * residual)


def confidence_cap(confidence: float, residual: float) -> float:
    return float(max(0.0, min(confidence, 1.0 - min(max(residual, 0.0), 1.0))))

"""Tensor factorization for glyph-atom discovery (SVD; NMF-optional)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def svd_factors(X: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (U_k, S_k * Vt_k, reconstruction_error)."""
    if X.size == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), 0.0
    rank = max(1, min(rank, X.shape[0], X.shape[1]))
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    U_k = U[:, :rank]
    Vt_k = (S[:rank, None] * Vt[:rank, :])
    reconstructed = U_k @ Vt_k
    residual = reconstruction_error(X, reconstructed)
    return U_k, Vt_k, residual


def reconstruction_error(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Normalized Frobenius residual in [0, inf)."""
    denom = float(np.linalg.norm(X))
    if denom <= 1e-12:
        return 0.0
    return float(np.linalg.norm(X - X_hat) / denom)

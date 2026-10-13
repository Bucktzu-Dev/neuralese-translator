"""Tensor factorization for glyph-atom discovery (SVD; NMF-optional)."""
from __future__ import annotations

import ctypes
import os
from typing import Tuple

import numpy as np

# Read at library load, which is usually before the first SVD in this process.
# The runtime setters below cover libraries that are already resident.
_BLAS_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_BLAS_THREAD_SYMBOLS = (
    "omp_set_num_threads",
    "openblas_set_num_threads",
    "MKL_Set_Num_Threads",
    "mkl_set_num_threads",
)


def pin_blas_threads(count: int = 1) -> None:
    """Force one BLAS/OpenMP/MKL thread so truncated SVD stays repeatable.

    Divide-and-conquer SVD can assign a different trailing subspace when the
    thread count changes. Column sign flips are handled separately and do not
    move squared-Euclidean k-means. A different BLAS build can still disagree
    with this one; the varied-prose locks record the single-thread result.
    """
    for key in _BLAS_THREAD_ENV:
        os.environ[key] = str(count)
    try:
        loaded = ctypes.CDLL(None)
    except OSError:
        loaded = None
    for name in _BLAS_THREAD_SYMBOLS:
        if loaded is not None:
            _call_thread_setter(loaded, name, count)
        for lib in ("libgomp.so.1", "libblas.so.3", "libopenblas.so.0", "libmkl_rt.so"):
            try:
                handle = ctypes.CDLL(lib)
            except OSError:
                continue
            if _call_thread_setter(handle, name, count):
                break


def _call_thread_setter(handle: ctypes.CDLL, name: str, count: int) -> bool:
    try:
        fn = getattr(handle, name)
    except AttributeError:
        return False
    fn.argtypes = [ctypes.c_int]
    fn.restype = None
    fn(count)
    return True


def svd_factors(X: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (U_k, S_k * Vt_k, reconstruction_error)."""
    if X.size == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), 0.0
    rank = max(1, min(rank, X.shape[0], X.shape[1]))
    pin_blas_threads(1)
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

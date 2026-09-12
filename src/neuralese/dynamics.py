"""Subjective Symbol Dynamics — per-subject Boltzmann process over codes.

P(z_t | z_{t-1}, s) ∝ exp{-β_s · E(z_t)} · T(z_{t-1}, z_t)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from neuralese.contracts import Symbol, SymbolPack
from neuralese.energy import compute_energy


class SubjectiveSymbolDynamics:
    def __init__(
        self,
        pack: SymbolPack,
        *,
        subject: str = "default",
        beta_s: float = 1.0,
    ) -> None:
        self.pack = pack
        self.subject = subject
        self.beta_s = float(beta_s)
        self.classes: List[Symbol] = list(pack.symbols)
        n = len(self.classes)
        if n == 0:
            self.transition = np.zeros((0, 0), dtype=np.float64)
        else:
            self.transition = np.full((n, n), 1.0 / n, dtype=np.float64)
        self._index = {s.class_id: i for i, s in enumerate(self.classes)}

    def update_from_bigrams(self, bigrams: Dict[tuple[int, int], int]) -> None:
        n = len(self.classes)
        counts = np.zeros((n, n), dtype=np.float64)
        for (src, dst), count in bigrams.items():
            i = self._index.get(int(src))
            j = self._index.get(int(dst))
            if i is None or j is None:
                continue
            counts[i, j] += float(count)
        for i in range(n):
            total = float(counts[i].sum())
            if total <= 0:
                self.transition[i] = 1.0 / max(n, 1)
            else:
                self.transition[i] = counts[i] / total

    def transition_probs(
        self,
        prev_class_id: Optional[int],
        context_embedding: Optional[List[float]] = None,
    ) -> Dict[int, float]:
        n = len(self.classes)
        if n == 0:
            return {}
        if prev_class_id is None or prev_class_id not in self._index:
            base = np.full(n, 1.0 / n)
        else:
            base = self.transition[self._index[prev_class_id]]
        energies = np.zeros(n, dtype=np.float64)
        for i, symbol in enumerate(self.classes):
            proto = symbol.proto_embedding
            ctx = context_embedding if context_embedding else proto
            energies[i] = compute_energy(
                ctx,
                proto,
                kappa=max(symbol.confidence, 0.0),
                residual=self.pack.reconstruction_error,
            )
        tilted = base * np.exp(-self.beta_s * energies)
        total = float(tilted.sum())
        if total <= 0:
            tilted = np.full(n, 1.0 / n)
        else:
            tilted = tilted / total
        return {self.classes[i].class_id: float(tilted[i]) for i in range(n)}

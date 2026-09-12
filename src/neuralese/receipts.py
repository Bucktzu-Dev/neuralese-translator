"""Receipt constructors for learn-pipeline steps."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from neuralese.contracts import Receipt


def create_receipt(
    step: str,
    ok: bool,
    *,
    kappa: Optional[float] = None,
    reconstruction_error: Optional[float] = None,
    delta_mdl_bits: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Receipt:
    return Receipt(
        step=step,
        ok=ok,
        timestamp=time.time(),
        kappa=kappa,
        reconstruction_error=reconstruction_error,
        delta_mdl_bits=delta_mdl_bits,
        metadata=metadata or {},
    )

"""Neuralese to English Translator."""

from neuralese.adapters import (
    load_activation_matrix,
    load_activations,
    load_alignment_texts,
    load_observations_jsonl,
    load_pack,
    load_stream,
    observations_from_activations,
    save_observations_jsonl,
    save_pack,
)
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify
from neuralese.contracts import (
    AuditCertificate,
    Gloss,
    Observation,
    Receipt,
    Symbol,
    SymbolPack,
    UncertifiedPackError,
)
from neuralese.dynamics import SubjectiveSymbolDynamics
from neuralese.translator import translate_stream

__version__ = "0.1.2"

__all__ = [
    "__version__",
    "Observation",
    "Symbol",
    "SymbolPack",
    "Gloss",
    "AuditCertificate",
    "Receipt",
    "LearnConfig",
    "learn_pack",
    "translate_stream",
    "certify",
    "load_activation_matrix",
    "load_activations",
    "load_alignment_texts",
    "load_observations_jsonl",
    "load_pack",
    "load_stream",
    "observations_from_activations",
    "save_observations_jsonl",
    "save_pack",
    "SubjectiveSymbolDynamics",
    "UncertifiedPackError",
]

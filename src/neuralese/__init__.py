"""Neuralese to English Translator."""

from neuralese.activations import load_activation_dump, observations_from_hidden_states
from neuralese.adapters import (
    load_observations_jsonl,
    load_pack,
    load_stream,
    save_observations_jsonl,
    save_pack,
)
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify, decodability_report
from neuralese.contracts import (
    AuditCertificate,
    Gloss,
    Observation,
    Receipt,
    Symbol,
    SymbolPack,
)
from neuralese.dynamics import SubjectiveSymbolDynamics
from neuralese.encode import ENCODER_CHAR_TRIGRAM, ENCODER_WORD_SENTENCE_SVD, word_sentence_svd
from neuralese.translator import translate_stream
from neuralese.unfold import UnfoldReport, unfold_code

__version__ = "0.6.0"

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
    "unfold_code",
    "UnfoldReport",
    "certify",
    "decodability_report",
    "load_observations_jsonl",
    "save_observations_jsonl",
    "load_activation_dump",
    "observations_from_hidden_states",
    "load_pack",
    "load_stream",
    "save_pack",
    "SubjectiveSymbolDynamics",
    "ENCODER_CHAR_TRIGRAM",
    "ENCODER_WORD_SENTENCE_SVD",
    "word_sentence_svd",
]

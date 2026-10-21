"""Typed claims and an independent chain-rule checker.

This package is not part of Neuralese ``learn``, ``translate``, or
``certify``. It does not admit a runtime, does not change the certificate
in ``neuralese.audit``, and does not change package version 0.7.0. The
existing certificate authority stays ``RUNTIME_NOT_AUTHORIZED``. Nothing
here emits ``RESEARCH_PASS``.

The parser turns a narrow claim language into a typed record. It does not
assign ``PROVED``, ``DISPROVED``, ``UNRESOLVED``, or ``INVALID_INPUT``. It
does not attach a certificate and it has no ``Error_Bound`` field. A later
``verify`` call is the only place that assigns an outcome. The parsed
assertion is not evidence. One checker, ``coordinate_chain_rule_v1``,
derives the chain rule under the assumptions written on the claim.

The executable checker uses one coordinate index. The handoff's general
argument is the sum over ``i``. A coordinate change preserves dimension.
This module does not verify the cited paper. The OpenAI ``interpolation.tex``
URL is stored as provenance text and is not fetched.

FFOC is not implemented. It is not claimed to be complete, terminating,
confluent, or canonical. No benchmark table, pass rate, timing, or
compression figure is produced. The nine SymPy 1.14.0 smoke checks in the
21 October 2026 handoff (script SHA-256
``98b836d67804a276e70ded04db119397f5ebdd9ad81053923c2333b28c83649c``,
output ``PASS: 9 assertions; SymPy 1.14.0``) are prior smoke checks of
coefficient algebra. That hash identifies that script, not this code.

Outcomes
--------
``PROVED``
    The checker derived the claim under the recorded assumptions.
``DISPROVED``
    A well-formed claim has an in-domain witness or a symbolic
    contradiction. Failure to prove is not this outcome.
``UNRESOLVED``
    A needed assumption, direction, symbol meaning, or supported theory
    is missing, or simplification stalled.
``INVALID_INPUT``
    The input is malformed, ill-typed, or forbidden by the grammar.
"""

from neuralese.claims.syntax import (
    deserialize_claim,
    parse_claim,
    parse_expr,
    parse_operator,
    serialize_claim,
)
from neuralese.claims.verify import (
    CHECKER_VERSION,
    DISPROVED,
    INVALID_INPUT,
    OUTCOMES,
    PAPER_PROVENANCE_URL,
    PROVED,
    SUPPORTED_THEORY,
    TACTIC_VERSION,
    TRUST_BOUNDARY,
    UNRESOLVED,
    apply_operator,
    replay_derivation,
    verify,
)

__all__ = [
    "CHECKER_VERSION",
    "DISPROVED",
    "INVALID_INPUT",
    "OUTCOMES",
    "PAPER_PROVENANCE_URL",
    "PROVED",
    "SUPPORTED_THEORY",
    "TACTIC_VERSION",
    "TRUST_BOUNDARY",
    "UNRESOLVED",
    "apply_operator",
    "deserialize_claim",
    "parse_claim",
    "parse_expr",
    "parse_operator",
    "replay_derivation",
    "serialize_claim",
    "verify",
]

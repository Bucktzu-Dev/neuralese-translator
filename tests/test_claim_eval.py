"""Frozen claim-evaluation cases.

Expected outcomes were written in docs/CLAIM_EVAL.md before these checks.
``verify`` computes each operator. It does not read this table.
"""

from __future__ import annotations

from pathlib import Path

from neuralese.claims import DISPROVED, INVALID_INPUT, PROVED, UNRESOLVED, parse_claim, verify
from neuralese.claims.verify import replay_derivation

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "examples" / "claims" / "eval"
TWO_INDEX = ROOT / "examples" / "claims" / "two_index_coordinate_change.txt"

# Labels frozen in docs/CLAIM_EVAL.md. verify() does not consult this table.
FROZEN = (
    ("e01_shear.txt", PROVED, None),
    ("e02_wrong_coefficient.txt", DISPROVED, "symbolic_contradiction"),
    ("e03_second_order.txt", UNRESOLVED, "unsupported_operator"),
    ("e04_malformed.txt", INVALID_INPUT, "malformed_claim"),
    ("e05_missing_log_branch.txt", UNRESOLVED, "missing_log_branch"),
    ("e06_missing_exclusion.txt", UNRESOLVED, "missing_domain"),
    ("e07_domain_includes_zero.txt", DISPROVED, "inverse_undefined_at_included_point"),
    ("e08_reversed_composition.txt", INVALID_INPUT, "illegal_operator_application"),
    ("e09_missing_direction.txt", UNRESOLVED, "missing_pullback_direction"),
    ("e11_two_index_false_target.txt", DISPROVED, "symbolic_contradiction"),
)


def _load(name: str) -> str:
    if name == "two_index_coordinate_change.txt":
        return TWO_INDEX.read_text(encoding="utf-8")
    return (EVAL / name).read_text(encoding="utf-8")


def test_frozen_outcomes_come_from_the_checker():
    checker = (ROOT / "src" / "neuralese" / "claims" / "verify.py").read_text(encoding="utf-8")
    for name, _outcome, _reason in FROZEN:
        assert name not in checker
    assert "eval-shear" not in checker
    assert "eval-two-index" not in checker
    for name, outcome, reason in FROZEN:
        result = verify(parse_claim(_load(name)))
        assert result.outcome == outcome
        if reason is not None:
            assert result.reason == reason


def test_unfamiliar_shear_is_derived():
    parsed = parse_claim(_load("e01_shear.txt"))
    result = verify(parsed)
    assert result.outcome == PROVED
    assert result.evidence["derived"]["V"] == "d/dp"
    assert result.evidence["proved_by"] == "chain_rule"
    assert result.residual == "0"
    assert replay_derivation(parsed, result) is True
    steps = [step for step in result.derivation if step["rule"] == "chain_rule_coefficient"]
    by_key = {(step["operator"], step["component"]): step["output"] for step in steps}
    assert by_key[("V", "p")] == "1"
    assert by_key[("V", "q")] == "0"


def test_wrong_coefficient_witness():
    result = verify(parse_claim(_load("e02_wrong_coefficient.txt")))
    assert result.outcome == DISPROVED
    assert result.evidence["derived"]["V"] == "d/dp"
    witness = result.witness
    assert witness["function"] == "p"
    assert witness["actual_output"] == "1"
    assert witness["claimed_output"] == "2"
    assert witness["residual"] == "-1"
    assert witness["kind"] == "symbolic"
    assert "point" not in witness


def test_missing_inputs_name_what_was_not_supplied():
    branch = verify(parse_claim(_load("e05_missing_log_branch.txt")))
    assert branch.evidence["not_supplied"] == ["log(1) = 0"]
    missing = verify(parse_claim(_load("e06_missing_exclusion.txt")))
    assert missing.evidence["not_supplied"] == ["Y != 0"]
    direction = parse_claim(_load("e09_missing_direction.txt"))
    assert direction.claim.maps[0].direction is None
    result = verify(direction)
    assert result.evidence["repaired"] is False
    assert result.evidence["direction"] is None


def test_reversed_order_is_not_reordered():
    result = verify(parse_claim(_load("e08_reversed_composition.txt")))
    assert result.outcome == INVALID_INPUT
    assert result.violated_obligation == "pullback_composition_order"
    assert "not reordered" in result.message
    assert result.span is not None


def test_domain_restriction_does_not_evaluate_the_inverse_at_zero():
    result = verify(parse_claim(_load("e07_domain_includes_zero.txt")))
    assert result.outcome == DISPROVED
    assert result.violated_obligation == "invertibility"
    assert result.evidence["inverse_evaluated_at_excluded_point"] is False


def test_malformed_eval_input_has_a_span_and_no_verdict_on_the_parse():
    parsed = parse_claim(_load("e04_malformed.txt"))
    assert parsed.claim is None
    assert parsed.syntax_error is not None
    result = verify(parsed)
    assert result.outcome == INVALID_INPUT
    assert result.span is not None
    assert result.span.start == 0


def test_two_index_pair():
    proved = verify(parse_claim(_load("two_index_coordinate_change.txt")))
    assert proved.outcome == PROVED
    assert proved.evidence["derived"]["D0"] == "(1 + t) * d/dt"
    assert proved.evidence["derived"]["D1"] == "d/du1"
    assert proved.evidence["derived"]["D2"] == "d/du2"
    assert proved.evidence["summed_coordinates"] == ["Y", "X1", "X2"]
    assert proved.evidence["summed_coordinate_count"] == 3
    assert proved.residual == "0"
    false = verify(parse_claim(_load("e11_two_index_false_target.txt")))
    assert false.outcome == DISPROVED
    assert false.evidence["derived"]["D0"] == "(1 + t) * d/dt"
    assert false.evidence["derived"]["D1"] == "d/du1"
    assert false.evidence["derived"]["D2"] == "d/du2"
    witness = false.witness
    assert witness["function"] == "t"
    assert witness["actual_output"] == "0"
    assert witness["claimed_output"] == "1"
    assert witness["residual"] == "-1"
    assert witness["kind"] == "symbolic"

"""Deterministic checks for the independent chain-rule claim verifier.

No network. The fixture outcomes do not import SymPy. Parse results have no
verdict and no Error_Bound.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import neuralese
from neuralese.claims import (
    DISPROVED,
    INVALID_INPUT,
    OUTCOMES,
    PAPER_PROVENANCE_URL,
    PROVED,
    UNRESOLVED,
    deserialize_claim,
    parse_claim,
    parse_expr,
    parse_operator,
    replay_derivation,
    serialize_claim,
    verify,
)
from neuralese.claims.syntax import Apply, Compose, MultiplyCoefficient, Partial
from neuralese.claims.verify import apply_operator, normalize_operator, operator_difference

PRIOR_SMOKE_SCRIPT_SHA256 = "98b836d67804a276e70ded04db119397f5ebdd9ad81053923c2333b28c83649c"

POSITIVE = f'''source "{PAPER_PROVENANCE_URL}"
constant c
assume analytic germs near Y = 1, log(1) = 0, neighborhood excludes Y = 0
space old
kind coordinates
coordinates Y, X
ring analytic_germs
regularity analytic
center Y = 1, X = c
neighborhood germ
exclude Y = 0
log_branch log(1) = 0
end
space new
kind coordinates
coordinates t, u
ring analytic_germs
regularity analytic
center t = 0
neighborhood germ
exclude Y = 0
log_branch log(1) = 0
end
map Phi
source new
target old
direction new_to_old
inverse Psi
obligations domain_compatibility, invertibility
Y = 1 + t
X = c + u + log(1 + t)
end
map Psi
source old
target new
direction old_to_new
inverse Phi
obligations domain_compatibility, invertibility
t = Y - 1
u = X - c - log(Y)
end
operator D0
domain old
codomain old
theory coordinate_chain_rule_v1
formula Y * d/dY + d/dX
end
operator D1
domain old
codomain old
theory coordinate_chain_rule_v1
formula d/dX
end
operator image_D0
domain new
codomain new
theory coordinate_chain_rule_v1
formula (1 + t) * d/dt
end
operator image_D1
domain new
codomain new
theory coordinate_chain_rule_v1
formula d/du
end
define T pullback of Phi
assert T o D0 o T^-1 = image_D0
assert T o D1 o T^-1 = image_D1
assert Phi o Psi = id
assert Psi o Phi = id
'''

EULER = f'''source "{PAPER_PROVENANCE_URL}"
assume rejected Euler chart t = log Y and u = X / Y
space old
kind coordinates
coordinates Y, X
ring analytic
regularity analytic
neighborhood chart
exclude Y = 0
log_branch log(1) = 0
end
space new
kind coordinates
coordinates t, u
ring analytic
regularity analytic
neighborhood chart
exclude Y = 0
log_branch log(1) = 0
contains t = 1
end
map Phi
source new
target old
direction new_to_old
inverse Psi
obligations domain_compatibility, invertibility
Y = exp(t)
X = u * exp(t)
end
map Psi
source old
target new
direction old_to_new
inverse Phi
obligations domain_compatibility, invertibility
t = log(Y)
u = X / Y
end
operator E0
domain old
codomain old
theory coordinate_chain_rule_v1
formula Y * d/dY + X * d/dX
end
operator E1
domain old
codomain old
theory coordinate_chain_rule_v1
formula d/dX
end
operator claimed_E0
domain new
codomain new
theory coordinate_chain_rule_v1
formula (1 + t) * d/dt
end
operator claimed_E1
domain new
codomain new
theory coordinate_chain_rule_v1
formula d/du
end
define T pullback of Phi
assert T o E0 o T^-1 = claimed_E0
assert T o E1 o T^-1 = claimed_E1
'''

ORDER_CLAIM = '''source "local"
space old
kind coordinates
coordinates x
ring analytic
regularity analytic
neighborhood germ
exclude x = 0
end
space new
kind coordinates
coordinates s
ring analytic
regularity analytic
neighborhood germ
exclude x = 0
end
map Phi
source new
target old
inverse Psi
x = s
end
map Psi
source old
target new
direction old_to_new
inverse Phi
s = x
end
operator Forth
domain old
codomain old
theory coordinate_chain_rule_v1
formula d/dx o *(x)
end
operator Back
domain old
codomain old
theory coordinate_chain_rule_v1
formula *(x) o d/dx
end
'''

UNKNOWN_SYMBOL = '''source "local"
space old
kind coordinates
coordinates Y
ring analytic
regularity analytic
neighborhood germ
exclude Y = 0
end
operator W
domain old
codomain old
theory coordinate_chain_rule_v1
formula q * d/dY
end
'''

FORBIDDEN_SYMBOL = '''source "local"
space old
kind coordinates
coordinates Y
ring analytic
regularity analytic
neighborhood germ
exclude Y = 0
end
operator W
domain old
codomain old
theory coordinate_chain_rule_v1
formula $q * d/dY
end
'''

MALFORMED = "assert conjugation\n"

INCOMPATIBLE_ASSIGNMENT = '''source "local"
space new
kind coordinates
coordinates t
ring analytic
regularity analytic
neighborhood germ
exclude t = 0
end
space old
kind coordinates
coordinates Y
ring analytic
regularity analytic
neighborhood germ
exclude Y = 0
end
map Phi
source new
target old
direction new_to_old
inverse Psi
Z = t
end
'''

DIMENSION_MISMATCH = '''source "local"
space new
kind coordinates
coordinates t, u
ring analytic
regularity analytic
neighborhood germ
exclude t = 0
end
space old
kind coordinates
coordinates Y
ring analytic
regularity analytic
neighborhood germ
exclude Y = 0
end
map Phi
source new
target old
direction new_to_old
Y = t
end
'''

ILLEGAL_APPLY = '''source "local"
space old
kind coordinates
coordinates Y, X
ring analytic
regularity analytic
neighborhood germ
exclude Y = 0
end
space new
kind coordinates
coordinates t, u
ring analytic
regularity analytic
neighborhood germ
exclude Y = 0
end
function g on new
operator D0
domain old
codomain old
theory coordinate_chain_rule_v1
formula d/dY
end
assert apply D0 to g
'''

LOG_AT_EXCLUDED = '''source "local"
space old
kind coordinates
coordinates Y
ring analytic
regularity analytic
neighborhood germ
exclude Y = 0
end
assert apply log at Y = 0
'''

INVERSE_ON_EXCLUDED_POINT = '''source "local"
space old
kind coordinates
coordinates Y, X
ring analytic
regularity analytic
neighborhood chart
includes Y = 0
end
space new
kind coordinates
coordinates t, u
ring analytic
regularity analytic
neighborhood chart
end
map Phi
source new
target old
direction new_to_old
inverse Psi
Y = exp(t)
X = u * exp(t)
end
map Psi
source old
target new
direction old_to_new
inverse Phi
t = log(Y)
u = X / Y
end
assert invertible Psi
'''

CONTRADICTORY_DOMAIN = '''source "local"
space old
kind coordinates
coordinates Y
ring analytic
regularity analytic
neighborhood germ
exclude Y = 0
includes Y = 0
end
'''

FOREIGN_THEORY = '''source "local"
space old
kind coordinates
coordinates Y
ring analytic
regularity analytic
neighborhood germ
exclude Y = 0
end
operator W
domain old
codomain old
theory exterior_calculus
formula d/dY
end
'''

NON_VECTOR_FIELD = '''source "local"
space line
kind coordinates
coordinates x
ring analytic
regularity analytic
neighborhood germ
exclude x = 0
end
space sline
kind coordinates
coordinates s
ring analytic
regularity analytic
neighborhood germ
exclude x = 0
end
map Phi
source sline
target line
direction new_to_old
inverse Psi
x = s
end
map Psi
source line
target sline
direction old_to_new
inverse Phi
s = x
end
operator D
domain line
codomain line
theory coordinate_chain_rule_v1
formula d/dx o *(x)
end
operator image
domain sline
codomain sline
theory coordinate_chain_rule_v1
formula d/ds
end
define T pullback of Phi
assert T o D o T^-1 = image
'''


def _keys(value, found):
    if isinstance(value, dict):
        found.update(value)
        for item in value.values():
            _keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _keys(item, found)


def _assert_parse_has_no_verdict(text: str):
    parsed = parse_claim(text)
    blob = serialize_claim(parsed)
    data = json.loads(blob)
    found = set()
    _keys(data, found)
    assert "status" not in found
    assert "Error_Bound" not in found
    assert "error_bound" not in found
    assert "certificate" not in found
    assert "Error_Bound" not in blob
    assert not hasattr(parsed, "status")
    assert not hasattr(parsed, "Error_Bound")
    if parsed.claim is not None:
        fields = set(parsed.claim.__dataclass_fields__)
        assert "status" not in fields
        assert "Error_Bound" not in fields
        assert parsed.claim.source_hash == hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert parsed.claim.source_hash != PRIOR_SMOKE_SCRIPT_SHA256
    before = blob
    verify(parsed)
    assert serialize_claim(parsed) == before
    return parsed


def test_package_version_and_production_boundary():
    assert neuralese.__version__ == "0.7.0"
    assert set(OUTCOMES) == {PROVED, DISPROVED, UNRESOLVED, INVALID_INPUT}
    root = Path(__file__).resolve().parents[1] / "src" / "neuralese"
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "neuralese.claims" not in source
        assert "from neuralese.claims" not in source


def test_positive_coordinate_change_is_proved_by_the_chain_rule():
    parsed = _assert_parse_has_no_verdict(POSITIVE)
    result = verify(parsed)
    assert result.outcome == PROVED
    assert result.evidence["derived"]["D0"] == "(1 + t) * d/dt"
    assert result.evidence["derived"]["D1"] == "d/du"
    assert result.evidence["proved_by"] == "chain_rule"
    assert result.evidence["primary"] == "chain_rule"
    assert result.evidence["paper_verified"] is False
    assert result.evidence["coordinate_index"] == "one"
    assert "sum over i" in result.evidence["general_argument"]
    assert result.residual == "0"
    assert result.witness is None
    steps = [step for step in result.derivation if step["rule"] == "chain_rule_coefficient"]
    by_key = {(step["operator"], step["component"]): step for step in steps}
    assert by_key[("D0", "t")]["before_substitution"] == "Y"
    assert by_key[("D0", "t")]["output"] == "1 + t"
    assert by_key[("D0", "u")]["output"] == "0"
    assert by_key[("D1", "t")]["output"] == "0"
    assert by_key[("D1", "u")]["before_substitution"] == "1"
    assert by_key[("D1", "u")]["output"] == "1"
    identities = [step for step in result.derivation if step["rule"] == "inverse_composition"]
    assert {(step["order"], step["component"], step["output"]) for step in identities} == {
        ("Phi o Psi", "Y", "Y"),
        ("Phi o Psi", "X", "X"),
        ("Psi o Phi", "t", "t"),
        ("Psi o Phi", "u", "u"),
    }
    assert result.evidence["consistency"]["bracket"] == "0"
    assert result.evidence["consistency"]["is_zero"] is True
    assert result.evidence["consistency"]["proves_coordinate_change"] is False
    assert result.evidence["commutator_proves_claim"] is False
    assert "sum over i" in " ".join(parsed.claim.interpretation_choices)
    assert replay_derivation(parsed, result) is True
    again = deserialize_claim(serialize_claim(parsed))
    assert verify(again).outcome == PROVED


def test_zero_commutator_is_not_a_proof():
    text = "evidence commutator_only\n" + POSITIVE
    parsed = _assert_parse_has_no_verdict(text)
    result = verify(parsed)
    assert result.outcome == UNRESOLVED
    assert result.reason == "commutator_is_not_a_proof"
    assert result.outcome != PROVED
    note = result.evidence["commutator"][0]
    assert note["operators"] == ["D0", "D1"]
    assert note["bracket"] == "0"
    assert note["is_zero"] is True
    assert note["proves_coordinate_change"] is False
    assert result.evidence["proved_by"] is None
    assert result.evidence["proves_coordinate_change"] is False


def test_euler_chart_is_disproved_by_an_in_domain_witness():
    parsed = _assert_parse_has_no_verdict(EULER)
    result = verify(parsed)
    assert result.outcome == DISPROVED
    assert result.reason == "in_domain_witness"
    assert result.evidence["derived"]["E0"] == "d/dt"
    assert result.evidence["derived"]["E1"] == "exp(-t) * d/du"
    assert result.evidence["primary"] == "witness"
    assert result.evidence["proved_by"] is None
    assert result.evidence["paper_verified"] is False
    witness = result.witness
    assert witness["function"] == "t"
    assert witness["actual_output"] == "1"
    assert witness["claimed_output"] == "1 + t"
    assert witness["residual"] == "-t"
    assert result.residual == "-t"
    assert witness["point"] == {"t": "1"}
    assert witness["actual_value"] == "1"
    assert witness["claimed_value"] == "2"
    assert witness["residual_value"] == "-1"
    assert witness["kind"] == "in_domain_witness"
    consistency = result.evidence["consistency"]
    assert consistency["bracket"] == "-d/dX"
    assert consistency["transformed_bracket"] == "-exp(-t) * d/du"
    assert consistency["proves_coordinate_change"] is False
    assert consistency["role"] == "consistency_only"
    assert replay_derivation(parsed, result) is True


def test_wrong_coefficient_is_a_symbolic_contradiction():
    text = POSITIVE.replace("formula (1 + t) * d/dt", "formula d/dt", 1)
    parsed = _assert_parse_has_no_verdict(text)
    result = verify(parsed)
    assert result.outcome == DISPROVED
    assert result.reason == "symbolic_contradiction"
    assert result.evidence["derived"]["D0"] == "(1 + t) * d/dt"
    witness = result.witness
    assert witness["function"] == "t"
    assert witness["actual_output"] == "1 + t"
    assert witness["claimed_output"] == "1"
    assert witness["residual"] == "t"
    assert result.residual == "t"
    assert witness["kind"] == "symbolic"
    assert "point" not in witness


def test_missing_domain_and_missing_log_branch_stay_unresolved():
    missing_domain = POSITIVE.replace("exclude Y = 0\n", "")
    parsed = _assert_parse_has_no_verdict(missing_domain)
    result = verify(parsed)
    assert result.outcome == UNRESOLVED
    assert result.reason == "missing_domain"
    assert result.evidence["not_supplied"] == ["Y != 0"]
    assert all(space.exclusions == () for space in parsed.claim.spaces)
    missing_branch = POSITIVE.replace("log_branch log(1) = 0\n", "")
    parsed = _assert_parse_has_no_verdict(missing_branch)
    result = verify(parsed)
    assert result.outcome == UNRESOLVED
    assert result.reason == "missing_log_branch"
    assert result.evidence["not_supplied"] == ["log(1) = 0"]
    assert any(space.exclusions for space in parsed.claim.spaces)


def test_missing_direction_is_not_repaired():
    text = POSITIVE.replace("direction new_to_old\n", "", 1)
    text = text.replace(
        "assert T o D0 o T^-1 = image_D0",
        "assert Phi^-1 o D0 o Phi = image_D0",
    )
    parsed = _assert_parse_has_no_verdict(text)
    phi = next(item for item in parsed.claim.maps if item.name == "Phi")
    assert phi.direction is None
    parts = parsed.claim.claimed_property[0].parts
    assert parts[0] == ("Phi", True)
    assert parts[1] == ("D0", False)
    assert parts[2] == ("Phi", False)
    result = verify(parsed)
    assert result.outcome == UNRESOLVED
    assert result.reason == "missing_pullback_direction"
    assert result.evidence["repaired"] is False
    assert result.evidence["direction"] is None
    again = deserialize_claim(serialize_claim(parsed))
    assert again.claim.maps[0].direction is None
    assert again.claim.claimed_property[0].parts == parts


def test_recorded_direction_accepts_only_the_written_pullback_order():
    text = POSITIVE.replace(
        "assert T o D0 o T^-1 = image_D0",
        "assert Phi o D0 o Phi^-1 = image_D0",
    ).replace(
        "assert T o D1 o T^-1 = image_D1",
        "assert Phi o D1 o Phi^-1 = image_D1",
    )
    parsed = _assert_parse_has_no_verdict(text)
    assert verify(parsed).outcome == PROVED
    rejected = POSITIVE.replace(
        "assert T o D0 o T^-1 = image_D0",
        "assert Phi^-1 o D0 o Phi = image_D0",
    )
    parsed = _assert_parse_has_no_verdict(rejected)
    assert parsed.claim.claimed_property[0].parts[0] == ("Phi", True)
    result = verify(parsed)
    assert result.outcome == INVALID_INPUT
    assert result.reason == "illegal_operator_application"
    assert result.violated_obligation == "pullback_composition_order"
    assert result.span is not None
    assert "not reordered" in result.message


def test_other_outcomes():
    cases = {
        "encoding ffoc\n" + POSITIVE: (UNRESOLVED, "ffoc_unimplemented"),
        UNKNOWN_SYMBOL: (UNRESOLVED, "unknown_symbol"),
        FOREIGN_THEORY: (UNRESOLVED, "unsupported_theory"),
        NON_VECTOR_FIELD: (UNRESOLVED, "unsupported_operator"),
        FORBIDDEN_SYMBOL: (INVALID_INPUT, "forbidden_unbound_symbol"),
        MALFORMED: (INVALID_INPUT, "malformed_claim"),
        INCOMPATIBLE_ASSIGNMENT: (INVALID_INPUT, "incompatible_spaces"),
        DIMENSION_MISMATCH: (INVALID_INPUT, "incompatible_spaces"),
        ILLEGAL_APPLY: (INVALID_INPUT, "illegal_operator_application"),
        LOG_AT_EXCLUDED: (INVALID_INPUT, "illegal_operator_application"),
        INVERSE_ON_EXCLUDED_POINT: (DISPROVED, "inverse_undefined_at_included_point"),
        CONTRADICTORY_DOMAIN: (INVALID_INPUT, "contradictory_domain"),
    }
    for text, (outcome, reason) in cases.items():
        parsed = _assert_parse_has_no_verdict(text)
        result = verify(parsed)
        assert result.outcome == outcome
        assert result.reason == reason
        assert result.outcome in OUTCOMES
    stalled = verify(parse_claim(POSITIVE), {"max_steps": 1})
    assert stalled.outcome == UNRESOLVED
    assert stalled.reason == "simplifier_stall"
    assert stalled.outcome != DISPROVED

    forbidden = verify(parse_claim(FORBIDDEN_SYMBOL))
    assert forbidden.span is not None
    assert forbidden.span.end == forbidden.span.start + 1
    malformed = verify(parse_claim(MALFORMED))
    assert malformed.span is not None
    assert malformed.span.start == 0
    assert malformed.reason == "malformed_claim"

    illegal = verify(parse_claim(ILLEGAL_APPLY))
    assert illegal.violated_obligation == "operator_domain"
    assert illegal.span is not None
    logged = verify(parse_claim(LOG_AT_EXCLUDED))
    assert logged.violated_obligation == "log_domain"
    assert logged.evidence["inverse_evaluated_at_excluded_point"] is False
    inverse = verify(parse_claim(INVERSE_ON_EXCLUDED_POINT))
    assert inverse.violated_obligation == "invertibility"
    assert inverse.evidence["inverse_evaluated_at_excluded_point"] is False
    assigned = verify(parse_claim(INCOMPATIBLE_ASSIGNMENT))
    assert assigned.violated_obligation == "coordinate_assignment"
    dimension = verify(parse_claim(DIMENSION_MISMATCH))
    assert dimension.violated_obligation == "dimension"
    assert "preserves dimension" in dimension.message


def test_multiplication_and_differentiation_do_not_commute():
    after_multiply = parse_operator("d/dx o *(x)")
    before_multiply = parse_operator("*(x) o d/dx")
    assert isinstance(after_multiply, Compose)
    assert isinstance(before_multiply, Compose)
    assert isinstance(after_multiply.parts[0], Partial)
    assert isinstance(after_multiply.parts[1], MultiplyCoefficient)
    assert isinstance(before_multiply.parts[0], MultiplyCoefficient)
    assert isinstance(before_multiply.parts[1], Partial)
    argument = parse_expr("f(x)")
    assert apply_operator(after_multiply, argument) == "f(x) + x * d(f)/d(x)"
    assert apply_operator(before_multiply, argument) == "x * d(f)/d(x)"
    assert operator_difference(after_multiply, before_multiply, argument) == "f(x)"
    normalized = normalize_operator(after_multiply)
    assert [type(part).__name__ for part in normalized.parts] == [
        "Partial",
        "MultiplyCoefficient",
    ]
    sample = parse_expr("f(x)")
    applied = Apply(after_multiply, sample)
    assert applied.to_data()["kind"] == "Apply"
    assert [part["kind"] for part in applied.to_data()["operator"]["parts"]] == [
        "Partial",
        "MultiplyCoefficient",
    ]


def test_serialization_preserves_composition_order_and_direction():
    parsed = _assert_parse_has_no_verdict(ORDER_CLAIM)
    again = deserialize_claim(serialize_claim(parsed))
    formulas = {item.name: item.formula for item in again.claim.operators}
    assert [type(part).__name__ for part in formulas["Forth"].parts] == [
        "Partial",
        "MultiplyCoefficient",
    ]
    assert [type(part).__name__ for part in formulas["Back"].parts] == [
        "MultiplyCoefficient",
        "Partial",
    ]
    directions = {item.name: item.direction for item in again.claim.maps}
    assert directions["Phi"] is None
    assert directions["Psi"] == "old_to_new"
    positive = deserialize_claim(serialize_claim(parse_claim(POSITIVE)))
    d0 = next(item for item in positive.claim.operators if item.name == "D0")
    assert d0.formula.to_data()["kind"] == "OperatorSum"
    first, second = d0.formula.terms
    assert first.to_data()["kind"] == "Compose"
    assert [part.to_data()["kind"] for part in first.parts] == [
        "MultiplyCoefficient",
        "Partial",
    ]
    assert second.to_data()["kind"] == "Partial"
    assert second.variable == "X"
    phi = next(item for item in positive.claim.maps if item.name == "Phi")
    assert phi.direction == "new_to_old"

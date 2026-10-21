"""Independent chain-rule checker.

The claim text is not evidence. A conjugation is ``PROVED`` only when this
module derives it from the recorded maps, under the exclusions and logarithm
branch written on the claim. A zero commutator is recorded as a consistency
note and is never itself a proof of the coordinate change.

The checker sums over the finite coordinates written on the claim. The
handoff's general argument is that sum over i. It does not verify the cited
paper, and it does not fetch the mutable main-branch URL stored as provenance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from neuralese.claims.algebra import (
    Assumptions,
    Budget,
    DivisionByZero,
    EMPTY_ASSUMPTIONS,
    Function,
    Span,
    Stall,
    Symbol,
    add_expr,
    canonicalize,
    contains_log,
    decide_equal,
    denominator_symbols,
    differentiate,
    eval_nf,
    format_nf,
    free_symbols,
    is_blocked,
    is_decided_nonzero,
    is_decided_zero,
    is_neg_one,
    is_one,
    lit,
    log_arguments,
    mul_expr,
    neg_expr,
    nf_add,
    nf_sub,
    substitute,
    symbols_in_nf,
)
from neuralese.claims.syntax import (
    Apply,
    ApplyAssert,
    ClaimRecord,
    Compose,
    ConjugationAssert,
    Identity,
    InvertibleAssert,
    MapComposeAssert,
    MapRecord,
    MultiplyCoefficient,
    OperatorRecord,
    OperatorSum,
    ParseResult,
    Partial,
    PullbackDef,
    SpaceRecord,
    parse_expr,
)

PROVED = "PROVED"
DISPROVED = "DISPROVED"
UNRESOLVED = "UNRESOLVED"
INVALID_INPUT = "INVALID_INPUT"
OUTCOMES = (PROVED, DISPROVED, UNRESOLVED, INVALID_INPUT)

CHECKER_VERSION = "chain-rule-v1"
TACTIC_VERSION = "coordinate-germ-finite-index-1"
SUPPORTED_THEORY = "coordinate_chain_rule_v1"
PAPER_PROVENANCE_URL = (
    "https://github.com/openai/math/blob/main/preprints/"
    "The-irrationality-exponent-of-pi-is-2-September-24-2026/build/sections/interpolation.tex"
)
TRUST_BOUNDARY = (
    "Independent stdlib checker coordinate_chain_rule_v1. "
    "The parsed claim is not evidence. Finite coordinate indices; the identity "
    "is the sum over i. This result does not verify the cited paper."
)
GENERAL_ARGUMENT = (
    "finite coordinate indices; the handoff's general m-coordinate identity is the sum over i"
)

DEFAULT_LIMITS = {"max_steps": 20000, "max_nodes": 20000}


class VerificationResult:
    def __init__(self, **kwargs):
        self.outcome = kwargs["outcome"]
        self.claim_hash = kwargs.get("claim_hash", "")
        self.assumptions = kwargs.get("assumptions", ())
        self.checker_version = CHECKER_VERSION
        self.tactic_version = TACTIC_VERSION
        self.resource_limits = kwargs.get("resource_limits", dict(DEFAULT_LIMITS))
        self.evidence = kwargs.get("evidence", {})
        self.residual = kwargs.get("residual")
        self.witness = kwargs.get("witness")
        self.reason = kwargs.get("reason")
        self.span = kwargs.get("span")
        self.violated_obligation = kwargs.get("violated_obligation")
        self.derivation = tuple(kwargs.get("derivation", ()))
        self.trust_boundary = TRUST_BOUNDARY
        self.message = kwargs.get("message", "")

    def to_dict(self) -> dict:
        span = None
        if self.span is not None:
            span = {"start": self.span.start, "end": self.span.end}
        return {
            "outcome": self.outcome,
            "claim_hash": self.claim_hash,
            "assumptions": list(self.assumptions),
            "checker_version": self.checker_version,
            "tactic_version": self.tactic_version,
            "resource_limits": self.resource_limits,
            "evidence": self.evidence,
            "residual": self.residual,
            "witness": self.witness,
            "reason": self.reason,
            "span": span,
            "violated_obligation": self.violated_obligation,
            "derivation": list(self.derivation),
            "trust_boundary": self.trust_boundary,
            "message": self.message,
        }


def _limits(overrides: Optional[dict]) -> dict:
    merged = dict(DEFAULT_LIMITS)
    if overrides:
        merged.update(overrides)
    return merged


def _invalid(claim_hash: str, assumptions, limits, reason: str, message: str, span: Optional[Span], obligation: Optional[str] = None, evidence: Optional[dict] = None) -> VerificationResult:
    return VerificationResult(
        outcome=INVALID_INPUT,
        claim_hash=claim_hash,
        assumptions=assumptions,
        resource_limits=limits,
        reason=reason,
        message=message,
        span=span,
        violated_obligation=obligation,
        evidence=evidence or {},
    )


def _unresolved(claim_hash: str, assumptions, limits, reason: str, message: str, evidence: Optional[dict] = None, derivation=(), span: Optional[Span] = None) -> VerificationResult:
    return VerificationResult(
        outcome=UNRESOLVED,
        claim_hash=claim_hash,
        assumptions=assumptions,
        resource_limits=limits,
        reason=reason,
        message=message,
        evidence=evidence or {},
        derivation=derivation,
        span=span,
    )


def _space(claim: ClaimRecord, name: str) -> Optional[SpaceRecord]:
    for space in claim.spaces:
        if space.name == name:
            return space
    return None


def _map(claim: ClaimRecord, name: str) -> Optional[MapRecord]:
    for item in claim.maps:
        if item.name == name:
            return item
    return None


def _operator(claim: ClaimRecord, name: str) -> Optional[OperatorRecord]:
    for item in claim.operators:
        if item.name == name:
            return item
    return None


def _pullback_map(claim: ClaimRecord, name: str) -> Optional[MapRecord]:
    for pullback in claim.pullbacks:
        if pullback.name == name:
            return _map(claim, pullback.map_name)
    return _map(claim, name)


def _declared_symbols(claim: ClaimRecord) -> set:
    names = set(claim.constants)
    for space in claim.spaces:
        names.update(space.coordinates)
    for function in claim.functions:
        names.add(function.name)
        names.update(function.deps)
    return names


def _walk_exprs(claim: ClaimRecord) -> List[object]:
    exprs = []
    for space in claim.spaces:
        for _name, expr in space.center:
            exprs.append(expr)
    for item in claim.maps:
        for _name, expr in item.formulas:
            exprs.append(expr)
    for item in claim.operators:
        exprs.extend(_coeff_exprs(item.formula))
    return exprs


def _coeff_exprs(formula: object) -> List[object]:
    if isinstance(formula, MultiplyCoefficient):
        return [formula.coefficient]
    if isinstance(formula, Compose):
        found = []
        for part in formula.parts:
            found.extend(_coeff_exprs(part))
        return found
    if isinstance(formula, OperatorSum):
        found = []
        for term in formula.terms:
            found.extend(_coeff_exprs(term))
        return found
    return []


def _unknown_symbol(claim: ClaimRecord) -> Optional[str]:
    declared = _declared_symbols(claim)
    for expr in _walk_exprs(claim):
        for name in sorted(free_symbols(expr)):
            if name not in declared:
                return name
        if isinstance(expr, Function) and expr.name not in declared and expr.name not in {"log", "exp"}:
            return expr.name
    for item in claim.operators:
        for variable in _partial_variables(item.formula):
            if variable not in declared:
                return variable
    return None


def _partial_variables(formula: object) -> List[str]:
    if isinstance(formula, Partial):
        return [formula.variable]
    if isinstance(formula, Compose):
        found = []
        for part in formula.parts:
            found.extend(_partial_variables(part))
        return found
    if isinstance(formula, OperatorSum):
        found = []
        for term in formula.terms:
            found.extend(_partial_variables(term))
        return found
    return []


def _structural_issue(claim: ClaimRecord) -> Optional[VerificationResult]:
    """Ill-typed structure. Direction is not inferred here."""
    limits = dict(DEFAULT_LIMITS)
    for space in claim.spaces:
        excluded = {name for name, _value in space.exclusions}
        included = {name for name, value in space.includes if value == "0"}
        both = excluded & included
        if both:
            return _invalid(
                claim.source_hash,
                claim.assumptions,
                limits,
                "contradictory_domain",
                f"space {space.name} both excludes and includes {sorted(both)}",
                space.span,
                "domain_consistency",
            )
    for item in claim.maps:
        if item.source_space and _space(claim, item.source_space) is None:
            return _invalid(claim.source_hash, claim.assumptions, limits, "unbound_reference", f"unknown space {item.source_space}", item.span, "domain_compatibility")
        if item.target_space and _space(claim, item.target_space) is None:
            return _invalid(claim.source_hash, claim.assumptions, limits, "unbound_reference", f"unknown space {item.target_space}", item.span, "domain_compatibility")
        source = _space(claim, item.source_space) if item.source_space else None
        target = _space(claim, item.target_space) if item.target_space else None
        if source and target and source.coordinates and target.coordinates:
            if len(source.coordinates) != len(target.coordinates):
                return _invalid(
                    claim.source_hash,
                    claim.assumptions,
                    limits,
                    "incompatible_spaces",
                    "a coordinate change preserves dimension",
                    item.span,
                    "dimension",
                )
        if item.direction and target is not None:
            target_coords = set(target.coordinates)
            for var, _expr in item.formulas:
                if target_coords and var not in target_coords:
                    return _invalid(
                        claim.source_hash,
                        claim.assumptions,
                        limits,
                        "incompatible_spaces",
                        f"formula assigns {var}, which is not a coordinate of target space {target.name}",
                        item.span,
                        "coordinate_assignment",
                    )
    for item in claim.operators:
        if _space(claim, item.domain) is None or _space(claim, item.codomain) is None:
            return _invalid(claim.source_hash, claim.assumptions, limits, "unbound_reference", f"operator {item.name} names an unknown space", item.span, "operator_domain")
        if item.domain != item.codomain:
            return _invalid(
                claim.source_hash,
                claim.assumptions,
                limits,
                "incompatible_spaces",
                f"operator {item.name} domain {item.domain} is not its codomain {item.codomain}",
                item.span,
                "operator_domain",
            )
    for prop in claim.claimed_property:
        if isinstance(prop, ApplyAssert) and prop.operator_name != "log":
            operator = _operator(claim, prop.operator_name)
            function = next((item for item in claim.functions if item.name == prop.target), None)
            if operator is None:
                return _invalid(claim.source_hash, claim.assumptions, limits, "unbound_reference", f"unknown operator {prop.operator_name}", prop.span, "operator_domain")
            if function is None:
                return _invalid(claim.source_hash, claim.assumptions, limits, "unbound_reference", f"unknown function {prop.target}", prop.span, "operator_domain")
            if function.space and function.space != operator.domain:
                return _invalid(
                    claim.source_hash,
                    claim.assumptions,
                    limits,
                    "illegal_operator_application",
                    f"{prop.operator_name} is defined on {operator.domain} and {prop.target} is on {function.space}",
                    prop.span,
                    "operator_domain",
                )
        if isinstance(prop, ApplyAssert) and prop.operator_name == "log" and prop.point is not None:
            var, value = prop.point
            for space in claim.spaces:
                if any(name == var and zero == "0" for name, zero in space.exclusions) and value == "0":
                    return _invalid(
                        claim.source_hash,
                        claim.assumptions,
                        limits,
                        "illegal_operator_application",
                        f"log is not defined at the excluded point {var} = 0",
                        prop.span,
                        "log_domain",
                        {"inverse_evaluated_at_excluded_point": False},
                    )
        if isinstance(prop, ConjugationAssert):
            if _operator(claim, prop.rhs) is None and _map(claim, prop.rhs) is None:
                # RHS is an operator name in the supported claims.
                if _operator(claim, prop.rhs) is None:
                    return _invalid(claim.source_hash, claim.assumptions, limits, "unbound_reference", f"unknown operator {prop.rhs}", prop.span, "claimed_operator")
        if isinstance(prop, MapComposeAssert):
            if _map(claim, prop.outer) is None or _map(claim, prop.inner) is None:
                return _invalid(claim.source_hash, claim.assumptions, limits, "unbound_reference", "map composition names an unknown map", prop.span, "domain_compatibility")
            outer = _map(claim, prop.outer)
            inner = _map(claim, prop.inner)
            if outer and inner and inner.target_space and outer.source_space and inner.target_space != outer.source_space:
                return _invalid(
                    claim.source_hash,
                    claim.assumptions,
                    limits,
                    "incompatible_spaces",
                    f"{prop.inner} lands in {inner.target_space}, not the source of {prop.outer}",
                    prop.span,
                    "domain_compatibility",
                )
        if isinstance(prop, InvertibleAssert):
            if _map(claim, prop.map_name) is None:
                return _invalid(claim.source_hash, claim.assumptions, limits, "unbound_reference", f"unknown map {prop.map_name}", prop.span, "invertibility")
    return None


def _symbols_requiring_exclusion(claim: ClaimRecord) -> List[str]:
    """Symbols that a written logarithm or denominator needs to keep off zero.

    The names come from the formulas. A missing exclusion is not filled in,
    and a claim that never mentions ``Y`` is not reported as missing ``Y``.
    """
    needed: List[str] = []
    seen = set()
    for item in claim.maps:
        for _name, expr in item.formulas:
            for arg in log_arguments(expr):
                if isinstance(arg, Symbol) and arg.name not in seen:
                    seen.add(arg.name)
                    needed.append(arg.name)
            for name in sorted(denominator_symbols(expr)):
                if name not in seen:
                    seen.add(name)
                    needed.append(name)
    return needed


def _summed_coordinates(claim: ClaimRecord) -> List[str]:
    """Coordinates the chain rule sums over, in the written order."""
    for prop in claim.claimed_property:
        if isinstance(prop, ConjugationAssert) and len(prop.parts) >= 2:
            operator = _operator(claim, prop.parts[1][0])
            if operator is not None:
                space = _space(claim, operator.domain)
                if space is not None:
                    return list(space.coordinates)
    for item in claim.operators:
        space = _space(claim, item.domain)
        if space is not None and space.coordinates:
            return list(space.coordinates)
    for space in claim.spaces:
        if space.kind == "coordinates" and space.coordinates:
            return list(space.coordinates)
    return []


def _needs_domain(claim: ClaimRecord) -> bool:
    for prop in claim.claimed_property:
        if isinstance(prop, (ConjugationAssert, MapComposeAssert, InvertibleAssert)):
            return True
    return False


def _formulas_need_log(claim: ClaimRecord) -> bool:
    for item in claim.maps:
        for _name, expr in item.formulas:
            if contains_log(expr):
                return True
    return False


def _singularity_hit(claim: ClaimRecord) -> Optional[Tuple[str, str]]:
    """A written includes-point at which a recorded inverse formula is undefined."""
    included = []
    for space in claim.spaces:
        for name, value in space.includes:
            if value == "0":
                included.append(name)
    if not included:
        return None
    for item in claim.maps:
        for _lhs, expr in item.formulas:
            logged = log_arguments(expr)
            denominators = denominator_symbols(expr)
            for name in included:
                for arg in logged:
                    if isinstance(arg, Symbol) and arg.name == name:
                        return name, item.name
                if name in denominators:
                    return name, item.name
    return None


def build_assumptions(claim: ClaimRecord, budget: Budget) -> Assumptions:
    excluded = set()
    recorded = False
    log1 = False
    contains: List[Tuple[str, str]] = []
    for space in claim.spaces:
        if space.exclude_line_present:
            recorded = True
        for name, _zero in space.exclusions:
            excluded.add(name)
        compact = space.log_branch.replace(" ", "")
        if compact == "log(1)=0":
            log1 = True
        contains.extend(space.contains)
    permissive = Assumptions(
        excluded_zero=frozenset(excluded),
        log1_is_0=log1,
        recorded_exclusions=recorded,
    )
    identifications = []
    for item in claim.maps:
        for name, expr in item.formulas:
            identifications.append((name, canonicalize(expr, permissive, budget, strict=False)))
    return Assumptions(
        excluded_zero=frozenset(excluded),
        log1_is_0=log1,
        contains=tuple(contains),
        identifications=tuple(identifications),
        recorded_exclusions=recorded,
        texts=claim.assumptions,
    )


def as_vector_field(formula: object) -> Optional[Dict[str, object]]:
    if isinstance(formula, OperatorSum):
        merged: Dict[str, object] = {}
        for term in formula.terms:
            part = as_vector_field(term)
            if part is None:
                return None
            for var, coeff in part.items():
                merged[var] = add_expr(merged[var], coeff) if var in merged else coeff
        return merged
    if isinstance(formula, Partial):
        return {formula.variable: lit(1, formula.span)}
    if (
        isinstance(formula, Compose)
        and len(formula.parts) == 2
        and isinstance(formula.parts[0], MultiplyCoefficient)
        and isinstance(formula.parts[1], Partial)
    ):
        return {formula.parts[1].variable: formula.parts[0].coefficient}
    return None


def _format_field(coeffs: Dict[str, object], order: Sequence[str], assumptions: Assumptions, budget: Budget) -> str:
    parts = []
    for var in order:
        coeff = coeffs.get(var)
        if coeff is None:
            continue
        normal = coeff if hasattr(coeff, "terms") else canonicalize(coeff, assumptions, budget)
        if is_decided_zero(normal):
            continue
        if is_one(normal):
            parts.append(f"d/d{var}")
        elif is_neg_one(normal):
            parts.append(f"-d/d{var}")
        else:
            text = format_nf(normal)
            if text.startswith("-") and " " not in text[1:]:
                parts.append(f"{text} * d/d{var}")
            elif " " in text or text.startswith("-"):
                text = f"({text})"
                parts.append(f"{text} * d/d{var}")
            else:
                parts.append(f"{text} * d/d{var}")
    return " + ".join(parts) if parts else "0"


def _table(vf: Dict[str, object], variables: Sequence[str], assumptions: Assumptions, budget: Budget) -> Dict[str, object]:
    table = {}
    for var in variables:
        expr = vf.get(var, lit(0))
        table[var] = canonicalize(expr, assumptions, budget)
    return table


def chain_rule_coefficients(
    operator_name: str,
    vf: Dict[str, object],
    inverse: MapRecord,
    forward: MapRecord,
    assumptions: Assumptions,
    budget: Budget,
) -> Tuple[Dict[str, object], List[dict]]:
    """(T D T^{-1}) coefficients via (D Psi_beta) composed with Phi."""
    substitution = {name: expr for name, expr in forward.formulas}
    substitution_print = {}
    for name, expr in forward.formulas:
        substitution_print[name] = format_nf(canonicalize(expr, assumptions, budget, strict=False))
    coeffs = {}
    steps = []
    for new_name, psi in inverse.formulas:
        total = lit(0)
        partials = []
        for old_name, coeff in vf.items():
            partial = differentiate(psi, old_name, budget)
            product = mul_expr(coeff, partial)
            total = add_expr(total, product)
            partials.append(
                {
                    "variable": old_name,
                    "coefficient": format_nf(canonicalize(coeff, assumptions, budget)),
                    "partial": format_nf(canonicalize(partial, assumptions, budget)),
                    "product": format_nf(canonicalize(product, assumptions, budget)),
                }
            )
        directional = canonicalize(total, assumptions, budget)
        # (D Psi) is simplified in the old coordinates, then composed with Phi.
        # A second pass substitutes the unsimplified tree, and a decided result wins.
        candidates = []
        if not directional.blocked:
            simplified_tree = substitute(_expr_of_nf(directional), substitution, budget)
            candidates.append(canonicalize(simplified_tree, assumptions, budget))
        candidates.append(canonicalize(substitute(total, substitution, budget), assumptions, budget))
        substituted = candidates[-1]
        for candidate in candidates:
            if not is_blocked(candidate):
                substituted = candidate
                break
        coeffs[new_name] = substituted
        steps.append(
            {
                "rule": "chain_rule_coefficient",
                "operator": operator_name,
                "component": new_name,
                "psi": format_nf(canonicalize(psi, assumptions, budget, strict=False)),
                "before_substitution": format_nf(directional),
                "output": format_nf(substituted),
                "partials": partials,
                "substitution": substitution_print,
                "blocked": list(substituted.blocked),
                "formula": "(D Psi) compose Phi",
            }
        )
    return coeffs, steps


def lie_bracket(left: Dict[str, object], right: Dict[str, object], variables: Sequence[str], assumptions: Assumptions, budget: Budget) -> Dict[str, object]:
    out = {}
    for var in variables:
        left_coeff = left.get(var, lit(0))
        right_coeff = right.get(var, lit(0))
        left_on_right = lit(0)
        right_on_left = lit(0)
        for coord in variables:
            left_on_right = add_expr(
                left_on_right,
                mul_expr(left.get(coord, lit(0)), differentiate(right_coeff, coord, budget)),
            )
            right_on_left = add_expr(
                right_on_left,
                mul_expr(right.get(coord, lit(0)), differentiate(left_coeff, coord, budget)),
            )
        out[var] = canonicalize(add_expr(left_on_right, neg_expr(right_on_left)), assumptions, budget)
    return out


def _compose_maps(outer: MapRecord, inner: MapRecord, assumptions: Assumptions, budget: Budget) -> Dict[str, object]:
    mapping = {name: expr for name, expr in inner.formulas}
    results = {}
    for name, expr in outer.formulas:
        results[name] = canonicalize(substitute(expr, mapping, budget), assumptions, budget)
    return results


def _identity_steps(outer: MapRecord, inner: MapRecord, results: Dict[str, object]) -> List[dict]:
    steps = []
    for name, normal in results.items():
        steps.append(
            {
                "rule": "inverse_composition",
                "order": f"{outer.name} o {inner.name}",
                "component": name,
                "output": format_nf(normal),
                "expected": name,
                "blocked": list(normal.blocked),
            }
        )
    return steps


def _apply_table(table: Dict[str, object], variable: str, assumptions: Assumptions, budget: Budget):
    expr = Symbol(variable)
    total = canonicalize(lit(0), assumptions, budget)
    for var, coeff in table.items():
        derivative = canonicalize(differentiate(expr, var, budget), assumptions, budget)
        total = nf_add(total, nf_mul_safe(coeff, derivative, assumptions))
    return total


def nf_mul_safe(left, right, assumptions: Assumptions):
    from neuralese.claims.algebra import nf_mul

    return nf_mul(left, right, assumptions)


def _witness_from_tables(actual: Dict[str, object], claimed: Dict[str, object], variables: Sequence[str], assumptions: Assumptions, budget: Budget) -> Optional[dict]:
    for variable in variables:
        actual_out = _apply_table(actual, variable, assumptions, budget)
        claimed_out = _apply_table(claimed, variable, assumptions, budget)
        residual = nf_sub(actual_out, claimed_out)
        if is_blocked(residual) or is_blocked(actual_out) or is_blocked(claimed_out):
            continue
        if is_decided_zero(residual):
            continue
        witness = {
            "function": variable,
            "actual_output": format_nf(actual_out),
            "claimed_output": format_nf(claimed_out),
            "residual": format_nf(residual),
            "kind": "symbolic",
        }
        point = {name: value for name, value in assumptions.contains}
        needed = symbols_in_nf(residual) | symbols_in_nf(actual_out) | symbols_in_nf(claimed_out)
        if needed and all(name in point for name in needed):
            numeric = {name: _fraction_of(point[name]) for name in needed}
            actual_value = eval_nf(actual_out, numeric)
            claimed_value = eval_nf(claimed_out, numeric)
            residual_value = eval_nf(residual, numeric)
            if actual_value is not None and claimed_value is not None and residual_value is not None:
                witness["point"] = {name: _format_fraction(numeric[name]) for name in sorted(needed)}
                witness["actual_value"] = _format_fraction(actual_value)
                witness["claimed_value"] = _format_fraction(claimed_value)
                witness["residual_value"] = _format_fraction(residual_value)
                witness["kind"] = "in_domain_witness"
        return witness
    return None


def _fraction_of(text: str):
    from fractions import Fraction

    if "/" in text:
        num, den = text.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(int(text))


def _format_fraction(value) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _pullback_arrow(item: MapRecord) -> Optional[Tuple[str, str]]:
    if item.direction in {"new_to_old", "old_to_new"}:
        # T(f) = f compose map sends functions on the target space to functions on the source space.
        return item.target_space, item.source_space
    return None


def _composition_type(claim: ClaimRecord, parts: Sequence[Tuple[str, bool]]) -> Tuple[str, Optional[Tuple[str, str]], Optional[str]]:
    """Return ok/missing/illegal, the arrow, and a message."""
    arrow = None
    for name, inverse in reversed(parts):
        pullback = _pullback_map(claim, name)
        operator = _operator(claim, name)
        if pullback is not None and (operator is None or name in {item.name for item in claim.pullbacks} or name in {item.name for item in claim.maps}):
            # A name that is both would be ambiguous. Maps and operators are distinct in the fixtures.
            if operator is not None and pullback is not None and _operator(claim, name) is not None and _map(claim, name) is None and not any(p.name == name for p in claim.pullbacks):
                piece_arrow = (operator.domain, operator.codomain)
            else:
                piece_arrow = _pullback_arrow(pullback)
                if piece_arrow is None:
                    return "missing", None, name
                if inverse:
                    piece_arrow = (piece_arrow[1], piece_arrow[0])
        elif operator is not None:
            piece_arrow = (operator.domain, operator.codomain)
            if inverse:
                return "illegal", None, f"operator {name} has no recorded inverse"
        else:
            return "illegal", None, f"unknown name {name}"
        if arrow is None:
            arrow = piece_arrow
        elif piece_arrow[0] != arrow[1]:
            return "illegal", None, f"{name} expects {piece_arrow[0]} but the inner operator outputs {arrow[1]}"
        else:
            arrow = (arrow[0], piece_arrow[1])
    return "ok", arrow, None


def _is_standard_pullback(parts: Sequence[Tuple[str, bool]]) -> bool:
    return (
        len(parts) == 3
        and parts[1][1] is False
        and parts[0][1] is False
        and parts[2][1] is True
        and parts[0][0] == parts[2][0]
    )


def normalize_operator(formula: object) -> object:
    """Simplify nothing that would reorder composition. Parts stay in source order."""
    if isinstance(formula, Compose):
        return Compose(tuple(normalize_operator(part) for part in formula.parts), formula.span)
    if isinstance(formula, OperatorSum):
        return OperatorSum(tuple(normalize_operator(term) for term in formula.terms), formula.span)
    return formula


def _apply_tree(operator: object, argument: object, budget: Budget) -> object:
    if isinstance(operator, MultiplyCoefficient):
        return mul_expr(operator.coefficient, argument, span=operator.span)
    if isinstance(operator, Partial):
        return differentiate(argument, operator.variable, budget)
    if isinstance(operator, Compose):
        current = argument
        for part in reversed(operator.parts):
            current = _apply_tree(part, current, budget)
        return current
    if isinstance(operator, OperatorSum):
        return add_expr(*(_apply_tree(term, argument, budget) for term in operator.terms), span=operator.span)
    if isinstance(operator, Identity):
        return argument
    if isinstance(operator, Apply):
        return _apply_tree(operator.operator, operator.argument, budget)
    raise TypeError(f"cannot apply {type(operator).__name__}")


def apply_operator(operator: object, argument: object, assumptions: Optional[Assumptions] = None) -> str:
    budget = Budget()
    tree = _apply_tree(operator, argument, budget)
    normal = canonicalize(tree, assumptions or EMPTY_ASSUMPTIONS, budget)
    return format_nf(normal)


def operator_difference(left: object, right: object, argument: object, assumptions: Optional[Assumptions] = None) -> str:
    budget = Budget()
    assumptions = assumptions or EMPTY_ASSUMPTIONS
    tree = add_expr(
        _apply_tree(left, argument, budget),
        neg_expr(_apply_tree(right, argument, budget)),
    )
    return format_nf(canonicalize(tree, assumptions, budget))


def verify(parsed: ParseResult, limits: Optional[dict] = None) -> VerificationResult:
    resource_limits = _limits(limits)
    if parsed.syntax_error is not None:
        issue = parsed.syntax_error
        return _invalid("", (), resource_limits, issue.reason, issue.message, issue.span)
    claim = parsed.claim
    assert claim is not None
    if claim.encoding == "ffoc":
        return _unresolved(
            claim.source_hash,
            claim.assumptions,
            resource_limits,
            "ffoc_unimplemented",
            "FFOC is not implemented. The claim was not checked by an FFOC encoding.",
            {"ffoc": "unimplemented", "paper_verified": False, "general_argument": GENERAL_ARGUMENT},
        )
    structural = _structural_issue(claim)
    if structural is not None:
        structural.resource_limits = resource_limits
        structural.claim_hash = claim.source_hash
        structural.assumptions = claim.assumptions
        return structural
    unknown = _unknown_symbol(claim)
    if unknown is not None:
        return _unresolved(
            claim.source_hash,
            claim.assumptions,
            resource_limits,
            "unknown_symbol",
            f"symbol {unknown} is well-formed and has no recorded meaning",
            {"symbol": unknown},
        )
    foreign = [item.name for item in claim.operators if item.theory != SUPPORTED_THEORY]
    if foreign:
        return _unresolved(
            claim.source_hash,
            claim.assumptions,
            resource_limits,
            "unsupported_theory",
            f"operators {foreign} are outside {SUPPORTED_THEORY}",
            {"operators": foreign, "paper_verified": False},
        )
    budget = Budget(resource_limits["max_steps"], resource_limits["max_nodes"])
    try:
        return _verify_claim(claim, budget, resource_limits)
    except Stall:
        return _unresolved(
            claim.source_hash,
            claim.assumptions,
            resource_limits,
            "simplifier_stall",
            "simplification exceeded the step budget",
            {"steps": budget.steps, "paper_verified": False},
        )
    except DivisionByZero:
        return _invalid(
            claim.source_hash,
            claim.assumptions,
            resource_limits,
            "division_by_zero",
            "a denominator simplified to zero",
            claim.span,
            "denominator",
        )


def _verify_claim(claim: ClaimRecord, budget: Budget, resource_limits: dict) -> VerificationResult:
    assumptions = build_assumptions(claim, budget)
    if claim.evidence_mode == "commutator_only":
        evidence = _commutator_evidence(claim, assumptions, budget)
        evidence["proves_coordinate_change"] = False
        evidence["proved_by"] = None
        evidence["paper_verified"] = False
        evidence["general_argument"] = GENERAL_ARGUMENT
        return _unresolved(
            claim.source_hash,
            claim.assumptions,
            resource_limits,
            "commutator_is_not_a_proof",
            "a commutator, including a zero commutator, does not prove the coordinate change",
            evidence,
        )
    missing_direction = _missing_direction(claim)
    if missing_direction is not None:
        return _unresolved(
            claim.source_hash,
            claim.assumptions,
            resource_limits,
            "missing_pullback_direction",
            f"map {missing_direction} has no direction; it was not inferred or reordered",
            {
                "map": missing_direction,
                "direction": None,
                "repaired": False,
                "paper_verified": False,
            },
        )
    typed = _type_conjugations(claim)
    if typed is not None:
        return typed if isinstance(typed, VerificationResult) else _invalid(
            claim.source_hash,
            claim.assumptions,
            resource_limits,
            typed[0],
            typed[1],
            typed[2],
            typed[3],
        )
    singularity = _singularity_hit(claim)
    claims_inverse = any(isinstance(prop, (InvertibleAssert, ConjugationAssert, MapComposeAssert)) for prop in claim.claimed_property)
    if singularity is not None and claims_inverse:
        variable, map_name = singularity
        return VerificationResult(
            outcome=DISPROVED,
            claim_hash=claim.source_hash,
            assumptions=claim.assumptions,
            resource_limits=resource_limits,
            reason="inverse_undefined_at_included_point",
            message=f"map {map_name} uses {variable} in log or a denominator, and the domain includes {variable} = 0",
            violated_obligation="invertibility",
            evidence={
                "variable": variable,
                "map": map_name,
                "inverse_evaluated_at_excluded_point": False,
                "primary": "violated_obligation",
                "paper_verified": False,
            },
            residual=None,
            witness=None,
        )
    if _needs_domain(claim) and not assumptions.recorded_exclusions:
        named = _symbols_requiring_exclusion(claim)
        not_supplied = [f"{name} != 0" for name in named] or ["an exclusion line"]
        return _unresolved(
            claim.source_hash,
            claim.assumptions,
            resource_limits,
            "missing_domain",
            "no exclusion was written on the claim; the checker did not supply "
            + ", ".join(not_supplied),
            {"not_supplied": not_supplied, "paper_verified": False},
        )
    if _needs_domain(claim) and _formulas_need_log(claim) and not assumptions.log1_is_0:
        return _unresolved(
            claim.source_hash,
            claim.assumptions,
            resource_limits,
            "missing_log_branch",
            "a logarithm appears and the claim does not record log(1) = 0",
            {"not_supplied": ["log(1) = 0"], "paper_verified": False},
        )
    return _prove(claim, assumptions, budget, resource_limits)


def _missing_direction(claim: ClaimRecord) -> Optional[str]:
    needed = set()
    for pullback in claim.pullbacks:
        needed.add(pullback.map_name)
    for prop in claim.claimed_property:
        if isinstance(prop, ConjugationAssert):
            for name, _inverse in prop.parts:
                mapped = _pullback_map(claim, name)
                if mapped is not None:
                    needed.add(mapped.name)
        if isinstance(prop, (MapComposeAssert, InvertibleAssert)):
            if isinstance(prop, InvertibleAssert):
                needed.add(prop.map_name)
    for name in needed:
        item = _map(claim, name)
        if item is not None and item.direction is None:
            return name
    return None


def _type_conjugations(claim: ClaimRecord):
    for prop in claim.claimed_property:
        if not isinstance(prop, ConjugationAssert):
            continue
        if not _is_standard_pullback(prop.parts):
            return (
                "illegal_operator_application",
                "written order is not T o D o T^-1; the factors were not reordered",
                prop.span,
                "pullback_composition_order",
            )
        status, arrow, detail = _composition_type(claim, prop.parts)
        if status == "missing":
            pullback_map = _pullback_map(claim, prop.parts[0][0])
            if (
                pullback_map is not None
                and pullback_map.direction is not None
                and pullback_map.direction not in {"new_to_old", "old_to_new"}
            ):
                return _unresolved(
                    claim.source_hash,
                    claim.assumptions,
                    dict(DEFAULT_LIMITS),
                    "unrecognized_pullback_direction",
                    (
                        f"map {pullback_map.name} has direction {pullback_map.direction}; "
                        "only new_to_old and old_to_new are supported, and neither was inferred"
                    ),
                    {
                        "map": pullback_map.name,
                        "direction": pullback_map.direction,
                        "repaired": False,
                        "paper_verified": False,
                    },
                )
            return None
        if status == "illegal":
            return (
                "illegal_operator_application",
                f"written composition is ill-typed: {detail}. The factors were not reordered.",
                prop.span,
                "pullback_composition_types",
            )
        rhs = _operator(claim, prop.rhs)
        if rhs is None:
            return ("unbound_reference", f"unknown operator {prop.rhs}", prop.span, "claimed_operator")
        if arrow != (rhs.domain, rhs.codomain):
            return (
                "incompatible_spaces",
                f"composition has type {arrow} and {prop.rhs} has type {(rhs.domain, rhs.codomain)}",
                prop.span,
                "conjugation_codomain",
            )
        pullback_map = _pullback_map(claim, prop.parts[0][0])
        if pullback_map is None or pullback_map.direction != "new_to_old":
            return _unresolved(
                claim.source_hash,
                claim.assumptions,
                dict(DEFAULT_LIMITS),
                "unsupported_pullback",
                "the supported conjugation is T o D o T^-1 for a map with direction new_to_old",
            )
        if not pullback_map.inverse_candidate or _map(claim, pullback_map.inverse_candidate) is None:
            return _unresolved(
                claim.source_hash,
                claim.assumptions,
                dict(DEFAULT_LIMITS),
                "missing_inverse",
                "the forward map has no recorded inverse candidate",
            )
    return None


def _commutator_evidence(claim: ClaimRecord, assumptions: Assumptions, budget: Budget) -> dict:
    pairs = []
    seen = set()
    for prop in claim.claimed_property:
        if isinstance(prop, ConjugationAssert):
            operator = _operator(claim, prop.parts[1][0]) if len(prop.parts) == 3 else None
            if operator is None:
                continue
            key = operator.name
            if key in seen:
                continue
            seen.add(key)
            pairs.append(operator)
    notes = []
    if len(pairs) >= 2:
        left_vf = as_vector_field(pairs[0].formula)
        right_vf = as_vector_field(pairs[1].formula)
        space = _space(claim, pairs[0].domain)
        variables = space.coordinates if space else ()
        if left_vf is not None and right_vf is not None and variables:
            bracket = lie_bracket(left_vf, right_vf, variables, assumptions, budget)
            notes.append(
                {
                    "operators": [pairs[0].name, pairs[1].name],
                    "bracket": _format_field(bracket, variables, assumptions, budget),
                    "is_zero": all(is_decided_zero(bracket[var]) for var in variables),
                    "role": "consistency_only",
                    "proves_coordinate_change": False,
                }
            )
    return {"commutator": notes, "primary": "commutator_only"}


def _prove(claim: ClaimRecord, assumptions: Assumptions, budget: Budget, resource_limits: dict) -> VerificationResult:
    if not any(
        isinstance(prop, (ConjugationAssert, MapComposeAssert, InvertibleAssert, ApplyAssert))
        for prop in claim.claimed_property
    ):
        return _unresolved(
            claim.source_hash,
            claim.assumptions,
            resource_limits,
            "no_claimed_equality",
            "the claim states no equality, conjugation, or invertibility to check",
            {
                "paper_verified": False,
                "general_argument": GENERAL_ARGUMENT,
                "coordinate_index": "finite",
            },
        )
    derivation: List[dict] = []
    derived = {}
    failures = []
    unresolved = None
    for prop in claim.claimed_property:
        if isinstance(prop, ConjugationAssert):
            report = _prove_conjugation(claim, prop, assumptions, budget)
            derivation.extend(report["steps"])
            derived[report["operator"]] = report["derived_print"]
            if report["state"] == "unresolved":
                unresolved = report
                break
            if report["state"] == "disproved":
                failures.append(report)
        elif isinstance(prop, MapComposeAssert):
            report = _prove_identity(claim, prop, assumptions, budget)
            derivation.extend(report["steps"])
            if report["state"] == "unresolved":
                unresolved = report
                break
            if report["state"] == "disproved":
                failures.append(report)
        elif isinstance(prop, InvertibleAssert):
            report = _prove_invertible_record(claim, prop, assumptions, budget)
            derivation.extend(report["steps"])
            if report["state"] == "unresolved":
                unresolved = report
                break
            if report["state"] == "disproved":
                failures.append(report)
        elif isinstance(prop, ApplyAssert):
            return _invalid(
                claim.source_hash,
                claim.assumptions,
                resource_limits,
                "illegal_operator_application",
                "this apply was not discharged by the type checker",
                prop.span,
                "operator_domain",
            )
    summed = _summed_coordinates(claim)
    evidence = {
        "derived": derived,
        "proved_by": None if failures or unresolved else "chain_rule",
        "commutator_proves_claim": False,
        "paper_verified": False,
        "general_argument": GENERAL_ARGUMENT,
        "coordinate_index": "finite",
        "summed_coordinates": summed,
        "summed_coordinate_count": len(summed),
        "symbolic_zero_meaning": "a simplified residual of 0 is equality inside coordinate_chain_rule_v1, not an error bound",
    }
    consistency = _consistency(claim, assumptions, budget)
    if consistency is not None:
        evidence["consistency"] = consistency
    if unresolved is not None:
        evidence["blocked"] = unresolved.get("blocked", [])
        return _unresolved(
            claim.source_hash,
            claim.assumptions,
            resource_limits,
            unresolved["reason"],
            unresolved["message"],
            evidence,
            derivation,
        )
    if failures:
        primary = failures[0]
        evidence["primary"] = primary.get("primary", "witness")
        evidence["failures"] = [
            {key: value for key, value in failure.items() if key in {"operator", "message", "witness", "derived_print", "claimed_print", "primary"}}
            for failure in failures
        ]
        return VerificationResult(
            outcome=DISPROVED,
            claim_hash=claim.source_hash,
            assumptions=claim.assumptions,
            resource_limits=resource_limits,
            reason=primary["reason"],
            message=primary["message"],
            evidence=evidence,
            residual=primary.get("residual"),
            witness=primary.get("witness"),
            violated_obligation=primary.get("violated_obligation"),
            derivation=derivation,
        )
    evidence["primary"] = "chain_rule"
    return VerificationResult(
        outcome=PROVED,
        claim_hash=claim.source_hash,
        assumptions=claim.assumptions,
        resource_limits=resource_limits,
        reason=None,
        message="derived by the chain rule under the recorded assumptions",
        evidence=evidence,
        residual="0",
        witness=None,
        derivation=derivation,
    )


def _prove_conjugation(claim: ClaimRecord, prop: ConjugationAssert, assumptions: Assumptions, budget: Budget) -> dict:
    operator = _operator(claim, prop.parts[1][0])
    claimed = _operator(claim, prop.rhs)
    forward = _pullback_map(claim, prop.parts[0][0])
    inverse = _map(claim, forward.inverse_candidate)
    vf = as_vector_field(operator.formula)
    claimed_vf = as_vector_field(claimed.formula)
    if vf is None or claimed_vf is None:
        return {
            "state": "unresolved",
            "reason": "unsupported_operator",
            "message": "the operator is not a first-order vector field in the supported theory",
            "steps": [],
            "operator": operator.name,
            "derived_print": "",
            "blocked": [],
        }
    actual, steps = chain_rule_coefficients(operator.name, vf, inverse, forward, assumptions, budget)
    new_space = _space(claim, claimed.domain)
    variables = list(new_space.coordinates) if new_space else list(actual.keys())
    for var in variables:
        actual.setdefault(var, canonicalize(lit(0), assumptions, budget))
    claimed_table = _table(claimed_vf, variables, assumptions, budget)
    derived_print = _format_field(actual, variables, assumptions, budget)
    claimed_print = _format_field(claimed_table, variables, assumptions, budget)
    blocked = []
    unequal = False
    for var in variables:
        if is_blocked(actual[var]) or is_blocked(claimed_table[var]):
            blocked.extend(actual[var].blocked)
            blocked.extend(claimed_table[var].blocked)
            continue
        comparison = decide_equal(actual[var], claimed_table[var])
        if comparison is None:
            blocked.append(var)
        elif comparison is False:
            unequal = True
    if blocked and not unequal:
        return {
            "state": "unresolved",
            "reason": "blocked_simplification",
            "message": "a coefficient could not be decided without an assumption the claim does not record",
            "steps": steps,
            "operator": operator.name,
            "derived_print": derived_print,
            "blocked": blocked,
        }
    if unequal or any(
        decide_equal(actual[var], claimed_table[var]) is False
        for var in variables
        if not is_blocked(actual[var]) and not is_blocked(claimed_table[var])
    ):
        witness = _witness_from_tables(actual, claimed_table, variables, assumptions, budget)
        if witness is None and blocked:
            return {
                "state": "unresolved",
                "reason": "blocked_simplification",
                "message": "the operators differ only through an expression the assumptions do not decide",
                "steps": steps,
                "operator": operator.name,
                "derived_print": derived_print,
                "blocked": blocked,
            }
        return {
            "state": "disproved",
            "reason": "symbolic_contradiction" if witness and witness.get("kind") == "symbolic" else "in_domain_witness",
            "message": f"chain rule derived {derived_print}, not {claimed_print}",
            "steps": steps,
            "operator": operator.name,
            "derived_print": derived_print,
            "claimed_print": claimed_print,
            "witness": witness,
            "residual": None if witness is None else witness.get("residual"),
            "primary": "witness",
            "violated_obligation": None,
        }
    return {
        "state": "proved",
        "steps": steps,
        "operator": operator.name,
        "derived_print": derived_print,
    }


def _prove_identity(claim: ClaimRecord, prop: MapComposeAssert, assumptions: Assumptions, budget: Budget) -> dict:
    outer = _map(claim, prop.outer)
    inner = _map(claim, prop.inner)
    results = _compose_maps(outer, inner, assumptions, budget)
    steps = _identity_steps(outer, inner, results)
    blocked = []
    for name, normal in results.items():
        if is_blocked(normal):
            blocked.extend(normal.blocked)
            continue
        expected = canonicalize(Symbol(name), assumptions, budget)
        comparison = decide_equal(normal, expected)
        if comparison is None:
            blocked.append(name)
        elif comparison is False:
            return {
                "state": "disproved",
                "reason": "inverse_not_identity",
                "message": f"{prop.outer} o {prop.inner} sends {name} to {format_nf(normal)}",
                "steps": steps,
                "residual": format_nf(nf_sub(normal, expected)),
                "witness": {
                    "function": name,
                    "actual_output": format_nf(normal),
                    "claimed_output": name,
                    "residual": format_nf(nf_sub(normal, expected)),
                    "kind": "symbolic",
                },
                "primary": "witness",
                "operator": f"{prop.outer} o {prop.inner}",
                "derived_print": format_nf(normal),
                "claimed_print": name,
                "violated_obligation": "invertibility",
            }
    if blocked:
        return {
            "state": "unresolved",
            "reason": "blocked_simplification",
            "message": f"{prop.outer} o {prop.inner} is not decided under the recorded assumptions",
            "steps": steps,
            "blocked": blocked,
        }
    return {"state": "proved", "steps": steps}


def _prove_invertible_record(claim: ClaimRecord, prop: InvertibleAssert, assumptions: Assumptions, budget: Budget) -> dict:
    item = _map(claim, prop.map_name)
    other_name = item.inverse_candidate
    other = _map(claim, other_name) if other_name else None
    if other is None:
        return {
            "state": "unresolved",
            "reason": "missing_inverse",
            "message": f"{prop.map_name} has no recorded inverse candidate",
            "steps": [],
            "blocked": [],
        }
    first = _prove_identity(claim, MapComposeAssert(prop.map_name, other.name, prop.span), assumptions, budget)
    second = _prove_identity(claim, MapComposeAssert(other.name, prop.map_name, prop.span), assumptions, budget)
    steps = first["steps"] + second["steps"]
    for report in (first, second):
        if report["state"] != "proved":
            report = dict(report)
            report["steps"] = steps
            return report
    return {"state": "proved", "steps": steps}


def _consistency(claim: ClaimRecord, assumptions: Assumptions, budget: Budget) -> Optional[dict]:
    operators = []
    for prop in claim.claimed_property:
        if isinstance(prop, ConjugationAssert) and len(prop.parts) == 3:
            operator = _operator(claim, prop.parts[1][0])
            if operator is not None:
                operators.append(operator)
    if len(operators) < 2:
        return None
    left = as_vector_field(operators[0].formula)
    right = as_vector_field(operators[1].formula)
    space = _space(claim, operators[0].domain)
    if left is None or right is None or space is None:
        return None
    variables = space.coordinates
    bracket = lie_bracket(left, right, variables, assumptions, budget)
    record = {
        "operators": [operators[0].name, operators[1].name],
        "bracket": _format_field(bracket, variables, assumptions, budget),
        "is_zero": all(is_decided_zero(bracket[var]) for var in variables),
        "role": "consistency_only",
        "proves_coordinate_change": False,
    }
    # Transform the bracket when the original fields are the two Euler-style or paper fields
    # and a new_to_old pullback is recorded. This is still not the proof.
    forward = None
    for item in claim.maps:
        if item.direction == "new_to_old" and item.inverse_candidate:
            forward = item
            break
    if forward is not None and _map(claim, forward.inverse_candidate) is not None:
        inverse = _map(claim, forward.inverse_candidate)
        transformed, _steps = chain_rule_coefficients(
            f"[{operators[0].name},{operators[1].name}]",
            {var: _expr_of_nf(bracket[var]) for var in variables},
            inverse,
            forward,
            assumptions,
            budget,
        )
        new_space = _space(claim, forward.source_space)
        order = new_space.coordinates if new_space else tuple(transformed)
        record["transformed_bracket"] = _format_field(transformed, order, assumptions, budget)
    return record


def _expr_of_nf(normal) -> object:
    """Rebuild a polynomial/exponential expression the chain rule can differentiate.

    The Lie bracket returns a normal form. The chain rule differentiates
    coefficient trees, so a decided normal form is reparsed from its printed
    form. A blocked form is left as zero only when it is decided zero; otherwise
    the printed form is still the auditable coefficient.
    """
    return parse_expr(format_nf(normal) if normal.terms or not normal.blocked else "0")


def replay_derivation(parsed: ParseResult, result: VerificationResult) -> bool:
    """Recompute recorded chain-rule steps from their printed inputs."""
    if parsed.claim is None:
        return False
    budget = Budget()
    try:
        assumptions = build_assumptions(parsed.claim, budget)
    except Stall:
        return False
    for step in result.derivation:
        if step.get("rule") != "chain_rule_coefficient":
            continue
        if step.get("blocked"):
            continue
        total = canonicalize(lit(0), assumptions, budget)
        for partial in step["partials"]:
            piece = canonicalize(parse_expr(partial["product"]), assumptions, budget)
            total = nf_add(total, piece)
        before = canonicalize(parse_expr(step["before_substitution"]), assumptions, budget)
        if decide_equal(total, before) is not True:
            return False
        mapping = {name: parse_expr(expr) for name, expr in step["substitution"].items()}
        substituted = canonicalize(substitute(parse_expr(step["before_substitution"]), mapping, budget), assumptions, budget)
        expected = canonicalize(parse_expr(step["output"]), assumptions, budget)
        if decide_equal(substituted, expected) is not True:
            return False
    for step in result.derivation:
        if step.get("rule") != "inverse_composition":
            continue
        if step.get("blocked"):
            continue
        if step["output"] != step["expected"] and decide_equal(
            canonicalize(parse_expr(step["output"]), assumptions, budget),
            canonicalize(Symbol(step["expected"]), assumptions, budget),
        ) is not True:
            return False
    return True

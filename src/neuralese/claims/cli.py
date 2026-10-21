"""Deterministic reader for one claim file.

``python3 -m neuralese.claims PATH`` prints the interpreted claim and then
the checker's result. The parser still assigns no verdict. Ambiguous or
incomplete input stays ``UNRESOLVED`` and the clarification names the missing
field without supplying it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from neuralese.claims.algebra import Add, Div, Exp, Function, IntLit, Log, Mul, Neg, Pow, Symbol
from neuralese.claims.syntax import (
    Compose,
    ConjugationAssert,
    FunctionDecl,
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
    parse_claim,
)
from neuralese.claims.verify import VerificationResult, verify


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="neuralese.claims",
        description="Interpret one coordinate-change claim and check it.",
    )
    parser.add_argument("claim", help="path to a claim file, or - to read stdin")
    args = parser.parse_args(argv)
    if args.claim == "-":
        text = sys.stdin.read()
    else:
        path = Path(args.claim)
        if not path.is_file():
            print(f"claim file not found: {path}", file=sys.stderr)
            return 2
        text = path.read_text(encoding="utf-8")
    parsed = parse_claim(text)
    result = verify(parsed)
    sys.stdout.write(render(parsed, result))
    return 0


def render(parsed: ParseResult, result: VerificationResult) -> str:
    lines = _claim_lines(parsed)
    lines.append("result")
    lines.extend(_result_lines(result))
    return "\n".join(lines) + "\n"


def _claim_lines(parsed: ParseResult) -> List[str]:
    lines = ["interpreted claim"]
    if parsed.syntax_error is not None:
        issue = parsed.syntax_error
        lines.append("ast: (not parsed)")
        lines.append("spaces: (not parsed)")
        lines.append("maps: (not parsed)")
        lines.append("assumptions: (not parsed)")
        lines.append(f"span: {issue.span.start}..{issue.span.end}")
        lines.append(f"syntax: {issue.reason}")
        lines.append(f"message: {issue.message}")
        return lines
    claim = parsed.claim
    assert claim is not None
    lines.append(f"source_id: {claim.source_id}")
    lines.append(f"source_hash: {claim.source_hash}")
    lines.append(f"span: {claim.span.start}..{claim.span.end}")
    lines.append("assumptions:")
    if claim.assumptions:
        lines.extend(f"  - {item}" for item in claim.assumptions)
    else:
        lines.append("  - (none)")
    lines.append("interpretation:")
    for item in claim.interpretation_choices:
        lines.append(f"  - {item}")
    lines.append("spaces:")
    if not claim.spaces:
        lines.append("  - (none)")
    for space in claim.spaces:
        lines.append(f"  - {_space_line(space)}")
    lines.append("maps:")
    if not claim.maps:
        lines.append("  - (none)")
    for item in claim.maps:
        lines.append(f"  - {_map_line(item)}")
        formulas = "; ".join(f"{name} = {format_expr(expr)}" for name, expr in item.formulas)
        if formulas:
            lines.append(f"    {formulas}")
    lines.append("operators:")
    if not claim.operators:
        lines.append("  - (none)")
    for item in claim.operators:
        lines.append(f"  - {_operator_line(item)}")
        lines.append(f"    formula {format_operator(item.formula)}")
        lines.append(f"    ast {format_operator_ast(item.formula)}")
    lines.append("pullbacks:")
    if not claim.pullbacks:
        lines.append("  - (none)")
    for item in claim.pullbacks:
        lines.append(f"  - {item.name} of {item.map_name}")
    lines.append("claims:")
    if not claim.claimed_property:
        lines.append("  - (none)")
    for prop in claim.claimed_property:
        lines.append(f"  - {_assert_line(prop)}")
    lines.append("ast:")
    if not claim.ast:
        lines.append("  - (none)")
    for node in claim.ast:
        lines.append(f"  - {_top_line(node)}")
    return lines


def _space_line(space: SpaceRecord) -> str:
    if space.exclude_line_present:
        exclude = ", ".join(f"{name}=0" for name, _zero in space.exclusions) or "none"
    else:
        exclude = "(absent)"
    contains = ", ".join(f"{name}={value}" for name, value in space.contains) or "(none)"
    includes = ", ".join(f"{name}={value}" for name, value in space.includes) or "(none)"
    coordinates = ", ".join(space.coordinates) or "(none)"
    branch = space.log_branch or "(absent)"
    return (
        f"{space.name} kind {space.kind} coordinates {coordinates} "
        f"exclude {exclude} log_branch {branch} contains {contains} includes {includes} "
        f"span {space.span.start}..{space.span.end}"
    )


def _map_line(item: MapRecord) -> str:
    direction = item.direction if item.direction is not None else "(absent)"
    inverse = item.inverse_candidate or "(absent)"
    return (
        f"{item.name} source {item.source_space} target {item.target_space} "
        f"direction {direction} inverse {inverse} span {item.span.start}..{item.span.end}"
    )


def _operator_line(item: OperatorRecord) -> str:
    return (
        f"{item.name} domain {item.domain} codomain {item.codomain} theory {item.theory} "
        f"span {item.span.start}..{item.span.end}"
    )


def _assert_line(prop: object) -> str:
    if isinstance(prop, ConjugationAssert):
        written = " o ".join(f"{name}^-1" if inverse else name for name, inverse in prop.parts)
        return f"Conjugation {written} = {prop.rhs} span {prop.span.start}..{prop.span.end}"
    if isinstance(prop, MapComposeAssert):
        return f"MapCompose {prop.outer} o {prop.inner} = id span {prop.span.start}..{prop.span.end}"
    if isinstance(prop, InvertibleAssert):
        where = f" on {prop.space_name}" if prop.space_name else ""
        return f"Invertible {prop.map_name}{where} span {prop.span.start}..{prop.span.end}"
    return f"{type(prop).__name__} span {getattr(prop, 'span').start}..{getattr(prop, 'span').end}"


def _top_line(node: object) -> str:
    if isinstance(node, SpaceRecord):
        return f"Space {node.name} coordinates {', '.join(node.coordinates)}"
    if isinstance(node, MapRecord):
        direction = node.direction if node.direction is not None else "(absent)"
        return f"Map {node.name} source {node.source_space} target {node.target_space} direction {direction}"
    if isinstance(node, OperatorRecord):
        return f"Operator {node.name} domain {node.domain} codomain {node.codomain} {_operator_ast_brief(node.formula)}"
    if isinstance(node, FunctionDecl):
        return f"Function {node.name}"
    if isinstance(node, PullbackDef):
        return f"Pullback {node.name} of {node.map_name}"
    if isinstance(node, (ConjugationAssert, MapComposeAssert, InvertibleAssert)):
        return _assert_line(node)
    return type(node).__name__


def _result_lines(result: VerificationResult) -> List[str]:
    lines = [f"outcome: {result.outcome}"]
    if result.reason:
        lines.append(f"reason: {result.reason}")
    if result.span is not None:
        lines.append(f"span: {result.span.start}..{result.span.end}")
    if result.violated_obligation:
        lines.append(f"violated_obligation: {result.violated_obligation}")
    if result.message:
        lines.append(f"message: {result.message}")
    if result.outcome == "UNRESOLVED":
        lines.append(f"clarification: {_clarification(result)}")
    if result.outcome == "PROVED":
        lines.append(f"residual: {result.residual}")
        lines.append("derivation:")
        lines.extend(_derivation_lines(result))
    elif result.outcome == "DISPROVED":
        lines.append("counterexample:")
        lines.extend(_counterexample_lines(result))
        if result.derivation:
            lines.append("derivation:")
            lines.extend(_derivation_lines(result))
    return lines


def _clarification(result: VerificationResult) -> str:
    reason = result.reason or ""
    if reason == "missing_pullback_direction":
        return "the map has no direction field; none was inferred"
    if reason == "unrecognized_pullback_direction":
        return "the direction is not new_to_old or old_to_new; neither was inferred"
    if reason == "missing_log_branch":
        return "no log_branch was written; log(1) = 0 was not assumed"
    if reason == "missing_domain":
        supplied = ", ".join(result.evidence.get("not_supplied") or [])
        return f"no exclusion was written; the checker did not supply {supplied}"
    if reason == "no_claimed_equality":
        return "state an equality, a conjugation, or invertibility"
    if reason == "unsupported_operator":
        return "the operator is outside first-order coordinate_chain_rule_v1"
    if reason == "unsupported_theory":
        return "the theory is not coordinate_chain_rule_v1"
    if reason == "ffoc_unimplemented":
        return "encoding ffoc has no term grammar in this repository; the claim was not checked"
    if reason == "commutator_is_not_a_proof":
        return "a commutator was the only evidence mode; it does not prove the claim"
    if reason in {"blocked_simplification", "simplifier_stall"}:
        return "simplification did not decide the claim; this is not a counterexample"
    if reason == "unknown_symbol":
        return "a symbol has no recorded meaning"
    if reason == "missing_inverse_proof":
        missing = ", ".join(result.evidence.get("missing_inverse_proof") or [])
        return (
            "the inverse identity was not established"
            + (f" ({missing})" if missing else "")
            + "; a supplied inverse is not a proof"
        )
    return "the checker cannot decide this well-formed claim"


def _derivation_lines(result: VerificationResult) -> List[str]:
    lines = []
    for step in result.derivation:
        if step.get("rule") == "chain_rule_coefficient":
            lines.append(
                f"  {step['operator']}.{step['component']}: "
                f"(D Psi) compose Phi :: {step['before_substitution']} -> {step['output']}"
            )
        elif step.get("rule") == "inverse_composition":
            lines.append(f"  {step['order']} {step['component']}: {step['output']}")
    if not lines:
        lines.append("  (none)")
    return lines


def _counterexample_lines(result: VerificationResult) -> List[str]:
    lines = []
    witness = result.witness or {}
    if witness:
        if witness.get("composition"):
            lines.append(f"  composition: {witness.get('composition')}")
        lines.append(f"  function: {witness.get('function')}")
        lines.append(f"  actual: {witness.get('actual_output')}")
        lines.append(f"  claimed: {witness.get('claimed_output')}")
        lines.append(f"  residual: {witness.get('residual')}")
        if witness.get("point"):
            point = ", ".join(f"{name}={value}" for name, value in witness["point"].items())
            lines.append(f"  point: {point}")
            lines.append(f"  actual_value: {witness.get('actual_value')}")
            lines.append(f"  claimed_value: {witness.get('claimed_value')}")
            lines.append(f"  residual_value: {witness.get('residual_value')}")
    else:
        lines.append(f"  {result.message}")
    derived = result.evidence.get("derived") or {}
    if derived:
        shown = "; ".join(f"{name} = {text}" for name, text in derived.items())
        lines.append(f"  derived: {shown}")
    if result.residual is not None:
        lines.append(f"  result_residual: {result.residual}")
    return lines


def format_expr(expr: object) -> str:
    if isinstance(expr, IntLit):
        if expr.value.denominator == 1:
            return str(expr.value.numerator)
        return f"{expr.value.numerator}/{expr.value.denominator}"
    if isinstance(expr, Symbol):
        return expr.name
    if isinstance(expr, Neg):
        inner = format_expr(expr.arg)
        if isinstance(expr.arg, (Add, Neg)):
            return f"-({inner})"
        return f"-{inner}"
    if isinstance(expr, Add):
        chunks: List[str] = []
        for arg in expr.args:
            if isinstance(arg, Neg):
                inner = format_expr(arg.arg)
                if isinstance(arg.arg, Add):
                    inner = f"({inner})"
                chunks.append(f"- {inner}" if chunks else f"-{inner}")
            elif isinstance(arg, IntLit) and arg.value < 0:
                piece = format_expr(IntLit(-arg.value, arg.span))
                chunks.append(f"- {piece}" if chunks else f"-{piece}")
            else:
                piece = format_expr(arg)
                chunks.append(piece if not chunks else f"+ {piece}")
        return " ".join(chunks) if chunks else "0"
    if isinstance(expr, Mul):
        return " * ".join(_paren_sum(arg) for arg in expr.args)
    if isinstance(expr, Div):
        return f"{_paren_sum(expr.num)} / {_paren_sum(expr.den)}"
    if isinstance(expr, Pow):
        base = _paren_sum(expr.base)
        if expr.exp < 0:
            return f"{base}^{expr.exp}"
        return f"{base}^{expr.exp}"
    if isinstance(expr, Log):
        return f"log({format_expr(expr.arg)})"
    if isinstance(expr, Exp):
        return f"exp({format_expr(expr.arg)})"
    if isinstance(expr, Function):
        return f"{expr.name}({', '.join(expr.deps)})"
    return type(expr).__name__


def _paren_sum(expr: object) -> str:
    text = format_expr(expr)
    if isinstance(expr, (Add, Neg)):
        return f"({text})"
    return text


def format_operator(formula: object) -> str:
    if isinstance(formula, OperatorSum):
        return " + ".join(format_operator(term) for term in formula.terms)
    if (
        isinstance(formula, Compose)
        and len(formula.parts) == 2
        and isinstance(formula.parts[0], MultiplyCoefficient)
        and isinstance(formula.parts[1], Partial)
    ):
        return f"{_paren_sum(formula.parts[0].coefficient)} * d/d{formula.parts[1].variable}"
    if isinstance(formula, Compose):
        return " o ".join(format_operator(part) for part in formula.parts)
    if isinstance(formula, Partial):
        return f"d/d{formula.variable}"
    if isinstance(formula, MultiplyCoefficient):
        return f"*({format_expr(formula.coefficient)})"
    return type(formula).__name__


def format_operator_ast(formula: object) -> str:
    if isinstance(formula, OperatorSum):
        inner = ", ".join(format_operator_ast(term) for term in formula.terms)
        return f"OperatorSum({inner})"
    if isinstance(formula, Compose):
        inner = ", ".join(format_operator_ast(part) for part in formula.parts)
        return f"Compose({inner})"
    if isinstance(formula, Partial):
        return f"Partial({formula.variable})"
    if isinstance(formula, MultiplyCoefficient):
        return f"MultiplyCoefficient({format_expr(formula.coefficient)})"
    return type(formula).__name__


def _operator_ast_brief(formula: object) -> str:
    return format_operator_ast(formula)

"""Narrow claim language. Parsing records provenance and does not assign a verdict.

The grammar does not invent a pullback direction, a logarithm branch, or an
exclusion. ``Error_Bound`` is not a field of a parse result. Operator
composition is stored in source order. A vector-field term ``c * d/dv`` is
the composition ``MultiplyCoefficient(c)`` then ``Partial(v)``, which acts
as ``c * df/dv``. That reading is an interpretation choice stored on the
claim. It is not a rewrite that swaps an explicit ``o`` composition.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from neuralese.claims.algebra import (
    NO_SPAN,
    Add,
    Div,
    Exp,
    Expr,
    Function,
    IntLit,
    Log,
    Mul,
    Neg,
    Pow,
    Span,
    Symbol,
    add_expr,
    expr_from_data,
    lit,
    mul_expr,
    neg_expr,
    sym,
)


@dataclass(frozen=True)
class SyntaxIssue:
    reason: str
    span: Span
    message: str
    excerpt: str

    def to_data(self) -> dict:
        return {
            "reason": self.reason,
            "span": self.span.to_data(),
            "message": self.message,
            "excerpt": self.excerpt,
        }


class ClaimSyntax(Exception):
    def __init__(self, issue: SyntaxIssue):
        super().__init__(issue.message)
        self.issue = issue


@dataclass(frozen=True)
class MultiplyCoefficient:
    coefficient: object
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {
            "kind": "MultiplyCoefficient",
            "coefficient": self.coefficient.to_data(),
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class Partial:
    variable: str
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {"kind": "Partial", "variable": self.variable, "span": self.span.to_data()}


@dataclass(frozen=True)
class Compose:
    """Mathematical composition in source order. ``parts[0]`` is outermost."""

    parts: Tuple[object, ...]
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {
            "kind": "Compose",
            "parts": [part.to_data() for part in self.parts],
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class Apply:
    operator: object
    argument: object
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {
            "kind": "Apply",
            "operator": self.operator.to_data(),
            "argument": self.argument.to_data(),
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class OperatorSum:
    terms: Tuple[object, ...]
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {
            "kind": "OperatorSum",
            "terms": [term.to_data() for term in self.terms],
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class Identity:
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {"kind": "Identity", "span": self.span.to_data()}


@dataclass(frozen=True)
class SpaceRecord:
    name: str
    kind: str
    coordinates: Tuple[str, ...]
    ring: str
    regularity: str
    center: Tuple[Tuple[str, object], ...]
    neighborhood: str
    exclusions: Tuple[Tuple[str, str], ...]
    log_branch: str
    contains: Tuple[Tuple[str, str], ...]
    includes: Tuple[Tuple[str, str], ...]
    exclude_line_present: bool
    span: Span

    def to_data(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "coordinates": list(self.coordinates),
            "ring": self.ring,
            "function_space": self.ring,
            "regularity": self.regularity,
            "center": [[name, expr.to_data()] for name, expr in self.center],
            "neighborhood": self.neighborhood,
            "exclusions": [list(item) for item in self.exclusions],
            "log_branch": self.log_branch,
            "contains": [list(item) for item in self.contains],
            "includes": [list(item) for item in self.includes],
            "exclude_line_present": self.exclude_line_present,
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class MapRecord:
    name: str
    source_space: str
    target_space: str
    direction: Optional[str]
    formulas: Tuple[Tuple[str, object], ...]
    inverse_candidate: Optional[str]
    obligations: Tuple[str, ...]
    span: Span

    def to_data(self) -> dict:
        return {
            "name": self.name,
            "source_space": self.source_space,
            "target_space": self.target_space,
            "direction": self.direction,
            "formulas": [[name, expr.to_data()] for name, expr in self.formulas],
            "inverse_candidate": self.inverse_candidate,
            "obligations": list(self.obligations),
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class OperatorRecord:
    name: str
    domain: str
    codomain: str
    theory: str
    formula: object
    span: Span

    def to_data(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "codomain": self.codomain,
            "theory": self.theory,
            "formula": self.formula.to_data(),
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class FunctionDecl:
    name: str
    space: Optional[str]
    deps: Tuple[str, ...]
    span: Span

    def to_data(self) -> dict:
        return {
            "name": self.name,
            "space": self.space,
            "deps": list(self.deps),
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class PullbackDef:
    name: str
    map_name: str
    span: Span

    def to_data(self) -> dict:
        return {"name": self.name, "map": self.map_name, "span": self.span.to_data()}


@dataclass(frozen=True)
class ConjugationAssert:
    parts: Tuple[Tuple[str, bool], ...]
    rhs: str
    span: Span

    def to_data(self) -> dict:
        return {
            "kind": "ConjugationAssert",
            "parts": [{"name": name, "inverse": inverse} for name, inverse in self.parts],
            "rhs": self.rhs,
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class MapComposeAssert:
    outer: str
    inner: str
    span: Span

    def to_data(self) -> dict:
        return {
            "kind": "MapComposeAssert",
            "outer": self.outer,
            "inner": self.inner,
            "rhs": "id",
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class InvertibleAssert:
    map_name: str
    space_name: Optional[str]
    span: Span

    def to_data(self) -> dict:
        return {
            "kind": "InvertibleAssert",
            "map": self.map_name,
            "space": self.space_name,
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class ApplyAssert:
    operator_name: str
    target: str
    point: Optional[Tuple[str, str]]
    span: Span
    apply: Apply

    def to_data(self) -> dict:
        return {
            "kind": "ApplyAssert",
            "operator": self.operator_name,
            "target": self.target,
            "point": list(self.point) if self.point else None,
            "span": self.span.to_data(),
            "apply": self.apply.to_data(),
        }


@dataclass(frozen=True)
class ClaimRecord:
    source_id: str
    source_hash: str
    source_text: str
    span: Span
    ast: Tuple[object, ...]
    interpretation_choices: Tuple[str, ...]
    assumptions: Tuple[str, ...]
    claimed_property: Tuple[object, ...]
    spaces: Tuple[SpaceRecord, ...]
    maps: Tuple[MapRecord, ...]
    operators: Tuple[OperatorRecord, ...]
    functions: Tuple[FunctionDecl, ...]
    constants: Tuple[str, ...]
    pullbacks: Tuple[PullbackDef, ...]
    encoding: Optional[str]
    evidence_mode: Optional[str]

    def to_data(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_hash": self.source_hash,
            "source_text": self.source_text,
            "span": self.span.to_data(),
            "ast": [_node_data(node) for node in self.ast],
            "interpretation_choices": list(self.interpretation_choices),
            "assumptions": list(self.assumptions),
            "claimed_property": [_node_data(node) for node in self.claimed_property],
            "spaces": [space.to_data() for space in self.spaces],
            "maps": [item.to_data() for item in self.maps],
            "operators": [item.to_data() for item in self.operators],
            "functions": [item.to_data() for item in self.functions],
            "constants": list(self.constants),
            "pullbacks": [item.to_data() for item in self.pullbacks],
            "encoding": self.encoding,
            "evidence_mode": self.evidence_mode,
        }


@dataclass(frozen=True)
class ParseResult:
    claim: Optional[ClaimRecord]
    syntax_error: Optional[SyntaxIssue]

    def to_data(self) -> dict:
        return {
            "claim": None if self.claim is None else self.claim.to_data(),
            "syntax_error": None if self.syntax_error is None else self.syntax_error.to_data(),
        }


def _node_data(node: object) -> dict:
    if hasattr(node, "to_data"):
        return node.to_data()
    raise TypeError(f"no serialization for {type(node).__name__}")


_TOKEN_RE = re.compile(
    r"\s+"
    r"|d/d[A-Za-z_][A-Za-z0-9_]*"
    r"|\^-1"
    r"|[A-Za-z_][A-Za-z0-9_]*"
    r"|\d+"
    r"|\^"
    r"|[+\-*/(),=]"
)


@dataclass
class _Tok:
    text: str
    start: int
    end: int


def _lex(text: str, base: int) -> List[_Tok]:
    tokens: List[_Tok] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        match = _TOKEN_RE.match(text, index)
        if match is None or match.start() != index:
            raise ClaimSyntax(
                SyntaxIssue(
                    reason="forbidden_unbound_symbol",
                    span=Span(base + index, base + index + 1),
                    message=f"forbidden token {text[index]!r}",
                    excerpt=text[index],
                )
            )
        piece = match.group(0)
        if not piece.isspace():
            tokens.append(_Tok(piece, base + match.start(), base + match.end()))
        index = match.end()
    return tokens


class _ExprParser:
    def __init__(self, tokens: List[_Tok], stop_vector_star: bool = False):
        self.tokens = tokens
        self.index = 0
        self.stop_vector_star = stop_vector_star

    def peek(self, offset: int = 0) -> Optional[_Tok]:
        pos = self.index + offset
        if pos >= len(self.tokens):
            return None
        return self.tokens[pos]

    def accept(self, text: str) -> Optional[_Tok]:
        tok = self.peek()
        if tok is not None and tok.text == text:
            self.index += 1
            return tok
        return None

    def parse(self) -> object:
        expr = self._parse_sum()
        return expr

    def _parse_sum(self) -> object:
        left = self._parse_product()
        while True:
            tok = self.peek()
            if tok is None or tok.text not in {"+", "-"}:
                break
            if tok.text == "-" and self.peek(1) is not None and self.peek(1).text == "1" and False:
                break
            self.index += 1
            right = self._parse_product()
            span = Span(getattr(left, "span").start, getattr(right, "span").end)
            if tok.text == "+":
                left = add_expr(left, right, span=span)
            else:
                left = add_expr(left, neg_expr(right, tok and Span(tok.start, tok.end)), span=span)
        return left

    def _parse_product(self) -> object:
        left = self._parse_power()
        while True:
            tok = self.peek()
            if tok is None or tok.text not in {"*", "/"}:
                break
            if (
                self.stop_vector_star
                and tok.text == "*"
                and self.peek(1) is not None
                and self.peek(1).text.startswith("d/d")
            ):
                break
            self.index += 1
            right = self._parse_power()
            span = Span(getattr(left, "span").start, getattr(right, "span").end)
            if tok.text == "*":
                left = mul_expr(left, right, span=span)
            else:
                left = Div(left, right, span)
        return left

    def _parse_power(self) -> object:
        base = self._parse_unary()
        tok = self.peek()
        if tok is not None and tok.text == "^-1":
            self.index += 1
            return Pow(base, -1, Span(getattr(base, "span").start, tok.end))
        if tok is not None and tok.text == "^":
            self.index += 1
            exp_tok = self.peek()
            if exp_tok is None or not exp_tok.text.isdigit():
                raise ClaimSyntax(
                    SyntaxIssue(
                        "malformed_claim",
                        Span(tok.start, tok.end),
                        "power needs an integer exponent",
                        tok.text,
                    )
                )
            self.index += 1
            return Pow(base, int(exp_tok.text), Span(getattr(base, "span").start, exp_tok.end))
        return base

    def _parse_unary(self) -> object:
        tok = self.peek()
        if tok is not None and tok.text == "-":
            self.index += 1
            arg = self._parse_unary()
            return neg_expr(arg, Span(tok.start, getattr(arg, "span").end))
        return self._parse_primary()

    def _parse_primary(self) -> object:
        tok = self.peek()
        if tok is None:
            raise ClaimSyntax(
                SyntaxIssue("malformed_claim", Span(0, 0), "expression ended early", "")
            )
        if tok.text == "(":
            self.index += 1
            inner = self._parse_sum()
            close = self.accept(")")
            if close is None:
                raise ClaimSyntax(
                    SyntaxIssue("malformed_claim", Span(tok.start, tok.end), "missing )", tok.text)
                )
            return inner
        if tok.text.isdigit():
            self.index += 1
            return lit(int(tok.text), Span(tok.start, tok.end))
        if tok.text.startswith("d/d"):
            raise ClaimSyntax(
                SyntaxIssue(
                    "malformed_claim",
                    Span(tok.start, tok.end),
                    "partial derivative is not a scalar expression",
                    tok.text,
                )
            )
        if re.match(r"[A-Za-z_]", tok.text):
            self.index += 1
            if self.accept("("):
                arg = self._parse_sum()
                close = self.peek()
                if close is None or close.text != ")":
                    raise ClaimSyntax(
                        SyntaxIssue(
                            "malformed_claim",
                            Span(tok.start, tok.end),
                            f"missing ) after {tok.text}",
                            tok.text,
                        )
                    )
                self.index += 1
                span = Span(tok.start, close.end)
                if tok.text == "log":
                    return Log(arg, span)
                if tok.text == "exp":
                    return Exp(arg, span)
                if isinstance(arg, Symbol):
                    return Function(tok.text, (arg.name,), span)
                raise ClaimSyntax(
                    SyntaxIssue(
                        "malformed_claim",
                        span,
                        "function calls take one symbol argument in this grammar",
                        tok.text,
                    )
                )
            return sym(tok.text, Span(tok.start, tok.end))
        raise ClaimSyntax(
            SyntaxIssue(
                "forbidden_unbound_symbol",
                Span(tok.start, tok.end),
                f"unbound symbol {tok.text}",
                tok.text,
            )
        )


def parse_expr(text: str, base: int = 0, stop_vector_star: bool = False) -> object:
    parser = _ExprParser(_lex(text, base), stop_vector_star=stop_vector_star)
    expr = parser.parse()
    if parser.peek() is not None:
        tok = parser.peek()
        raise ClaimSyntax(
            SyntaxIssue(
                "malformed_claim",
                Span(tok.start, tok.end),
                f"unexpected {tok.text}",
                tok.text,
            )
        )
    return expr


def _parse_operator_atom(parser: _ExprParser) -> object:
    tok = parser.peek()
    if tok is None:
        raise ClaimSyntax(SyntaxIssue("malformed_claim", Span(0, 0), "missing operator", ""))
    if tok.text.startswith("d/d"):
        parser.index += 1
        return Partial(tok.text[3:], Span(tok.start, tok.end))
    if tok.text == "*" and parser.peek(1) is not None and parser.peek(1).text == "(":
        start = tok.start
        parser.index += 1
        parser.accept("(")
        coeff = parser._parse_sum()
        close = parser.accept(")")
        if close is None:
            raise ClaimSyntax(
                SyntaxIssue("malformed_claim", Span(tok.start, tok.end), "missing ) after coefficient", "*")
            )
        return MultiplyCoefficient(coeff, Span(start, close.end))
    start_index = parser.index
    coeff = parser._parse_product()
    star = parser.peek()
    partial = parser.peek(1) if star is not None else None
    if star is None or star.text != "*" or partial is None or not partial.text.startswith("d/d"):
        raise ClaimSyntax(
            SyntaxIssue(
                "malformed_claim",
                Span(tok.start, tok.end),
                "operator term must be a partial, a coefficient times a partial, or a composition",
                tok.text,
            )
        )
    parser.index += 2
    mul = MultiplyCoefficient(coeff, Span(getattr(coeff, "span").start, getattr(coeff, "span").end))
    part = Partial(partial.text[3:], Span(partial.start, partial.end))
    return Compose((mul, part), Span(mul.span.start, part.span.end))


def _parse_compose(parser: _ExprParser) -> object:
    first = _parse_operator_atom(parser)
    parts = [first]
    while parser.accept("o"):
        parts.append(_parse_operator_atom(parser))
    if len(parts) == 1:
        return first
    return Compose(tuple(parts), Span(parts[0].span.start, parts[-1].span.end))


def parse_operator(text: str, base: int = 0) -> object:
    parser = _ExprParser(_lex(text, base), stop_vector_star=True)
    if parser.peek() is None:
        raise ClaimSyntax(SyntaxIssue("malformed_claim", Span(base, base + len(text)), "empty formula", text))
    first = _parse_compose(parser)
    terms = [first]
    while parser.accept("+"):
        terms.append(_parse_compose(parser))
    if parser.peek() is not None:
        tok = parser.peek()
        raise ClaimSyntax(
            SyntaxIssue("malformed_claim", Span(tok.start, tok.end), f"unexpected {tok.text}", tok.text)
        )
    if len(terms) == 1:
        return terms[0]
    return OperatorSum(tuple(terms), Span(terms[0].span.start, terms[-1].span.end))


def _strip_comment(line: str) -> str:
    in_string = False
    for index, char in enumerate(line):
        if char == '"':
            in_string = not in_string
        elif char == "#" and not in_string:
            return line[:index]
    return line


def _logical_lines(text: str) -> List[Tuple[int, str, str]]:
    """Return (offset, stripped line without comment, raw line)."""
    rows = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        body = raw[:-1] if raw.endswith("\n") else raw
        if body.endswith("\r"):
            body = body[:-1]
        stripped = _strip_comment(body).strip()
        if stripped:
            content_start = len(body) - len(body.lstrip())
            rows.append((offset + content_start, stripped, body))
        offset += len(raw)
    return rows


def _bad_char(line: str, offset: int) -> None:
    in_string = False
    for index, char in enumerate(line):
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        allowed = char.isalnum() or char in " \t_+-*/^()=,:"
        if not allowed:
            raise ClaimSyntax(
                SyntaxIssue(
                    "forbidden_unbound_symbol",
                    Span(offset + index, offset + index + 1),
                    f"the grammar forbids {char!r}",
                    char,
                )
            )


def _split_top(text: str, sep: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == sep and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _parse_equations(text: str, base: int) -> List[Tuple[str, object]]:
    rows = []
    for piece in _split_top(text, ","):
        if "=" not in piece:
            raise ClaimSyntax(
                SyntaxIssue("malformed_claim", Span(base, base + len(text)), f"expected name = expr in {piece}", piece)
            )
        name, rhs = piece.split("=", 1)
        name = name.strip()
        if not re.match(r"[A-Za-z_][A-Za-z0-9_]*$", name):
            raise ClaimSyntax(
                SyntaxIssue("forbidden_unbound_symbol", Span(base, base + len(name)), f"bad name {name}", name)
            )
        expr = parse_expr(rhs.strip(), base + text.index(rhs))
        rows.append((name, expr))
    return rows


def _span_of(offset: int, text: str) -> Span:
    return Span(offset, offset + len(text))


@dataclass
class _Builder:
    source_id: str = ""
    encoding: Optional[str] = None
    evidence_mode: Optional[str] = None
    constants: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    spaces: List[SpaceRecord] = field(default_factory=list)
    maps: List[MapRecord] = field(default_factory=list)
    operators: List[OperatorRecord] = field(default_factory=list)
    functions: List[FunctionDecl] = field(default_factory=list)
    pullbacks: List[PullbackDef] = field(default_factory=list)
    claims: List[object] = field(default_factory=list)
    ast: List[object] = field(default_factory=list)
    interpretations: List[str] = field(default_factory=list)


def _parse_block(header: str, body: List[Tuple[int, str]], kind: str, builder: _Builder, span: Span) -> None:
    if kind == "space":
        name = header.split(None, 1)[1]
        data = {
            "kind": "",
            "coordinates": (),
            "ring": "",
            "regularity": "",
            "center": (),
            "neighborhood": "",
            "exclusions": [],
            "log_branch": "",
            "contains": [],
            "includes": [],
            "exclude_line_present": False,
        }
        for offset, line in body:
            head, _, rest = line.partition(" ")
            if head == "kind":
                data["kind"] = rest.strip()
            elif head == "coordinates":
                data["coordinates"] = tuple(part.strip() for part in rest.split(",") if part.strip())
            elif head == "ring":
                data["ring"] = rest.strip()
            elif head == "regularity":
                data["regularity"] = rest.strip()
            elif head == "neighborhood":
                data["neighborhood"] = rest.strip()
            elif head == "center":
                data["center"] = tuple(_parse_equations(rest.strip(), offset + line.index(rest)))
            elif head == "exclude":
                data["exclude_line_present"] = True
                if rest.strip() != "none":
                    for var, expr in _parse_equations(rest.strip(), offset + line.index(rest)):
                        if not (isinstance(expr, IntLit) and expr.value == 0):
                            raise ClaimSyntax(
                                SyntaxIssue(
                                    "malformed_claim",
                                    _span_of(offset, line),
                                    "exclude expects name = 0 or none",
                                    line,
                                )
                            )
                        data["exclusions"].append((var, "0"))
            elif head == "log_branch":
                data["log_branch"] = rest.strip()
            elif head == "contains":
                for var, expr in _parse_equations(rest.strip(), offset + line.index(rest)):
                    data["contains"].append((var, _expr_number(expr)))
            elif head == "includes":
                for var, expr in _parse_equations(rest.strip(), offset + line.index(rest)):
                    data["includes"].append((var, _expr_number(expr)))
            else:
                raise ClaimSyntax(
                    SyntaxIssue("malformed_claim", _span_of(offset, line), f"unknown space field {head}", line)
                )
        if data["kind"] not in {"coordinates", "scalar_field"}:
            raise ClaimSyntax(
                SyntaxIssue("malformed_claim", span, "space kind must be coordinates or scalar_field", header)
            )
        record = SpaceRecord(
            name=name,
            kind=data["kind"],
            coordinates=data["coordinates"],
            ring=data["ring"],
            regularity=data["regularity"],
            center=data["center"],
            neighborhood=data["neighborhood"],
            exclusions=tuple(data["exclusions"]),
            log_branch=data["log_branch"],
            contains=tuple(data["contains"]),
            includes=tuple(data["includes"]),
            exclude_line_present=data["exclude_line_present"],
            span=span,
        )
        builder.spaces.append(record)
        builder.ast.append(record)
        if record.exclude_line_present:
            builder.assumptions.append(
                f"exclude {', '.join(f'{name}=0' for name, _ in record.exclusions) or 'none'} on {record.name}"
            )
        if record.log_branch:
            builder.assumptions.append(f"log_branch {record.log_branch} on {record.name}")
        return

    if kind == "map":
        name = header.split(None, 1)[1]
        source_space = ""
        target_space = ""
        direction: Optional[str] = None
        inverse = None
        obligations: Tuple[str, ...] = ()
        formulas: List[Tuple[str, object]] = []
        direction_written = False
        for offset, line in body:
            if line.startswith("source "):
                source_space = line.split(None, 1)[1].strip()
            elif line.startswith("target "):
                target_space = line.split(None, 1)[1].strip()
            elif line.startswith("direction "):
                direction_written = True
                direction = line.split(None, 1)[1].strip()
            elif line.startswith("inverse "):
                inverse = line.split(None, 1)[1].strip()
            elif line.startswith("obligations "):
                obligations = tuple(part.strip() for part in line.split(None, 1)[1].split(",") if part.strip())
            elif "=" in line:
                var, rhs = line.split("=", 1)
                formulas.append((var.strip(), parse_expr(rhs.strip(), offset + line.index(rhs))))
            else:
                raise ClaimSyntax(
                    SyntaxIssue("malformed_claim", _span_of(offset, line), f"unknown map field {line}", line)
                )
        if not direction_written:
            direction = None
            builder.interpretations.append(
                f"map {name} has no direction field; pullback direction was not inferred"
            )
        record = MapRecord(
            name=name,
            source_space=source_space,
            target_space=target_space,
            direction=direction,
            formulas=tuple(formulas),
            inverse_candidate=inverse,
            obligations=obligations,
            span=span,
        )
        builder.maps.append(record)
        builder.ast.append(record)
        return

    if kind == "operator":
        name = header.split(None, 1)[1]
        domain = ""
        codomain = ""
        theory = ""
        formula = None
        formula_span = span
        for offset, line in body:
            if line.startswith("domain "):
                domain = line.split(None, 1)[1].strip()
            elif line.startswith("codomain "):
                codomain = line.split(None, 1)[1].strip()
            elif line.startswith("theory "):
                theory = line.split(None, 1)[1].strip()
            elif line.startswith("formula "):
                formula_text = line.split(None, 1)[1]
                formula_span = Span(offset + line.index(formula_text), offset + len(line))
                formula = parse_operator(formula_text, offset + line.index(formula_text))
                _note_operator_reading(formula, builder)
            else:
                raise ClaimSyntax(
                    SyntaxIssue("malformed_claim", _span_of(offset, line), f"unknown operator field {line}", line)
                )
        if formula is None or not domain or not codomain or not theory:
            raise ClaimSyntax(
                SyntaxIssue("malformed_claim", span, f"operator {name} is missing a required field", header)
            )
        record = OperatorRecord(name, domain, codomain, theory, formula, Span(span.start, formula_span.end))
        builder.operators.append(record)
        builder.ast.append(record)
        return

    raise ClaimSyntax(SyntaxIssue("malformed_claim", span, f"unknown block {kind}", header))


def _note_operator_reading(formula: object, builder: _Builder) -> None:
    nodes = formula.terms if isinstance(formula, OperatorSum) else (formula,)
    for node in nodes:
        if (
            isinstance(node, Compose)
            and len(node.parts) == 2
            and isinstance(node.parts[0], MultiplyCoefficient)
            and isinstance(node.parts[1], Partial)
        ):
            builder.interpretations.append(
                "vector-field term c * d/dv is Compose(MultiplyCoefficient(c), Partial(v)); "
                "explicit o composition is not reordered"
            )
        elif isinstance(node, Compose):
            builder.interpretations.append(
                "composition token o is stored in source order in Compose.parts"
            )


def _expr_number(expr: object) -> str:
    if isinstance(expr, IntLit):
        return str(expr.value.numerator) if expr.value.denominator == 1 else f"{expr.value.numerator}/{expr.value.denominator}"
    if isinstance(expr, Neg) and isinstance(expr.arg, IntLit):
        return "-" + _expr_number(expr.arg)
    raise ClaimSyntax(
        SyntaxIssue("malformed_claim", getattr(expr, "span", NO_SPAN), "expected an integer", "")
    )


def _parse_assert(offset: int, line: str, builder: _Builder) -> None:
    body = line[len("assert ") :].strip()
    span = _span_of(offset, line)
    if body.startswith("invertible "):
        rest = body[len("invertible ") :].strip()
        space_name = None
        if " on " in rest:
            map_name, space_name = rest.split(" on ", 1)
            map_name = map_name.strip()
            space_name = space_name.strip()
        else:
            map_name = rest.strip()
        if not map_name:
            raise ClaimSyntax(SyntaxIssue("malformed_claim", span, "invertible needs a map name", line))
        node = InvertibleAssert(map_name, space_name, span)
        builder.claims.append(node)
        builder.ast.append(node)
        return
    if body.startswith("apply "):
        rest = body[len("apply ") :].strip()
        if rest.startswith("log at "):
            point_text = rest[len("log at ") :].strip()
            var, _, value = point_text.partition("=")
            if not value:
                raise ClaimSyntax(SyntaxIssue("malformed_claim", span, "apply log at needs name = integer", line))
            point = (var.strip(), value.strip())
            argument = parse_expr(value.strip(), offset + line.index(value))
            apply = Apply(Log(Symbol("Y"), span), argument, span)
            node = ApplyAssert("log", "point", point, span, apply)
            builder.claims.append(node)
            builder.ast.append(node)
            return
        if " to " not in rest:
            raise ClaimSyntax(SyntaxIssue("malformed_claim", span, "apply needs 'to' or 'log at'", line))
        op_name, target = rest.split(" to ", 1)
        op_name = op_name.strip()
        target = target.strip()
        apply = Apply(Symbol(op_name, span), Symbol(target, span), span)
        node = ApplyAssert(op_name, target, None, span, apply)
        builder.claims.append(node)
        builder.ast.append(node)
        return
    if " = " not in body:
        raise ClaimSyntax(SyntaxIssue("malformed_claim", span, "assert needs an equality or a keyword", line))
    left, rhs = body.split(" = ", 1)
    rhs = rhs.strip()
    pieces = [piece.strip() for piece in left.split(" o ")]
    if len(pieces) == 2 and rhs == "id":
        node = MapComposeAssert(pieces[0], pieces[1], span)
        builder.claims.append(node)
        builder.ast.append(node)
        builder.interpretations.append(
            f"map composition {pieces[0]} o {pieces[1]} keeps source order; outer map is {pieces[0]}"
        )
        return
    parts: List[Tuple[str, bool]] = []
    for piece in pieces:
        inverse = piece.endswith("^-1")
        name = piece[:-3] if inverse else piece
        if not re.match(r"[A-Za-z_][A-Za-z0-9_]*$", name):
            raise ClaimSyntax(SyntaxIssue("malformed_claim", span, f"bad operator piece {piece}", piece))
        parts.append((name, inverse))
    node = ConjugationAssert(tuple(parts), rhs, span)
    builder.claims.append(node)
    builder.ast.append(node)
    written = " o ".join(("{}{}".format(name, "^-1" if inverse else "")) for name, inverse in parts)
    builder.interpretations.append(
        f"generic or pullback composition {written} is stored in that written order and is not reordered"
    )


def parse_claim(text: str) -> ParseResult:
    """Parse ``text``. The result has no verification status and no error bound."""
    try:
        claim = _parse_claim_text(text)
    except ClaimSyntax as exc:
        return ParseResult(claim=None, syntax_error=exc.issue)
    return ParseResult(claim=claim, syntax_error=None)


def _parse_claim_text(text: str) -> ClaimRecord:
    rows = _logical_lines(text)
    for offset, line, raw in rows:
        content_offset = raw.find(line) if line in raw else 0
        # ``offset`` already points at the stripped content when built that way.
        _bad_char(line, offset)
    builder = _Builder()
    index = 0
    while index < len(rows):
        offset, line, _raw = rows[index]
        if line.startswith("source "):
            quoted = line.split(None, 1)[1].strip()
            if not (quoted.startswith('"') and quoted.endswith('"')):
                raise ClaimSyntax(SyntaxIssue("malformed_claim", _span_of(offset, line), "source needs a quoted id", line))
            builder.source_id = quoted[1:-1]
            index += 1
            continue
        if line.startswith("encoding "):
            builder.encoding = line.split(None, 1)[1].strip()
            index += 1
            continue
        if line.startswith("evidence "):
            builder.evidence_mode = line.split(None, 1)[1].strip()
            index += 1
            continue
        if line.startswith("constant "):
            for name in line.split(None, 1)[1].split(","):
                builder.constants.append(name.strip())
            index += 1
            continue
        if line.startswith("assume "):
            builder.assumptions.append(line[len("assume ") :].strip())
            index += 1
            continue
        if line.startswith("function "):
            builder.functions.append(_parse_function(offset, line))
            builder.ast.append(builder.functions[-1])
            index += 1
            continue
        if line.startswith("define "):
            match = re.match(r"define\s+([A-Za-z_][A-Za-z0-9_]*)\s+pullback\s+of\s+([A-Za-z_][A-Za-z0-9_]*)$", line)
            if match is None:
                raise ClaimSyntax(SyntaxIssue("malformed_claim", _span_of(offset, line), "define needs 'name pullback of map'", line))
            node = PullbackDef(match.group(1), match.group(2), _span_of(offset, line))
            builder.pullbacks.append(node)
            builder.ast.append(node)
            builder.interpretations.append(
                f"T-like name {node.name} means f compose {node.map_name}; direction is read only from that map's direction field"
            )
            index += 1
            continue
        if line.startswith("assert "):
            _parse_assert(offset, line, builder)
            index += 1
            continue
        if line.startswith("space ") or line.startswith("map ") or line.startswith("operator "):
            kind = line.split(None, 1)[0]
            block_start = offset
            index += 1
            body: List[Tuple[int, str]] = []
            while index < len(rows) and rows[index][1] != "end":
                body.append((rows[index][0], rows[index][1]))
                index += 1
            if index >= len(rows):
                raise ClaimSyntax(SyntaxIssue("malformed_claim", _span_of(block_start, line), f"unclosed {kind} block", line))
            end_offset, end_line, _end_raw = rows[index]
            _parse_block(line, body, kind, builder, Span(block_start, end_offset + len(end_line)))
            index += 1
            continue
        raise ClaimSyntax(SyntaxIssue("malformed_claim", _span_of(offset, line), f"unknown statement {line}", line))

    if not builder.interpretations:
        builder.interpretations.append(
            "finite coordinate indices are the supported reading; the chain-rule argument is the sum over i"
        )
    else:
        builder.interpretations.append(
            "finite coordinate indices are the supported reading; the chain-rule argument is the sum over i"
        )
    # Preserve order and drop duplicate interpretation sentences.
    seen = set()
    choices = []
    for item in builder.interpretations:
        if item not in seen:
            seen.add(item)
            choices.append(item)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ClaimRecord(
        source_id=builder.source_id,
        source_hash=digest,
        source_text=text,
        span=Span(0, len(text)),
        ast=tuple(builder.ast),
        interpretation_choices=tuple(choices),
        assumptions=tuple(builder.assumptions),
        claimed_property=tuple(builder.claims),
        spaces=tuple(builder.spaces),
        maps=tuple(builder.maps),
        operators=tuple(builder.operators),
        functions=tuple(builder.functions),
        constants=tuple(builder.constants),
        pullbacks=tuple(builder.pullbacks),
        encoding=builder.encoding,
        evidence_mode=builder.evidence_mode,
    )


def _parse_function(offset: int, line: str) -> FunctionDecl:
    body = line[len("function ") :].strip()
    span = _span_of(offset, line)
    if " on " in body:
        name, space = body.split(" on ", 1)
        return FunctionDecl(name.strip(), space.strip(), (), span)
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$", body)
    if match is None:
        raise ClaimSyntax(SyntaxIssue("malformed_claim", span, "function needs 'name on space' or name(var)", line))
    deps = tuple(part.strip() for part in match.group(2).split(",") if part.strip())
    return FunctionDecl(match.group(1), None, deps, span)


def serialize_claim(parsed: ParseResult) -> str:
    return json.dumps(parsed.to_data(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _op_from_data(data: dict) -> object:
    kind = data["kind"]
    span = Span(data["span"][0], data["span"][1])
    if kind == "MultiplyCoefficient":
        return MultiplyCoefficient(expr_from_data(data["coefficient"]), span)
    if kind == "Partial":
        return Partial(data["variable"], span)
    if kind == "Compose":
        return Compose(tuple(_op_from_data(part) for part in data["parts"]), span)
    if kind == "Apply":
        argument = data["argument"]
        arg = _op_from_data(argument) if argument.get("kind") in {
            "MultiplyCoefficient",
            "Partial",
            "Compose",
            "Apply",
            "OperatorSum",
            "Identity",
        } else expr_from_data(argument)
        return Apply(_op_from_data(data["operator"]), arg, span)
    if kind == "OperatorSum":
        return OperatorSum(tuple(_op_from_data(term) for term in data["terms"]), span)
    if kind == "Identity":
        return Identity(span)
    return expr_from_data(data)


def _assert_from_data(data: dict) -> object:
    span = Span(data["span"][0], data["span"][1])
    kind = data["kind"]
    if kind == "ConjugationAssert":
        parts = tuple((item["name"], bool(item["inverse"])) for item in data["parts"])
        return ConjugationAssert(parts, data["rhs"], span)
    if kind == "MapComposeAssert":
        return MapComposeAssert(data["outer"], data["inner"], span)
    if kind == "InvertibleAssert":
        return InvertibleAssert(data["map"], data.get("space"), span)
    if kind == "ApplyAssert":
        point = tuple(data["point"]) if data.get("point") else None
        return ApplyAssert(data["operator"], data["target"], point, span, _op_from_data(data["apply"]))
    raise ValueError(kind)


def deserialize_claim(blob: str) -> ParseResult:
    data = json.loads(blob)
    error = data.get("syntax_error")
    if data.get("claim") is None:
        issue = None
        if error:
            issue = SyntaxIssue(
                error["reason"],
                Span(error["span"][0], error["span"][1]),
                error["message"],
                error["excerpt"],
            )
        return ParseResult(None, issue)
    claim = data["claim"]
    spaces = []
    for item in claim["spaces"]:
        spaces.append(
            SpaceRecord(
                name=item["name"],
                kind=item["kind"],
                coordinates=tuple(item["coordinates"]),
                ring=item["ring"],
                regularity=item["regularity"],
                center=tuple((name, expr_from_data(expr)) for name, expr in item["center"]),
                neighborhood=item["neighborhood"],
                exclusions=tuple(tuple(pair) for pair in item["exclusions"]),
                log_branch=item["log_branch"],
                contains=tuple(tuple(pair) for pair in item["contains"]),
                includes=tuple(tuple(pair) for pair in item["includes"]),
                exclude_line_present=item["exclude_line_present"],
                span=Span(item["span"][0], item["span"][1]),
            )
        )
    maps = []
    for item in claim["maps"]:
        maps.append(
            MapRecord(
                name=item["name"],
                source_space=item["source_space"],
                target_space=item["target_space"],
                direction=item["direction"],
                formulas=tuple((name, expr_from_data(expr)) for name, expr in item["formulas"]),
                inverse_candidate=item["inverse_candidate"],
                obligations=tuple(item["obligations"]),
                span=Span(item["span"][0], item["span"][1]),
            )
        )
    operators = []
    for item in claim["operators"]:
        operators.append(
            OperatorRecord(
                name=item["name"],
                domain=item["domain"],
                codomain=item["codomain"],
                theory=item["theory"],
                formula=_op_from_data(item["formula"]),
                span=Span(item["span"][0], item["span"][1]),
            )
        )
    record = ClaimRecord(
        source_id=claim["source_id"],
        source_hash=claim["source_hash"],
        source_text=claim["source_text"],
        span=Span(claim["span"][0], claim["span"][1]),
        ast=tuple(_op_from_data(node) if "formula" not in node else node for node in []),
        interpretation_choices=tuple(claim["interpretation_choices"]),
        assumptions=tuple(claim["assumptions"]),
        claimed_property=tuple(_assert_from_data(item) for item in claim["claimed_property"]),
        spaces=tuple(spaces),
        maps=tuple(maps),
        operators=tuple(operators),
        functions=tuple(
            FunctionDecl(item["name"], item["space"], tuple(item["deps"]), Span(item["span"][0], item["span"][1]))
            for item in claim["functions"]
        ),
        constants=tuple(claim["constants"]),
        pullbacks=tuple(
            PullbackDef(item["name"], item["map"], Span(item["span"][0], item["span"][1]))
            for item in claim["pullbacks"]
        ),
        encoding=claim["encoding"],
        evidence_mode=claim["evidence_mode"],
    )
    # Rebuild ast from the structured records so composition nodes round-trip.
    ast: List[object] = []
    ast.extend(record.spaces)
    ast.extend(record.maps)
    ast.extend(record.operators)
    ast.extend(record.functions)
    ast.extend(record.pullbacks)
    ast.extend(record.claimed_property)
    record = ClaimRecord(
        source_id=record.source_id,
        source_hash=record.source_hash,
        source_text=record.source_text,
        span=record.span,
        ast=tuple(ast),
        interpretation_choices=record.interpretation_choices,
        assumptions=record.assumptions,
        claimed_property=record.claimed_property,
        spaces=record.spaces,
        maps=record.maps,
        operators=record.operators,
        functions=record.functions,
        constants=record.constants,
        pullbacks=record.pullbacks,
        encoding=record.encoding,
        evidence_mode=record.evidence_mode,
    )
    return ParseResult(record, None)

"""Expression trees and a decidable normal form for the supported theory.

Scalar multiplication is commutative. That fact is not applied to operator
composition; composition lives in ``syntax`` and is never reordered here.
Cancellation of a symbol in a denominator is performed only when that symbol
is listed in the claim's exclusions. ``exp`` is never zero. ``log(1) = 0``
and the inverse identities ``log(exp(t)) = t`` and ``exp(log(Y)) = Y`` are
used only when the claim records that branch and the log argument is known
to avoid zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class Stall(Exception):
    """Simplification exceeded the configured step budget."""


class DivisionByZero(Exception):
    """A denominator simplified to the constant zero."""


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def to_data(self) -> list:
        return [self.start, self.end]


NO_SPAN = Span(-1, -1)


@dataclass
class Budget:
    max_steps: int = 20000
    max_nodes: int = 20000
    steps: int = 0

    def tick(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps or self.steps > self.max_nodes:
            raise Stall()


@dataclass(frozen=True)
class IntLit:
    value: Fraction
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {"kind": "IntLit", "value": _fmt_coeff(self.value), "span": self.span.to_data()}


@dataclass(frozen=True)
class Symbol:
    name: str
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {"kind": "Symbol", "name": self.name, "span": self.span.to_data()}


@dataclass(frozen=True)
class Add:
    args: Tuple[object, ...]
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {"kind": "Add", "args": [a.to_data() for a in self.args], "span": self.span.to_data()}


@dataclass(frozen=True)
class Mul:
    args: Tuple[object, ...]
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {"kind": "Mul", "args": [a.to_data() for a in self.args], "span": self.span.to_data()}


@dataclass(frozen=True)
class Div:
    num: object
    den: object
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {
            "kind": "Div",
            "num": self.num.to_data(),
            "den": self.den.to_data(),
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class Pow:
    base: object
    exp: int
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {
            "kind": "Pow",
            "base": self.base.to_data(),
            "exp": self.exp,
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class Neg:
    arg: object
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {"kind": "Neg", "arg": self.arg.to_data(), "span": self.span.to_data()}


@dataclass(frozen=True)
class Log:
    arg: object
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {"kind": "Log", "arg": self.arg.to_data(), "span": self.span.to_data()}


@dataclass(frozen=True)
class Exp:
    arg: object
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {"kind": "Exp", "arg": self.arg.to_data(), "span": self.span.to_data()}


@dataclass(frozen=True)
class Function:
    """An indeterminate function. ``deps`` are the variables it may depend on."""

    name: str
    deps: Tuple[str, ...]
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {
            "kind": "Function",
            "name": self.name,
            "deps": list(self.deps),
            "span": self.span.to_data(),
        }


@dataclass(frozen=True)
class Derivative:
    name: str
    deps: Tuple[str, ...]
    orders: Tuple[str, ...]
    span: Span = NO_SPAN

    def to_data(self) -> dict:
        return {
            "kind": "Derivative",
            "name": self.name,
            "deps": list(self.deps),
            "orders": list(self.orders),
            "span": self.span.to_data(),
        }


Expr = object


def lit(value: int | Fraction, span: Span = NO_SPAN) -> IntLit:
    return IntLit(Fraction(value), span)


def sym(name: str, span: Span = NO_SPAN) -> Symbol:
    return Symbol(name, span)


def _flatten_add(args: Iterable[object]) -> Tuple[object, ...]:
    flat: List[object] = []
    for arg in args:
        if isinstance(arg, Add):
            flat.extend(arg.args)
        elif isinstance(arg, IntLit) and arg.value == 0:
            continue
        else:
            flat.append(arg)
    return tuple(flat)


def _flatten_mul(args: Iterable[object]) -> Tuple[object, ...]:
    flat: List[object] = []
    for arg in args:
        if isinstance(arg, Mul):
            flat.extend(arg.args)
        elif isinstance(arg, IntLit) and arg.value == 1:
            continue
        else:
            flat.append(arg)
    return tuple(flat)


def add_expr(*args: object, span: Span = NO_SPAN) -> object:
    flat = _flatten_add(args)
    if not flat:
        return lit(0, span)
    if len(flat) == 1:
        return flat[0]
    return Add(flat, span)


def mul_expr(*args: object, span: Span = NO_SPAN) -> object:
    flat = _flatten_mul(args)
    if not flat:
        return lit(1, span)
    if len(flat) == 1:
        return flat[0]
    return Mul(flat, span)


def neg_expr(arg: object, span: Span = NO_SPAN) -> object:
    if isinstance(arg, Neg):
        return arg.arg
    if isinstance(arg, IntLit):
        return IntLit(-arg.value, span)
    return Neg(arg, span)


@dataclass(frozen=True)
class NF:
    """Sum of monomials. ``blocked`` means a needed cancellation is not allowed."""

    terms: Tuple[Tuple["Mono", Fraction], ...] = ()
    blocked: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Mono:
    sym: Tuple[Tuple[str, int], ...] = ()
    logs: Tuple[Tuple[NF, int], ...] = ()
    exps: Tuple[NF, ...] = ()
    funs: Tuple[Tuple[str, int], ...] = ()
    ders: Tuple[Tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Assumptions:
    """Only facts written on the claim. Nothing here is invented by the checker."""

    excluded_zero: frozenset[str] = frozenset()
    log1_is_0: bool = False
    contains: Tuple[Tuple[str, str], ...] = ()
    identifications: Tuple[Tuple[str, NF], ...] = ()
    recorded_exclusions: bool = False
    texts: Tuple[str, ...] = ()


EMPTY_ASSUMPTIONS = Assumptions()


def _fmt_coeff(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _blocked(*parts: Iterable[str]) -> Tuple[str, ...]:
    found = []
    seen = set()
    for part in parts:
        for item in part:
            if item not in seen:
                seen.add(item)
                found.append(item)
    return tuple(found)


def nf_zero() -> NF:
    return NF((), ())


def nf_const(value: Fraction | int) -> NF:
    value = Fraction(value)
    if value == 0:
        return nf_zero()
    return NF(((Mono(), value),), ())


def is_blocked(nf: NF) -> bool:
    return bool(nf.blocked)


def is_decided_zero(nf: NF) -> bool:
    return not nf.terms and not nf.blocked


def is_decided_nonzero(nf: NF) -> bool:
    return bool(nf.terms) and not nf.blocked


def _mono_is_empty(mono: Mono) -> bool:
    return not (mono.sym or mono.logs or mono.exps or mono.funs or mono.ders)


def is_one(nf: NF) -> bool:
    if nf.blocked or len(nf.terms) != 1:
        return False
    mono, coeff = nf.terms[0]
    return coeff == 1 and _mono_is_empty(mono)


def is_neg_one(nf: NF) -> bool:
    if nf.blocked or len(nf.terms) != 1:
        return False
    mono, coeff = nf.terms[0]
    return coeff == -1 and _mono_is_empty(mono)


def is_symbol_nf(nf: NF, name: Optional[str] = None) -> bool:
    if nf.blocked or len(nf.terms) != 1:
        return False
    mono, coeff = nf.terms[0]
    if coeff != 1 or mono.logs or mono.exps or mono.funs or mono.ders:
        return False
    if len(mono.sym) != 1 or mono.sym[0][1] != 1:
        return False
    return name is None or mono.sym[0][0] == name


def symbol_name(nf: NF) -> Optional[str]:
    if not is_symbol_nf(nf):
        return None
    return nf.terms[0][0].sym[0][0]


def is_exp_nf(nf: NF) -> bool:
    if nf.blocked or len(nf.terms) != 1:
        return False
    mono, coeff = nf.terms[0]
    return coeff == 1 and not mono.sym and not mono.logs and not mono.funs and not mono.ders and len(mono.exps) == 1


def is_log_nf(nf: NF) -> bool:
    if nf.blocked or len(nf.terms) != 1:
        return False
    mono, coeff = nf.terms[0]
    return (
        coeff == 1
        and not mono.sym
        and not mono.exps
        and not mono.funs
        and not mono.ders
        and len(mono.logs) == 1
        and mono.logs[0][1] == 1
    )


def negate_nf(nf: NF) -> NF:
    if not nf.terms:
        return NF((), nf.blocked)
    return NF(tuple((mono, -coeff) for mono, coeff in nf.terms), nf.blocked)


def _sort_terms(terms: Dict[Mono, Fraction]) -> Tuple[Tuple[Mono, Fraction], ...]:
    items = [(mono, coeff) for mono, coeff in terms.items() if coeff != 0]
    items.sort(key=lambda item: (_mono_rank(item[0]), format_mono(item[0]), _fmt_coeff(item[1])))
    return tuple(items)


def nf_add(left: NF, right: NF) -> NF:
    acc: Dict[Mono, Fraction] = {}
    for mono, coeff in left.terms:
        acc[mono] = acc.get(mono, Fraction(0)) + coeff
    for mono, coeff in right.terms:
        acc[mono] = acc.get(mono, Fraction(0)) + coeff
    return NF(_sort_terms(acc), _blocked(left.blocked, right.blocked))


def known_nonzero(nf: NF, assumptions: Assumptions, seen: Optional[set] = None) -> bool:
    """True when the claim's written exclusions make ``nf`` nowhere zero."""
    if is_blocked(nf):
        return False
    if len(nf.terms) == 1:
        mono, coeff = nf.terms[0]
        if coeff != 0 and _mono_is_empty(mono):
            return True
        if is_exp_nf(nf):
            return True
        name = symbol_name(nf)
        if name is not None and name in assumptions.excluded_zero:
            return True
    key = format_nf(nf)
    seen = set() if seen is None else seen
    if key in seen:
        return False
    seen.add(key)
    for ident_name, rhs in assumptions.identifications:
        if ident_name in assumptions.excluded_zero and format_nf(rhs) == key:
            return True
        if format_nf(rhs) == key and known_nonzero(nf_symbol(ident_name), assumptions, seen):
            return True
    return False


def nf_symbol(name: str) -> NF:
    return NF(((Mono(sym=((name, 1),)), Fraction(1)),), ())


def _fun_key(name: str, deps: Sequence[str]) -> str:
    return f"{name}({','.join(deps)})"


def _der_key(name: str, orders: Sequence[str]) -> str:
    return f"{name}:{','.join(orders)}"


def mono_mul(left: Mono, right: Mono, assumptions: Assumptions) -> Tuple[Mono, Tuple[str, ...]]:
    blocked: List[str] = []
    sym_powers: Dict[str, int] = {}
    left_sym = dict(left.sym)
    right_sym = dict(right.sym)
    for name, power in list(left.sym) + list(right.sym):
        sym_powers[name] = sym_powers.get(name, 0) + power
    sym_out: List[Tuple[str, int]] = []
    for name in sorted(sym_powers):
        power = sym_powers[name]
        p1 = left_sym.get(name, 0)
        p2 = right_sym.get(name, 0)
        crosses_zero = bool(p1 and p2 and ((p1 > 0) != (p2 > 0)))
        if (crosses_zero or power < 0) and name not in assumptions.excluded_zero:
            blocked.append(f"denominator:{name}")
        if power != 0:
            sym_out.append((name, power))

    log_powers: Dict[NF, int] = {}
    for arg, power in list(left.logs) + list(right.logs):
        log_powers[arg] = log_powers.get(arg, 0) + power
    logs_out: List[Tuple[NF, int]] = []
    for arg, power in log_powers.items():
        if power != 0:
            logs_out.append((arg, power))
    logs_out.sort(key=lambda item: (format_nf(item[0]), item[1]))

    exp_args = list(left.exps) + list(right.exps)
    exps_out: Tuple[NF, ...] = ()
    if exp_args:
        total = exp_args[0]
        for arg in exp_args[1:]:
            total = nf_add(total, arg)
        if total.blocked:
            blocked.extend(total.blocked)
        if not is_decided_zero(total):
            if total.terms or total.blocked:
                exps_out = (total,)

    fun_powers: Dict[str, int] = {}
    for name, power in list(left.funs) + list(right.funs):
        fun_powers[name] = fun_powers.get(name, 0) + power
    funs_out = tuple(sorted((name, power) for name, power in fun_powers.items() if power != 0))

    der_powers: Dict[str, int] = {}
    for name, power in list(left.ders) + list(right.ders):
        der_powers[name] = der_powers.get(name, 0) + power
    ders_out = tuple(sorted((name, power) for name, power in der_powers.items() if power != 0))

    return Mono(tuple(sym_out), tuple(logs_out), exps_out, funs_out, ders_out), _blocked(blocked)


def nf_mul(left: NF, right: NF, assumptions: Assumptions) -> NF:
    if is_decided_zero(left) or is_decided_zero(right):
        return NF((), _blocked(left.blocked, right.blocked))
    acc: Dict[Mono, Fraction] = {}
    extra: List[str] = []
    for mono_l, coeff_l in left.terms:
        for mono_r, coeff_r in right.terms:
            mono, blocked = mono_mul(mono_l, mono_r, assumptions)
            extra.extend(blocked)
            acc[mono] = acc.get(mono, Fraction(0)) + coeff_l * coeff_r
    return NF(_sort_terms(acc), _blocked(left.blocked, right.blocked, extra))


def _mono_inv(mono: Mono) -> Mono:
    return Mono(
        sym=tuple((name, -power) for name, power in mono.sym),
        logs=tuple((arg, -power) for arg, power in mono.logs),
        exps=tuple(negate_nf(arg) for arg in mono.exps),
        funs=tuple((name, -power) for name, power in mono.funs),
        ders=tuple((name, -power) for name, power in mono.ders),
    )


def nf_div(num: NF, den: NF, assumptions: Assumptions) -> NF:
    if is_decided_zero(den):
        raise DivisionByZero()
    if len(den.terms) != 1:
        if (
            not den.blocked
            and not num.blocked
            and num.terms == den.terms
            and known_nonzero(den, assumptions)
        ):
            return nf_const(1)
        reason = ("sum_denominator",)
        if not den.blocked and not num.blocked and num.terms == den.terms:
            reason = ("cancelled_unexcluded_denominator",)
        return NF(num.terms, _blocked(num.blocked, den.blocked, reason))
    den_mono, den_coeff = den.terms[0]
    inv = NF(((_mono_inv(den_mono), Fraction(1) / den_coeff),), ())
    return nf_mul(num, inv, assumptions)


def _log_atom(arg: NF, assumptions: Assumptions, strict: bool) -> NF:
    blocked: Tuple[str, ...] = ()
    if strict and not assumptions.log1_is_0:
        blocked = ("missing_log_branch",)
    elif strict and not known_nonzero(arg, assumptions):
        blocked = ("log_undefined",)
    mono = Mono(logs=((arg, 1),))
    return NF(((mono, Fraction(1)),), _blocked(arg.blocked, blocked))


def _exp_atom(arg: NF) -> NF:
    if is_decided_zero(arg):
        return nf_const(1)
    if not arg.terms and arg.blocked:
        return NF((), arg.blocked)
    return NF(((Mono(exps=(arg,)), Fraction(1)),), arg.blocked)


def format_mono(mono: Mono) -> str:
    nums: List[str] = []
    dens: List[str] = []
    for name, power in mono.sym:
        if power > 0:
            nums.append(name if power == 1 else f"{name}^{power}")
        elif power < 0:
            power = -power
            dens.append(name if power == 1 else f"{name}^{power}")
    for arg, power in mono.logs:
        piece = f"log({format_nf(arg)})"
        if power > 1:
            piece = f"({piece})^{power}"
        elif power < 0:
            power = -power
            piece = f"log({format_nf(arg)})" if power == 1 else f"(log({format_nf(arg)}))^{power}"
            dens.append(piece)
            continue
        nums.append(piece)
    for arg in mono.exps:
        nums.append(f"exp({format_nf(arg)})")
    for name, power in mono.funs:
        nums.append(name if power == 1 else f"{name}^{power}")
    for name, power in mono.ders:
        fun, orders = name.split(":", 1)
        label = f"d({fun})/d({orders})"
        if power > 1:
            label = f"({label})^{power}"
        if power < 0:
            dens.append(label if power == -1 else f"({label})^{-power}")
        else:
            nums.append(label)
    num_s = " * ".join(nums) if nums else "1"
    if not dens:
        return "" if num_s == "1" and not nums else num_s
    den_s = " * ".join(dens)
    if len(dens) > 1 or " " in den_s:
        den_s = f"({den_s})"
    if num_s == "1":
        return f"1 / {den_s}"
    return f"{num_s} / {den_s}"


def _mono_rank(mono: Mono) -> int:
    return 0 if _mono_is_empty(mono) else 1


def format_nf(nf: NF) -> str:
    if not nf.terms:
        return "0"
    pos: List[Tuple[Mono, Fraction]] = []
    neg: List[Tuple[Mono, Fraction]] = []
    ordered = sorted(nf.terms, key=lambda item: (_mono_rank(item[0]), format_mono(item[0])))
    for mono, coeff in ordered:
        if coeff > 0:
            pos.append((mono, coeff))
        elif coeff < 0:
            neg.append((mono, -coeff))

    def positive_piece(mono: Mono, coeff: Fraction) -> str:
        mono_s = format_mono(mono)
        if not mono_s:
            return _fmt_coeff(coeff)
        if coeff == 1:
            return mono_s
        return f"{_fmt_coeff(coeff)} * {mono_s}"

    chunks: List[str] = []
    if pos:
        chunks.append(positive_piece(pos[0][0], pos[0][1]))
        for mono, coeff in pos[1:]:
            chunks.append(" + " + positive_piece(mono, coeff))
        for mono, coeff in neg:
            chunks.append(" - " + positive_piece(mono, coeff))
        return "".join(chunks)
    first = True
    for mono, coeff in neg:
        piece = positive_piece(mono, coeff)
        chunks.append(("-" if first else " - ") + piece)
        first = False
    return "".join(chunks) if chunks else "0"


def differentiate(expr: object, variable: str, budget: Budget) -> object:
    budget.tick()
    span = getattr(expr, "span", NO_SPAN)
    if isinstance(expr, IntLit):
        return lit(0, span)
    if isinstance(expr, Symbol):
        return lit(1, span) if expr.name == variable else lit(0, span)
    if isinstance(expr, Add):
        return add_expr(*(differentiate(arg, variable, budget) for arg in expr.args), span=span)
    if isinstance(expr, Mul):
        parts = expr.args
        summands = []
        for index, part in enumerate(parts):
            factors = []
            for other, piece in enumerate(parts):
                if other == index:
                    factors.append(differentiate(piece, variable, budget))
                else:
                    factors.append(piece)
            summands.append(mul_expr(*factors, span=span))
        return add_expr(*summands, span=span)
    if isinstance(expr, Div):
        left = mul_expr(differentiate(expr.num, variable, budget), expr.den, span=span)
        right = mul_expr(expr.num, differentiate(expr.den, variable, budget), span=span)
        numer = add_expr(left, neg_expr(right, span), span=span)
        return Div(numer, Pow(expr.den, 2, span), span)
    if isinstance(expr, Pow):
        if expr.exp == 0:
            return lit(0, span)
        power = mul_expr(
            lit(expr.exp, span),
            Pow(expr.base, expr.exp - 1, span) if expr.exp != 1 else expr.base,
            differentiate(expr.base, variable, budget),
            span=span,
        )
        return power
    if isinstance(expr, Neg):
        return neg_expr(differentiate(expr.arg, variable, budget), span)
    if isinstance(expr, Log):
        return Div(differentiate(expr.arg, variable, budget), expr.arg, span)
    if isinstance(expr, Exp):
        return mul_expr(Exp(expr.arg, span), differentiate(expr.arg, variable, budget), span=span)
    if isinstance(expr, Function):
        if variable in expr.deps:
            return Derivative(expr.name, expr.deps, (variable,), span)
        return lit(0, span)
    if isinstance(expr, Derivative):
        if variable in expr.deps:
            return Derivative(expr.name, expr.deps, expr.orders + (variable,), span)
        return lit(0, span)
    raise TypeError(f"cannot differentiate {type(expr).__name__}")


def substitute(expr: object, mapping: Dict[str, object], budget: Budget) -> object:
    budget.tick()
    if isinstance(expr, Symbol) and expr.name in mapping:
        return mapping[expr.name]
    if isinstance(expr, IntLit) or isinstance(expr, Symbol):
        return expr
    span = getattr(expr, "span", NO_SPAN)
    if isinstance(expr, Add):
        return add_expr(*(substitute(arg, mapping, budget) for arg in expr.args), span=span)
    if isinstance(expr, Mul):
        return mul_expr(*(substitute(arg, mapping, budget) for arg in expr.args), span=span)
    if isinstance(expr, Div):
        return Div(substitute(expr.num, mapping, budget), substitute(expr.den, mapping, budget), span)
    if isinstance(expr, Pow):
        return Pow(substitute(expr.base, mapping, budget), expr.exp, span)
    if isinstance(expr, Neg):
        return neg_expr(substitute(expr.arg, mapping, budget), span)
    if isinstance(expr, Log):
        return Log(substitute(expr.arg, mapping, budget), span)
    if isinstance(expr, Exp):
        return Exp(substitute(expr.arg, mapping, budget), span)
    if isinstance(expr, Function) or isinstance(expr, Derivative):
        return expr
    raise TypeError(f"cannot substitute in {type(expr).__name__}")


def canonicalize(
    expr: object,
    assumptions: Assumptions,
    budget: Budget,
    strict: bool = True,
) -> NF:
    budget.tick()
    if isinstance(expr, IntLit):
        return nf_const(expr.value)
    if isinstance(expr, Symbol):
        return nf_symbol(expr.name)
    if isinstance(expr, Neg):
        return negate_nf(canonicalize(expr.arg, assumptions, budget, strict))
    if isinstance(expr, Add):
        total = nf_zero()
        for arg in expr.args:
            total = nf_add(total, canonicalize(arg, assumptions, budget, strict))
        return total
    if isinstance(expr, Mul):
        total = nf_const(1)
        for arg in expr.args:
            total = nf_mul(total, canonicalize(arg, assumptions, budget, strict), assumptions)
        return total
    if isinstance(expr, Div):
        return nf_div(
            canonicalize(expr.num, assumptions, budget, strict),
            canonicalize(expr.den, assumptions, budget, strict),
            assumptions,
        )
    if isinstance(expr, Pow):
        base = canonicalize(expr.base, assumptions, budget, strict)
        if expr.exp == 0:
            return nf_const(1)
        if expr.exp < 0:
            return nf_div(nf_const(1), canonicalize(Pow(expr.base, -expr.exp, expr.span), assumptions, budget, strict), assumptions)
        total = nf_const(1)
        for _ in range(expr.exp):
            total = nf_mul(total, base, assumptions)
        return total
    if isinstance(expr, Log):
        arg = canonicalize(expr.arg, assumptions, budget, strict)
        if assumptions.log1_is_0 and is_one(arg):
            return nf_zero()
        if assumptions.log1_is_0 and is_exp_nf(arg):
            return arg.terms[0][0].exps[0]
        return _log_atom(arg, assumptions, strict)
    if isinstance(expr, Exp):
        arg = canonicalize(expr.arg, assumptions, budget, strict)
        if assumptions.log1_is_0 and is_log_nf(arg):
            inner = arg.terms[0][0].logs[0][0]
            if known_nonzero(inner, assumptions):
                return inner
        return _exp_atom(arg)
    if isinstance(expr, Function):
        key = _fun_key(expr.name, expr.deps)
        return NF(((Mono(funs=((key, 1),)), Fraction(1)),), ())
    if isinstance(expr, Derivative):
        key = _der_key(expr.name, expr.orders)
        return NF(((Mono(ders=((key, 1),)), Fraction(1)),), ())
    raise TypeError(f"cannot canonicalize {type(expr).__name__}")


def nf_sub(left: NF, right: NF) -> NF:
    return nf_add(left, negate_nf(right))


def decide_equal(left: NF, right: NF) -> Optional[bool]:
    difference = nf_sub(left, right)
    if difference.blocked:
        return None
    return not difference.terms


def free_symbols(expr: object, found: Optional[set] = None) -> set:
    found = set() if found is None else found
    if isinstance(expr, Symbol):
        found.add(expr.name)
    elif isinstance(expr, Function):
        found.add(expr.name)
        found.update(expr.deps)
    elif isinstance(expr, Derivative):
        found.add(expr.name)
    elif isinstance(expr, (Add, Mul)):
        for arg in expr.args:
            free_symbols(arg, found)
    elif isinstance(expr, Div):
        free_symbols(expr.num, found)
        free_symbols(expr.den, found)
    elif isinstance(expr, Pow):
        free_symbols(expr.base, found)
    elif isinstance(expr, (Neg, Log, Exp)):
        free_symbols(expr.arg, found)
    return found


def contains_log(expr: object) -> bool:
    if isinstance(expr, Log):
        return True
    if isinstance(expr, (Add, Mul)):
        return any(contains_log(arg) for arg in expr.args)
    if isinstance(expr, Div):
        return contains_log(expr.num) or contains_log(expr.den)
    if isinstance(expr, Pow):
        return contains_log(expr.base)
    if isinstance(expr, (Neg, Exp)):
        return contains_log(expr.arg)
    return False


def contains_division(expr: object) -> bool:
    if isinstance(expr, (Div,)):
        return True
    if isinstance(expr, Pow) and expr.exp < 0:
        return True
    if isinstance(expr, (Add, Mul)):
        return any(contains_division(arg) for arg in expr.args)
    if isinstance(expr, (Neg, Log, Exp)):
        return contains_division(expr.arg)
    return False


def log_arguments(expr: object, found: Optional[List[object]] = None) -> List[object]:
    found = [] if found is None else found
    if isinstance(expr, Log):
        found.append(expr.arg)
        log_arguments(expr.arg, found)
    elif isinstance(expr, (Add, Mul)):
        for arg in expr.args:
            log_arguments(arg, found)
    elif isinstance(expr, Div):
        log_arguments(expr.num, found)
        log_arguments(expr.den, found)
    elif isinstance(expr, Pow):
        log_arguments(expr.base, found)
    elif isinstance(expr, (Neg, Exp)):
        log_arguments(expr.arg, found)
    return found


def denominator_symbols(expr: object, found: Optional[set] = None) -> set:
    """Symbol names that appear as a whole denominator or a negative power."""
    found = set() if found is None else found
    if isinstance(expr, Div) and isinstance(expr.den, Symbol):
        found.add(expr.den.name)
        denominator_symbols(expr.num, found)
    elif isinstance(expr, Div):
        denominator_symbols(expr.num, found)
        denominator_symbols(expr.den, found)
    elif isinstance(expr, Pow) and expr.exp < 0 and isinstance(expr.base, Symbol):
        found.add(expr.base.name)
    elif isinstance(expr, (Add, Mul)):
        for arg in expr.args:
            denominator_symbols(arg, found)
    elif isinstance(expr, (Neg, Log, Exp, Pow)):
        target = expr.arg if not isinstance(expr, Pow) else expr.base
        denominator_symbols(target, found)
    return found


def eval_nf(nf: NF, point: Dict[str, Fraction]) -> Optional[Fraction]:
    if nf.blocked:
        return None
    total = Fraction(0)
    for mono, coeff in nf.terms:
        if mono.logs or mono.exps or mono.funs or mono.ders:
            return None
        value = coeff
        for name, power in mono.sym:
            if name not in point:
                return None
            value *= point[name] ** power
        total += value
    return total


def symbols_in_nf(nf: NF) -> set:
    names = set()
    for mono, _coeff in nf.terms:
        for name, _power in mono.sym:
            names.add(name)
        for arg, _power in mono.logs:
            names.update(symbols_in_nf(arg))
        for arg in mono.exps:
            names.update(symbols_in_nf(arg))
    return names


def expr_from_data(data: dict) -> object:
    kind = data["kind"]
    span = Span(data["span"][0], data["span"][1]) if "span" in data else NO_SPAN
    if kind == "IntLit":
        text = str(data["value"])
        if "/" in text:
            num, den = text.split("/", 1)
            return IntLit(Fraction(int(num), int(den)), span)
        return IntLit(Fraction(int(text)), span)
    if kind == "Symbol":
        return Symbol(data["name"], span)
    if kind == "Add":
        return Add(tuple(expr_from_data(arg) for arg in data["args"]), span)
    if kind == "Mul":
        return Mul(tuple(expr_from_data(arg) for arg in data["args"]), span)
    if kind == "Div":
        return Div(expr_from_data(data["num"]), expr_from_data(data["den"]), span)
    if kind == "Pow":
        return Pow(expr_from_data(data["base"]), int(data["exp"]), span)
    if kind == "Neg":
        return Neg(expr_from_data(data["arg"]), span)
    if kind == "Log":
        return Log(expr_from_data(data["arg"]), span)
    if kind == "Exp":
        return Exp(expr_from_data(data["arg"]), span)
    if kind == "Function":
        return Function(data["name"], tuple(data["deps"]), span)
    if kind == "Derivative":
        return Derivative(data["name"], tuple(data["deps"]), tuple(data["orders"]), span)
    raise ValueError(f"unknown expression kind {kind}")

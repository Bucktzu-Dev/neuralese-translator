# Claim verifier

This is a separate subsystem under `src/neuralese/claims/`. It turns a narrow
piece of mathematical text into a typed claim and then checks one class of
statement: the chain rule for a coordinate change with a single index. It is
not part of Neuralese `learn`, `translate`, or `certify`. It does not admit a
runtime. It does not change the certificate in `neuralese.audit`. That
certificate stays `RUNTIME_NOT_AUTHORIZED`. The package version stays
`0.7.0`. Nothing here emits `RESEARCH_PASS`.

## Baseline inspection

Observed before any further change to this checker, on commit `ea4ac03`
(`Name the disproof as the rejected Euler transformation claim`). The newer
wording commits `bfb3993` and `ea4ac03` stay in the history. They were not
reset.

Command: `python3 -m pytest -q`.

Result: `193 passed` in 20.35s. The suite was green. There were no failures.
The previously reported count of 193 matches this run; it is no longer an
unverified report.

Assumptions this inspection found:

1. One coordinate index. The writeup, the interpretation sentence, and
   `evidence["coordinate_index"]` say the executable checker has one index.
   The chain-rule code itself sums every coordinate written on the operator
   and on the inverse. A two-index probe with `D0 = Y d/dY + d/dX1 + d/dX2`,
   `Di = d/dXi`, `Y = 1 + t`, `Xi = ci + ui + log(1 + t)`, and
   `t = Y - 1`, `ui = Xi - ci - log(Y)`, under the germ assumptions, already
   returned `PROVED` with images `(1 + t) d/dt` and `d/dui`. The "one index"
   label is a scope sentence, not a second proof.
2. A commutator is not a proof. `evidence commutator_only` stays
   `UNRESOLVED` when the bracket is zero.
3. The parser has no verdict. `parse_claim` does not assign `PROVED`,
   `DISPROVED`, `UNRESOLVED`, or `INVALID_INPUT`.
4. `Error_Bound` is absent. It is not a field of a parse result, and a later
   `verify` call does not add it to that result.
5. Explicit reseal of identity-1 is unrelated and must stay untouched. That
   path is the certificate profile in `audit.py` and `contracts.py`. Those
   files, `encode.py`, `clustering.py`, `alphabet.py`, the translator, the
   production taus, and package version `0.7.0` are outside this checker.

Questionable behavior found in the same inspection, still unchanged at the
time of this note:

- A text that states no equality is `PROVED`, with residual `0` and
  `proved_by` `chain_rule`. No equality was established.
- Direction `sideways` is not `new_to_old` or `old_to_new`. The checker still
  proved `T o D o T^-1`. An unrecognized direction was treated as the
  supported pullback.
- A missing exclusion always reports `not_supplied: ["Y != 0"]`, including
  when the claim never mentions `Y`.

## What was implemented

The parser records provenance and an abstract syntax tree. It does not assign
an outcome, attach a certificate, or store `Error_Bound`. There is no
`Error_Bound` field on a parse result. `verify` is the only function that
assigns an outcome. The sentence in the claim is not evidence.

The checker implements

```text
(T D T^{-1}) g = sum_beta ((D Psi_beta) composed with Phi) * d/dz_beta g
```

where `Phi` sends the new coordinates to the old coordinates, `T(f) = f`
composed with `Phi`, and `z = Psi(x)` are the inverse coordinates. The
executable checker uses one coordinate index. The handoff's general argument
is the sum over `i`. A coordinate change preserves dimension. A proved
residual of `0` means the two operators are equal inside this theory. It is
not an error bound, and it is not written onto the parse result.

A claimed equality is `PROVED` only when this derivation succeeds under the
exclusions and logarithm branch written on the claim. A well-formed false
claim is `DISPROVED` when the checker finds an in-domain witness or a
symbolic contradiction. Failure to prove is `UNRESOLVED`. Malformed syntax,
incompatible types, and ill-typed operator application are `INVALID_INPUT`
and carry a source span and a reason.

## Supported language

Blocks end with `end`. `#` comments are ignored outside quoted source ids.
The statements are:

- `source "..."`, an immutable hash of the claim text, and the exact span
- `constant`, `assume` (recorded prose; it does not invent an exclusion)
- `space` with kind `coordinates` or `scalar_field`, coordinates, ring,
  regularity, center, neighborhood, `exclude name = 0`, `log_branch`,
  `contains`, and `includes`
- `map` with source, target, formulas, optional `direction`, inverse
  candidate, and obligations
- `operator` with domain, codomain, theory, and an ordered formula
- `function`, `define NAME pullback of MAP`
- `assert` of a conjugation, a map composition equal to `id`, invertibility,
  or an operator application

Expressions are integers, names, `+`, `-`, `*`, `/`, `^` with an integer
exponent, `log`, `exp`, and parentheses. Operator formulas are partials
`d/dv`, a coefficient times a partial, explicit `o` composition, and sums.
The node kinds `MultiplyCoefficient`, `Compose`, and `Apply` stay distinct.
`c * d/dv` is the composition of multiplication by `c` after `d/dv` in
application order: the source `Compose` stores the coefficient first.
Explicit `o` is stored in source order and is not reordered. Multiplication
by `x` and `d/dx` do not commute; `d/dx` after multiplication by `x` differs
from the reverse composition by the identity.

`T(f) = f` composed with `Phi`. For `direction new_to_old`, the supported
conjugation is `T o D o T^-1` (or the map's own name in that same order).
A generic `B^-1 o O o B` is not rewritten into that order. If the direction
field is absent, the claim is `UNRESOLVED`. The checker does not fill in a
missing domain, a missing `log(1) = 0` branch, or a missing direction.

The supported algebraic theory is `coordinate_chain_rule_v1`: first-order
vector fields, the chain rule above, and the inverse compositions on the
recorded branch. `log(1) = 0`, `log(exp(t)) = t`, and `exp(log(Y)) = Y` are
used only when the claim records the branch and the logarithm argument is
known to be nonzero from a written exclusion. Cancellation in a denominator
uses that same exclusion. An operator outside this theory, a simplification
that stalls, or a symbol that is well-formed but has no recorded meaning is
`UNRESOLVED`.

## Outcomes

| Outcome | When the checker returns it |
| --- | --- |
| `PROVED` | The chain rule derives the claim under the recorded assumptions. |
| `DISPROVED` | A well-formed claim has an in-domain witness or a symbolic contradiction. |
| `UNRESOLVED` | An assumption, direction, meaning, or supported theory is missing, or simplification stalled. |
| `INVALID_INPUT` | The input is malformed, ill-typed, or forbidden by the grammar. The result names a span and a reason. |

## Fixtures

The positive fixture is the one-index coordinate change. It is not a
verification of the paper. The mutable URL

`https://github.com/openai/math/blob/main/preprints/The-irrationality-exponent-of-pi-is-2-September-24-2026/build/sections/interpolation.tex`

is stored as provenance text and is not fetched.

```text
D0 = Y d/dY + d/dX
D1 = d/dX
Phi: (t, u) -> (Y, X),  Y = 1 + t,  X = c + u + log(1 + t)
Psi: t = Y - 1,  u = X - c - log(Y)
```

The domain is analytic germs near `Y = 1`, with `log(1) = 0` and `Y = 0`
excluded. Under those assumptions the checker derives `T D0 T^{-1} = (1 + t)
d/dt`, `T D1 T^{-1} = d/du`, and both inverse compositions equal the
identity. `[D0, D1] = 0` is stored as a consistency note. A claim whose only
evidence mode is the commutator stays `UNRESOLVED`.

The rejected Euler transformation claim is not the positive case. The chart
is valid: `t = log Y`, `u = X / Y`, the operators `E0` and `E1`, and the map
are a well-formed coordinate change. The witness disproves the proposed
target operators in the claim that

```text
E0 = Y d/dY + X d/dX
E1 = d/dX
t = log Y,  u = X / Y
```

sends `T E0 T^{-1}` and `T E1 T^{-1}` to `(1 + t) d/dt` and `d/du`. The
checker derives the actual images `T E0 T^{-1} = d/dt` and `T E1 T^{-1} =
exp(-t) d/du` on that valid chart. On `g(t, u) = t` the derived output is
`1` and the claimed output is `1 + t`. The residual is `-t`. The recorded
domain contains `t = 1`, where those outputs are `1` and `2`. That witness
is the disproof. `[E0, E1] = -E1`, and the transformed bracket is
`-exp(-t) d/du`; that note also disagrees with the claim and is not the
primary evidence.

A wrong coefficient, including a missing chain-rule factor, is `DISPROVED`
with the witness or the symbolic residual attached. A claim that asserts
invertibility on a domain that includes a point where the written inverse
uses `log` or division by that coordinate is `DISPROVED`, and the inverse is
not evaluated there. Applying `log` at an excluded point, or applying an
operator on the wrong space, is `INVALID_INPUT` and names the violated
obligation.

## What remains unimplemented

FFOC is not implemented. The grammar recognizes `encoding ffoc` and returns
`UNRESOLVED`. FFOC is not claimed to be complete, terminating, confluent, or
canonical.

There is no benchmark table, pass rate, timing, or compression figure. This
module does not detect hallucinations, does not reduce dimension, and does
not verify an arbitrary paper. The nine SymPy 1.14.0 smoke checks in the
21 October 2026 handoff are prior smoke checks of coefficient algebra. The
script SHA-256 is
`98b836d67804a276e70ded04db119397f5ebdd9ad81053923c2333b28c83649c` and the
reported output was `PASS: 9 assertions; SymPy 1.14.0`. That hash identifies
the earlier script, not this package. The checked-in verifier uses the
standard library and does not require SymPy or a network call.

The executable theory is one coordinate index. Extending the same identity
to the handoff's sum over `i`, and checking a paper-scale argument, are not
done.

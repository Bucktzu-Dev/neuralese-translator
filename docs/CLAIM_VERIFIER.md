# Claim verifier

This is a separate subsystem under `src/neuralese/claims/`. It turns a narrow
piece of mathematical text into a typed claim and then checks one class of
statement: the chain rule for a coordinate change on a finite list of
indices. It is
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
composed with `Phi`, and `z = Psi(x)` are the inverse coordinates. The sum
runs over every coordinate written on the claim. The handoff's general
argument is that sum over `i`. A coordinate change preserves dimension. A
proved residual of `0` is written only after this derivation establishes
equality inside the theory. It is not an error bound, and it is not written
onto the parse result.

The three questionable readings in the baseline inspection are no longer
what the checker does. A text with no equality is `UNRESOLVED`
(`no_claimed_equality`) and carries no residual. A direction other than
`new_to_old` or `old_to_new` is `UNRESOLVED`
(`unrecognized_pullback_direction`) and is not inferred. A missing exclusion
names a symbol that actually occurs as a logarithm argument or a denominator.
If no such symbol is written, the gap is `an exclusion line`, not a supplied
`Y != 0`.

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
field is absent, or the written direction is neither `new_to_old` nor
`old_to_new`, the claim is `UNRESOLVED`. The checker does not fill in a
missing domain, a missing `log(1) = 0` branch, or a missing direction.
Ambiguous input stays `UNRESOLVED` and the command prints a clarification.
It does not invent the missing field. A simplification that stalls is
`UNRESOLVED`, not `DISPROVED`.

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

The same identity with two indices is
`examples/claims/two_index_coordinate_change.txt`. `D0 = Y d/dY + d/dX1 +
d/dX2`, `Di = d/dXi`, `Y = 1 + t`, `Xi = ci + ui + log(1 + t)`, and
`t = Y - 1`, `ui = Xi - ci - log(Y)`, with the same germ assumptions. The
checker derives `T D0 T^{-1} = (1 + t) d/dt` and `T Di T^{-1} = d/dui` by
the chain rule above. A zero commutator on that pair is still not a proof.

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

## Input grammar

This is the whole input language. It is not unrestricted English. A line
whose first word is outside this grammar is `INVALID_INPUT`. `#` starts a
comment outside a quoted source id. Blocks end with a line `end`. Outside
quotes the allowed characters are letters, digits, space, tab, and
`_+-*/^()=, :`.

```text
claim        = { statement }
statement    = source | encoding | evidence | constant | assume
             | function | define | assert | space | map | operator
source       = "source" quoted-id
encoding     = "encoding" word
evidence     = "evidence" word
constant     = "constant" name { "," name }
assume       = "assume" rest-of-line
function     = "function" name "on" name
             | "function" name "(" name { "," name } ")"
define       = "define" name "pullback" "of" name
assert       = "assert" assertion
assertion    = "invertible" name [ "on" name ]
             | "apply" "log" "at" name "=" integer
             | "apply" name "to" name
             | name "o" name "=" "id"
             | piece { "o" piece } "=" name
piece        = name [ "^-1" ]
space        = "space" name newline field-line { field-line } "end"
map          = "map" name newline map-line { map-line } "end"
operator     = "operator" name newline op-line { op-line } "end"
field-line   = "kind" ( "coordinates" | "scalar_field" )
             | "coordinates" name { "," name }
             | "ring" word | "regularity" word | "neighborhood" word
             | "center" equations
             | "exclude" ( "none" | name "=" "0" { "," name "=" "0" } )
             | "log_branch" rest-of-line
             | "contains" equations | "includes" equations
map-line     = "source" name | "target" name | "direction" word
             | "inverse" name | "obligations" name { "," name }
             | name "=" expr
op-line      = "domain" name | "codomain" name | "theory" word
             | "formula" vector
equations    = name "=" expr { "," name "=" expr }
expr         = term { ( "+" | "-" ) term }
term         = factor { ( "*" | "/" ) factor }
factor       = [ "-" ] primary [ "^" unsigned | "^-1" ]
primary      = integer | name | "log" "(" expr ")" | "exp" "(" expr ")"
             | name "(" name ")" | "(" expr ")"
vector       = op-term { "+" op-term }
op-term      = op-atom { "o" op-atom }
op-atom      = "d/d" name
             | coefficient "*" "d/d" name
             | "*(" expr ")"
name         = letter { letter | digit | "_" }
word         = name
unsigned     = digit { digit }
```

`assume` records prose. It does not create an exclusion or a logarithm
branch. `c * d/dv` is stored as `Compose(MultiplyCoefficient(c), Partial(v))`.
Explicit `o` stays in source order. Two names joined by `o` and equal to
`id` are a map composition, with the left name outer. A negative integer
literal inside a sum is stored as that integer; the command prints it with
a minus (`Y - 1`).

`encoding ffoc` is accepted as a directive and is not an FFOC term language.
See the gap below.

## Mathematical scope

The supported theory is `coordinate_chain_rule_v1`.

- First-order vector fields on a finite coordinate tuple. The checked cases
  are two coordinates (`Y`, `X`) and three coordinates (`Y`, `X1`, `X2`).
  The chain rule sums over the coordinates written on the operator's domain.
- Pullback `T(f) = f` composed with `Phi`, and only the written order
  `T o D o T^-1`, when `Phi` has `direction new_to_old`.
- Inverse compositions on the recorded branch.
- `log(1) = 0`, `log(exp(t)) = t`, and `exp(log(Y)) = Y` only when
  `log_branch log(1) = 0` is written and the logarithm argument is kept off
  zero by a written exclusion.
- A `contains` point may turn a symbolic residual into an in-domain witness.
  The point has to lie in the declared domain. A germ claim is not proved by
  an arbitrary sample.
- A commutator, including a zero commutator, is a consistency note. It does
  not prove the coordinate change.

Outside this scope the checker returns `UNRESOLVED` or `INVALID_INPUT`. It
does not try a second theory. Second-order operators such as `d/dx o d/dx`
are unresolved. There is no paper-scale proof.

## Command

From the repository root, after `pip install -e .` or with `PYTHONPATH=src`:

```text
python3 -m neuralese.claims examples/claims/one_index_coordinate_change.txt
```

This command produced the printout below.

```text
interpreted claim
source_id: https://github.com/openai/math/blob/main/preprints/The-irrationality-exponent-of-pi-is-2-September-24-2026/build/sections/interpolation.tex
source_hash: 1d53048d2dc4349ed5608e7a5a857900b538ac3690d124d7f88d91dcbf887934
span: 0..1349
assumptions:
  - analytic germs near Y = 1, log(1) = 0, neighborhood excludes Y = 0
  - exclude Y=0 on old
  - log_branch log(1) = 0 on old
  - exclude Y=0 on new
  - log_branch log(1) = 0 on new
interpretation:
  - vector-field term c * d/dv is Compose(MultiplyCoefficient(c), Partial(v)); explicit o composition is not reordered
  - T-like name T means f compose Phi; direction is read only from that map's direction field
  - generic or pullback composition T o D0 o T^-1 is stored in that written order and is not reordered
  - generic or pullback composition T o D1 o T^-1 is stored in that written order and is not reordered
  - map composition Phi o Psi keeps source order; outer map is Phi
  - map composition Psi o Phi keeps source order; outer map is Psi
  - finite coordinate indices are the supported reading; the chain-rule argument is the sum over i
spaces:
  - old kind coordinates coordinates Y, X exclude Y=0 log_branch log(1) = 0 contains (none) includes (none) span 234..395
  - new kind coordinates coordinates t, u exclude Y=0 log_branch log(1) = 0 contains (none) includes (none) span 396..550
maps:
  - Phi source new target old direction new_to_old inverse Psi span 551..698
    Y = 1 + t; X = c + u + log(1 + t)
  - Psi source old target new direction old_to_new inverse Phi span 699..842
    t = Y - 1; u = X - c - log(Y)
operators:
  - D0 domain old codomain old theory coordinate_chain_rule_v1 span 843..934
    formula Y * d/dY + d/dX
    ast OperatorSum(Compose(MultiplyCoefficient(Y), Partial(Y)), Partial(X))
  - D1 domain old codomain old theory coordinate_chain_rule_v1 span 939..1019
    formula d/dX
    ast Partial(X)
  - image_D0 domain new codomain new theory coordinate_chain_rule_v1 span 1024..1120
    formula (1 + t) * d/dt
    ast Compose(MultiplyCoefficient(1 + t), Partial(t))
  - image_D1 domain new codomain new theory coordinate_chain_rule_v1 span 1125..1211
    formula d/du
    ast Partial(u)
pullbacks:
  - T of Phi
claims:
  - Conjugation T o D0 o T^-1 = image_D0 span 1241..1272
  - Conjugation T o D1 o T^-1 = image_D1 span 1273..1304
  - MapCompose Phi o Psi = id span 1305..1326
  - MapCompose Psi o Phi = id span 1327..1348
ast:
  - Space old coordinates Y, X
  - Space new coordinates t, u
  - Map Phi source new target old direction new_to_old
  - Map Psi source old target new direction old_to_new
  - Operator D0 domain old codomain old OperatorSum(Compose(MultiplyCoefficient(Y), Partial(Y)), Partial(X))
  - Operator D1 domain old codomain old Partial(X)
  - Operator image_D0 domain new codomain new Compose(MultiplyCoefficient(1 + t), Partial(t))
  - Operator image_D1 domain new codomain new Partial(u)
  - Pullback T of Phi
  - Conjugation T o D0 o T^-1 = image_D0 span 1241..1272
  - Conjugation T o D1 o T^-1 = image_D1 span 1273..1304
  - MapCompose Phi o Psi = id span 1305..1326
  - MapCompose Psi o Phi = id span 1327..1348
result
outcome: PROVED
message: derived by the chain rule under the recorded assumptions
residual: 0
derivation:
  D0.t: (D Psi) compose Phi :: Y -> 1 + t
  D0.u: (D Psi) compose Phi :: 0 -> 0
  D1.t: (D Psi) compose Phi :: 0 -> 0
  D1.u: (D Psi) compose Phi :: 1 -> 1
  Phi o Psi Y: Y
  Phi o Psi X: X
  Psi o Phi t: t
  Psi o Phi u: u
```

The two-index file uses the same command:

```text
python3 -m neuralese.claims examples/claims/two_index_coordinate_change.txt
```

Its result, copied from that run, is `PROVED` with residual `0`. The
derivation is `D0.t: Y -> 1 + t`, `D0.u1 -> 0`, `D0.u2 -> 0`, `D1.u1: 1 ->
1`, `D2.u2: 1 -> 1`, and both inverse compositions are the identity on
`Y, X1, X2` and `t, u1, u2`. The printed formulas are `Y = 1 + t`,
`Xi = ci + ui + log(1 + t)`, `t = Y - 1`, and `ui = Xi - ci - log(Y)`.

`python3 -m neuralese.claims -` reads stdin. A missing path exits `2` and
prints no verdict. A printed verdict exits `0`. The outcome is the text, not
the exit code.

## Tests

```text
python3 -m pytest -q
```

The run after this checker reported `216 passed`. The count before the
finite-index command and the evaluation cases was `193 passed`. Evaluation
labels and observed outcomes are in `docs/CLAIM_EVAL.md`.

## What remains unimplemented

FFOC is not implemented. The grammar above accepts the directive
`encoding ffoc` and the checker returns `UNRESOLVED` with reason
`ffoc_unimplemented`. That is the whole FFOC surface in this repository. No
source file and no document defines an FFOC term grammar: there is no sort,
no production for an FFOC expression, no reduction rule, and no concrete
FFOC term beyond the keyword. The 21 October 2026 handoff cited in this file
records nine SymPy 1.14.0 smoke checks of coefficient algebra, script
SHA-256 `98b836d67804a276e70ded04db119397f5ebdd9ad81053923c2333b28c83649c`,
output `PASS: 9 assertions; SymPy 1.14.0`. It does not state an FFOC grammar.
Because that grammar is missing, none was invented. FFOC is not claimed to
be complete, terminating, confluent, or canonical. No speed or compression
figure is reported.

There is no benchmark table, pass rate, timing, or compression figure. This
module does not detect hallucinations, does not reduce dimension, and does
not verify an arbitrary paper. The checked-in verifier uses the standard
library and does not require SymPy or a network call.

The executable theory is the coordinate chain rule on a finite list of
indices that fits the step budget. A paper-scale argument is not checked.
Nothing here admits a runtime. Authority stays `RUNTIME_NOT_AUTHORIZED`.
The checker is not imported by `learn`, `translate`, or `certify`.

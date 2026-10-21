# Claim evaluation

These labels were written down before changing the checker to satisfy them.
The outcome column is frozen. A later run may not replace a frozen outcome
with `UNRESOLVED` just because the checker refused the case. Inputs are the
claim language in `docs/CLAIM_VERIFIER.md`. They are not unrestricted
English. `verify` is not given a table from these ids to outcomes.

The one-index paper chart and the rejected Euler transformation claim stay
in `tests/test_claim_verifier.py`. They are reference cases, not rows below.
The Euler chart is valid. The false statement is the rejected claim that the
chart sends `T E0 T^{-1}` and `T E1 T^{-1}` to `(1 + t) d/dt` and `d/du`.

## Frozen cases

### E1 unfamiliar correct shear

Not the paper fixture. No logarithm.

```text
V = d/dx + d/dy
Phi: x = p, y = q + p
Psi: p = x, q = y - x
direction new_to_old on Phi
exclude x = 0 is written
assert T o V o T^-1 = d/dp
assert both inverse compositions are id
```

Chain rule: `V(p) = 1`, `V(q) = 1 - 1 = 0`, and substitution of `Phi` leaves
those coefficients. Both inverse compositions cancel by addition.

Expected outcome: `PROVED`.

Reason: the derivation shows `T V T^{-1} = d/dp` and both inverses are the
identity under the written exclusion. No logarithm branch is required.

### E2 wrong coefficient

Same shear and assumptions as E1. The claimed image is `2 d/dp`. One
assertion, the conjugation. No sample point is declared with `contains`.

Expected outcome: `DISPROVED`.

Reason: the derived coefficient of `d/dp` is `1`. On `g = p` the derived
value is `1`, the claimed value is `2`, and the residual is `-1`. That is a
symbolic contradiction. A failed simplification would not be this outcome.

### E3 unsupported operation

Conjugation of `d/dx o d/dx` through `x = s`, claiming `d/ds`. The theory
field is `coordinate_chain_rule_v1`. Exclusion `x = 0` is written. Direction
`new_to_old` is written.

Expected outcome: `UNRESOLVED`.

Reason: `d/dx o d/dx` is second order. It is not a first-order vector field
in the supported theory. The checker has no rule for it.

### E4 malformed input

```text
assert conjugation
```

Expected outcome: `INVALID_INPUT`.

Reason: the line is not an equality, `invertible`, or `apply`. The parser
records a syntax issue and does not assign a verdict. The verification
result names that reason and a span.

### E5 missing logarithm branch

```text
Y = exp(t), t = log(Y), D = Y d/dY, claimed image d/dt
exclude Y = 0 is written
log_branch is absent
direction new_to_old
```

Expected outcome: `UNRESOLVED`.

Reason: a logarithm is written and `log(1) = 0` is not. The checker must not
supply the branch. This is not a disproof of the formula.

### E6 missing exclusion

Same radial chart as E5, with `log_branch log(1) = 0` written and no
`exclude` line.

Expected outcome: `UNRESOLVED`.

Reason: the inverse uses `log(Y)` and no exclusion is written. The missing
fact is `Y != 0` because `Y` is the written logarithm argument. The checker
must not add that exclusion.

### E7 domain includes the logarithm singularity

```text
includes Y = 0
Psi: t = log(Y)
Phi: Y = exp(t)
assert invertible Psi
```

Expected outcome: `DISPROVED`.

Reason: the claimed inverse uses `log` at a point the domain includes. The
inverse is not evaluated at `Y = 0`. A germ claim needs a locally valid
argument, not a sample outside that obligation.

### E8 reversed composition

E1's shear, direction `new_to_old` written, claimed image the true
`d/dp`, but the assertion is `Phi^-1 o V o Phi`.

Expected outcome: `INVALID_INPUT`.

Reason: the written order is not `T o D o T^-1`. The factors are not
reordered into the supported pullback. Matching the true image is not
enough.

### E9 missing pullback direction

E1's shear and the true image `d/dp`, with `Phi`'s `direction` line omitted.
The assertion stays `T o V o T^-1`.

Expected outcome: `UNRESOLVED`.

Reason: no direction was written. The checker must not infer `new_to_old`.

### E10 two-index paper transformation

```text
D0 = Y d/dY + d/dX1 + d/dX2
D1 = d/dX1
D2 = d/dX2
Phi: Y = 1 + t, Xi = ci + ui + log(1 + t)
Psi: t = Y - 1, ui = Xi - ci - log(Y)
direction new_to_old
analytic germs, log(1) = 0, Y = 0 excluded
assert T Di T^-1 equals the images below, and both inverses are id
```

Expected outcome: `PROVED`.

Reason: the same chain rule as the one-index chart. `D0` of `t` is `Y`,
which `Phi` sends to `1 + t`. `D0` of each `ui` is `-1 + 1 = 0`. `Di` of
`ui` is `1` and the other new coordinates are `0`. So `T D0 T^{-1} = (1 + t)
d/dt` and `T Di T^{-1} = d/dui`. The inverses cancel on the recorded branch
because `Y != 0` and `log(1) = 0` are written. The outcome string is not a
substitute for that derivation.

### E11 false target on the two-index pair

E10, except the claimed image of `D1` is `d/dt` rather than `d/du1`.
`D0` and `D2` keep the images from E10. No `contains` point is written.

Expected outcome: `DISPROVED`.

Reason: the chain rule still sends `D1` to `d/du1`. Against `d/dt`, the
coordinate `t` is a witness: derived value `0`, claimed value `1`, residual
`-1`. That disagreement is inside the stated germ. A zero commutator is not
the evidence.

## Observed run

Command: `python3 -m pytest -q tests/test_claim_eval.py tests/test_claim_verifier.py tests/test_claim_cli.py`, then the full `python3 -m pytest -q`.

The full suite reported `216 passed`. The frozen outcome column was not
edited after this run. The checker computed each row. `verify` does not
contain these source ids.

| Id | Expected | Observed | What the checker reported |
| --- | --- | --- | --- |
| E1 | `PROVED` | `PROVED` | `V` derived as `d/dp`. Coefficients `p: 1`, `q: 0`. Residual `0`. |
| E2 | `DISPROVED` | `DISPROVED` | Derived `d/dp`. On `g = p`, actual `1`, claimed `2`, residual `-1`. Symbolic. |
| E3 | `UNRESOLVED` | `UNRESOLVED` | `unsupported_operator`. Second order is not a vector field. |
| E4 | `INVALID_INPUT` | `INVALID_INPUT` | `malformed_claim`. Parse has no claim and no verdict. Span starts at `0`. |
| E5 | `UNRESOLVED` | `UNRESOLVED` | `missing_log_branch`. Not supplied: `log(1) = 0`. |
| E6 | `UNRESOLVED` | `UNRESOLVED` | `missing_domain`. Not supplied: `Y != 0`, the written logarithm argument. |
| E7 | `DISPROVED` | `DISPROVED` | `inverse_undefined_at_included_point`. Inverse not evaluated at `Y = 0`. |
| E8 | `INVALID_INPUT` | `INVALID_INPUT` | `illegal_operator_application`. Order not reordered. |
| E9 | `UNRESOLVED` | `UNRESOLVED` | `missing_pullback_direction`. Direction left absent. |
| E10 | `PROVED` | `PROVED` | `D0 = (1 + t) d/dt`, `D1 = d/du1`, `D2 = d/du2`. Summed coordinates `Y, X1, X2`. Residual `0`. |
| E11 | `DISPROVED` | `DISPROVED` | Derived `D1 = d/du1`. On `g = t`, actual `0`, claimed `1`, residual `-1`. |

False acceptances: none. No frozen `DISPROVED`, `UNRESOLVED`, or
`INVALID_INPUT` row was returned as `PROVED`.

False rejections: none. E1 and E10 were `PROVED`, matching the frozen labels.
The one-index paper chart in `tests/test_claim_verifier.py` was also
`PROVED`, and the rejected Euler transformation claim was `DISPROVED` by the
in-domain witness `g = t` (actual `1`, claimed `1 + t`, residual `-t`, and
at `t = 1` the values `1` and `2`).

Interpretation errors: none on this list. Directions, domains, and the
logarithm branch that were written stayed as written. Omitted direction was
not filled in. The reversed composition was not reordered.

Unresolved cases: E3, E5, E6, and E9. Each was already labeled `UNRESOLVED`.
They were counted, not relabeled.

Supported-case coverage: two correct claims inside the theory were proved
(E1, the shear, and E10, the two-index chart), and the one-index reference
chart was proved. The checker did not refuse every claim. False claims E2,
E7, and E11 were disproved rather than left open. E11's disproof is the
derived operator, not a table from the fixture id.

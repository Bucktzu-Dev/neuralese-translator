# Audit protocol

`neuralese certify` is the public fail-closed gate. A pack that does not pass is not a lexicon. It is a draft.

## Certificate

`AuditCertificate` fields:

| field | type | meaning |
|---|---|---|
| `pack_id` | str | pack identity |
| `pack_checksum` | str | sealed checksum claimed by the pack |
| `expected_checksum` | str | checksum recomputed by the auditor |
| `passed` | bool | all gates true |
| `addressable` | bool | codebook + aliases resolve |
| `unfoldable` | bool | every live symbol has observation ids |
| `gloss_bound` | bool | English sealed; checksum matches |
| `residual_ok` | bool | reconstruction error ≤ τ |
| `fail_closed` | bool | always true for this protocol version |
| `failures` | list[str] | human-readable gate failures |
| `timestamp` | float | unix time of certification |
| `details` | object | counts, thresholds, residual |

CLI:

```bash
neuralese audit pack.json
neuralese certify pack.json --fail-on-undecodable
```

`audit` prints the certificate and exits 0 if the file was readable.
`certify --fail-on-undecodable` exits 1 when `passed` is false.

## Gates

### Addressable

Fail if:

- a codebook `code → class_id` has no matching symbol
- a symbol `code` is missing from the codebook
- an alias chain cycles
- an alias resolves to a missing code
- integer keys were lost in JSON (loaders must restore `int` keys)

### Unfoldable

For each symbol with `quarantined=false`:

- `observation_ids` must be a non-empty list of strings

Quarantined symbols are allowed to lack a fold path; they must not appear as `state=ok` in translation.

### Gloss-bound

- Recompute checksum from canonical pack content.
- Fail if it differs from `pack.checksum` (tamper / unsealed mutation).
- Fail if a live symbol has an empty `definition` when `require_gloss=true` (default).

Checksum covers: `pack_id`, codebook, aliases, each symbol’s `class_id`, `code`, `observation_ids`, `definition`, `quarantined`, and rounded `reconstruction_error`.

It does **not** cover wall-clock timestamps, so certification is deterministic.

### Residual

Default `tau_residual = 0.55` (mean normalized reconstruction error).

Override:

```bash
neuralese certify pack.json --tau-residual 0.4 --fail-on-undecodable
```

Confidence cap used by the translator:

```text
confidence := min(symbol.confidence, 1 - clip(residual, 0, 1))
```

### Parent packs (evolution)

If `parent_pack_id` is set, the new pack should record:

- `aliases` from previous codes to current codes
- `delta_mdl_bits` on the finalize receipt (`new_mdl - parent_mdl`)
- a reject when `delta_mdl_bits > 0` unless metadata marks an explicit exception

This leaf implements ΔMDL against the previous pack’s `mdl_bits`. It does not silently inherit stubbed deltas from any origin engine.

## Translation states

| state | when | English |
|---|---|---|
| `ok` | resolved, not quarantined, definition present | bound definition |
| `aliased` | input code rewritten, then ok | bound definition of the *resolved* code |
| `quarantined` | resolved to a quarantined class | `[quarantined: …]` |
| `unknown` | no resolution | `[undecodable: no symbol for code N]` |

No other states. No default “probably this word.”

## Receipts

Each learn run appends receipts:

| step | records |
|---|---|
| `factorization` | reconstruction error, atom/rank count |
| `clustering` | class count, min survival |
| `finalize` | decision `accept` \| `accept_provisional` \| `reject`, ΔMDL |

Receipts are evidence of the *update*. The certificate is the *gate*. Both ship inside `SymbolPack`.

## Fail-closed policy

1. Missing data fails. It does not become a zero.
2. Unknown codes fail translation honesty (explicit `unknown`), and they fail certify only if they appear *inside* the pack’s own codebook — not because a user later asked about an unused integer.
3. Quarantine is not a pass. It is a labeled hold.
4. `passed` is a conjunction, never a majority vote.

## Thresholds (defaults)

| name | default | role |
|---|---|---|
| `tau_kappa` | 0.35 | min mean cosine of members to prototype |
| `tau_residual` | 0.55 | max normalized reconstruction error |
| `tau_persist` | 0.80 | min cluster survival vs parent (1.0 on first pack) |
| `min_cluster_size` | 2 | below this, class is quarantined |

These are knobs for auditors. They are not moral facts. Publish the values you used with the certificate.

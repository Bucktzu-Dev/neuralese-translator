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
| `schema_version` | str | `"0.2"` on a certificate `certify` writes. A loaded object that omits the field is `unversioned`. This is not the package version |
| `package_version` | str or null | informational copy of `neuralese.__version__` when `certify` writes the certificate. Not a gate. Null on a legacy object that did not store it |
| `authority` | str | `RUNTIME_NOT_AUTHORIZED`. No other value is defined |
| `decision` | str | `RESEARCH_FAIL` when a structural gate fails, `UNRESOLVED` when the four structural gates pass. Never `RESEARCH_PASS` while numerical and semantic gates are unimplemented |
| `evidence` | object | `structural` (the four booleans, the failure list, and `digest_does_not_bind`), `numerical` (`NOT IMPLEMENTED` / `UNRESOLVED`), `semantic` (`NOT IMPLEMENTED` / `UNRESOLVED`) |

`passed` is the structural conjunction of `addressable`, `unfoldable`, `gloss_bound`, and `residual_ok`. It is not a numerical reconstruction and it is not a semantic certificate. `evidence.numerical` forbids reading the reservoir centroid residual, or gloss text, as a reconstructed activation. `evidence.semantic` forbids fluency, the confidence heuristic, kappa cosine, cluster purity, and cosine similarity as inputs. Those two gates stay unimplemented.

A certificate dict from before schema 0.2 still loads. Missing `schema_version` is `unversioned`. That loader does not turn a reservoir-less pack into `passed=true`. There is no v0.1 mode. An empty reservoir still fails unfold.

CLI:

```bash
neuralese audit pack.json
neuralese certify pack.json --fail-on-undecodable
```

`audit` prints the certificate and exits 0 if the file was readable.
`certify --fail-on-undecodable` exits 1 when `passed` is false.

```bash
neuralese report pack.json
```

`report` prints a decodability summary and exits 0 when the pack file loads. It does not apply `tau_residual` and it does not decide `passed`. Fields: `n_live`, `n_quarantined`, `residual`, `reservoir_size`, `unfold_failures`, `gloss_coverage`, `mdl_bits`, `decision` (null when learn did not record one). `gloss_coverage` is the fraction of live symbols whose definition is non-empty and is not an `[unglossed: ...]` marker. `--topics FILE` is optional. When it is passed, the JSON also contains `cluster_purity`. The file is JSONL observations with `metadata.topic`, or a JSON object of observation id to topic. A live cluster with majority purity under 0.5 has `label` `mixed`. `certify` does not read the file and does not grow a purity field.

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

For every symbol that lists observation ids, including quarantined ones:

- each id resolves to an observation stored on the pack
- that observation has an embedding
- the mean of those embeddings matches the stored prototype (L2 ≤ 1e-5)

Quarantined symbols are allowed to lack a fold path; they must not appear as `state=ok` in translation. A listed id that does not resolve is still a failure.

`neuralese unfold pack.json --code 0` prints the reservoir rows for a code.

### Gloss-bound

- Recompute checksum from canonical pack content.
- Fail if it differs from `pack.checksum` (tamper / unsealed mutation).
- Fail if a live symbol has an empty `definition` when `require_gloss=true` (default).

Checksum covers: `pack_id`, codebook, aliases, each symbol’s `class_id`, `code`, `observation_ids`, `definition`, `quarantined`, rounded `reconstruction_error`, and the reservoir (observation id, text, rounded embedding, metadata).

It does **not** cover wall-clock timestamps, so certification is deterministic. It also does not cover `proto_embedding`, `confidence`, `examples`, `survival`, symbol metadata / `kappa`, `guards`, `receipts`, `parent_pack_id`, `mdl_bits`, or pack `metadata`. Schema 0.2 lists that set on `evidence.structural.digest_does_not_bind`. Unfold still compares a stored prototype to the reservoir mean (L2 ≤ 1e-5). That check is not the digest. A drift inside the tolerance can pass, and putting the prototype into the digest is not this schema.

### Residual

Default `tau_residual = 0.55` (mean normalized reconstruction error).

Override:

```bash
neuralese certify pack.json --tau-residual 0.4 --fail-on-undecodable
```

The sealed residual must also be at least the cluster residual recomputed from reservoir embeddings and stored prototypes. A pack that reports a tighter error than its own rows support fails `residual_ok`.

Confidence cap used by the translator:

```text
confidence := min(symbol.confidence, 1 - clip(residual, 0, 1))
```

### Parent packs (evolution)

If `parent_pack_id` is set, the new pack records:

- `aliases` from previous codes to current codes
- `delta_mdl_bits` on the finalize receipt (`new_mdl - parent_mdl`)
- `decision=reject` when `delta_mdl_bits > 0`, unless `metadata.mdl_exception` is a non-empty reason

```bash
neuralese learn obs.jsonl -o child.json --parent parent.json --mdl-exception "split for a new domain"
```

The exception admits `accept_provisional`. It does not flip `pass_mdl` to true. A first pack has no parent and records `delta_mdl_bits = 0`.

### Retired codes

`learn --parent` does not drop a parent code.

- A parent symbol whose prototype matches a child cluster keeps its code on that cluster.
- A parent symbol with no match is appended as `quarantined=true`. Its observation rows are copied into the child reservoir when the id is not already there.
- Alias keys from the parent are flattened onto the surviving code.
- Integers used by parent codes or alias keys are not given to a new cluster.

Translating a retired code yields `state=quarantined`. It does not yield `unknown`, and it does not yield the gloss of whichever new cluster happened to receive that integer.

## Translation states

| state | when | English |
|---|---|---|
| `ok` | resolved, not quarantined | bound definition, or `[unglossed: …]` when the definition is empty |

`state=ok` does not call `certify`. It is not `decision=RESEARCH_PASS`, and it is not a statement that numerical or semantic evidence is resolved. The four states above are the whole public gloss vocabulary.
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

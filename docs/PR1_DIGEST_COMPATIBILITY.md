# PR-1 digest compatibility

Recommendation: **NONE**. Authority: **RUNTIME_NOT_AUTHORIZED**. Certificate schema **0.2** stands. Package **0.7.0** stands. The research checkpoint stands. Activation-codec experiments stay closed. This note does not add a codec, quantizer, layer, or slot-retrieval trial.

This commit specifies a versioned identity digest. It does not implement it. `compute_checksum`, `seal`, `SymbolPack.to_dict`, `SymbolPack.from_dict`, reservoir canonicalization, and the bytes any current fixture writes are unchanged. The package version is unchanged.

The profile name below is `digest_profile`. Today's profile is `unbound`. The proposed profile is `identity-1`. Those names are not the package version and not certificate `schema_version` `"0.2"`. [PR1_CERTIFICATION_DISCREPANCY.md](PR1_CERTIFICATION_DISCREPANCY.md) does not name a digest profile. [PR1_RESEARCH_CHECKPOINT.md](PR1_RESEARCH_CHECKPOINT.md) closes the activation-codec line at recommendation `NONE` and authority `RUNTIME_NOT_AUTHORIZED`.

## What the current code does

`canonical_dumps` (`src/neuralese/contracts.py`) is `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`. `sha16` is SHA-256 of that UTF-8 text, truncated to 16 hex characters.

`SymbolPack.compute_checksum` hashes one object. Its keys are exactly:

- `pack_id`
- `codebook` (stringified int keys, sorted)
- `aliases` (stringified int keys, sorted)
- `reconstruction_error`, `round(float(...), 8)`
- `reservoir`, from `_reservoir_rows`: each observation's `observation_id`, `text`, `embedding` as `round(float(x), 8)`, and `metadata`, in sorted observation-id order
- `symbols`, sorted by `class_id`, each object limited to `class_id`, `code`, `observation_ids`, `definition` (`_normalize_definition`: strip, and a blank string becomes null), and `quarantined`

`proto_embedding` is not a key of that object. Neither are `confidence`, `examples`, `survival`, symbol `metadata`, `checksum`, `parent_pack_id`, `receipts`, `guards`, `mdl_bits`, `timestamp`, or pack `metadata`.

`seal` assigns `self.checksum = self.compute_checksum()` and returns the pack. `to_dict` / `from_dict` round-trip the stored checksum string. They do not recompute it.

`DIGEST_DOES_NOT_BIND` in the same file is the list schema 0.2 records:

`proto_embedding`, `confidence`, `examples`, `survival`, `symbol.metadata`, `kappa`, `guards`, `receipts`, `parent_pack_id`, `mdl_bits`, `pack.metadata`.

`certify` (`src/neuralese/audit.py`) sets `passed` only when `addressable`, `unfoldable`, `gloss_bound`, and `residual_ok` are all true. A structural pass sets `decision` to `UNRESOLVED`. A structural miss sets `decision` to `RESEARCH_FAIL`. `decision` is never `RESEARCH_PASS`. `authority` is `RUNTIME_NOT_AUTHORIZED`. `schema_version` written by `certify` is `"0.2"`. `evidence.structural.digest_does_not_bind` is that tuple. `evidence.numerical` and `evidence.semantic` are `status: NOT IMPLEMENTED`, `decision: UNRESOLVED`. `gloss_bound` is `compute_checksum() == pack.checksum` with a non-empty checksum, plus a non-empty stripped definition on every live symbol when `require_gloss` is true.

`PROTO_ATOL` is `1e-5` and `RESIDUAL_ATOL` is `1e-4` (`src/neuralese/unfold.py`). `unfold_failures` fails unfold when the L2 distance from `proto_embedding` to the member mean is greater than `PROTO_ATOL`. It fails residual honesty when the recomputed cluster residual is greater than the sealed `reconstruction_error` plus `RESIDUAL_ATOL`. Those comparisons are acceptability of a reconstruction. They do not ask whether the stored prototype is the sealed prototype.

Section 4 of [PR1_CERTIFICATION_DISCREPANCY.md](PR1_CERTIFICATION_DISCREPANCY.md) leaves closing the prototype gap as later work. Section 8 records that `proto_embedding` was not added to `compute_checksum`, that the `1e-6` nudge still passes, and that the larger prototype assignment still fails unfold and residual honesty with `gloss_bound` true. This note is that later specification. It is not that implementation.

## Integrity and tolerance

Integrity: the certified numerical object is exactly the one originally sealed.

Tolerance: a modified numerical object remains within an acceptable reconstruction error.

A tolerance check is not an integrity check. `PROTO_ATOL` and `RESIDUAL_ATOL` answer acceptability. They do not answer whether the object changed.

## 1. Inventory

Read from this tree. "Digest" means a key of the object inside `compute_checksum`. "Certify", "unfold", and "translate" mean `certify`, `unfold_code` / `unfold_failures`, and `translate_stream`.

`SubjectiveSymbolDynamics.transition_probs` reads `proto_embedding` and `confidence`. That path is not one of the three columns.

### SymbolPack

| field | today's digest | certify | unfold | translate |
|---|---|---|---|---|
| `pack_id` | yes | copied onto the certificate; also inside the digest | no | no |
| `symbols` | partial; see the symbol table | yes | yes | yes, after code resolution |
| `codebook` | yes | addressable gate | `symbol_by_code` | `symbol_by_code` |
| `aliases` | yes | addressable gate | `resolve_code` | `resolve_code` |
| `reconstruction_error` | yes, 8 decimals | `residual_ok` and `details` | residual honesty against the recomputed cluster residual | `confidence_cap` |
| `checksum` | no; it is the digest output, not an input | compared with `compute_checksum()` | no | copied onto `Gloss.pack_checksum`; not recomputed |
| `parent_pack_id` | no | no | no | no |
| `receipts` | no | no | no | no |
| `guards` | no | no | no | no |
| `mdl_bits` | no | copied into `details` only; not a gate | no | no |
| `timestamp` | no | no | no | no |
| `metadata` | no | no. `decodability_report` reads `metadata["decision"]`; `certify` does not | no | no |
| `observations` | yes, as `reservoir` rows | digest, and `unfold_failures` | yes | no |

### Symbol

Ten fields. Five are keys of each symbol object in the digest: `class_id`, `code`, `observation_ids`, `definition`, `quarantined`.

| field | today's digest | certify | unfold | translate |
|---|---|---|---|---|
| `class_id` | yes | addressable | failure text and the unfold report | `Gloss.class_id` |
| `code` | yes | addressable | yes | yes |
| `proto_embedding` | **no** | only as an L2 input inside `unfold_failures` | yes, L2 against the member mean | no |
| `observation_ids` | yes | live-symbol gate and unfold | yes | copied onto the gloss |
| `definition` | yes, after strip; blank becomes null | `require_gloss` | `english_bound` on `unfold_code` | the English string |
| `examples` | no | no | no | no |
| `confidence` | no | no | no | `confidence_cap` |
| `quarantined` | yes | live-symbol filter and the quarantined count | gloss state | gloss state |
| `survival` | no | no | no | no |
| `metadata` | no | no | no | no |

### Observation

All four fields are inside `reservoir`. Embedding coordinates use 8 decimals.

| field | today's digest | certify | unfold | translate |
|---|---|---|---|---|
| `observation_id` | yes | digest and missing-id failures | yes | no |
| `embedding` | yes, 8 decimals | digest and unfold geometry | yes | no |
| `text` | yes | digest only. `unfold_failures` does not compare text | carried back on the observation object; not compared | no |
| `metadata` | yes | digest only | carried back; not compared | no |

### Receipt and GuardSnapshot

Neither object is in the digest. `certify`, unfold, and `translate_stream` do not read them.

Receipt fields: `step`, `ok`, `timestamp`, `kappa`, `reconstruction_error`, `delta_mdl_bits`, `metadata`.

GuardSnapshot fields: `kappa_avg`, `reconstruction_error`, `delta_mdl`, `min_survival`, `pass_kappa`, `pass_residual`, `pass_mdl`, `pass_persist`, `pass_compat`, `pass_all`.

`kappa` on the symbol is `symbol.metadata["kappa"]` as `learn_pack` writes it. It is covered by symbol `metadata` above. `certify` does not reapply `tau_kappa`.

### The prototype gap, re-verified

On `tests/packutil.py` `make_pack()` (prototype `[1.0, 0.0, 0.0]`, sealed residual `0.08`), `compute_checksum()` is `279f2a077bf752b2`. After `proto_embedding[0] += 1e-6`, and after replacing that prototype with `[0.0, 0.0, 0.0]`, the recomputed checksum is still `279f2a077bf752b2`. The digest omits the prototype, so resealing also writes a digest that still omits it.

`1e-6` is below `PROTO_ATOL` `1e-5`, so unfold accepts it. `tests/test_pr1_certification.py::test_prototype_drift_inside_atol_passes` locks that: checksum unchanged, `passed` true, `gloss_bound` true. That pass is a tolerance result.

The assignment to `[0.0, 0.0, 0.0]` is L2 `1.0` from the reservoir mean. Unfold fails (`l2=1.000000`) and residual honesty fails (recomputed cluster residual about `0.707107` against sealed `0.080000`). `gloss_bound` stays true. `tests/test_pr1_certification.py::test_prototype_tamper_without_reseal` locks the no-reseal case. `tests/test_unfold.py::test_certify_fails_when_prototype_drifts_from_reservoir` locks the same assignment after `seal()`, including `gloss_bound` true.

`tests/test_pr1_certification.py::test_confidence_tamper_is_outside_the_digest` locks the same omission for `confidence`: checksum unchanged, `passed` true, translate confidence `0.92`.

## 2. Proposed digest, specified and not implemented

`digest_profile` value `identity-1`. The digest value is a new field, `identity_digest`. `pack.checksum` keeps meaning today's `compute_checksum()` output for as long as compatibility with current packs matters.

`seal()` keeps assigning today's checksum and only that checksum until a later commit. This note does not change `seal()` output.

### Payload

`identity-1` uses the current checksum object plus one added key on each symbol: `proto_embedding`, each coordinate `round(float(x), 8)`, the same call the reservoir already uses. Symbols stay sorted by `class_id`. An empty prototype is `[]`, so a missing vector and a present vector are different canonical values. The prototype is inside the symbol object. There is no second top-level document and no second JSON dialect.

Top-level keys stay `pack_id`, `codebook`, `aliases`, `reconstruction_error`, `reservoir`, `symbols`. Symbol keys become `class_id`, `code`, `observation_ids`, `definition`, `quarantined`, and `proto_embedding`. Reservoir row keys stay `observation_id`, `text`, `embedding`, `metadata`. Definition normalization, codebook and alias sorting, and `reconstruction_error` rounded to 8 decimals stay as they are.

`identity_digest = sha16(payload)` with the existing `canonical_dumps` and `sha16`. Same key sort, same separators `( ",", ":" )`, same `ensure_ascii=True`, same 16-hex truncation. The profile name and the field name distinguish it from `pack.checksum`. The truncation width is the current function, not a new security claim.

### What `identity-1` binds

Bind a field only when the certificate claims that field's identity as part of the sealed object.

Already claimed, because they are the object `gloss_bound` recomputes today, and `identity-1` keeps them:

| field | class |
|---|---|
| `pack_id`, `codebook`, `aliases` | identity-bound |
| symbol `class_id`, `code`, `observation_ids`, normalized `definition`, `quarantined` | identity-bound |
| `reconstruction_error` at 8 decimals | identity-bound. The comparisons to `tau_residual` and to the recomputed residual plus `RESIDUAL_ATOL` stay tolerance checks beside that binding |
| reservoir `observation_id`, `text`, `embedding` at 8 decimals, observation `metadata` | identity-bound |

`proto_embedding` at 8 decimals is identity-bound under `identity-1`. Unfold's L2 check stays tolerance-only and keeps running. Equality of the rounded prototype is the integrity claim. `PROTO_ATOL` is not that claim.

### What stays excluded

These are the schema 0.2 `digest_does_not_bind` names other than `proto_embedding`, plus pack `timestamp`, which is already outside the checksum and is not in that tuple. They stay out of the `identity-1` payload. A later implementation must not add them in order to lengthen the list.

| field | class | why it stays excluded |
|---|---|---|
| `confidence` | informational, excluded | `certify` does not read it. Translate applies `confidence_cap` to a learn-time heuristic (`gloss.py`). The certificate does not claim that heuristic's identity. Binding it would turn a confidence edit into an integrity failure |
| `examples` | informational, excluded | `certify`, unfold, and `translate_stream` do not read examples |
| `survival` | informational, excluded | none of those three read it |
| `symbol.metadata` and `kappa` | informational, excluded | `kappa` is a learn-time cosine stored on symbol metadata. `certify` and `translate_stream` do not read it and do not reapply `tau_kappa`. Binding metadata would claim kappa |
| `guards` | informational, excluded | `certify` does not read `pass_all`, `pass_kappa`, or any other guard flag |
| `receipts` | informational, excluded | step logs, including timestamps. The certificate does not claim them. Hashing them would also make the digest depend on wall-clock receipt times |
| `parent_pack_id` | informational, excluded | lineage pointer. `certify`, unfold, and translate do not read it |
| `mdl_bits` | informational, excluded | copied into certificate `details` and not a gate. The certificate does not claim its identity |
| pack `metadata` | informational, excluded | includes `decision` and learn config. `certify` ignores `decision=reject`. Binding it would certify learn metadata the structural gates do not claim |
| pack `timestamp` | informational, excluded | already outside the checksum so re-audit stays deterministic (`docs/AUDIT_PROTOCOL.md`). The certificate does not claim it |
| `checksum` and `identity_digest` | outputs, excluded from the payload | a digest does not hash itself |

`identity-1` removes only `proto_embedding` from `digest_does_not_bind`, and only once `identity_digest` exists. The other names stay on the list under both profiles.

### Rounding

Reservoir embeddings and `reconstruction_error` use 8 decimals: `round(float(x), 8)`. The identity-bound prototype uses that same rounding. Identity is equality of the canonical value after that rounding. It is not L2 `<= 1e-5`.

The last decimal place is `1e-8`. Half a unit in that place is `5e-9`. Python `round` is applied to the float. The worked object is one symbol under `canonical_dumps`, which is the function `sha16` already uses, so a change in this fragment changes any payload that contains it.

Coordinate `1.0`, canonical embedding `[1.0,0.0,0.0]`:

```text
{"class_id":0,"code":0,"definition":"Symbol for hello.","observation_ids":["obs-hello"],"proto_embedding":[1.0,0.0,0.0],"quarantined":false}
```

`sha16` of that object is `24e6fb0ffe3617eb`.

Adding `1e-6` produces the float `1.00000099999999991773`. `round(..., 8)` is `1.000001`. The canonical `proto_embedding` is `[1.000001,0.0,0.0]`. `sha16` is `fe16d0fdc2b8a2e1`. The `1e-6` nudge survives 8-decimal rounding and changes the identity digest. Today's `compute_checksum()` does not change, as re-verified above. `PROTO_ATOL` still accepts the nudge. Those are different questions.

Adding `1e-9` produces the float `1.00000000100000008274`. That displacement is smaller than `5e-9`. `round(..., 8)` returns `1.0`. The canonical JSON and `sha16` stay `24e6fb0ffe3617eb`. The identity digest is unchanged because the canonical value is unchanged. That is a limit of rounding to 8 decimals, not a tolerance pass.

### Where the new field sits

During compatibility, `pack.checksum` remains today's digest. `identity_digest` is a new optional string. `digest_profile` is a new optional label. Absent, null, or blank `digest_profile` means `unbound` and means `identity_digest` is not required.

Do not replace `pack.checksum` in place. Do not rewrite stored checksums. Current `seal()` output stays the assignment of today's checksum until a later commit adds the new field without changing that assignment.

## 3. Legacy profile

`unbound` is a named profile. It is what today's packs already are. It is not a pass.

Loader rule, analogous to schema version and not a replacement for it:

- Omitted `schema_version` still loads as `unversioned` (`AuditCertificate.from_dict`). Keep that. There is still no v0.1 mode.
- Omitted, null, or blank `digest_profile` loads as `unbound`.
- `digest_profile` `"identity-1"` selects the identity profile and requires `identity_digest`.
- Any other `digest_profile` string is rejected.
- The two axes are independent. Schema `"0.2"` with `digest_profile` `unbound` is the legacy certificate. Encoding the digest profile into `schema_version` would collide with `"0.2"` and `"unversioned"`.

A legacy pack is one whose `digest_profile` is absent or `unbound`. `certify` on that pack keeps today's structural gates. `passed` is still only those four gates. A structural pass is still `decision` `UNRESOLVED`. A structural miss is still `RESEARCH_FAIL`. `decision` is never `RESEARCH_PASS`. `authority` stays `RUNTIME_NOT_AUTHORIZED`. `package_version` stays informational `0.7.0` when `certify` writes it.

The legacy profile surfaces the missing prototype binding in the place schema 0.2 already uses: `evidence.structural.digest_does_not_bind` keeps `proto_embedding` on the list. `unbound` must not report prototype identity as satisfied. A structural `passed` true under `unbound` is the four gates. It is not an identity pass and not `RESEARCH_PASS`.

Age does not create a pass. A missing reservoir still fails unfold (`missing reservoir observation …`) with `passed` false and `decision` `RESEARCH_FAIL`. Resealing an empty reservoir still does not make `passed` true. An unversioned certificate JSON still loads without recomputing an omitted `decision` into a pass; an omitted `decision` stays `UNRESOLVED`. A stored `RESEARCH_PASS` stays rejected.

`evidence.numerical` and `evidence.semantic` stay `NOT IMPLEMENTED` / `UNRESOLVED` on `unbound` and on `identity-1`. An integrity match is not numerical research validity and not semantic validity.

Under `identity-1`, once the digest exists, `digest_does_not_bind` drops `proto_embedding` and keeps every excluded name in the table above. `passed` is the four gates and `identity_bound`. `identity_bound` is true only when `identity_digest` is non-empty and equals `sha16` of the identity payload. A missing `identity_digest` on a named `identity-1` profile fails closed (`passed` false, `decision` `RESEARCH_FAIL`). It does not fall through to `unbound`. `gloss_bound` stays the comparison of `pack.checksum` to today's payload on both profiles, so a prototype-only edit can leave `gloss_bound` true while `identity_bound` is false.

## 4. Negative tests to add later

No new tests are added in this commit. The identity digest does not exist yet. Current locks stay as they are.

Already locked, and not duplicated:

| case | test | what it locks |
|---|---|---|
| prototype coordinate `+1e-6`, no reseal, current profile | `tests/test_pr1_certification.py::test_prototype_drift_inside_atol_passes` | `PROTO_ATOL == 1e-5`, checksum unchanged, `passed` true, `gloss_bound` true, unfold and residual true. `decision` on a structural pass is `UNRESOLVED` via `test_structural_pass_is_schema_v02_and_not_a_research_pass` |
| prototype replaced by `[0.0, 0.0, 0.0]`, no reseal | `tests/test_pr1_certification.py::test_prototype_tamper_without_reseal` | checksum unchanged, `gloss_bound` true, unfold false (`l2=1.000000`), residual false (about `0.707107`), `passed` false |
| same assignment, then `seal()` | `tests/test_unfold.py::test_certify_fails_when_prototype_drifts_from_reservoir` | unfold still fails, `gloss_bound` stays true because the resealed digest still omits the prototype |

Future tests, to be added only with the implementation of `identity-1`:

| test | input | expected |
|---|---|---|
| `test_identity1_1e6_fails_integrity_not_atol` | `make_pack()` sealed under `digest_profile` `identity-1`, then `proto_embedding[0] += 1e-6`, no reseal | recomputed `identity_digest` changes (`1.0` to `1.000001` at 8 decimals). `identity_bound` false. `passed` false. `decision` `RESEARCH_FAIL`. `gloss_bound` true, because `pack.checksum` is unchanged. `unfoldable` true, because L2 `1e-6` is inside `PROTO_ATOL`. The failure string is an identity mismatch. It is not the unfold sentence `prototype does not match reservoir mean`. `evidence.numerical` and `evidence.semantic` stay `NOT IMPLEMENTED` / `UNRESOLVED` |
| `test_identity1_far_prototype_fails_digest_and_unfold` | `make_pack()`, `digest_profile` `identity-1`, set `symbols[0].proto_embedding = [0.0, 0.0, 0.0]`, no reseal | unfold false (`l2=1.000000`), residual false, and `identity_bound` false, `passed` false. The legacy half of this case is already locked by `test_prototype_tamper_without_reseal`: `gloss_bound` true when the rest of today's payload is intact. The new assertion is the additional digest failure |
| `test_identity1_sub_ulp_nudge_keeps_the_canonical_value` | `make_pack()`, `digest_profile` `identity-1`, add `1e-9` to `proto_embedding[0]`, no reseal | `round(..., 8)` stays `1.0`. `identity_digest` unchanged. `identity_bound` true. Structural `passed` true. `decision` `UNRESOLVED`. The record of this case says the digest is unchanged because the canonical value is unchanged. It does not say the nudge passed `PROTO_ATOL` |
| `test_identity1_does_not_bind_excluded_fields` | `make_pack()`, `digest_profile` `identity-1`, edit `confidence` to `1.0`, set `metadata["kappa"]` to `0.0`, and append a receipt, no reseal | `identity_digest` unchanged, `identity_bound` true, `passed` true, `decision` `UNRESOLVED`. `digest_does_not_bind` still contains `confidence`, `kappa`, and `receipts`. Translate confidence stays `0.92` |
| `test_identity1_missing_digest_fails_closed` | profile named `identity-1`, `identity_digest` absent or empty, prototype untouched | `passed` false, `decision` `RESEARCH_FAIL`. The pack does not load as `unbound` and does not inherit a pass from the four gates alone |
| `test_unbound_omission_is_not_an_identity_pass` | certificate or pack dict with no `digest_profile` key, including a twelve-field certificate and a pack that fails unfold because the reservoir id is missing | loads as `unbound`. `digest_does_not_bind` still lists `proto_embedding`. Missing reservoir still fails unfold. `decision` is not `RESEARCH_PASS`. Omitted `schema_version` still loads as `unversioned` |

`passed` under `identity-1` is false for the `1e-6` case because the sealed object is not the object now present. `passed` under `unbound` stays true for that case. Setting it false on the legacy profile would change today's contract, which `test_prototype_drift_inside_atol_passes` locks.

On every row, including a row where integrity passes, numerical and semantic evidence stay `NOT IMPLEMENTED` / `UNRESOLVED`.

## 5. What an in-place checksum change would break

Changing `compute_checksum` tomorrow so that its payload includes `proto_embedding`, while `seal` still stores that result in `pack.checksum`, redefines the field every current caller treats as the digest.

### Tests

No test locks a frozen 16-hex literal of `pack.checksum`. The locks are equality against the string `seal()` just wrote, or a checksum-mismatch failure after a mutation.

These assertions fail if the current payload starts hashing `proto_embedding`, because they require the checksum to be unchanged when only the prototype changes:

- `tests/test_pr1_certification.py::test_prototype_tamper_without_reseal` (`compute_checksum() == sealed` after `[0.0, 0.0, 0.0]`, and `expected_checksum == sealed`)
- `tests/test_pr1_certification.py::test_prototype_drift_inside_atol_passes` (the same equality after `+1e-6`, plus `passed` true)

`tests/test_unfold.py::test_certify_fails_when_prototype_drifts_from_reservoir` reseals before `certify`. `gloss_bound` stays true after a reseal even if the new payload includes the prototype, because `seal` would write the new digest. That test locks the unfold failure and the post-reseal `gloss_bound` flag. It does not lock a checksum string. It would still pass under an in-place change. The no-reseal tests above would not.

These tests lock checksum behavior for fields that are already in the payload. They keep passing if the prototype is added and those fields stay in the payload. They fail if `pack.checksum` stops being `compute_checksum()` of that payload:

- `tests/test_audit.py::test_certify_detects_checksum_tamper`
- `tests/test_pr1_certification.py::test_alias_tamper_without_reseal`
- `tests/test_pr1_certification.py::test_observation_embedding_tamper_without_reseal`
- `tests/test_pr1_certification.py::test_missing_reservoir_id_without_reseal`
- `tests/test_unfold.py::test_mutating_reservoir_text_breaks_the_seal`
- `tests/test_prose_corpus.py` (definition rewritten without resealing; `gloss_bound` false)
- `tests/test_varied_prose.py` (the same definition rewrite)
- `tests/test_translator.py::test_ok_and_unknown_states` (`Gloss.pack_checksum == pack.checksum`)
- `tests/test_unfold.py::test_reservoir_survives_json_roundtrip` (`loaded.compute_checksum() == pack.checksum` after `save_pack` / `load_pack`)

`tests/test_pr1_certification.py::test_confidence_tamper_is_outside_the_digest` fails if `confidence` is added to the same payload. That is one reason section 2 leaves `confidence` excluded.

`tests/test_pr1_certification.py::test_structural_pass_is_schema_v02_and_not_a_research_pass` locks `digest_does_not_bind` against `DIGEST_DOES_NOT_BIND`. Removing `proto_embedding` from that tuple without updating the test fails it. The list is written by `certify` in `src/neuralese/audit.py`.

`tests/test_activation_source.py` and `tests/test_activation_boundary.py` lock SHA-256 file bytes in `PRODUCTION_SHA256` for `src/neuralese/encode.py`, `src/neuralese/clustering.py`, `src/neuralese/alphabet.py`, `src/neuralese/audit.py`, `src/neuralese/__init__.py`, and `pyproject.toml`. `src/neuralese/contracts.py` is not in that map, so an edit confined to `compute_checksum` does not by itself fail those hashes. An edit of `audit.py` that changes the evidence list or `passed` does fail them. The `__init__.py` and `pyproject.toml` hashes lock package `0.7.0`.

`tests/test_symbol_residual.py`, `tests/test_symbol_addressing.py`, `tests/test_activation_source.py`, and `tests/test_activation_boundary.py` also lock activation-array SHA-256 values (`ACTIVATION_SHA256` and the `report["checksums"]` strings). Those are hashes of fixture arrays, not `SymbolPack.compute_checksum`.

No test snapshots sealed pack bytes against a golden file.

### CI

`.github/workflows/tests.yml` job `pytest` has these steps: `Install`, `Test`, `Public pack`, `Prose pack`, `Varied prose`, `Activation stack`.

`Test` runs `python -m pytest -q`. It does not compare a digest of its own. It runs the tests named above.

`Public pack`, `Prose pack`, `Varied prose`, and `Activation stack` each run `neuralese learn`, which calls `seal()`, and then `certify` on that fresh pack. `Public pack` also learns `examples/mdl_delta/observations.jsonl` twice and compares `delta_mdl_bits`, `metadata["decision"] == "reject"`, quarantined symbols, and live codes. `Prose pack`, `Varied prose`, and `Activation stack` compare report counts, residual bands, `decision`, and `mdl_bits`. `Varied prose` also reads `pack.metadata["config"]` and `pack.guards`. None of the four steps compare a checksum hex string or a sealed-pack byte string.

A self-consistent in-place change, where `seal` and `certify` both use the new payload, would still certify a pack that the workflow has just learned. The workflow would fail indirectly because `Test` runs the prototype equality tests. It would not fail because a stored digest in the workflow file stopped matching.

### On-disk examples

These paths do not store a `SymbolPack` checksum:

- `examples/public_domain/` (`observations.jsonl`, `hidden_states.jsonl`)
- `examples/prose_corpus/` (`observations.jsonl`, `anchors.json`)
- `examples/mdl_delta/` (`observations.jsonl`)
- `examples/varied_prose/`, `examples/toy_stream/`, `examples/generalization/` (observation and topic files)
- `examples/activation_stack/hidden_states.npz`

`examples/activation_source/capture_meta.json` and `examples/activation_boundary/capture_meta.json` store a `checksums` object of SHA-256 strings for activation arrays (64 hex characters). Those strings are not `pack.checksum`. Changing `compute_checksum` does not make them mismatch. The npz fixtures beside them are arrays, not sealed packs.

There is no checked-in sealed pack whose `checksum` field would go stale. Any pack already sealed outside this tree, and any pack a test holds after `seal()`, still carries today's digest. Redefining `pack.checksum` makes `certify` report `gloss_bound` false on those packs until they are resealed, including packs whose prototype was never edited.

### Translate

`translate_stream` does not read `proto_embedding`. Gloss state, English text, and confidence are functions of code resolution, `quarantined`, `definition`, `confidence`, and `reconstruction_error`. A prototype edit does not change those. The gloss copies the stored `pack.checksum` string and does not recompute it. Under no reseal, that copied string stays the pre-edit checksum whether or not a future identity digest changes.

### Certify

If `pack.checksum` is redefined in place, every pack whose stored checksum was produced by today's payload fails `gloss_bound` until `seal()` runs again. That includes every pack that currently passes with an untouched prototype. `passed` becomes false and `decision` becomes `RESEARCH_FAIL` on those packs. After reseal, the new string adopts the prototype that happens to be present, including a `1e-6` nudge, and `gloss_bound` can pass again. The integrity failure and a later reseal are then the same field as today's tamper check.

If `identity_digest` is an additional field and `pack.checksum` stays today's payload, `gloss_bound` and `passed` for current packs stay what the four gates already say. The `1e-6` case stays `passed` true under `unbound`. `identity-1` is opt-in. Existing checksum strings still match.

## Recommendation

Do not replace `pack.checksum` in place. Add the identity digest as `digest_profile` `identity-1` and field `identity_digest`. Leave `compute_checksum`, `seal`, and the current pack bytes as they are until a follow-up implements that profile and can migrate callers without rewriting the legacy meaning of the current checksum.

`identity-1` binds the fields today's checksum already binds, plus `proto_embedding` rounded to 8 decimals. It leaves `confidence`, `examples`, `survival`, symbol `metadata` / `kappa`, `guards`, `receipts`, `parent_pack_id`, `mdl_bits`, pack `metadata`, and pack `timestamp` excluded, because the certificate does not claim their identity.

Under `unbound`, the `1e-6` prototype nudge keeps today's checksum, `passed` true, and `decision` `UNRESOLVED`, and `digest_does_not_bind` still lists `proto_embedding`. Under `identity-1`, that nudge changes `identity_digest`, `identity_bound` is false, and `passed` is false, while unfold still accepts the L2 distance. A nudge of `1e-9` rounds away at 8 decimals and leaves the identity digest unchanged. A prototype replaced by a far vector fails unfold and residual honesty on both profiles; `identity-1` also fails the digest; `unbound` keeps `gloss_bound` true when the rest of today's payload is intact.

Recommendation **NONE**. Authority **RUNTIME_NOT_AUTHORIZED**. Schema **0.2** unchanged. Package **0.7.0** unchanged. Serialization was not changed. No tests were added.

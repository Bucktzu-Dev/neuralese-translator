# PR-1 certification discrepancy report

Status: **SECTIONS 4 AND 5 IMPLEMENTED**. Tracking: ECHO-NLA-001 / GitHub issue #6.

Sections 1–3 are the reading of the certificate before schema 0.2. Section 8 records what was implemented from sections 4 and 5, and what was left unspecified. Package version stays **0.7.0** (`pyproject.toml`, `src/neuralese/__init__.py`). That number is the package. Certificate `schema_version` `"0.2"` is a different number.

The pre-implementation audit decision stays **RESEARCH_FAIL**. Schema 0.2 does not emit `RESEARCH_PASS`. Authority stays **RUNTIME_NOT_AUTHORIZED**. Numerical and semantic gates stay **NOT IMPLEMENTED**. `SymbolPack.compute_checksum` is unchanged. `learn_pack`, the translator's four gloss states, and the adapters are unchanged.

## External context

Anthropic, *Natural Language Autoencoders* (7 May 2026), reconstructs an activation from generated text. Neuralese currently certifies that a symbol unfolds to stored observations. Those are different evidence classes. Reservoir unfold is not text-only activation reconstruction. Issue #6 records the same boundary as an echo (`ECHO_RECORDED` / `COMPARISON_REVIEWED` / `REPRODUCTION_NOT_RUN` / `INTEGRATION_NOT_AUTHORIZED`). This report does not treat that publication as implementation authority.

## 1. Current certificate fields and what `passed` means

This section describes the twelve fields as they were before schema 0.2. The fields added in section 8 sit beside them. `passed` is still the same four-gate conjunction.

`AuditCertificate` (`src/neuralese/contracts.py`) had twelve fields and no schema, evidence-class, or authority field:

| field | what it stores today |
|---|---|
| `pack_id` | `SymbolPack.pack_id` |
| `pack_checksum` | the digest stored on the pack |
| `expected_checksum` | `pack.compute_checksum()` at audit time |
| `passed` | conjunction below |
| `addressable` | codebook and alias resolution |
| `unfoldable` | live symbols have ids, and listed ids unfold to stored rows whose mean matches the prototype |
| `gloss_bound` | stored digest is non-empty and matches the recomputed digest, and every live symbol has a non-empty definition when `require_gloss` is true |
| `residual_ok` | sealed `reconstruction_error` is finite and ≤ `tau_residual` (default 0.55), and is not tighter than the residual recomputed from stored rows |
| `fail_closed` | always `True` |
| `failures` | human-readable gate strings |
| `timestamp` | `time.time()` of the audit call |
| `details` | counts, `tau_residual`, `require_gloss`, `mdl_bits`, `reservoir_size`, `recomputed_cluster_residual` |

`certify().passed` is true only when `addressable` and `unfoldable` and `gloss_bound` and `residual_ok` are all true (`src/neuralese/audit.py`). It does not read `guards`, `metadata["decision"]`, `kappa`, `confidence`, `examples`, `survival`, or `parent_pack_id`. A pack whose learn metadata says `reject` still certifies when those four gates pass. `passed` is a structural seal of the current gates. It is not a per-symbol semantic certificate and it is not a numerical reconstruction from text.

Gloss states stay the public four: `ok`, `aliased`, `quarantined`, `unknown` (`GLOSS_STATES` in `src/neuralese/contracts.py`). `translate_stream` (`src/neuralese/translator.py`) assigns them as follows:

| state | when | English |
|---|---|---|
| `unknown` | `symbol_by_code` misses | `[undecodable: no symbol for code N]`, confidence 0 |
| `quarantined` | resolved symbol has `quarantined=true` | `[quarantined: symbol N is not certified unfoldable]`, confidence 0 |
| `aliased` | alias chain rewrites the code, symbol is live | bound definition, or `[unglossed: …]` if the definition is empty |
| `ok` | resolved, live, not an alias rewrite | bound definition, or `[unglossed: …]` if the definition is empty |

`state=ok` does not call `certify`. The gloss copies the stored `pack.checksum`; it does not recompute it. An empty definition is still `state=ok` with an explicit `[unglossed: …]` string. On the current `make_pack` helper, that gloss is `ok` while `certify` sets `gloss_bound=false` (`class 0 has no bound English gloss`). A successfully translated gloss is not a fully certified symbol.

`[unglossed: class N]` written at learn time (`src/neuralese/alphabet.py`) is a non-empty definition, so `certify` treats it as gloss-bound. `decodability_report` excludes that prefix from `gloss_coverage`. `certify` does not.

## 2. Checksum inclusion

`SymbolPack.compute_checksum` hashes one canonical object (`src/neuralese/contracts.py`):

- `pack_id`
- `codebook` (stringified int keys, sorted)
- `aliases` (stringified int keys, sorted)
- `reconstruction_error` rounded to 8 decimals
- `reservoir`: each observation's id, text, embedding rounded to 8 decimals, and metadata
- each symbol's `class_id`, `code`, `observation_ids`, stripped `definition` (empty becomes null), `quarantined`

The digest is SHA-256 truncated to 16 hex characters. Timestamps are outside it, so re-auditing a sealed pack is deterministic (`docs/AUDIT_PROTOCOL.md`).

Checked against `certify()` on this tree with `tests/packutil.py` `make_pack()` (default residual 0.08, two live symbols, reservoir rows copied from the prototypes). "Detected" means `passed` is false.

### SymbolPack fields

| field | in checksum | without reseal | after reseal |
|---|---|---|---|
| `pack_id` | yes | digest mismatch, `gloss_bound` false | reseal adopts the new id |
| `codebook` | yes | digest mismatch; `addressable` also fails if the map is inconsistent | reseal adopts it; `addressable` still checks consistency |
| `aliases` | yes | digest mismatch. A new well-formed alias (`7 → 0`) leaves `addressable` true. `translate_stream` still emits `aliased` | reseal adopts it. A cycle or a dangling target still fails `addressable`. A cycle sealed in the original pack already fails `addressable` (`tests/test_audit.py`) |
| `reconstruction_error` | yes, 8 decimals | digest mismatch, and `residual_ok` fails if the value is above `tau_residual` or below the recomputed cluster residual by more than `1e-4` | reseal adopts it; both residual checks still run |
| `checksum` | no (it is the digest) | empty digest fails `gloss_bound`; a forged digest fails `gloss_bound` | `seal()` writes `compute_checksum()` |
| `parent_pack_id` | no | not detected | not detected |
| `receipts` | no | not detected | not detected |
| `guards` | no | not detected. `certify` does not read `pass_all` or `pass_kappa` | not detected |
| `mdl_bits` | no | not detected | not detected |
| `timestamp` | no | not detected | not detected |
| `metadata` | no | not detected, including `decision=reject` | not detected |
| `observations` | yes, as reservoir rows | see observation rows | see observation rows |
| `symbols` | partial | see symbol rows | see symbol rows |

### Symbol fields

| field | in checksum | without reseal | after reseal |
|---|---|---|---|
| `class_id`, `code`, `observation_ids`, `quarantined` | yes | digest mismatch; unfold or addressable checks may also fail | reseal adopts them; those checks still run |
| `definition` | yes, after strip; blank becomes null | digest mismatch, `gloss_bound` false. Existing test: `test_certify_detects_checksum_tamper` | reseal adopts a non-empty definition and `gloss_bound` can pass. An empty definition still fails `require_gloss` |
| `proto_embedding` | **no** | drift of L2 1.0 (`[1,0,0]` → `[0,0,0]`) leaves the digest unchanged and `gloss_bound` true. `unfoldable` is false (`prototype does not match reservoir mean (l2=1.000000)`). `residual_ok` is false (recomputed cluster residual 0.707107 > sealed 0.080000). A `1e-6` shift on the first coordinate still **passes** (`PROTO_ATOL` is `1e-5`) | same two failures. Reseal rewrites a digest that still omits the prototype, so `gloss_bound` stays true. Current code catches above-tolerance drift. It does not catch it as a checksum failure |
| `examples` | no | not detected. Translator does not read examples | not detected |
| `confidence` | no | not detected. Setting confidence to 1.0 keeps `passed` true. `translate_stream` emits 0.92 because `confidence_cap` clips to `1 - residual` | not detected |
| `survival` | no | not detected | not detected |
| `metadata` (`kappa`, `size`) | no | not detected. Setting `kappa` to 0.0 keeps `passed` true. `certify` does not reapply `tau_kappa` | not detected |

### Observation fields

| field | in checksum | without reseal | after reseal |
|---|---|---|---|
| `observation_id` | yes | deleting `obs-hello` fails the digest and fails unfold (`missing reservoir observation obs-hello`). `residual_ok` stays true: an incomplete reservoir makes `recomputed_cluster_residual` return `None`, and the honesty check only runs when that value exists. `translate_stream` of code 0 stays `state=ok` | clearing the reservoir and resealing fails unfold with `gloss_bound` still true. Existing test: `test_certify_fails_when_reservoir_rows_are_removed` |
| `text` | yes | digest mismatch, `gloss_bound` false. Unfold and residual stay true; unfold does not compare text. Existing test: `test_mutating_reservoir_text_breaks_the_seal` | reseal adopts the new text. Unfold can pass. There is no semantic gate on the new sentence |
| `embedding` | yes, 8 decimals | replacing `[1,0,0]` with `[0,1,0]` fails the digest, fails unfold (L2 1.414214), and fails residual honesty (recomputed residual 1.0 > 0.08). A `1e-10` nudge rounds away and **passes** | reseal clears the digest failure. Unfold and residual honesty still fail while the prototype and sealed error stay put. Matching the prototype to the new mean, raising `reconstruction_error` to cover it and stay ≤ `tau_residual`, and resealing can pass. That is a new seal of a new geometry |
| `metadata` | yes | digest mismatch | reseal adopts it. `certify` does not interpret observation metadata |

Definition whitespace is inside the strip, so `"  Symbol for hello.  "` keeps the same digest and `passed` stays true.

Counts on this tree: 5 of 10 `Symbol` fields are in the digest (`class_id`, `code`, `observation_ids`, `definition`, `quarantined`). All 4 `Observation` fields are in the digest, with embedding coordinates rounded to 8 decimals. Of the pack fields, `parent_pack_id`, `receipts`, `guards`, `mdl_bits`, `timestamp`, and `metadata` are outside the digest. `checksum` is the digest, not an input.

## 3. Discrepancies against the six rules

### Rule 1. Preserve the public gloss contract

**Held by the current translator.** States are only `ok` / `aliased` / `quarantined` / `unknown`. An unknown code stays `[undecodable: no symbol for code N]` with confidence 0 (`src/neuralese/translator.py`, `tests/test_translator.py::test_ok_and_unknown_states`). Missing codes do not receive invented English.

Two honesty gaps sit inside that contract. They are recorded here and not changed:

- A live symbol with `definition=None` translates as `state=ok` and `[unglossed: symbol N has no bound English]`, while `certify` fails `gloss_bound`. `ok` is not "certified".
- A quarantined symbol whose reservoir matches its prototype still translates as `[quarantined: symbol N is not certified unfoldable]`, and `certify().passed` is true with `unfoldable` true. Quarantine at learn time is `size < min_cluster_size` or `kappa < tau_kappa` (`src/neuralese/alphabet.py`), which is not the same predicate as unfold failure. The state is `quarantined`, not `ok`. The sentence overclaims the reason.

Rule status: the state machine holds. The v0.2 program does not inherit a pass from that fact.

### Rule 2. Keep decodability, numerical reconstruction, and semantic correctness separate

**RESEARCH_FAIL.** One boolean, `passed`, folds addressability, reservoir unfold, gloss binding, and residual honesty together. `AuditCertificate` has no evidence-class tag. `gloss_bound` mixes a structural digest with "definition is non-empty", which is not a semantic check. `residual_ok` compares stored embeddings to stored prototypes (`recomputed_cluster_residual` in `src/neuralese/unfold.py`). That is provenance geometry. It is not a numerical reconstruction from gloss text, and the certificate does not say so.

### Rule 3. Do not infer semantic truth from fluency or cosine alone

**UNRESOLVED.** `certify` does not set `passed` from fluency or from cosine. That part holds.

What the tree actually computes:

- Gloss confidence is a text-count heuristic: `min(1.0, 0.25 + 0.15 * text_count + 0.05 * len(keywords))` (`src/neuralese/gloss.py`). Quarantine zeros it at learn time. Translation then applies `confidence_cap` (`src/neuralese/energy.py`): `min(confidence, 1 - clip(residual, 0, 1))`. The cap is residual, not cosine. The likely "cosine-capped confidence" reading is false on this tree.
- `kappa` is the mean cosine of member embeddings to the prototype (`src/neuralese/alphabet.py`), stored on `symbol.metadata["kappa"]`, compared to `tau_kappa` (default 0.35) only while learning. `certify` and `translate_stream` do not read it.
- `cluster_purity` is a report diagnostic. `certify` does not read it (`src/neuralese/audit.py`).

Confidence and kappa are unlabeled on the certificate. A gloss can show 0.92 after a confidence edit that `certify` accepts. Nothing in the certificate says that number is not semantic evidence. No semantic gate exists to fail.

### Rule 4. A stored reservoir is not text-only reconstruction

**RESEARCH_FAIL** as a certificate claim. **The implementation retrieves stored rows.**

`unfold_code` looks up `pack.observations` and returns those objects (`src/neuralese/unfold.py`). `unfold_failures` checks that each listed id exists, that the embedding is non-empty and the same width as the prototype, and that the member mean is within L2 `1e-5` of `proto_embedding`. `recomputed_cluster_residual` is the Frobenius ratio of `(embedding - prototype)` over stored rows. None of these functions build an activation from the English gloss.

`docs/DECODABILITY.md` describes the reservoir as the fold path for a lossy symbol. `LIMITATIONS.md` says the reservoir is part of the seal. The certificate itself only exposes `passed` plus `recomputed_cluster_residual`. A reader can treat that pair as successful numerical reconstruction. The object that would forbid that reading is missing.

There is no text-to-activation reconstructor in this package. This report does not add one.

### Rule 5. Unknown or unsupported content stays explicitly unresolved

**RESEARCH_FAIL** for the certificate. The translator path for an unknown code is held (rule 1).

Unsupported certificate content is absent rather than labeled:

- No `schema_version`. `AuditCertificate.to_dict()` keys are the twelve fields in section 1.
- A pack sealed with an empty `observations` map and live ids fails unfold (`missing reservoir observation …`) with `gloss_bound` still true, because the digest was computed with `"reservoir": []`. There is no `v0.1` branch in `certify` or `unfold_failures`. `LIMITATIONS.md` says packs sealed before the reservoir fail closed until relearned. That is today's behavior: fail unfold (and, if the stored digest was minted without a reservoir payload, also fail `gloss_bound`). It is not an explicit compatibility mode.
- Numerical and semantic questions have no `UNRESOLVED` slot. They are simply not fields.

### Rule 6. New capabilities do not acquire authority

**RESEARCH_FAIL.** Nothing on `AuditCertificate` or `SymbolPack` stores `RUNTIME_NOT_AUTHORIZED`. `passed=true` is the only success bit. Logit lens is `next` on `docs/ROADMAP.md` and is unstarted (`LIMITATIONS.md`). Absence is not an authority field. A later experiment must not treat `passed` as permission to override CSLP, SLAR, or this translator.

## 4. Proposed v0.2 certificate schema

The field list below is what section 8 implemented. Package version stays 0.7.0. Schema version and package version are different numbers. The sentences in this section that say the fields are absent describe the tree before that implementation.

A v0.2 certificate would add, beside the current twelve fields:

| field | value in the proposal |
|---|---|
| `schema_version` | `"0.2"` when a future sealer writes it. Absent on every certificate this tree emits today |
| `package_version` | informational copy of `neuralese.__version__`. Not a gate. Today that string is `"0.7.0"` |
| `authority` | `RUNTIME_NOT_AUTHORIZED`. No other value is defined by PR-1 |
| `decision` | `RESEARCH_FAIL` or `UNRESOLVED` while numerical and semantic gates are unimplemented. The whole-certificate decision must not be `RESEARCH_PASS` |
| `evidence.structural` | today's four booleans, the failure list, and an explicit list of fields the digest does not bind (`proto_embedding`, `confidence`, `examples`, `survival`, symbol `metadata` / `kappa`, `guards`, `receipts`, `parent_pack_id`, `mdl_bits`, pack `metadata`) |
| `evidence.numerical` | `status: NOT IMPLEMENTED`, `decision: UNRESOLVED`. Forbidden reading: reservoir centroid residual, or gloss text, scored as a reconstructed activation |
| `evidence.semantic` | `status: NOT IMPLEMENTED`, `decision: UNRESOLVED`. Forbidden inputs: fluency, the confidence heuristic, kappa cosine, cluster purity, cosine similarity |

`evidence.structural` may record that the four current gates passed. That record stays a structural note. It does not promote `decision` to `RESEARCH_PASS`, and it does not cover the unbound prototype. Closing the prototype gap (put `proto_embedding` in the digest, or state in the schema that unfold, not the digest, binds it) is PR-2 work.

### Legacy packs

A legacy pack here means an unversioned pack: no `schema_version`, and typically no reservoir, with a digest minted before `reservoir` was part of `compute_checksum`.

Current behavior, to preserve until a loader change is accepted:

- Missing `observations` loads as `{}` (`_load_observations(None)` in `src/neuralese/contracts.py`).
- Live symbols with ids that are absent fail `unfoldable`.
- Live symbols with an empty id list fail `unfoldable` (`has no observation_ids`). Quarantined symbols may omit ids.
- `compute_checksum` always hashes `reservoir`. An old digest that omitted that key fails `gloss_bound`.

Proposed handling, not implemented:

- Missing `schema_version` is labeled `unversioned`. Loaders keep accepting today's certificate JSON. `AuditCertificate.from_dict` currently requires the existing keys; making `schema_version` required would reject those objects. That breaking loader change is escalated and not taken. The compatible choice is an optional field that defaults to `unversioned`.
- Do not add a mode that turns today's unfold failure into `passed=true`.
- Whole-certificate `decision` for a reservoir-less pack is `RESEARCH_FAIL` on the structural reservoir gate and `UNRESOLVED` on numerical and semantic gates.
- Migration is to re-learn so the reservoir rows are stored, the prototype matches their mean, and a future sealer writes `schema_version: "0.2"`. Resealing an empty reservoir only refreshes an unversioned digest; it does not create a v0.2 certificate.

A fully certified symbol, under this proposal, would need a structural pass that includes that symbol and numerical and semantic decisions other than `UNRESOLVED`. No symbol in this package meets that bar. `translate_stream` state `ok` remains the public gloss state and is the weaker claim.

## 5. Regression-test proposal

The tests proposed below are now in `tests/test_pr1_certification.py` and use `make_pack` from `tests/packutil.py`. Expectations were checked against `certify` / `translate_stream` before schema 0.2 and still describe the digest and the four structural gates. Section 8 notes the one assertion that changed because `certify` now writes `schema_version`.

| test | file today | setup and mutation | expected result | status |
|---|---|---|---|---|
| `test_certify_detects_checksum_tamper` | `tests/test_audit.py` | `make_pack()`, set `symbols[0].definition` to a new sentence, no `seal()` | `passed` false, `gloss_bound` false, a failure containing `checksum` | **exists** |
| `test_alias_tamper_without_reseal` | proposed `tests/test_pr1_certification.py` | `make_pack()`, set `aliases[7] = 0`, no `seal()` | `passed` false, `gloss_bound` false, `addressable` true, failure contains `checksum`. `translate_stream(pack, [7])` is `state=aliased`, `resolved_code=0` | **absent**. `test_certify_fails_alias_cycle` seals a cycle at construction and only checks `addressable` |
| `test_mutating_reservoir_text_breaks_the_seal` | `tests/test_unfold.py` | `make_pack()`, set `observations["obs-hello"].text`, no `seal()` | `gloss_bound` false, failure contains `checksum`. Unfold stays true | **exists** |
| `test_observation_embedding_tamper_without_reseal` | proposed | `make_pack()`, set `observations["obs-hello"].embedding = [0.0, 1.0, 0.0]`, no `seal()` | `passed` false, `gloss_bound` false, `unfoldable` false (L2 about 1.414214), `residual_ok` false (recomputed residual 1.0 > 0.08) | **absent** |
| `test_prototype_tamper_without_reseal` | proposed | `make_pack()`, set `symbols[0].proto_embedding = [0.0, 0.0, 0.0]`, no `seal()` | digest unchanged, `gloss_bound` true, `unfoldable` false (`l2=1.000000`), `residual_ok` false (recomputed residual about 0.707107), `passed` false | **absent**. Current code catches this drift via unfold and residual honesty, not via the checksum |
| `test_certify_fails_when_prototype_drifts_from_reservoir` | `tests/test_unfold.py` | same prototype assignment, then `seal()` | `unfoldable` false, failure contains `prototype`. Additional assertion to lock: `gloss_bound` stays true because the prototype is outside the digest | **exists** for the unfold failure. The `gloss_bound` assertion is **absent**. Current code catches the resealed drift the same way |
| `test_prototype_drift_inside_atol_passes` | proposed | `make_pack()`, add `1e-6` to `proto_embedding[0]`, no `seal()` | `passed` true. Documents `PROTO_ATOL = 1e-5` | **absent** |
| `test_missing_reservoir_id_without_reseal` | proposed | `make_pack()`, `del observations["obs-hello"]`, no `seal()` | `passed` false, `gloss_bound` false, `unfoldable` false, failure contains `missing reservoir observation obs-hello`, `residual_ok` true. `translate_stream(pack, [0])` stays `state=ok` | **absent**. `test_certify_fails_when_reservoir_rows_are_removed` clears every row and reseals, so `gloss_bound` stays true and unfold fails |
| `test_ok_and_unknown_states` | `tests/test_translator.py` | `make_pack()`, codes `[0, 99]` | code 0 is `ok`; code 99 is `unknown`, English starts with `[undecodable:`, confidence 0. `certify` on the untouched pack stays passed; an unknown query is not a certify input | **exists** for translate. `tests/test_unfold.py::test_unfold_unknown_code_has_no_rows` covers unfold state `unknown` |
| `test_quarantined_state_does_not_use_definition_as_ok` | `tests/test_translator.py` | one quarantined symbol, definition `"should not appear as ok"` | `state=quarantined`, English starts with `[quarantined:` | **exists** |
| `test_unglossed_ok_is_not_a_certificate` | proposed | `make_pack` with `definition=None` and a matching one-row reservoir | translate `state=ok`, English starts with `[unglossed:`. `certify` has `gloss_bound` false and `passed` false | translate half **exists** as `test_unglossed_live_symbol_is_explicit_gap`. The paired `certify` assertion is **absent** |
| `test_confidence_tamper_is_outside_the_digest` | proposed | `make_pack()`, set `symbols[0].confidence = 1.0`, no `seal()` | `passed` true, digest unchanged, translate confidence is 0.92 | **absent**. Locks the gap until a schema change decides to bind confidence |
| `test_unversioned_empty_reservoir_fails_unfold` | proposed | `make_pack(observations={}, symbols=[one live symbol with id "obs-1" and a definition])` so `seal()` runs on an empty map | `passed` false, `unfoldable` false, `gloss_bound` true, failure contains `missing reservoir observation obs-1`. `schema_version` is not an attribute of the certificate | **absent**. Documents legacy failure with no v0.1 mode |

Also already present, not duplicated: `test_sealed_residual_cannot_undercut_reservoir` (residual honesty when the mean matches and the sealed error is too small) and `test_quarantined_without_ids_still_passes_unfold` (a quarantined symbol may omit ids).

## 6. What this report does not authorize

| later slice | state |
|---|---|
| PR-2, versioned certificate fields and the tests in section 5 | `RUNTIME_NOT_AUTHORIZED` until this report is accepted. Schema text in section 4 is a proposal |
| PR-3, a text-to-activation reconstructor | `RUNTIME_NOT_AUTHORIZED`. Out of scope. Reservoir unfold must not be extended into that reconstructor under this note |
| PR-4, a held-out activation eval | `RUNTIME_NOT_AUTHORIZED` |
| Logit lens, SAE, hub runtime, learned model, or any change that gives a new experiment authority over CSLP, SLAR, or `translate_stream` | `RUNTIME_NOT_AUTHORIZED` |

Numerical and semantic gates stay **NOT IMPLEMENTED**.

## 7. Recommended next action

After this report is accepted, implement only the PR-1 tests in section 5 and the versioned certificate fields in section 4. Keep numerical and semantic gates `NOT IMPLEMENTED` and `UNRESOLVED`. Keep `authority` at `RUNTIME_NOT_AUTHORIZED`. Leave the public gloss states as they are. Do not start PR-3.

That action is what section 8 records. It is not a wider certificate.

## 8. Implementation note

Schema 0.2 is what `certify()` writes. `passed` is still true only when `addressable`, `unfoldable`, `gloss_bound`, and `residual_ok` are all true. A structural miss sets `decision` to `RESEARCH_FAIL`. A structural pass sets `decision` to `UNRESOLVED`, because numerical and semantic gates are unimplemented. `decision` is never `RESEARCH_PASS`. `authority` is `RUNTIME_NOT_AUTHORIZED`. `package_version` is an informational copy of `neuralese.__version__` (`0.7.0`) and is not a gate.

`evidence.structural` records the four booleans, the failure list, and `digest_does_not_bind`: `proto_embedding`, `confidence`, `examples`, `survival`, `symbol.metadata`, `kappa`, `guards`, `receipts`, `parent_pack_id`, `mdl_bits`, `pack.metadata`. Those fields were not added to `compute_checksum`. `evidence.numerical` and `evidence.semantic` are `status: NOT IMPLEMENTED`, `decision: UNRESOLVED`, with the forbidden reading and the forbidden inputs from section 4.

Legacy certificate JSON that omits `schema_version` loads as `unversioned`. `AuditCertificate.from_dict` still accepts the original twelve fields. An omitted `decision` stays `UNRESOLVED` and is not recomputed into a pass. An omitted `authority` loads as `RUNTIME_NOT_AUTHORIZED`. An omitted `package_version` stays null. A stored `RESEARCH_PASS`, any other authority, or a schema label other than `0.2` or `unversioned` is rejected. There is no v0.1 mode. A pack sealed with an empty reservoir still fails unfold (`missing reservoir observation …`) with `passed` false and `decision` `RESEARCH_FAIL`. Resealing that empty reservoir does not make `passed` true.

Section 5 tests that were absent are in `tests/test_pr1_certification.py`. They lock the published fail/pass results, including the prototype nudge inside `PROTO_ATOL` `1e-5` (`passed` true) and the confidence edit (`passed` true, digest unchanged, translate confidence 0.92). `tests/test_unfold.py::test_certify_fails_when_prototype_drifts_from_reservoir` now also asserts `gloss_bound` stays true after reseal, because the prototype is outside the digest. The expected unfold failure did not become a pass.

`test_unversioned_empty_reservoir_fails_unfold` was specified against a certificate that had no `schema_version` attribute. `certify()` now writes `schema_version` `"0.2"` on that failure. The unfold failure, `gloss_bound` true, and `passed` false are unchanged. The loader test covers a twelve-field dict that still has no schema field.

The activation-source and activation-boundary tests lock a sha256 of `src/neuralese/audit.py`. That digest now names the schema 0.2 file. The encode, clustering, alphabet, package init, and `pyproject.toml` digests in those tests are unchanged. No held-out slot count, residual, or admission byte was retuned.

### Not specified, and not implemented

- A per-symbol certified flag, and any gloss state besides `ok` / `aliased` / `quarantined` / `unknown`. The plan distinguishes a translated gloss from a certified symbol by the certificate: a full certification would need a structural pass and numerical and semantic decisions other than `UNRESOLVED`, and no symbol meets that bar. `translate_stream` state `ok` stays the weaker public gloss. The exact extra gloss fields were not defined, so none were added.
- Putting `proto_embedding` in the digest, or declaring that unfold rather than the digest closes that gap. Section 4 names that closure as later work. The 1e-6 prototype nudge still passes. The larger prototype assignment still fails unfold and residual honesty with `gloss_bound` true.
- Binding `confidence`, `examples`, `survival`, symbol metadata / `kappa`, `guards`, `receipts`, `parent_pack_id`, `mdl_bits`, or pack `metadata`. The plan lists them as unbound. The tests lock that.
- A legacy mode named v0.1 that turns a missing reservoir into `passed=true`.
- Rewording the quarantined gloss. Section 3 records that `[quarantined: symbol N is not certified unfoldable]` overclaims the reason, and says the state machine is not changed. The sentence is unchanged.
- Changing `state=ok` when the definition is empty or a reservoir id was deleted. Those cases stay `ok` with an explicit gap string, and `certify` still fails when the structural gate fails. `tau_residual` 0.55 and `tau_kappa` 0.35 are unchanged. The production encoder, `word_sentence_svd`, farthest-first k-means, `learn_pack` clustering, and the research codecs' frozen constants are unchanged.
- A package bump. Schema 0.2 is not a package version. The package stays 0.7.0. This repository has no changelog file; this section is the record.

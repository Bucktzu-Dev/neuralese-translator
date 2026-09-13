# Audit protocol

`neuralese certify` is the public fail-closed gate. A pack that does not pass is a **draft**, not a lexicon. Translation refuses to emit English unless a named policy passes.

## Certificate

v0.1.1 splits three authorities. `passed` is a conjunction under the named policy, never a majority vote.

| field | meaning |
|---|---|
| `integrity_valid` | schema + addressable aliases + full SHA-256 seal + residual + gloss binding |
| `evidence_valid` | live ids are non-blank, present in the sealed evidence map with SHA-256 digests; if `--observations` is supplied, digests are recomputed from content |
| `admission_valid` | learning `decision` is not `reject`; guards/finalize agree with admission |
| `passed` | policy conjunction of the above |
| `policy` | `default` \| `strict` \| `integrity` |

Kept detail flags: `addressable`, `unfoldable`, `gloss_bound`, `residual_ok`.

CLI:

```bash
neuralese audit pack.json
neuralese certify pack.json --fail-on-undecodable
neuralese certify pack.json --observations observations.jsonl --fail-on-undecodable
neuralese translate pack.json stream.json          # fails if certify fails; certificate JSON on stderr
neuralese translate pack.json stream.json --allow-uncertified   # debug only
```

`learn` writes the pack always. Exit `2` when `decision=reject` (explicit draft).

## Policies

| policy | `passed` when |
|---|---|
| `default` | integrity ∧ evidence ∧ admission (`accept` or `accept_provisional`) |
| `strict` | default plus `decision=accept` and `guards.pass_all` |
| `integrity` | seal/schema only (does **not** authorize translation) |

`translate_stream(..., policy="default", require_certified=True)` is the library default. `policy="integrity"` is for inspecting a seal; `translate` and `translate_stream` reject it.

## Seal (full SHA-256)

Checksum is 64 hex characters over the complete semantic manifest:

- `pack_id`, `decoder_version`, `decision`
- `parent_pack_id`, `parent_checksum`
- codebook, versioned aliases, evidence map
- every symbol: class/code, prototype, observation ids, definition, example hashes, confidence, survival, quarantine, metadata
- guards, receipts, MDL, residual, pack metadata

Wall-clock `timestamp` on the pack object is **not** in the checksum. Receipt timestamps **are** (receipts are append-only once sealed).

This is self-consistency, not publisher authenticity. Sign or externally anchor the manifest if you need that. A pack whose `decoder_version` is not this decoder (`0.1.1`) fails integrity; the field is reported and enforced.

## Evidence / unfold

A nonempty string in `observation_ids` is not enough. Each live id must be non-blank and appear in `pack.evidence` as `id → SHA-256(observation_id, embedding, text)`.

Ids missing from the map fail. Blank ids fail. Mutating the manifest without resealing fails integrity.

A well-formed digest in the map is **self-consistency**, not proof the tensors existed. Pass the original observations (`certify(..., observations=...)` / `--observations`) to recompute hashes. Fabricated ids with attacker-chosen 64-hex digests fail that check. Duplicate `observation_id` values fail during `learn` and during content verification; they are not a fold path. The same live observation id on two classes fails evidence. A supplied observation with neither embedding nor text fails `evidence_valid` instead of crashing. `guards.pass_all` must equal the conjunction of the component flags; a sealed `pass_all=true` with a false component fails integrity.

Every `example_hashes` value must be a full 64-character SHA-256 hex **string**. A public pack that stores `"raw secret"`, a digest plus a trailing newline, or a JSON number (including a 64-digit decimal) in that field fails integrity; the field itself is not a place to store plaintext, and non-strings are not coerced with `str()`. Those hashes are unsalted SHA-256 of the canonical JSON object `{"example": <text>}` (the same `canonical_dumps` envelope as the pack seal), not SHA-256 of the UTF-8 bytes alone. `learn_pack` hashes the original nonblank observation text when no private examples are stored; leading and trailing whitespace is part of that commitment. This is a self-consistency commitment, not a confidentiality control. Guessable strings can be recovered by hashing candidates through `example_hash`. Keyed or per-pack commitments are out of scope for v0.1.1. Changing the envelope would reseal existing packs, so v0.1.1 keeps this function.

Public packs store hashes, not raw example text (`include_private=false` by default). Heuristic glosses then use keywords that do not reproduce an observation, or an explicit `[unglossed]` / quarantine marker. They do not copy the first observation verbatim into `definition`, and a single-token observation such as `TOPSECRET1234` is not kept as a public keyword. Optional LLM glosses for public packs are prompted from those remaining keywords only; a response that echoes raw observation text of any length is discarded. A recorded `[unglossed]` sentinel keeps confidence at 0 and translates as an explicit gap, not as bound English. Default certification treats that sentinel as unglossed, so `gloss_bound` fails unless `--allow-unglossed` is set. An explicit empty `decoder_version` or `decision` is preserved on load and fails integrity rather than being silently replaced with this decoder's defaults. Unknown `decision` values fail schema even under `--policy integrity`. A pack whose `metadata` is `null` loads as `{}` instead of raising. Explicit `metadata` that is not a mapping (`[]`, `false`, `""`) fails load rather than becoming `{}` and resealing to the empty-object checksum. A pack whose `evidence` is a non-mapping (`[]`, `null`) also loads as `{}` and then fails evidence instead of raising during `load_pack`. Guard flags, receipt `ok`, `quarantined`, and `include_private` must be JSON booleans; `"false"`/`"true"` strings are not coerced and fail schema/admission. Symbol `confidence` and `survival` must be real numbers in `[0, 1]`; strings such as `"0.5"` and booleans are not coerced with `float()` and fail integrity. Load preserves those present values (and `reconstruction_error`) instead of converting them, so a sealed pack rewritten with `"confidence": "0.5"` cannot reload as the original number and pass. Explicit JSON `null` for `confidence`, `survival`, or `reconstruction_error` is also preserved and fails schema; it is not rewritten to `0.0`/`1.0`. `reconstruction_error` must be a real number before residual arithmetic; a string residual fails schema instead of raising `TypeError`. If schema fails, `certify` does not recompute the checksum, so `confidence="not-a-number"` still yields a failed certificate rather than an exception. Private packs (`include_private=true`) with nonempty `examples` must have matching `example_hashes` of equal length; an unrelated 64-hex digest is not a commitment to that text. A scalar `"examples": "secret"` fails load rather than becoming one example per character. Constructed packs whose `example_hashes` is `null` fail schema instead of raising during iteration. A non-string `checksum` such as `123` fails integrity instead of raising `TypeError` in the SHA-256 check. Constructed packs that mutate `definition` to a non-string fail schema instead of raising `AttributeError` while sealing. Constructed `examples=None` fails schema instead of raising while sealing. Constructed non-mapping `evidence` (`[]`, `None`) fails schema before checksum or evidence lookups, so `certify` returns a failed certificate instead of `AttributeError`. Loading a public pack (`include_private` is not exactly `true`) drops any serialized `examples` from live symbols; a constructed public pack that still holds raw examples fails schema. `learn_pack(previous=...)` requires a sealed parent checksum; an empty checksum is not recorded as lineage and is not used as an alias source key. Empty or non-SHA-256 `parent_checksum` fails schema. Private `learn` examples keep the original nonblank `Observation.text`, including leading and trailing whitespace; stripped copies are used only for tokenization and prompting. `include_private: "false"` is therefore not loaded as a private pack, and save/re-serialize uses an exact-`True` check so that string cannot leak `Symbol.examples`. Evidence digest values that are not strings (`null`, `123`, a 64-digit JSON number) fail `evidence_valid` instead of raising `TypeError` or being `str()`-coerced into hex. Malformed or empty `--observations` files fail `audit`/`certify` with a CLI error instead of a traceback. Syntactically valid non-object JSONL records (`[]`, `null`, a string) also fail that way, including on `learn`. Object records with unusable field types (for example `"embedding": 1`, `"embedding": "1"`, or `"text": 1`) are the same clean CLI error, not a traceback. Symbol `definition` must be a string or null; a numeric definition fails `load_pack` with a CLI error. A top-level pack that is `[]`/`null`, or a `symbols` entry that is not an object, fails `load_pack` as an invalid pack rather than an `AttributeError` traceback. An explicit `symbols` value that is not an array (`0`, `""`, `{}`) is the same failure; only a missing key defaults to an empty list. `guards` may be omitted or `null`; `[]`, `{}`, `false`, or `""` do not mean "no guards" and fail load. Explicit `aliases` values that are not mappings (`[]`, `false`, `""`) fail load; only a missing or `null` field means no aliases. Mixed alias shapes (one nested table and one scalar) fail the same way. Stream files whose `codes` are `null` or contain `null`, a non-integral number such as `1.9`, or a boolean fail `translate` as a CLI error. Public echo filtering collapses whitespace before comparing observation windows, so `alpha   beta` cannot sneak through as `alpha beta`. An empty private gloss is recorded as `[unglossed]` at confidence 0, the same gap contract as the public sentinel. `Observation.content_hash()` uses the same text-only n-gram fill as `learn`/`certify`, so a JSONL row without an embedding reproduces the sealed evidence digest. Symbol `observation_ids` must be an array of strings; `[null]` is not coerced to `"None"`, and a scalar string is not iterated into characters. Constructed packs that mutate `observation_ids` to `[None]` fail integrity and evidence; `certify` does not `str()` the id before looking it up in `pack.evidence`. Reloaded audit certificates require JSON booleans for `passed` and the authority flags; `"false"` is not coerced with `bool()`. Failed `translate` writes that certificate JSON to stderr, not stdout, so a redirected translation result cannot be a certificate object.

## Aliases

Aliases are version-scoped: `{source_pack_checksum: {old_code: new_code}}`.

Current codebook codes are never rewritten. A map `{0: 1}` does not steal live code `0`. Historical codes absent from the current codebook follow hops (`rewrite_stream` is multi-hop, bounded by the alias table size so an acyclic chain is not truncated).

Pass `--source-pack <checksum>` to select a parent table explicitly. An explicit source, including empty string, never merges other tables and never selects a table keyed by `""`. Alias source keys must be non-empty. Explicit `aliases` that are not a mapping (`[]`, `false`, `""`) fail load rather than becoming an empty table; only a missing or `null` field means no aliases. A flat `{old: new}` map is treated as the unscoped `legacy` table: with an explicit source it is not applied, including the reserved selector `legacy` itself. With no source, `alias_table` / `translate_stream` use only `legacy` and ignore versioned tables. Standalone `rewrite_stream` is stricter: a versioned-only mapping with no `source_pack_checksum` raises `ValueError`. Mixed `legacy` + versioned still uses only `legacy` when no source is given. `audit` and `certify` reject unknown `--policy` values at the CLI.

## Admission

| `decision` | meaning | default certify |
|---|---|---|
| `accept` | admitted lexicon | pass if guards allow |
| `accept_provisional` | admitted with notes | pass unless `--policy strict` or the latest `finalize` receipt has `ok` other than `true` |
| `reject` | draft | **fail** |

Learning rejection and certification now agree.

## Fail-closed

1. Missing data fails. It does not become a zero.
2. Translation does not outrun certification.
3. Quarantine is a labeled hold, not a pass.
4. Negative residuals, duplicate identities, and unsealed mutations fail integrity.

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
neuralese translate pack.json stream.json          # fails if certify fails
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

A well-formed digest in the map is **self-consistency**, not proof the tensors existed. Pass the original observations (`certify(..., observations=...)` / `--observations`) to recompute hashes. Fabricated ids with attacker-chosen 64-hex digests fail that check. Duplicate `observation_id` values fail during `learn` and during content verification; they are not a fold path. A supplied observation with neither embedding nor text fails `evidence_valid` instead of crashing.

Every `example_hashes` value must be a full 64-character SHA-256 hex digest. A public pack that stores `"raw secret"` (or a digest plus a trailing newline) in that field fails integrity; hashes are not a privacy side channel for plaintext.

Public packs store hashes, not raw example text (`include_private=false` by default). Heuristic glosses then use keywords that do not reproduce an observation, or an explicit `[unglossed]` / quarantine marker. They do not copy the first observation verbatim into `definition`, and a single-token observation such as `TOPSECRET1234` is not kept as a public keyword. Optional LLM glosses for public packs are prompted from those remaining keywords only; a response that echoes raw observation text of any length is discarded. A recorded `[unglossed]` sentinel keeps confidence at 0 and translates as an explicit gap, not as bound English. An explicit empty `decoder_version` or `decision` is preserved on load and fails integrity rather than being silently replaced with this decoder's defaults. Unknown `decision` values fail schema even under `--policy integrity`. A pack whose `metadata` is `null` loads as `{}` instead of raising. Malformed or empty `--observations` files fail `audit`/`certify` with a CLI error instead of a traceback. Syntactically valid non-object JSONL records (`[]`, `null`, a string) also fail that way, including on `learn`.

## Aliases

Aliases are version-scoped: `{source_pack_checksum: {old_code: new_code}}`.

Current codebook codes are never rewritten. A map `{0: 1}` does not steal live code `0`. Historical codes absent from the current codebook follow hops (`rewrite_stream` is multi-hop, bounded by the alias table size so an acyclic chain is not truncated).

Pass `--source-pack <checksum>` to select a parent table explicitly. An explicit source, including empty string, never merges other tables. With no source, only the unscoped `legacy` table is used (`rewrite_stream` and `SymbolPack.alias_table` agree: versioned tables are ignored unless a source is given). `audit` and `certify` reject unknown `--policy` values at the CLI.

## Admission

| `decision` | meaning | default certify |
|---|---|
| `accept` | admitted lexicon | pass if guards allow |
| `accept_provisional` | admitted with notes | pass unless `--policy strict` |
| `reject` | draft | **fail** |

Learning rejection and certification now agree.

## Fail-closed

1. Missing data fails. It does not become a zero.
2. Translation does not outrun certification.
3. Quarantine is a labeled hold, not a pass.
4. Negative residuals, duplicate identities, and unsealed mutations fail integrity.

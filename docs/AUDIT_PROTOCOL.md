# Audit protocol

`neuralese certify` is the public fail-closed gate. A pack that does not pass is a **draft**, not a lexicon. Translation refuses to emit English unless a named policy passes.

## Certificate

v0.1.1 splits three authorities. `passed` is a conjunction under the named policy, never a majority vote.

| field | meaning |
|---|---|
| `integrity_valid` | schema + addressable aliases + full SHA-256 seal + residual + gloss binding |
| `evidence_valid` | every live `observation_id` resolves in the sealed evidence manifest with a SHA-256 digest |
| `admission_valid` | learning `decision` is not `reject`; guards/finalize agree with admission |
| `passed` | policy conjunction of the above |
| `policy` | `default` \| `strict` \| `integrity` |

Kept detail flags: `addressable`, `unfoldable`, `gloss_bound`, `residual_ok`.

CLI:

```bash
neuralese audit pack.json
neuralese certify pack.json --fail-on-undecodable
neuralese translate pack.json stream.json          # fails if certify fails
neuralese translate pack.json stream.json --allow-uncertified   # debug only
```

`learn` writes the pack always. Exit `2` when `decision=reject` (explicit draft).

## Policies

| policy | `passed` when |
|---|---|
| `default` | integrity ∧ evidence ∧ admission (`accept` or `accept_provisional`) |
| `strict` | default plus `decision=accept` and `guards.pass_all` |
| `integrity` | seal/schema only (does not authorize translation) |

`translate_stream(..., policy="default", require_certified=True)` is the library default.

## Seal (full SHA-256)

Checksum is 64 hex characters over the complete semantic manifest:

- `pack_id`, `decoder_version`, `decision`
- `parent_pack_id`, `parent_checksum`
- codebook, versioned aliases, evidence map
- every symbol: class/code, prototype, observation ids, definition, example hashes, confidence, survival, quarantine, metadata
- guards, receipts, MDL, residual, pack metadata

Wall-clock `timestamp` on the pack object is **not** in the checksum. Receipt timestamps **are** (receipts are append-only once sealed).

This is self-consistency, not publisher authenticity. Sign or externally anchor the manifest if you need that.

## Evidence / unfold

A nonempty string in `observation_ids` is not enough. Each live id must appear in `pack.evidence` as `id → SHA-256(observation_id, embedding, text)`.

Fabricated ids fail. Mutating the manifest without resealing fails integrity.

Public packs store hashes, not raw example text (`include_private=false` by default).

## Aliases

Aliases are version-scoped: `{source_pack_checksum: {old_code: new_code}}`.

Current codebook codes are never rewritten. A map `{0: 1}` does not steal live code `0`. Historical codes absent from the current codebook follow hops (`rewrite_stream` is multi-hop).

Pass `--source-pack <checksum>` to select a parent table explicitly.

## Admission

| `decision` | meaning | default certify |
|---|---|---|
| `accept` | admitted lexicon | pass if guards allow |
| `accept_provisional` | admitted with notes | pass unless `--policy strict` |
| `reject` | draft | **fail** |

Learning rejection and certification now agree.

## Fail-closed

1. Missing data fails. It does not become a zero.
2. Translation does not outrun certification.
3. Quarantine is a labeled hold, not a pass.
4. Negative residuals, duplicate identities, and unsealed mutations fail integrity.

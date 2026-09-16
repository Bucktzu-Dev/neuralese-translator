# Changelog

## 0.1.2

Activation ingest. Pack schema and `decoder_version` stay `0.1.1`. Authority-gate semantics are unchanged.

- CLI: `neuralese ingest ACTIVATIONS -o observations.jsonl [--texts TEXTS] [--layer N] [--source LABEL]`
- Inputs: `.npy` 2-D `(n, dim)` or a 1-D vector (one observation); `.npz` with `hidden_states` / `activations` / `embeddings` / `last_hidden_state` or exactly one matrix (optional 1-D `texts`/`prompts` and `observation_ids`/`ids`); activation JSONL; activation `.json` (array of objects/vectors, a single record, `{ "activations": ... }`, or a 2-D `hidden_states` matrix)
- JSON/JSONL dump aliases: `id` (`observation_id`), `prompt` (`text`), `activation`/`vector` (`hidden_state`/`embedding`). Colliding aliases fail closed.
- `metadata.source` is the activations filename or `--source`, not an absolute path
- Fail closed on rank-3+ dumps, non-finite values, colliding named npz matrices, colliding JSON `activations`/`rows`, colliding dump aliases, non-string ids, blank ids, duplicate ids, recursive/non-finite metadata, non-string metadata keys, `--texts` on JSON, ingesting a file onto itself, and truncated npz member reads
- Observation JSONL is serialized fully before the output file is replaced
- JSONL/JSON loaders accept UTF-8 BOM and bare numeric arrays per row
- No in-tree HuggingFace client. Dump recipe: `examples/activations/README.md`. Shipped demo: `examples/activations/states.jsonl`

## 0.1.1

Fail-closed certification: translation is gated on `certify`; the seal is full SHA-256 over the semantic manifest.

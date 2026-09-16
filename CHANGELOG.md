# Changelog

## 0.1.2

Activation ingest. Pack schema and `decoder_version` stay `0.1.1`. Authority-gate semantics are unchanged.

- CLI: `neuralese ingest ACTIVATIONS -o observations.jsonl [--texts TEXTS] [--layer N] [--source LABEL]`
- Inputs: 2-D `.npy`; `.npz` with `hidden_states` / `activations` / `embeddings` or exactly one matrix (optional 1-D `texts` / `observation_ids`); activation JSONL; activation `.json` (array of objects/vectors, a single `hidden_state` record, `{ "activations": ... }`, or a 2-D `hidden_states` matrix)
- `metadata.source` is the activations filename or `--source`, not an absolute path
- Fail closed on rank-3+ dumps, non-finite values, colliding named npz matrices, colliding JSON `activations`/`rows`, non-string ids, blank ids, duplicate ids, recursive/non-finite metadata, non-string metadata keys, `--texts` on JSON, ingesting a file onto itself, and truncated npz member reads
- Observation JSONL is serialized fully before the output file is replaced
- JSONL/JSON loaders accept UTF-8 BOM and bare numeric arrays per row
- No in-tree HuggingFace client. Dump recipe: `examples/activations/README.md`. Shipped demo: `examples/activations/states.jsonl`

## 0.1.1

Fail-closed certification: translation is gated on `certify`; the seal is full SHA-256 over the semantic manifest.

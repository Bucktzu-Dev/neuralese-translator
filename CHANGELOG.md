# Changelog

## 0.1.2

Activation ingest plus structural hardening. Pack schema and `decoder_version` stay `0.1.1`.

- CLI: `neuralese ingest ACTIVATIONS -o observations.jsonl [--texts TEXTS] [--layer N] [--source LABEL]`
- Inputs: `.npy` 2-D `(n, dim)` or a 1-D vector (one observation); `.npz` with `hidden_states` / `activations` / `embeddings` / `last_hidden_state` or exactly one matrix (optional 1-D `texts`/`prompts` and `observation_ids`/`ids`); activation JSONL; activation `.json` (array of objects/vectors, a single record, `{ "activations": ... }`, or a 2-D `hidden_states` matrix)
- JSON/JSONL dump aliases: `id` (`observation_id`), `prompt` (`text`), `activation`/`vector` (`hidden_state`/`embedding`). Colliding aliases fail closed.
- `metadata.source` is the activations filename or `--source`, not an absolute path
- Fail closed on rank-3+ dumps, non-finite values, colliding named npz matrices, colliding JSON `activations`/`rows`, colliding dump aliases, mapping texts/ids, non-string ids, blank ids, duplicate ids, NaN npz ids, recursive/non-finite metadata, non-object metadata, non-string metadata keys, `--texts` on JSON, ingesting onto the activations path or `--texts`, truncated npz member reads (`BadZipFile` / `EOFError`), and record-row `hidden_state` plus `hidden_states`
- CLI ingest receipt reports `layer: null` when record-level layers differ
- Observation JSONL is streamed to a tempfile and replaced only after every row serializes; `--texts` does not skip colliding npz alignment aliases
- `.npz` alignment arrays and the matrix are read from one archive snapshot; NaN npz `texts`/`prompts` stay missing
- `--texts` JSON objects use exclusive `texts`/`prompts`; a validated npy/npz matrix is not finite-scanned again
- Observation JSONL save validates `observation_id`, `text`, and `embedding` against the loader schema before replacing the output
- A named npz matrix plus another unrecognized array fails closed
- Historical alias sources that collide with a current codebook code fail closed (`{0: 1}` is ambiguous without `(pack_checksum, code)` identity)
- Python `translate_stream` / `resolve_code` reject the same non-integer stream codes as the CLI (`True`, `0.9`, `"1"`); NumPy integer scalars remain valid codes; oversized reals (`Fraction`) fail as `ValueError`, not `OverflowError`
- `learn`/`ingest` `-o` must differ from the evidence path, including hard links (`Path.samefile`)
- Default/strict certification requires at least one admitted (non-quarantined) symbol; the activation demo uses `--n-symbols 2`
- JSONL/JSON loaders accept UTF-8 BOM and bare numeric arrays per row
- No in-tree HuggingFace client. Dump recipe: `examples/activations/README.md`. Shipped demo: `examples/activations/states.jsonl`

## 0.1.1

Fail-closed certification: translation is gated on `certify`; the seal is full SHA-256 over the semantic manifest.

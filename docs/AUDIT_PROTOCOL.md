# Audit protocol

This protocol is what a third-party auditor should be able to run against a sealed translation pack without access to Eris internals.

## Inputs

- A translation pack JSON file.
- Optional: the original observation JSONL if evidence hashes need to be re-derived.

## Steps

1. Confirm `schema_version` is `neuralese.pack.v1`.
2. Confirm `decoder_version` equals the decoder that produced the pack (`0.1.1` for this release). A sealed checksum does not authorize a newer decoder.
3. Recompute the canonical SHA-256 checksum and compare it to `checksum`. The digest must be the full 64-character lowercase hex string; truncated hashes fail closed.
4. If `parent_checksum` is present, confirm it is a full SHA-256 hex digest.
5. Confirm every atom has a non-empty `atom_id`, `form`, and `definition`.
6. Confirm every `evidence` value is a 64-character SHA-256 hex digest. If original observations are available, re-hash them and confirm each atom evidence id still resolves.
7. Confirm observation ids are unique and observation text is non-empty. Integer ids such as `1` and string ids such as `"1"` are the same id after load.
8. Confirm `decision` is one of `certified`, `needs_review`, or `rejected`. Unknown values fail closed even during an integrity-only check.
9. Confirm alias rewrites are bounded by the alias table size, so a long unique chain still resolves and a cycle still fails closed.
10. Run `neuralese audit --pack pack.json`.
11. Run `neuralese certify --pack pack.json` and confirm the written `decision`.
12. Attempt `neuralese translate` against an uncertified pack and confirm it is refused unless `--allow-uncertified` is explicit.
13. Confirm `neuralese audit --policy integrity` and `neuralese certify --policy integrity` do not authorize translation. Integrity only checks the seal; `default` and `strict` are the policies that can mark a pack certified.
14. Confirm public packs omit raw `examples` and that public definitions/keywords do not reproduce observation text, including 8-character windows and public keywords of length 8 or more that are substrings of an observation.

## Fail closed

Any checksum mismatch, unknown decoder, unknown decision, duplicate observation id, empty observation, malformed evidence hash, or uncertified translate request must fail. Do not repair the pack in place during an audit.

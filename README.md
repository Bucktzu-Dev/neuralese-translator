# Neuralese to English Translator

Inner representations are not a mythical language. They are a **compact symbol stream**. This tool reconstructs that alphabet, glosses it into English, and **refuses to call a symbol real unless it stays decodable**.

If a code cannot be unfolded to the observations that produced it, it is residue — not a symbol.

Public review repo: [Bucktzu-Dev/neuralese-translator](https://github.com/Bucktzu-Dev/neuralese-translator). See [LIMITATIONS.md](LIMITATIONS.md) for what v0.1 does not claim.

## Why this exists

People talk about model internals as if they were an unknowable boogieman. That story only survives if nobody can *read* the codes.

This package gives auditors, researchers, and other model operators a public function:

1. Learn a compact alphabet from embeddings and/or text (`neuralese learn`).
2. Translate a code stream into English glosses (`neuralese translate`).
3. Certify that every live symbol is addressable, unfoldable, and bound to a receipt (`neuralese certify`).

English here is a **receipt**, not a vibe. An LLM may help write a gloss; it cannot replace the certificate.

Hidden-state dumps become observations with `neuralese adapt`. The command reads arrays already on disk. It does not download or run a model.

## 5-minute start

```bash
pip install -e .
neuralese learn examples/toy_stream/observations.jsonl -o pack.json
neuralese translate pack.json examples/toy_stream/stream.json
neuralese unfold pack.json --code 0
neuralese audit pack.json
neuralese certify pack.json --fail-on-undecodable
```

JSONL observations:

```json
{"observation_id": "obs-1", "text": "hello there", "embedding": [1, 0, 0, 0]}
```

If `embedding` is omitted, the tool builds a deterministic hashed n-gram vector from `text`.

A stream file is `{"codes": [0, 1, 0]}` or a JSON list of integers.

## What a gloss looks like

Each token becomes:

| field | meaning |
|---|---|
| `code` | input code |
| `class_id` | symbol class, if resolved |
| `english` | bound gloss, or an explicit gap |
| `confidence` | 0–1, capped by reconstruction residual |
| `state` | `ok` \| `aliased` \| `quarantined` \| `unknown` |
| `observation_ids` | unfold path to source observations |
| `pack_checksum` | sealed pack identity |

`unknown` is a stated gap (`[undecodable: …]`). The translator does not invent a sentence for a missing symbol.

## The audit contract

A symbol is admitted only if it stays:

1. **Addressable** — every code resolves through the codebook or an alias map.
2. **Unfoldable** — every admitted code lists observation ids, and those rows (text and embeddings) are sealed in the pack so an auditor can recompute the prototype.
3. **Gloss-bound** — English is sealed into the pack checksum. Mutating prose without resealing fails certify.
4. **Fail-closed** — unknown, quarantined, and aliased are explicit states. A child pack keeps every parent code: a match inherits it, and an unmatched code stays quarantined with its reservoir rows. The integer is not reused for a new cluster.
5. **Residual-honest** — reconstruction error is reported; gloss confidence may not exceed what that residual allows.

Read [docs/DECODABILITY.md](docs/DECODABILITY.md) and [docs/AUDIT_PROTOCOL.md](docs/AUDIT_PROTOCOL.md).

## Activation dumps

```bash
neuralese adapt dump.npz -o observations.jsonl --layout hf_stack --layer -1 --pool last
neuralese learn observations.jsonl -o pack.json
```

| layout | shape | meaning |
|---|---|---|
| `vectors` | `(n, d)` | one vector per observation |
| `tokens` | `(n, seq, d)` | pool the sequence (`last` or `mean`) |
| `layers` | `(n, layers, d)` | pick `--layer` |
| `hf_layers` | `(layers, n, d)` | stacked pooled layers |
| `hf_stack` | `(layers, n, seq, d)` | `numpy.stack` of a HuggingFace `hidden_states` tuple |

A checked-in public dump lives in [examples/public_domain](examples/public_domain). Parent-pack compression examples live in [examples/mdl_delta](examples/mdl_delta).

## Library

```python
from neuralese import load_observations_jsonl, learn_pack, translate_stream, unfold_code, certify

obs = load_observations_jsonl("examples/toy_stream/observations.jsonl")
pack = learn_pack(obs)
glosses = translate_stream(pack, [0, 1, 99])
report = unfold_code(pack, 0)
cert = certify(pack)
assert report.missing_ids == []
assert cert.passed
```

## What this is not

- Not a full mechanistic-interpretability suite (no logit lens, no SAE training zoo).
- Not a claim that every hidden state is a word.
- Not permission to delete symbols. Replacement is a new forward commit; erasure is not an inverse.

## Origin

Alphabet reconstruction and subject-aware symbol dynamics were developed inside Eris SLAR. This leaf is a generalized, MIT-licensed extract with public names (`SymbolPack`, `Gloss`, `AuditCertificate`). Mapping to internal SLAR names is in [docs/SLAR_MAPPING.md](docs/SLAR_MAPPING.md) and is not required to use the tool.

License: MIT (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).

# Prose corpus

320 original English sentences: 40 each for harbor, orchard, ledger, bakery, joinery, weather, apiary, and pottery. Nothing here is copied from a book or a lyric. `observations.jsonl` has text and a topic label. It has no embeddings, so `learn` builds them with the default hashed character trigram embedder (`hashed_ngram_vector`, `dim=32`, `n=3`).

Each row is one varied sentence plus one repeated workplace sentence for that topic. The repeated sentences are listed in `anchors.json`. They are inside the row text, so they are inside the pack checksum. The classes on this run follow those repeated sentences. The varied sentences alone do not certify; see the measurements below and `docs/ROADMAP.md`.

This is the text path. `examples/toy_stream`, `examples/public_domain`, and `examples/mdl_delta` are unit fixtures.

## Published thresholds

Library defaults. They are passed through by omitting the flags. They were not loosened inside `certify`.

| knob | value |
|---|---|
| `tau_residual` | 0.55 |
| `tau_kappa` | 0.35 |
| `n_symbols` | 8 |
| `min_cluster_size` | 2 |
| embedding dim | 32 |
| n-gram order | 3 |
| seed | 0 |

## What seed 0 measured

Anchored rows (this file), `LearnConfig(n_symbols=8, seed=0)`:

| quantity | value |
|---|---|
| residual | 0.4223 |
| mean kappa | 0.9063 |
| live / quarantined | 8 / 0 |
| decision | `accept` |
| `certify` at the defaults | pass |

Varied sentences only (anchor suffix removed), same config:

| quantity | value |
|---|---|
| residual | 0.7710 |
| mean kappa | 0.6357 |
| `certify` at `tau_residual` 0.55 | fail (`residual_ok` is false) |
| decision | `reject` |
| mean majority-topic purity | about 0.31 |

Do not raise the library residual to make the second table pass.

Glosses are the heuristic keyword receipt (`Symbol for ...`). They are sealed. They are not a claim that the class is a dictionary definition.

## Commands

The pack is learned where you run this. It is not committed: `pack_id` is a random uuid.

```bash
python -m neuralese learn examples/prose_corpus/observations.jsonl -o /tmp/prose-pack.json --n-symbols 8 --seed 0
python -m neuralese report /tmp/prose-pack.json
python -m neuralese certify /tmp/prose-pack.json --fail-on-undecodable
```

`report` prints JSON: `n_live`, `n_quarantined`, `residual`, `reservoir_size`, `unfold_failures`, `gloss_coverage`, `mdl_bits`, `decision`. It does not apply `tau_residual`. `certify` does.

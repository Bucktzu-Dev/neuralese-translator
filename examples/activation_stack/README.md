# Activation stack

Checked-in hidden-state dump derived from [examples/prose_corpus](../prose_corpus). It is not a toy fixture and it is not a model activation. `export_hidden_states.py` hashes the 320 prose rows offline. There is no weight download, no HuggingFace hub call, and no hand-placed orthogonal centers.

`hidden_states.npz` is layout `hf_layers`, shape `(3, 320, 32)`:

| index | name | how it is built |
|---|---|---|
| 0 | `char_trigram` | `hashed_ngram_vector`, dim 32, character n-grams of order 3, on the full row text |
| 1 | `word_unigram` | signed SHA-256 bin per whitespace word, dim 32, L2-normalized |
| 2 | `sum_l2` | per-row L2-normalized sum of layers 0 and 1 |

The npz also stores `texts`, `observation_ids`, `layout`, and `layer_names`. The zip timestamp is 1980-01-01. Regenerating with the script rewrites the same arrays.

`hf_layers` has no sequence axis. `--pool` is part of the adapt command and does not change these vectors. The published slice is layer 0.

## Published thresholds

Library defaults. They are passed through by omitting the flags on `learn` and `certify`. They were not loosened inside `certify`.

| knob | value |
|---|---|
| layout | `hf_layers` |
| layer | `0` |
| pool | `last` |
| `tau_residual` | 0.55 |
| `tau_kappa` | 0.35 |
| `n_symbols` | 8 |
| `min_cluster_size` | 2 |
| embedding dim | 32 |
| seed | 0 |

## What seed 0 measured

Layer 0 after `adapt`, then `LearnConfig(n_symbols=8, seed=0)`:

| quantity | value |
|---|---|
| residual | 0.4223 |
| mean kappa | 0.9063 |
| live / quarantined | 8 / 0 |
| reservoir | 320 |
| decision | `accept` |
| `certify` at the defaults | pass |
| majority-topic purity | 1.0 |

Layer 0 is the same character-trigram hash the prose `learn` path builds, written out so `adapt` can load it. The repeated workplace sentence in the prose rows is still inside that hash. This file does not certify the varied sentences with that sentence removed. Those sentences live in [examples/varied_prose](../varied_prose) and are embedded there with `word_sentence_svd`, a deterministic local TF-IDF SVD. This dump is not that encoder.

The other layers also meet the default residual and kappa, and they are not the published command:

| layer | residual | mean kappa | live / quarantined | majority-topic purity | `certify` at the defaults |
|---|---|---|---|---|---|
| 1 `word_unigram` | 0.4372 | 0.8952 | 8 / 0 | about 0.93 | pass |
| 2 `sum_l2` | 0.3793 | 0.9380 | 7 / 1 | about 0.92 | pass |

Layer 2 quarantines a singleton (`min_cluster_size` is 2). Do not treat that layer as eight topic classes.

Glosses are the heuristic keyword receipt. An unknown code stays `unknown`.

## Commands

The pack is learned where you run this. It is not committed: `pack_id` is a random uuid.

```bash
python3 examples/activation_stack/export_hidden_states.py
python -m neuralese adapt examples/activation_stack/hidden_states.npz -o /tmp/act-obs.jsonl --layout hf_layers --layer 0 --pool last
python -m neuralese learn /tmp/act-obs.jsonl -o /tmp/act-pack.json --n-symbols 8 --seed 0
python -m neuralese report /tmp/act-pack.json
python -m neuralese certify /tmp/act-pack.json --fail-on-undecodable
```

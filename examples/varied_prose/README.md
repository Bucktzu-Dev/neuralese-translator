# Varied prose

The 320 sentences from [examples/prose_corpus](../prose_corpus) with the repeated workplace sentence removed. Nothing here is copied from a book or a lyric. `observations.jsonl` has text and a topic label. It has no embeddings. No shared suffix was added.

This file is the character-trigram failure from the prose README, checked in on its own. The default encoder (`char_trigram`, `hashed_ngram_vector`, `dim=32`, `n=3`) still rejects it. The published command below uses `--encoder word_sentence_svd`, a deterministic local encoder. It is a TF-IDF word-sentence SVD of the texts in this file, after a fixed English stop list. It is not a model hidden state, a logit lens, or an SAE. Hashed n-grams remain the default for other text-only JSONL.

Topic ids are not an input to the vector. Majority-topic purity below is a diagnostic computed after clustering. It is not a certify gate. `neuralese report --topics` can print the same diagnostic from a label file. `certify` does not read that file.

## Encoder

`word_sentence_svd`, width `--n-symbols` (8 on the published command):

1. Lowercase tokens matching `[a-z0-9']+`. Stop words and one-character tokens are dropped. The stop list is `neuralese.encode.FUNCTION_WORDS`: the Glasgow Information Retrieval Group English list (the list scikit-learn ships), used whole, plus `did` and `does`. It is not a list of these eight topic names, and it was not fit by document frequency on this file. The list is broader than closed-class function words. Content-like entries such as `fire` stay because the list is not trimmed.
2. Term frequency is the raw count. IDF is `ln(N / df) + 1`.
3. L2-normalize each row. Truncated SVD of that width, with BLAS, OpenMP, and MKL pinned to one thread. Scale by the singular values. Fix each left singular vector so its largest-magnitude coordinate is positive (a tie keeps the lowest index). L2-normalize each row again. A column sign flip leaves squared Euclidean k-means unchanged. The thread pin is what keeps one BLAS build from reshuffling the subspace between runs.

The basis is a function of this file. A different file gets a different basis.

v0.6 used a shorter closed-class list on this same SVD. That run's bakery-majority cluster was 22 of 55, purity 0.4000, and its size-weighted purity was 0.6594. The numbers below are the Glasgow-list measurement on reference BLAS/LAPACK 3.12 with one thread. Commit 5527cdc, on another BLAS, measured a different subspace of this same encoder: residual 0.4394, mean kappa 0.9044, 8 live, unweighted purity 0.7572, size-weighted 0.7250 (232 of 320), worst cluster pottery-majority 31 of 55, purity 0.5636. Singular values 8 and 9 of this file differ by about 0.022, so that gap is enough for two LAPACK builds to hand k-means different coordinates.

## Published thresholds

Library defaults. They are passed through by omitting the tau flags. They were not loosened inside `certify`.

| knob | value |
|---|---|
| encoder | `word_sentence_svd` |
| `tau_residual` | 0.55 |
| `tau_kappa` | 0.35 |
| `n_symbols` | 8 |
| `min_cluster_size` | 2 |
| SVD width | 8 |
| seed | 0 |

## What seed 0 measured

`LearnConfig(n_symbols=8, seed=0, encoder="word_sentence_svd")`:

| quantity | value |
|---|---|
| residual | 0.5239 |
| mean kappa | 0.8814 |
| live / quarantined | 7 / 1 |
| decision | `accept` |
| `certify` at the defaults | pass |
| majority-topic purity (unweighted mean of live clusters) | 0.6837 |
| size-weighted majority purity | 0.6395 (204 of 319 live rows) |
| lowest live cluster majority | 0.4531 |

One cluster has a single row (`orchard-031`) and is quarantined because `min_cluster_size` is 2. Its kappa is 1. It is left out of the purity table. Ledger is not the majority of a live cluster.

Per live cluster, majority topic after the fact, in code order:

| rows | majority | count | purity | label |
|---|---|---|---|---|
| 50 | bakery | 32 | 0.6400 | bakery |
| 59 | pottery | 31 | 0.5254 | pottery |
| 42 | joinery | 31 | 0.7381 | joinery |
| 45 | orchard | 28 | 0.6222 | orchard |
| 32 | harbor | 27 | 0.8438 | harbor |
| 64 | weather | 29 | 0.4531 | mixed |
| 27 | apiary | 26 | 0.9630 | apiary |

The weather-majority cluster is 29 of 64, purity 0.4531, so `cluster_purity` labels it `mixed`. The other live clusters are at or above 0.5. Mean purity 0.6837 is partly topical. It is not seven pure workplaces. Codes are whatever k-means assigned. They are not topic names. The 5527cdc table (8 live, pottery 31 of 55 at 0.5636, bakery 32 of 46 at 0.6957, apiary 26 of 26) is the other BLAS measurement, recorded above.

The same file with the default character-trigram encoder, same `n_symbols` and seed:

| quantity | value |
|---|---|
| residual | 0.7710 |
| mean kappa | 0.6357 |
| `certify` at `tau_residual` 0.55 | fail (`residual_ok` is false) |
| decision | `reject` |
| majority-topic purity | 0.3085 |

That rejection stays locked. Do not raise the library residual to make the trigram table pass.

Glosses are the heuristic keyword receipt (`Symbol for ...`). They are sealed. They are not a claim that the class is a dictionary definition. Code `99999` stays `unknown`.

## Commands

The pack is learned where you run this. It is not committed: `pack_id` is a random uuid.

```bash
python -m neuralese learn examples/varied_prose/observations.jsonl -o /tmp/varied-pack.json --encoder word_sentence_svd --n-symbols 8 --seed 0
python -m neuralese report /tmp/varied-pack.json
python -m neuralese report /tmp/varied-pack.json --topics examples/varied_prose/observations.jsonl
python -m neuralese certify /tmp/varied-pack.json --fail-on-undecodable
```

Omit `--encoder` and the same file is the character-trigram reject above. `report` prints JSON and does not apply `tau_residual`. Without `--topics` the JSON has no `cluster_purity` field. `certify` applies the residual threshold and does not read the topic file.

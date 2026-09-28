# Roadmap

Tracked plan for this repository. Statuses are `done`, `now`, `next`, and `later`. There are no calendar estimates.

A milestone is `done` only when the tree matches it. `now` is empty after the varied-prose slice below. `next` is what a follow-up should pick up. `later` stays out of scope until that follow-up opens it on purpose.

The audit contract does not change with the status labels: `unknown` and `quarantined` stay required states, gloss text stays inside the pack checksum, and tests stay deterministic.

## Done

### Reservoir, unfold, and certify

Status: **done** (v0.4).

Observation rows are sealed in the pack. `certify` recomputes prototypes from those rows. `neuralese unfold` lists them. Packs sealed before the reservoir fail closed until relearned.

### Activation dumps

Status: **done** (v0.4).

`neuralese adapt` reads `.npy`, `.npz`, and `.jsonl` hidden states, including `hf_stack`. It does not download weights or run a model.

### Parent-pack description length and code retention

Status: **done** (v0.4).

The first pack records `delta_mdl_bits = 0`. A child records `new_mdl - parent_mdl`. A positive delta rejects unless `--mdl-exception` names the reason, and `pass_mdl` stays false when the description grows. The child keeps every parent code. A matched prototype inherits that code. An unmatched code is quarantined and its reservoir rows are copied. A new cluster does not reuse those integers. Parent aliases still resolve.

### Unglossed live symbols

Status: **done** (v0.4).

A live symbol with no gloss is sealed as `[unglossed: class N]` at learn time. The translator does not invent a sentence for a missing code.

### Toy fixtures

Status: **done**. They stay unit fixtures. They are not the text path.

| path | what it is |
|---|---|
| `examples/toy_stream` | hand-placed near-orthogonal embeddings |
| `examples/public_domain` | 12 original sentences with hand-placed 8-d near-orthogonal vectors |
| `examples/mdl_delta` | two-cluster ΔMDL fixture |

### Prose alphabet and decodability report

Status: **done** (v0.5). This was the first open milestone.

`examples/prose_corpus` is 320 original sentences, 40 for each of eight topics. Rows do not carry hand-placed embeddings. `learn` hashes the text with the default character trigram embedder (`dim=32`, `n=3`) at `--n-symbols 8 --seed 0`.

Published thresholds, which are the library defaults and were not loosened:

| knob | published value | measured on this run |
|---|---|---|
| `tau_residual` | 0.55 | residual 0.4223 |
| `tau_kappa` | 0.35 | mean kappa 0.9063 |
| `n_symbols` | 8 | 8 live, 0 quarantined |
| embedding dim | 32 | hashed trigrams |
| seed | 0 | fixed |

`certify` passes at those defaults. `neuralese report` prints `n_live`, `n_quarantined`, `residual`, `reservoir_size`, `unfold_failures`, `gloss_coverage`, `mdl_bits`, and `decision`. CI learns the pack (it is not committed) and runs `report` and `certify --fail-on-undecodable`.

Each topic repeats one workplace sentence (`examples/prose_corpus/anchors.json`). That sentence is part of the row text, so it is inside the checksum. The varied sentences with the anchor removed do **not** certify: residual 0.7710, mean kappa 0.6357, mean cluster purity about 0.31, `decision=reject` at `tau_residual` 0.55. A test locks both runs.

Must not be claimed:

- These codes are model hidden states, a logit lens, or an SAE.
- The varied sentences alone meet the default residual.
- A heuristic keyword gloss is a human definition of the workplace.
- `report` is a second, looser gate. `certify` still decides.
- The toy fixtures above are this result.

### Checked-in activation matrix

Status: **done** (v0.5).

`examples/activation_stack/hidden_states.npz` is a deterministic `(3, 320, 32)` `hf_layers` dump. `examples/activation_stack/export_hidden_states.py` builds it from the prose corpus text. There is no weight download and no hand-placed orthogonal center. The zip timestamp is fixed at 1980-01-01.

| layer | contents |
|---|---|
| 0 `char_trigram` | hashed character trigrams, dim 32, order 3, full row text |
| 1 `word_unigram` | hashed word unigrams, dim 32 |
| 2 `sum_l2` | per-row L2-normalized sum of layers 0 and 1 |

The published command is `adapt --layout hf_layers --layer 0 --pool last`, then `learn --n-symbols 8 --seed 0`. `hf_layers` has no sequence axis, so the pool does not change the vectors. Library defaults were not loosened: `tau_residual` 0.55, `tau_kappa` 0.35.

Layer 0 measured residual 0.4223, mean kappa 0.9063, 8 live, 0 quarantined, `decision=accept`. `certify` passes. Those figures match the anchored prose hash because layer 0 is that hash stored in the dump. CI adapts, learns, reports, and certifies. The pack is not committed. An unknown code stays `unknown`.

Layer 1 also meets the defaults (residual 0.4372, mean kappa 0.8952, 8 live) with majority-topic purity about 0.93. Layer 2 meets them for its live symbols (residual 0.3793, mean kappa 0.9380, 7 live, 1 quarantined) with purity about 0.92. Neither replaces the published layer 0 command.

Must not be claimed:

- These arrays are model hidden states, a logit lens, or an SAE.
- This dump certifies the varied sentences with the workplace anchor removed. That file is [examples/varied_prose](../examples/varied_prose), and the hash stack is not its encoder.
- `report` is a second, looser gate. `certify` still decides.

### Varied prose without the repeated workplace sentence

Status: **done** (v0.6).

`examples/varied_prose` is `examples/prose_corpus` with the repeated workplace sentence removed from each row. No shared suffix was written back in. The checked-in rows are text and a topic label. They have no embeddings.

`learn --encoder word_sentence_svd --n-symbols 8 --seed 0` embeds that file with a deterministic local encoder: TF-IDF of the words in the file (`ln(N / df) + 1` on raw counts), row L2 normalization, then a truncated SVD whose width is `n_symbols`. Coordinates are scaled by the singular values and L2-normalized again. Topic ids are not an input. Hashed character trigrams remain the default for text-only JSONL. Library `tau_residual` 0.55 and `tau_kappa` 0.35 were not changed.

| knob | published value | measured on this run |
|---|---|---|
| `tau_residual` | 0.55 | residual 0.5057 |
| `tau_kappa` | 0.35 | mean kappa 0.8713 |
| `n_symbols` | 8 | 8 live, 0 quarantined |
| SVD width | 8 | same as `n_symbols` |
| seed | 0 | fixed |
| decision | | `accept` |
| `certify` | library defaults | pass |

Majority-topic purity is a diagnostic, not a certify gate. On this run the unweighted mean is 0.7084 and the size-weighted mean is 0.6594 (211 of 320 rows sit in a cluster whose majority topic is their own). Every topic is the majority of one live cluster. The bakery-majority cluster is 22 of 55 rows, purity 0.4000. The geometry is decodable under the published thresholds and only partly topical. It is not eight pure topic classes.

The character-trigram encoder on this same file still rejects at the library defaults: residual 0.7710, mean kappa 0.6357, majority-topic purity 0.3085, `decision=reject`, `residual_ok` false. That locked test stays.

Must not be claimed:

- These coordinates are model hidden states, a logit lens, or an SAE.
- Purity was a training target. Topic ids are not in the vector.
- The bakery-majority symbol is the bakery, or any symbol is a pure workplace.
- Hashed character trigrams now certify the varied file.
- `report` is a second, looser gate. `certify` still decides.

## Now

Nothing is in progress in the tree beyond the done varied-prose slice.

## Next

### Logit lens

Status: **next**. Pick this up first.

Acceptance: a certified pack's codes are compared to a checked-in next-token distribution that was exported offline. No HuggingFace hub runtime. `certify` remains fail-closed. Until that exists, do not claim a code predicts a token. Opening it still requires an update to `LIMITATIONS.md`. This milestone is not started.

## Later

These stay out of scope. Varied prose now certifies under the published defaults with the local word-sentence SVD encoder, which was the precondition for even considering them. Opening one still requires an update to `LIMITATIONS.md`. None of them are started. Logit lens is the next slice, above.

### SAE zoo

Status: **later**.

Acceptance: a deterministic check on a checked-in feature matrix, with no SAE training loop and no weight download in CI. Until that exists, do not claim sparse features.

### Live hub inference

Status: **later**.

Acceptance: would mean this package downloads or runs model weights. That is a product decision this roadmap does not grant. `neuralese adapt` stays a reader of files already on disk.

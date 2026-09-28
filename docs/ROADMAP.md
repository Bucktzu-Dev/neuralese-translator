# Roadmap

Tracked plan for this repository. Statuses are `done`, `now`, `next`, and `later`. There are no calendar estimates.

A milestone is `done` only when the tree matches it. `now` is empty after the prose slice below. `next` is what a follow-up should pick up. `later` stays out of scope until that follow-up opens it on purpose.

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

## Now

Nothing is in progress in the tree beyond the done prose slice.

## Next

### Checked-in activation matrix that is not a toy

Status: **next**. Pick this up first.

Leave `examples/public_domain` as the hand-placed unit fixture. Add a checked-in matrix that was produced offline, without a weight download in CI, and without hand-placed orthogonal centers. Load it with `neuralese adapt`. Learn with a fixed seed. Run `neuralese report` and `neuralese certify`.

Acceptance:

- CI adapts and learns the matrix. It does not commit a pack whose `pack_id` is a random uuid.
- Every live symbol unfolds to reservoir rows, prototypes match, and an unknown code stays `unknown`.
- Thresholds actually used (`tau_residual`, `tau_kappa`, `n_symbols`, embedding width) are written in that example's README and, if they differ from the library defaults, in `LIMITATIONS.md`.
- If an honest threshold cannot pass, the measured residual and kappa are locked in a test within a tight tolerance and this milestone stays open. Do not loosen `certify` inside the library to force a pass.

Must not be claimed: live hub inference, a logit lens, or an SAE. Do not import Eris internals.

### Varied prose without the repeated workplace sentence

Status: **next**, after the activation matrix if a follow-up has to choose one.

The blocker is measured, not theoretical. Stripping `anchors.json` from `examples/prose_corpus` and learning at `--n-symbols 8 --seed 0` yields residual 0.7710 and mean kappa 0.6357. `certify` fails `residual_ok` at the default `tau_residual` 0.55. Mean majority-topic purity is about 0.31, so the classes are not the eight topics.

Acceptance:

- A deterministic embedder already in the repo, or a checked-in text-only change, learns the varied sentences with a fixed seed.
- `certify` passes at a published threshold that is still the library default, or the example README records a tighter explicit call-site threshold that the run actually meets.
- Cluster membership is reported. A pass that keeps purity near 0.31 is not this milestone.
- The anchored prose test still passes. Unknown codes stay unknown.

Must not be claimed: raising `tau_residual` in the library, or calling the current unanchored reject a certification.

## Later

These stay out of scope. The prose pack now certifies under the published defaults, which was the precondition for even considering them. Opening one still requires an update to `LIMITATIONS.md`. None of them are started.

### Logit lens

Status: **later**.

Acceptance: a certified pack's codes are compared to a checked-in next-token distribution that was exported offline. No HuggingFace hub runtime. `certify` remains fail-closed. Until that exists, do not claim a code predicts a token.

### SAE zoo

Status: **later**.

Acceptance: a deterministic check on a checked-in feature matrix, with no SAE training loop and no weight download in CI. Until that exists, do not claim sparse features.

### Live hub inference

Status: **later**.

Acceptance: would mean this package downloads or runs model weights. That is a product decision this roadmap does not grant. `neuralese adapt` stays a reader of files already on disk.

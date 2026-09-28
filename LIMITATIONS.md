# Limitations (v0.5)

This is a working public extract, not a complete mechanistic-interpretability suite.

- **Heuristic glosses.** English is keyword/example based unless you inject an LLM client. A fluent sentence is not extra evidence.
- **No logit lens / SAE zoo.** The alphabet is clustering + SVD on observation embeddings (or hashed n-grams from text). Those stay later on [docs/ROADMAP.md](docs/ROADMAP.md).
- **No hub runtime.** `neuralese adapt` reads hidden-state dumps (`.npy`, `.npz`, `.jsonl`), including the stacked shape of a HuggingFace `hidden_states` tuple. It does not download weights or run a model.
- **Prose is hashed text, not a hidden state.** [examples/prose_corpus](examples/prose_corpus) is 320 original sentences. `learn` embeds them with the default 32-d character trigram hash (`--n-symbols 8 --seed 0`). Default `tau_residual` 0.55 and `tau_kappa` 0.35 hold for that file (residual about 0.422, mean kappa about 0.906). Each topic repeats one workplace sentence (`anchors.json`). The varied sentences alone miss the default residual (about 0.771, mean kappa about 0.636) and are not certified. `neuralese report` prints the measurement. It does not relax `certify`.
- **Toy fixtures stay toys.** `examples/toy_stream`, `examples/public_domain`, and `examples/mdl_delta` are unit fixtures. The public-domain file uses hand-placed 8-d near-orthogonal vectors. They are not the prose result.
- **Reservoir is part of the seal.** Packs store the observations used at mint time and `certify` recomputes prototypes from them. Packs sealed before the reservoir fail closed until relearned.
- **First-pack ΔMDL is zero.** There is no parent to compare. A child pack records `new_mdl - parent_mdl`. An increase rejects unless `--mdl-exception` names the reason. `pass_mdl` stays false when the description grows.
- **Codes are local to a pack lineage.** Without `--parent`, code `0` is whatever k-means assigned. With a parent, a matched prototype keeps its code, and an unmatched code stays quarantined instead of being reused. The integer is still not a universal label across unrelated packs.
- **Not Eris.** Subject-aware dynamics are included as an optional module. The mothership SLAR pipeline, event bus, and identity system are not in this repo.

If a claim is not certified by `neuralese certify`, it is not part of the audit contract.

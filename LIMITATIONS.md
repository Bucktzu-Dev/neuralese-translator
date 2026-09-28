# Limitations (v0.4)

This is a working public extract, not a complete mechanistic-interpretability suite.

- **Heuristic glosses.** English is keyword/example based unless you inject an LLM client. A fluent sentence is not extra evidence.
- **No logit lens / SAE zoo.** The alphabet is clustering + SVD on observation embeddings (or hashed n-grams from text).
- **No hub runtime.** `neuralese adapt` reads hidden-state dumps (`.npy`, `.npz`, `.jsonl`), including the stacked shape of a HuggingFace `hidden_states` tuple. It does not download weights or run a model.
- **Reservoir is part of the seal.** Packs store the observations used at mint time and `certify` recomputes prototypes from them. Packs sealed before the reservoir fail closed until relearned.
- **First-pack ΔMDL is zero.** There is no parent to compare. A child pack records `new_mdl - parent_mdl`. An increase rejects unless `--mdl-exception` names the reason. `pass_mdl` stays false when the description grows.
- **Codes are local to a pack lineage.** Without `--parent`, code `0` is whatever k-means assigned. With a parent, a matched prototype keeps its code, and an unmatched code stays quarantined instead of being reused. The integer is still not a universal label across unrelated packs.
- **Not Eris.** Subject-aware dynamics are included as an optional module. The mothership SLAR pipeline, event bus, and identity system are not in this repo.

If a claim is not certified by `neuralese certify`, it is not part of the audit contract.

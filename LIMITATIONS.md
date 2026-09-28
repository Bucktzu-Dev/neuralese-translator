# Limitations (v0.2)

This is a working public extract, not a complete mechanistic-interpretability suite.

- **Heuristic glosses.** English is keyword/example based unless you inject an LLM client. A fluent sentence is not extra evidence.
- **No logit lens / SAE zoo.** The alphabet is clustering + SVD on observation embeddings (or hashed n-grams from text).
- **No model-hub adapter yet.** You bring JSONL observations or a sealed pack. Hidden-state dumps from HuggingFace are not a built-in loader.
- **Reservoir is part of the seal.** v0.2 stores the observations used at mint time and `certify` recomputes prototypes from them. Packs sealed by v0.1 have no reservoir and fail closed until relearned.
- **First-pack ΔMDL is zero.** Compression accounting vs a parent pack is implemented; the first pack records `delta_mdl_bits = 0`.
- **Cluster ids are not stable labels.** Code `0` is whatever k-means assigned, not a universal “hello” token across runs unless you keep the sealed pack.
- **Not Eris.** Subject-aware dynamics are included as an optional module. The mothership SLAR pipeline, event bus, and identity system are not in this repo.

If a claim is not certified by `neuralese certify`, it is not part of the audit contract.

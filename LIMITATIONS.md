# Limitations (v0.1.1)

This is a working public extract, not a complete mechanistic-interpretability suite.

Authority-gap repair in 0.1.1: translation is gated on certification; the seal is full SHA-256 over the semantic manifest; evidence ids must resolve; reject learning is a draft.

Still not claimed:

- **Heuristic glosses.** English is keyword/example based unless you inject an LLM client. A fluent sentence is not extra evidence. One `definition` per class, not plural SSD hypotheses.
- **No logit lens / SAE zoo.** The alphabet is clustering + SVD on observation embeddings (or hashed n-grams from text).
- **No model-hub adapter yet.** You bring JSONL observations or a sealed pack. Hidden-state dumps from HuggingFace are not a built-in loader.
- **Checksum is self-consistency, not a publisher signature.** Full SHA-256 binds the manifest. It does not prove who sealed it. Evidence ids that merely have a well-formed digest are also self-consistency; recompute them with `--observations` against the original JSONL if you need content verification.
- **Public packs hash examples; they do not keep a full private reservoir.** `--include-private` serializes raw text. Without it, heuristic fallback must not copy observation text into `definition`, and an optional LLM is prompted from keywords only (echoed raw text is discarded). Keyword glosses still derive from tokens in the observations. Do not use `--include-private` on real subjective-symbol data until v0.2 public/private auditor views exist.
- **First-pack ΔMDL is zero.** Compression accounting vs a parent pack is implemented; the first pack records `delta_mdl_bits = 0`.
- **Cluster ids are not stable labels.** Code `0` is whatever k-means assigned, not a universal “hello” token across runs unless you keep the sealed pack.
- **Not Eris.** Subject-aware dynamics are included as an optional module. The mothership SLAR pipeline, event bus, and identity system are not in this repo.

If a claim is not certified by `neuralese certify`, it is not part of the audit contract. If `translate` emits English under `default` or `strict`, the pack passed that policy. `--allow-uncertified` is an explicit debug override and is not an audit.

# How to keep symbols decodable

Neuralese is the compact code a system uses internally after it compresses observations. English is a gloss of those codes. **Audit is the requirement that the map stays readable.**

A symbol that cannot be unfolded is not mysterious. It is an accounting failure.

## What a symbol is

An observation stream \(H\) is coarse-grained by a symbolization map \(\Pi: H \rightarrow S\). That map is a quotient: many histories share one code. Compression is the point. Irreversibility of \(\Pi\) itself is expected — you cannot recover the full micro-history from the code alone.

Decodability is the other half of that fact:

> If \(\Pi\) is lossy, the only honest way to read a symbol later is to keep a **reservoir**: the observation ids (and embeddings/text) that were present at creation.

Without that reservoir, reversal needs extra bits you did not store. With it, an auditor can unfold \(s\) to the evidence that minted it.

That is the whole trick. Inner language is not a boogieman. It is a codebook plus a fold path.

## Five invariants

### 1. Addressable

Every live code must resolve to a class through the current codebook **or** a finite alias chain.

- Codes may move when the alphabet is rebuilt.
- Old streams stay readable by rewriting `old_code → new_code`.
- Alias cycles and dangling aliases fail closed.

Deletion is not an inverse. If a class dies, either quarantine it or issue a new forward commit. Do not pretend the old code never existed.

### 2. Unfoldable

Every admitted (non-quarantined) symbol lists `observation_ids` that produced it.

This is the reservoir. An auditor must be able to:

1. Take a code.
2. Look up its class.
3. Retrieve the observations (text and/or embeddings) used at mint time.
4. Recompute the prototype and residual.

If step 3 is empty, the symbol is not certified. Quarantine it or refuse the pack.

### 3. Gloss is a receipt

English is sealed into the pack checksum with the class id.

- A gloss without a pack checksum is commentary, not an audit.
- Changing the sentence without resealing **must** fail `certify()`.
- An optional LLM may draft the sentence. The certificate does not depend on the vendor.

Unglossed but unfoldable symbols are still better than hallucinated English. The translator emits `[unglossed: …]` rather than a plausible lie.

### 4. Fail-closed evolution

Alphabet updates (merge, split, hot-swap) are new packs with a `parent_pack_id`.

Admit a new pack only if:

- coherence of prototypes stays above threshold,
- reconstruction residual stays below threshold,
- description length does not grow without a recorded exception,
- aliases have no collisions that drop a live class on the floor.

Unknown / quarantined / aliased are first-class states. The translator never invents a fourth, silent “looks fine” path.

### 5. Residual-honest

Factorization or clustering leaves a reconstruction error \(\|X - \hat{X}\|\). Report it. Cap gloss `confidence` so it cannot exceed \(1 - \mathrm{clip}(\mathrm{residual})\).

A fluent sentence with a large residual is not more true. It is more confident than the geometry allows.

## Practical checklist

When you mint or accept a pack:

1. Every codebook entry points at a symbol object.
2. Every non-quarantined symbol has ≥1 observation id.
3. Every non-quarantined symbol has a bound English string **or** an explicit unglossed marker recorded in the pack (not invented at read time).
4. Checksum covers codebook, aliases, observation ids, definitions, residual.
5. `certify(pack)` returns `passed=true` before you ship the pack to another team.

When you translate a stream:

1. Rewrite through aliases.
2. Emit `unknown` for missing codes.
3. Emit `quarantined` for symbols that failed persistence/unfold.
4. Never fill gaps with an unconstrained language model.

## What auditors should distrust

- Gloss files stored beside the pack with no checksum.
- “We deleted the old codes to tidy the vocab.”
- Confidence 1.0 on a pack whose residual is large.
- A decoder that always produces grammatical English for every integer.

Those are how inner language becomes mythology: the map is lossy, the reservoir was dropped, and a sentence was generated anyway.

## Further reading

- [AUDIT_PROTOCOL.md](AUDIT_PROTOCOL.md) — certificate fields and fail-closed rules.
- [SLAR_MAPPING.md](SLAR_MAPPING.md) — names used in the originating SLAR engine (optional).

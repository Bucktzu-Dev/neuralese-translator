# PR-1 research checkpoint

Status: the activation-codec line is closed. Recommendation: **NONE**. Authority: **RUNTIME_NOT_AUTHORIZED**. No production admission follows from this checkpoint.

This note closes the line that ran from the activation-source capture ([PR1_ACTIVATION_SOURCE.md](PR1_ACTIVATION_SOURCE.md)) through the boundary trial ([PR1_ACTIVATION_BOUNDARY.md](PR1_ACTIVATION_BOUNDARY.md)), the symbol-residual trial ([PR1_SYMBOL_RESIDUAL.md](PR1_SYMBOL_RESIDUAL.md), commit `01b500469f89f3ec7d8292e7e7e3b2f730629baf`), and the symbolic-addressing trial ([PR1_SYMBOL_ADDRESSING.md](PR1_SYMBOL_ADDRESSING.md), commit `b87f1f77dc0975b350bc225d0b3090843cf3d7c6`). It does not search for another codec, quantizer width, layer, or slot-retrieval variant. It does not open a compositional, cross-context, or unfolding experiment. The numbers below are the fractions already published in those ledgers. They were not refit for this note.

Two earned findings stay separate. Neither is revoked:

| finding | ledger | commit | recommendation | authority |
|---|---|---|---|---|
| `RESIDUAL_DISTINCTION` | [PR1_SYMBOL_RESIDUAL.md](PR1_SYMBOL_RESIDUAL.md) | `01b500469f89f3ec7d8292e7e7e3b2f730629baf` | `NONE` | `RUNTIME_NOT_AUTHORIZED` |
| `QUANTIZED_MEMORY_SUFFICIENT` | [PR1_SYMBOL_ADDRESSING.md](PR1_SYMBOL_ADDRESSING.md) | `b87f1f77dc0975b350bc225d0b3090843cf3d7c6` | `NONE` | `RUNTIME_NOT_AUTHORIZED` |

The boundary trial stays `GEOMETRY_ONLY` with recommendation `NONE`. The activation-source trial stays `PASS_FOR_LATER_PR` with recommendation `LATER_ISOLATED_PR`. That source recommendation is not production admission, and this checkpoint does not turn it into one. Authority on every ledger in the line stays `RUNTIME_NOT_AUTHORIZED`.

## Three outcomes

### 1. Structural symbolic certification works within its declared contract

`certify` passing is structural only.

The activation-source ledger ran the existing `learn_pack` and `certify` on one pinned capture. On that training pack, `certify` passes: 5 live symbols, 3 quarantined, sealed residual `0.22755261731526205`, learn decision `accept`. An unknown code stays `unknown`. The boundary ledger states the same limit in its own words: a certify pass remains structural, and that trial did not add one. Passing those gates does not certify a gloss as a definition, a coordinate as a word, or a pack as a runtime.

### 2. Low-bit numerical representations preserve controlled-slot retrieval on the tested fixtures

The slot counts are held-out probe accuracy on the frozen final-layer fixtures. Condition 2 is the varied-terminal pair frames (64 held-out rows). Condition 4 is the context-matched shared-terminal pairs (16 held-out rows). These fractions are copied from [PR1_SYMBOL_ADDRESSING.md](PR1_SYMBOL_ADDRESSING.md).

| arm | condition 2 | condition 4 |
|---|---|---|
| standalone 4-bit | 58/64 | 15/16 |
| symbol plus residual | 57/64 | 12/16 |
| residual only | 61/64 | 14/16 |

Admission bytes are system bytes divided by the held-out count, as that ledger defines them. Condition 2, then condition 4:

| arm | admission bytes |
|---|---|
| standalone 4-bit | 210 and 258 |
| symbol plus residual | 341 and 773 |
| residual only | 338 and 770 |

### 3. Symbolic addressing has not earned extra accuracy or overall storage efficiency

The verdict is **QUANTIZED_MEMORY_SUFFICIENT**. The recommendation is **NONE**. The symbol id is not the carrier. The clause that fired is `quantized_memory_matches_or_beats_symbol_on_both_conditions`. The symbol arm is below standalone 4-bit memory on both conditions and below residual-only retrieval on both conditions. The tolerance was 0 and was not loosened.

**RESIDUAL_DISTINCTION** still stands, and is not revoked. A 4-bit residual preserved a distinction the centroid alone lost on condition 2 (57/64 vs 38/64). Condition 4 showed reconstruction error falling while slot accuracy stayed 12/16: the residual ledger records symbol-only mean squared Euclidean `838.2705645480764` and symbol-plus-residual `182.00555880726898`, with both slot counts at 12/16. A smaller reconstruction error is not a slot result, and it is not admission of condition 4.

## Measurement caveat

The addressing trial's admission latency excludes query quantization and codebook learning. The varied-terminal latency advantage (median about 0.00129 s versus 0.00226 s and 0.00235 s) is not an end-to-end throughput claim. Condition 4 latency ratios were about 0.886 and 0.826, short of the frozen 4/5 bar. A cheaper restricted search did not satisfy the joint gate.

Those medians are the scoring sample published in the addressing ledger. Tests there lock the repeat count, the warmup, and the definition. They do not lock the wall-clock sample. The accuracy clauses do not move with that sample, and they already block both conditions.

## What the slot readout is

The slot readout searches training payloads and returns that row's probe. It is not an English gloss and not an independently decoded semantic state. Cheap quantized memory and useful symbolic interpretation are different achievements. The next hypothesis that would give a symbol something to earn (stable concepts across contexts, compositional queries, selective unfolding, traceable explanations) is not opened here.

## What this checkpoint does not authorize

- A production admission of the residual codec, the symbol arm, the standalone 4-bit quantizer, or `certify` as a semantic or numerical pass.
- Revoking `RESIDUAL_DISTINCTION` or `QUANTIZED_MEMORY_SUFFICIENT`, or folding them into one verdict.
- Another codec, quantizer width, layer, clusterer, or slot-retrieval variant.
- A compositional, cross-context, or selective-unfolding experiment.
- A logit lens, an SAE, or a hub runtime.

Recommendation **NONE**. Authority **RUNTIME_NOT_AUTHORIZED**.

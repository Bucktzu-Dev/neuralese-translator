# PR-1 symbolic addressing

Status: `PLAN_ONLY` / `RESEARCH_ONLY`. Parent: the symbol residual trial at commit `01b500469f89f3ec7d8292e7e7e3b2f730629baf`, [PR1_SYMBOL_RESIDUAL.md](PR1_SYMBOL_RESIDUAL.md), PR #5. Entering verdict: `RESIDUAL_DISTINCTION`. Entering recommendation: `NONE`. Authority: `RUNTIME_NOT_AUTHORIZED`.

This note asks whether the symbol byte delivers measurable value beyond ordinary quantized numerical memory, after the residual and the lookup index are accounted for. The vectors are the final-layer rows already stored in [PR1_ACTIVATION_BOUNDARY.md](PR1_ACTIVATION_BOUNDARY.md). They were not recaptured. `word_sentence_svd`, `learn_pack`, `clustering.kmeans`, `certify`, `tau_residual` 0.55, `tau_kappa` 0.35, the CLI, and package version `0.7.0` stay as they are. The frozen residual codec in `src/neuralese/symbol_residual.py` is called and is not edited. `ACCURACY_MARGIN`, `SHUFFLE_FALL`, `STORAGE_RATIO_MAX`, the signed int4 grid, and the tests that lock 57/64 and 12/16 stay as they are. No new model training, no runtime integration, and no production admission.

The trial verdict is **QUANTIZED_MEMORY_SUFFICIENT**. The recommendation is **NONE**. Authority stays **RUNTIME_NOT_AUTHORIZED**. The entering result **RESIDUAL_DISTINCTION** is retained and is not revoked. The independent behavioral target is **INDEPENDENT_TARGET_UNAVAILABLE**. The clause that fired is `quantized_memory_matches_or_beats_symbol_on_both_conditions`.

The slot readout searches training payloads and returns the probe label stored on the chosen training row. It is a nearest-training-example task. It is not an English gloss and not an independently decoded semantic state. This ledger does not compare that readout to Anthropic NLA, or to any other semantic decoder. Matching a probe label is not a claim that a coordinate means the word.

## Frozen rules

The precision, the byte-accounting rules, the distance, the latency procedure, the tolerance, and the verdict order were written in `src/neuralese/symbol_addressing.py` before any held-out slot score from this trial was computed. They were not edited after the scores. Probe labels did not enter the activation matrix, the k-means fit, the scales, the quantization bins, or the choice of arm.

| field | value |
|---|---|
| model | `EleutherAI/pythia-14m` |
| revision | `cf967c0a9a04383db6f7b1108d86b2962634b4ac` |
| weight sha256 | `116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1` |
| forward dtype | float32 |
| layer | hidden_states index 6, the final layer norm |
| conditions | 2 and 4 only |
| condition 1 | omitted; those prompts have no cat/dog slot |
| condition 3 | omitted; this trial does not score the intermediate layer |
| controlled property | which of cat (probe 0, token id 5798) or dog (probe 1, token id 4370) sits at token index 1 |
| distance | squared Euclidean on dequantized payloads |
| dequant formula | `code * scale / 7`, the residual codec's `dequantize` |
| quantizer | `rint`, then clip to `[-7, 7]` |
| scale | per-coordinate maximum absolute value on the training matrix only; a zero coordinate uses scale 1 |
| standalone source | the activation itself |
| residual source | activation minus the assigned centroid |
| tie break | majority probe of exact distance ties; a tie of probes takes the smaller probe id |
| trial combiner | both conditions; a win on only one condition is not trial-level success |
| slot tolerance versus residual-only | `0.0` |
| admission storage figure | system bytes divided by the held-out count |
| second amortization | system bytes divided by train plus held-out; reported, not the gate |
| parent 65/512 cap | the parent trial's gate, not this trial's gate |
| cost margin | strictly under `4/5` of the other arm (`value * 5 < other * 4`) |
| latency warmup | 3 unrecorded full-path passes |
| latency repeats | 31, odd, so the median is one order statistic |
| admission latency | median of paired assignment-plus-search samples when the arm assigns a centroid; median of search samples for the standalone quantizer |
| quantize line | recorded, and not in the admission latency |
| learn time | recorded, and not in the admission latency |
| coordinate ops | a supplement, not the latency gate |
| reconstruction | reported, not an admission gate |
| independent target | `INDEPENDENT_TARGET_UNAVAILABLE` |

Four bits on the activation uses the same grid and the same denominator as the residual codec. The standalone scale is the train-only max-abs of the activation, not of a centroid residual. Held-out rows do not fit centroids, scales, or bins. The probe is read only after the nearest-payload assignment. An empty training symbol would use the global training majority. This run assigned no query to an empty training symbol. Exact distance ties: 0 queries on every arm, so the majority-of-ties rule did not change a prediction.

The parent 65-byte figure is a per-row payload, not the system footprint. Centroids are 8 times 128 float64 values, 8192 bytes, when the arm needs them. Scales are 128 float64 values, 1024 bytes, when the arm needs them. The labeled training store is what the nearest-training-example decoder requires: payload bytes, a symbol byte if that byte is stored, and one probe-label byte. The label is not a symbol id.

| arm | shared centroids | shared scales | training row | query row |
|---|---|---|---|---|
| standalone 4-bit | 0 | 1024 | 64 payload + 1 label = 65 | 64 |
| symbol plus residual | 8192 | 1024 | 64 payload + 1 symbol + 1 label = 66 | 65 |
| residual only | 8192 | 1024 | 64 payload + 1 label = 65 | 64 |

The standalone arm has no centroid table. It does have the scale table, because reconstruction and the decoder distance both use `code * scale / 7`. The residual-only arm does not store a symbol byte. Forming its query residual still assigns a centroid. That assignment is compute, not a stored symbol byte.

System bytes are shared tables plus the training store plus the query payloads. The admission storage figure is that total divided by the held-out count. System bytes divided by the number of train rows plus held-out rows is the second amortization. A cost win is admission bytes strictly under 4/5 of both other arms, or admission latency strictly under 4/5 of both other arms. Beating one arm on bytes and the other on latency is not a cost win.

A positive symbolic-addressing result requires every one of the following on both conditions. The tolerance is zero: the symbol arm must not be less accurate than residual-only. A nonzero tolerance would hide the restriction the parent trial already measured.

- Slot accuracy of the symbol arm is at least the standalone arm's accuracy.
- Slot accuracy of the symbol arm is at least the residual-only arm's accuracy.
- And a cost win against both other arms, on the admission storage figure or on the admission latency.
- A loss to residual-only blocks that condition, including when the symbol search is cheaper.

Verdict order, also fixed before the scores:

1. A checksum, reproduction, or isolation failure is `UNRESOLVED` / `NONE`.
2. The symbolic-addressing gate on both conditions is `SYMBOL_ADDRESSING_VALUE` / `NONE`. There is no independent behavioral target and no production admission, so the recommendation stays `NONE`.
3. Otherwise, if the standalone arm or the residual-only arm matches or beats the symbol arm on slot accuracy on both conditions, the verdict is `QUANTIZED_MEMORY_SUFFICIENT` / `NONE`. That does not revoke `RESIDUAL_DISTINCTION`. It says the symbol id is not the carrier.
4. Otherwise `RESEARCH_FAIL` / `NONE`, with the failing clause named.

Authority is `RUNTIME_NOT_AUTHORIZED` on every branch.

## What the tests load

`examples/activation_boundary/activations.npz` is the fixture from the activation boundary. Tests call `numpy.load` and this research module. They do not import torch, and they do not download a model. The four float32 checksums were checked before the slot scores. They match.

| array | shape | sha256 |
|---|---|---|
| condition 2 train | `(128, 128)` | `f9140300c4308ea67da83fae727b67ce93cd26fb55c396d9f57139fcb3ede999` |
| condition 2 held-out | `(64, 128)` | `33fe6c348437a28c6496fc52c5cb6e0a6d8dad3c67ab885dfa1253f2f607d9c4` |
| condition 4 train | `(32, 128)` | `dc17fbdffd8deb25cdb393239fba9fe26d6b27c0cf5280522a98406ffc629d64` |
| condition 4 held-out | `(16, 128)` | `fb23bebcfda1407a299c11ba48ebe04ebac68b319000f0b03e61a4ca3ce6f91c` |

The published parent counts were reproduced by calling the frozen runner before the new arm entered the verdict. Explicit integrity failures: none.

| condition | symbol only | symbol plus residual | residual without symbol id | nearest train row |
|---|---|---|---|---|
| 2 | 38/64 | 57/64 | 61/64 | 59/64 |
| 4 | 12/16 | 12/16 | 14/16 | 14/16 |

The symbol-plus-residual arm in this trial calls the same decoder and reproduces 57/64 and 12/16. The residual-only arm calls the same decoder with the symbol restriction removed and reproduces 61/64 and 14/16. The symbol arm's reconstruction error matches the parent ledger exactly: mean squared Euclidean 13.72172380594111 on condition 2 and 182.00555880726898 on condition 4.

The standalone quantizer occupies the full signed grid. Train and held-out codes on both conditions span `-7` through `7`. Held-out coordinates that land on code 0 are 594/8192 in condition 2 and 87/2048 in condition 4. The train scale is positive on every coordinate. Condition 2 activation scales run from 4.444333076477051 to 59.3287239074707. Condition 4 activation scales run from 2.664839506149292 to 53.46266555786133. The zero-scale fallback did not fire.

## Slot accuracy

Held-out probe accuracy. The decoder returns the probe of the nearest training payload. It does not read a gloss.

| arm | condition 2 | condition 4 |
|---|---|---|
| standalone 4-bit | 58/64 = 0.90625 | 15/16 = 0.9375 |
| symbol plus residual | 57/64 = 0.890625 | 12/16 = 0.75 |
| residual only | 61/64 = 0.953125 | 14/16 = 0.875 |

The symbol arm is below the standalone arm on both conditions: by 1/64 on condition 2 and by 3/16 on condition 4. It is below residual-only on both conditions: by 4/64 on condition 2 and by 2/16 on condition 4. The tolerance is 0. Both accuracy clauses fail on both conditions. The clause names are `symbol_accuracy_below_standalone_quant` and `symbol_accuracy_below_residual_only`.

The symbol arm does not beat standalone 4-bit memory. The symbol arm does not beat residual-only retrieval. The symbol search is a restriction of the same residual payloads. On this probe it is not an information gain. Condition 2's loss to residual-only blocks that condition even though the recorded search was cheaper. Condition 4's loss blocks that condition as well.

## Byte ledger

Shared state, the labeled training store, and the query payloads are kept separate. The admission column is system bytes divided by the held-out count. The last column divides by train plus held-out and is not the gate.

| arm | condition | centroids | scales | training store | query payloads | system bytes | admission bytes | bytes per row of both splits |
|---|---|---|---|---|---|---|---|---|
| standalone 4-bit | 2 | 0 | 1024 | 8320 | 4096 | 13440 | 210 | 70 |
| symbol plus residual | 2 | 8192 | 1024 | 8448 | 4160 | 21824 | 341 | 21824/192 |
| residual only | 2 | 8192 | 1024 | 8320 | 4096 | 21632 | 338 | 21632/192 |
| standalone 4-bit | 4 | 0 | 1024 | 2080 | 1024 | 4128 | 258 | 86 |
| symbol plus residual | 4 | 8192 | 1024 | 2112 | 1040 | 12368 | 773 | 12368/48 |
| residual only | 4 | 8192 | 1024 | 2080 | 1024 | 12320 | 770 | 12320/48 |

Condition 2 training rows are 128 and held-out rows are 64. Condition 4 training rows are 32 and held-out rows are 16. The symbol arm's admission bytes are 341 against 210 and 338 on condition 2, and 773 against 258 and 770 on condition 4. Each ratio is above 4/5: 341/210, 341/338, 773/258, and 773/770. The byte clause does not fire on either condition. The second amortization ranks the arms the same way and is still not the gate.

## Reconstruction

Fidelity uses the parent definitions. `mean_squared_euclidean` is the mean over rows of the squared Euclidean distance. `normalized_frobenius` is the Frobenius norm of the error divided by the Frobenius norm of the activation matrix. Neither number is an admission gate. The parent already recorded condition 4's residual error falling while slot accuracy stayed 12/16. A similar split in this table is not success.

| arm | condition 2 mean squared Euclidean | condition 2 normalized Frobenius | condition 4 mean squared Euclidean | condition 4 normalized Frobenius |
|---|---|---|---|---|
| standalone 4-bit | 55.89721257565742 | 0.06973300427023361 | 67.12070775462495 | 0.07561271737035702 |
| symbol plus residual | 13.72172380594111 | 0.03454996701578748 | 182.00555880726898 | 0.12451134043861779 |
| residual only, centroid added back | 13.72172380594111 | 0.03454996701578748 | 182.00555880726898 | 0.12451134043861779 |
| residual only, centroid omitted | 11205.26815787999 | 0.9873120271274967 | 11479.817513712027 | 0.9888583261121884 |

The residual-only row with the centroid added back is the same reconstruction as the symbol arm. The error matches exactly, so the residual code is the same code. The row without a centroid is not a reconstruction of the activation. Its normalized Frobenius error is 0.9873120271274967 on condition 2 and 0.9888583261121884 on condition 4. Omitting the symbol id removes the stored name of the centroid, and with it the reference frame required to rebuild the activation. The slot count does not require that frame: the same residual payloads, searched without the class restriction, score 61/64 and 14/16.

Condition 2 is the separation between these measurements. The symbol arm's mean squared error is 13.72172380594111 and the standalone arm's is 55.89721257565742, while the symbol arm's slot count is the lower one, 57/64 against 58/64. Condition 4 moves the other way: the standalone error is 67.12070775462495 against the symbol arm's 182.00555880726898, and the standalone slot count is 15/16 against 12/16. The lower error is not the result either way.

## Latency

The admission latency is a median of 31 paired samples after 3 unrecorded warmups. For the two arms that assign a centroid, each sample is assignment time plus search time. For the standalone quantizer, each sample is search time. The quantize line is the query's `rint`-clip and dequantize, and on the residual arms it also subtracts the assigned centroid. That line is not in the admission figure. K-means learn time is not in it. Coordinate-op counts are not it. A one-sample clock is not it. The median of the paired sums is not the sum of the two medians.

The seconds below are the scoring sample recorded with this note. Tests lock the repeat count, the warmup, and the definition. They do not lock these wall-clock samples. A later process can record a different median. On this sample, condition 4's symbol latency sat near the 4/5 line against residual-only, so that sample's latency-win boolean can move. The accuracy clauses cannot, and they already block both conditions.

| arm | condition | median assignment seconds | median quantize seconds | median search seconds | admission latency seconds |
|---|---|---|---|---|---|
| standalone 4-bit | 2 | 0.0 | 3.7809997593285516e-05 | 0.0022552109985554125 | 0.0022552109985554125 |
| symbol plus residual | 2 | 8.301499838125892e-05 | 4.377000004751608e-05 | 0.0012066010021953844 | 0.0012923100039188284 |
| residual only | 2 | 8.27530020615086e-05 | 4.3644999095704406e-05 | 0.0022628010010521393 | 0.0023452419991372153 |
| standalone 4-bit | 4 | 0.0 | 1.6219000826822594e-05 | 0.0003508970003167633 | 0.0003508970003167633 |
| symbol plus residual | 4 | 2.3335000150837004e-05 | 1.9046001398237422e-05 | 0.00028682000265689567 | 0.0003109739991487004 |
| residual only | 4 | 2.3081000108504668e-05 | 1.8601000192575157e-05 | 0.00035317299989401363 | 0.00037645399788743816 |

On this sample the condition 2 symbol admission latency is under 4/5 of both other arms: `5 * 0.0012923100039188284` is below `4 * 0.0022552109985554125` and below `4 * 0.0023452419991372153`. The latency clause fires on condition 2. The byte clause does not. A cost win on condition 2 is not admission, because the symbol arm is below both other arms on slot accuracy, and a loss to residual-only blocks the condition.

On this sample the condition 4 symbol admission latency is not under 4/5 of either other arm. `5 * 0.0003109739991487004` is above `4 * 0.0003508970003167633` and above `4 * 0.00037645399788743816`. The ratios are about 0.886 against standalone and about 0.826 against residual-only. The latency clause does not fire. The byte clause does not fire. The cost clause name on this sample is `no_cost_win_versus_both_arms`. The accuracy clauses would block the condition even if a later sample cleared 4/5.

Learn seconds in this sample are 0.0012541049982246477 on condition 2 and 0.000582468001084635 on condition 4. They are not a gate.

Coordinate touches are a supplement. One touch is one coordinate in a squared-Euclidean comparison or in the held-out quantize. The symbol arm's search touches match the parent decode total once the quantize touches are added back: condition 2 is 186368 plus 8192, which is 194560, and condition 4 is 11648 plus 2048, which is 13696. Residual-only search is the full training store: 1048576 and 65536.

| arm | condition 2 assignment | condition 2 quantize | condition 2 search | condition 4 assignment | condition 4 quantize | condition 4 search |
|---|---|---|---|---|---|---|
| standalone 4-bit | 0 | 8192 | 1048576 | 0 | 2048 | 65536 |
| symbol plus residual | 65536 | 8192 | 186368 | 16384 | 2048 | 11648 |
| residual only | 65536 | 8192 | 1048576 | 16384 | 2048 | 65536 |

## What failed

Checksums matched. The published slot counts reproduced. The called symbol arm reproduced 57/64 and 12/16. The called residual-only arm reproduced 61/64 and 14/16. Isolation failures: none. Verdict step 1 did not fire.

The symbolic-addressing gate did not fire. On both conditions the symbol arm is below standalone 4-bit memory and below residual-only retrieval. The tolerance is 0 and was not loosened. Condition 2's latency win does not override that loss. Condition 4, on the recorded sample, also misses the cost clause. A pass on one condition would not have been trial-level success under the frozen `and`.

Verdict step 3 did fire. The standalone arm matches or beats the symbol arm on both conditions, and the residual-only arm matches or beats the symbol arm on both conditions. Either fact would have been enough. The clause name is `quantized_memory_matches_or_beats_symbol_on_both_conditions`.

Explicit accuracy failures, both conditions:

- `symbol_accuracy_below_standalone_quant`
- `symbol_accuracy_below_residual_only`

The recorded sample also carries `condition_4:no_cost_win_versus_both_arms`. That cost flag is a property of the wall-clock sample. The accuracy flags are properties of the frozen decoder.

## Independent target

`INDEPENDENT_TARGET_UNAVAILABLE`.

No behavioral target was constructed that is not a function of the activation vector. Next-token logits of the hidden state were not stored and were not scored. Centroid cosine and reconstruction error were not substituted for a target. The slot probe is a label on the frozen strings. The readout that uses it is nearest-training-example retrieval.

The recommendation stays `NONE`. `QUANTIZED_MEMORY_SUFFICIENT` does not raise it. The branch that would authorize a later runtime does not exist in this verdict order. Authority stays `RUNTIME_NOT_AUTHORIZED`.

## What this does not authorize

- Revoking `RESIDUAL_DISTINCTION`, or editing the residual codec, its constants, or the tests that lock 57/64 and 12/16.
- Replacing `word_sentence_svd`, the default character-trigram path, or the production k-means routine.
- Editing `clustering.py`, the certificate gates, the taus, the CLI, or the package version.
- Treating 58/64, 57/64, 61/64, 15/16, 12/16, or 14/16 as a semantic reading of the hidden state, as an English gloss, or as a definition of the symbol.
- Treating the standalone quantizer's slot count as a compressed code that this trial admitted.
- Treating a lower reconstruction error, a nonzero code, or a fluent gloss as a result. No gloss was read.
- Using the parent 65/512 per-row cap as the gate of this trial.
- Admitting the symbol arm because its condition 2 search was cheaper than an unrestricted scan.
- Shipping a hub runtime, an SAE, or a logit lens in this package.
- Fitting a new checkpoint, a new layer, or a new quantizer width and calling it this result.

The text-encoder trials stay frozen. The activation-boundary verdict stays `GEOMETRY_ONLY` with recommendation `NONE`. The residual-trial verdict stays `RESIDUAL_DISTINCTION` with recommendation `NONE`.

This experimental line is closed. `QUANTIZED_MEMORY_SUFFICIENT` stands and does not revoke `RESIDUAL_DISTINCTION`. The checkpoint is [PR1_RESEARCH_CHECKPOINT.md](PR1_RESEARCH_CHECKPOINT.md). No further addressing, residual, codec, quantizer, layer, or slot-retrieval trial is opened by this note.

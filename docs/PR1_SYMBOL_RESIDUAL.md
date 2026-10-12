# PR-1 symbol residual

Status: `PLAN_ONLY` / `RESEARCH_ONLY`. Parent: PR #5, activation boundary at commit `81077ec`. Entering verdict: `GEOMETRY_ONLY`. Entering recommendation: `NONE`. Authority: `RUNTIME_NOT_AUTHORIZED`.

This note asks whether a bounded residual attached to an existing farthest-first symbol preserves a controlled slot distinction that centroid assignment loses. The vectors are the final-layer rows already stored in [PR1_ACTIVATION_BOUNDARY.md](PR1_ACTIVATION_BOUNDARY.md). They were not recaptured. `word_sentence_svd`, `learn_pack`, `clustering.kmeans`, `certify`, `tau_residual` 0.55, `tau_kappa` 0.35, the CLI, and package version `0.7.0` stay as they are. No runtime integration and no production admission.

The trial verdict is **RESIDUAL_DISTINCTION**. The recommendation is **NONE**. Authority stays **RUNTIME_NOT_AUTHORIZED**. The independent behavioral target is **INDEPENDENT_TARGET_UNAVAILABLE**. Condition 2 meets the admission gate. Condition 4 does not. A passing geometry on one condition does not authorize a runtime.

## Frozen codec

The precision, the byte budget, the fitting rule, and the admission constants were written in `src/neuralese/symbol_residual.py` before any held-out slot score from this trial was computed. They were not edited after the scores. Probe labels did not enter the activation matrix, the k-means fit, the residual scale, the quantization bins, or the choice of precision.

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
| distance | squared Euclidean |
| codebook | production `kmeans`, `k = 8`, seed argument 0 |
| residual | activation minus the assigned centroid |
| scale | per-coordinate maximum absolute training residual; a zero coordinate would use scale 1 |
| quantizer | `rint`, then clip to `[-7, 7]`; the train scale is reused on held-out rows |
| precision | signed 4-bit |
| symbol storage | 1 byte, uint8; 3 information bits |
| residual storage | 64 bytes; dimension `2i` in the low nibble, `2i+1` in the high nibble |
| bytes per row | 65 |
| full vector | float32, 128 dimensions, 512 bytes |
| storage cap | at most `0.25` of 512, which is 128 bytes |
| accuracy margin | `0.10` above symbol-only, same condition, continuity with the boundary trial |
| shuffle fall | `0.10` absolute drop of the within-class residual shuffle below the intact residual arm |
| shuffle seeds | assignment permutation seed 0; within-class payload permutation seed 1 |
| decoder | nearest training payload inside the assigned symbol, then the majority probe of exact distance ties |
| tie break | smaller probe id |
| independent target | `INDEPENDENT_TARGET_UNAVAILABLE` |

Four bits is the widest signed packing that lands on a nibble, keeps a separate symbol byte, and stays within 128 bytes. An 8-bit residual plus a symbol byte is 129 bytes, which misses the cap. The cap is the predeclared advantage. Wall-clock time is recorded below and is not a threshold. Shared centroids and scales are recorded and are not added into the per-row payload.

Farthest-first initialization does not read the seed argument. This trial calls that routine. It does not edit it, and it does not call `learn_pack`. The research labels are k-means class ids. They are not pack codes. `min_cluster_size` quarantine is not applied. The structural certificate was not re-run.

The slot strings are probes. They are not features. Nothing in the verdict reads a gloss. Matching a probe label is not a claim that a coordinate means the word.

Held-out rows do not fit centroids, scales, bins, or a decomposition. The symbol-to-probe map is the majority probe among training rows of that symbol. A tie takes the smaller probe id. The residual decoder reads probes only after the nearest-payload assignment. An empty training symbol would use the global training majority. This run assigned no held-out row to an empty training symbol. Exact distance ties on the residual decoder: 0 queries in every residual arm, so the majority-of-ties rule did not change a residual prediction.

## Gates

These constants sit above `run_symbol_residual`.

| constant | value | requirement |
|---|---|---|
| `ACCURACY_MARGIN` | 0.10 | symbol-plus-residual held-out accuracy is at least this far above symbol-only |
| `SHUFFLE_FALL` | 0.10 | within-class residual-shuffle accuracy is at least this far below the intact residual arm |
| `STORAGE_RATIO_MAX` | 0.25 | 65 stored bytes is at most this fraction of the 512-byte float32 vector |
| `SHUFFLE_ASSIGNMENT_SEED` | 0 | one permutation of the held-out symbol ids |
| `WITHIN_CLASS_RESIDUAL_SHUFFLE_SEED` | 1 | payload permutation within each symbol, symbols visited in ascending id |
| `INDEPENDENT_TARGET_STATUS` | `INDEPENDENT_TARGET_UNAVAILABLE` | logits, centroid cosine, and reconstruction error are disallowed as behavior |

A condition is admitted only when all three numerical clauses hold on the symbol-plus-residual arm. Nearest-row and full-vector accuracies are references. They do not admit the residual. Verdict order, also fixed before the scores:

1. A checksum or isolation failure, or a parent symbol boundary that does not reproduce, is `UNRESOLVED` / `NONE`.
2. Admission on condition 2 or condition 4 is `RESIDUAL_DISTINCTION` / `NONE`. There is no independent behavioral target, so the recommendation is not raised.
3. An accuracy gain that misses the storage cap, or an accuracy gain that survives the within-class residual shuffle, is `GEOMETRY_ONLY` / `NONE`.
4. No accuracy gain over symbol-only is `RESEARCH_FAIL` / `NONE`.

The recommendation is `NONE` on every branch. Authority stays `RUNTIME_NOT_AUTHORIZED`.

## What the tests load

`examples/activation_boundary/activations.npz` is the fixture from the parent boundary. Tests call `numpy.load` and this research module. They do not import torch, and they do not download a model. The four float32 checksums were checked before the slot scores. They match.

| array | shape | sha256 |
|---|---|---|
| condition 2 train | `(128, 128)` | `f9140300c4308ea67da83fae727b67ce93cd26fb55c396d9f57139fcb3ede999` |
| condition 2 held-out | `(64, 128)` | `33fe6c348437a28c6496fc52c5cb6e0a6d8dad3c67ab885dfa1253f2f607d9c4` |
| condition 4 train | `(32, 128)` | `dc17fbdffd8deb25cdb393239fba9fe26d6b27c0cf5280522a98406ffc629d64` |
| condition 4 held-out | `(16, 128)` | `fb23bebcfda1407a299c11ba48ebe04ebac68b319000f0b03e61a4ca3ce6f91c` |

Parent slot counts reproduced. Explicit integrity failures: none.

| condition | symbols | train mean | shuffled assignment | nearest train row |
|---|---|---|---|---|
| 2 | 38/64 | 32/64 | 38/64 | 59/64 |
| 4 | 12/16 | 8/16 | 8/16 | 14/16 |

Those are the published boundary counts. This trial does not replace them. Train and held-out symbol counts match the boundary note as well. Condition 2 train counts are 32, 17, 2, 9, 32, 15, 14, 7. Condition 2 held-out counts are 16, 9, 0, 3, 16, 7, 8, 5. Condition 4 train counts are 6, 2, 6, 3, 2, 2, 7, 4. Condition 4 held-out counts are 3, 0, 7, 2, 0, 0, 3, 1.

Train probe counts per symbol, probe 0 then probe 1, are a description of the majority map. They are not a gate. Condition 2: `[16, 16], [4, 13], [1, 1], [7, 2], [16, 16], [12, 3], [6, 8], [2, 5]`. Symbols 0 and 4 are tied, so the smaller probe id is the symbol-only prediction for every row in those symbols. Condition 4: `[2, 4], [1, 1], [4, 2], [1, 2], [1, 1], [1, 1], [3, 4], [3, 1]`.

The quantizer occupies the full signed grid. Train and held-out codes on both conditions span `-7` through `7`. Held-out coordinates that land on code 0 are 1176/8192 in condition 2 and 158/2048 in condition 4. The train scale is positive on every coordinate. Condition 2 scales run from 1.6162757312550262 to 9.433844595270998. Condition 4 scales run from 0.8565953118460516 to 5.456004157662392. The zero-scale fallback did not fire.

## Slot accuracy

Held-out probe accuracy. The full-vector readout is the nearest training row, because no probe model is fit on the vector. Its accuracy therefore equals the nearest-row reference. Its storage does not. The within-class control permutes held-out residual payloads among rows that share a symbol and leaves the symbol byte on the row. The payload-only control drops that byte and searches every training payload.

| arm | condition 2 | condition 4 | bytes per row |
|---|---|---|---|
| symbol only | 38/64 = 0.59375 | 12/16 = 0.75 | 1 |
| symbol plus residual | 57/64 = 0.890625 | 12/16 = 0.75 | 65 |
| nearest training row | 59/64 = 0.921875 | 14/16 = 0.875 | 512 |
| full vector | 59/64 = 0.921875 | 14/16 = 0.875 | 512 |
| train mean | 32/64 = 0.5 | 8/16 = 0.5 | 1 |
| shuffled assignment | 38/64 = 0.59375 | 8/16 = 0.5 | 1 |
| within-class residual shuffle | 35/64 = 0.546875 | 12/16 = 0.75 | 65 |
| residual without symbol id | 61/64 = 0.953125 | 14/16 = 0.875 | 64 |

Condition 2 margin over symbol-only is 19/64 = 0.296875. The required margin is 0.10. The within-class shuffle falls by 22/64 = 0.34375. The required fall is 0.10. Stored bytes are 65, and 65/512 = 0.126953125, which is below 0.25. All three clauses hold. The clause name is `admission_gate_met`.

On condition 2 the residual decoder disagrees with the symbol majority on 19 held-out rows. All 19 disagreements are rows the symbol missed and the residual hit. No symbol-correct row becomes wrong. Both wrong: 7. Both correct: 38.

Condition 4 margin is 0. The required margin is 0.10. The clause name is `no_accuracy_gain_over_symbol_only`. The within-class shuffle also stays at 12/16, so the fall is 0. The frozen order does not use that fall as the condition clause, because the accuracy margin was not met. The residual decoder disagrees with the symbol majority on 4 held-out rows: 2 become correct, 2 become wrong, and the net count stays 12. Storage on this condition is the same 65 bytes. Storage alone does not admit it.

The trial clause is `admission_gate_met_without_independent_target`, because the frozen order admits the trial when either condition passes. Condition 4 remains an explicit miss: `condition_4:no_accuracy_gain_over_symbol_only`.

Nearest-row and the full vector clear the 0.10 accuracy margin on both conditions. Condition 2 nearest-row margin over symbols is 21/64 = 0.328125. Condition 4 is 2/16 = 0.125. Both references store 512 bytes per row. They are not compressed solutions, and the verdict function does not read them as admission.

The payload-only control clears the same 0.10 margin over symbol-only on both conditions: 61/64 on condition 2 and 14/16 on condition 4. Under the predeclared reporting rule, the residual without the symbol id still carries the slot. That flag is not an admission input. On these rows the symbol restriction is not required for a nearest-payload probe readout. Condition 2's symbol-restricted arm is 57/64, below the payload-only 61/64. Condition 4's symbol-restricted arm stays at the symbol-only count while the payload-only arm matches the nearest row at 14/16. The admitted object is still the symbol-plus-residual arm on condition 2. The control says the symbol byte is not what carries the slot.

## Reconstruction fidelity

Fidelity is the distance from a reconstructed vector to the original activation. `mean_squared_euclidean` is the mean over rows of the squared Euclidean distance. `normalized_frobenius` is the Frobenius norm of the error divided by the Frobenius norm of the activation matrix. Neither number is an admission gate. The full vector is stored as itself, so both of its fidelity numbers are 0. The payload-only arm does not reconstruct the activation: the centroid is named by the symbol byte, and that byte is omitted. Its payload quantization error is the same vector as the intact arm's activation error, because the centroid cancels.

| arm | condition 2 mean squared Euclidean | condition 2 normalized Frobenius | condition 4 mean squared Euclidean | condition 4 normalized Frobenius |
|---|---|---|---|---|
| symbol only | 500.0086570588038 | 0.20856050137538953 | 838.2705645480764 | 0.2672136815462288 |
| symbol plus residual | 13.72172380594111 | 0.03454996701578748 | 182.00555880726898 | 0.12451134043861779 |
| nearest training row | 467.14654083402917 | 0.20159041039987236 | 809.714322027339 | 0.26262283965537503 |
| full vector | 0.0 | 0.0 | 0.0 | 0.0 |
| train mean | 2170.3090957849786 | 0.43451432184730054 | 982.3489764858955 | 0.2892673925478527 |
| shuffled assignment | 3876.3199747084736 | 0.5807017166937707 | 1060.7607981003152 | 0.3005905430929568 |
| within-class residual shuffle | 650.5428166908108 | 0.23789276999537842 | 718.5497704888743 | 0.24739733092973376 |

Condition 4 is the separation between these two measurements. Mean squared error falls from 838.2705645480764 at the centroid to 182.00555880726898 with the residual, and held-out slot accuracy stays 12/16. A smaller reconstruction error is not a slot result. Condition 2's within-class shuffle raises mean squared error from 13.72172380594111 to 650.5428166908108 and drops slot accuracy from 57/64 to 35/64. The accuracy drop is the control. The error number is not.

## Bytes and cost

Per-row payload against the float32 vector is the storage gate. Shared state is extra accounting. Centroids are 8 times 128 float64 values, 8192 bytes. Scales are 128 float64 values, 1024 bytes. Those totals are not divided into the per-row gate.

Assignment cost and decode cost count coordinate touches. One touch is one coordinate in a squared-Euclidean comparison, or one coordinate in the held-out quantize. A majority-map lookup is a row operation, not a coordinate operation. Capture time is not included. The production k-means fit is learn time, one sample, and it is not a gate.

| arm | condition 2 assignment ops | condition 2 decode ops | condition 4 assignment ops | condition 4 decode ops |
|---|---|---|---|---|
| symbol only | 65536 | 0 | 16384 | 0 |
| symbol plus residual | 65536 | 194560 | 16384 | 13696 |
| nearest training row | 1048576 | 0 | 65536 | 0 |
| full vector | 1048576 | 0 | 65536 | 0 |
| train mean | 8192 | 0 | 2048 | 0 |
| shuffled assignment | 0 | 0 | 0 | 0 |
| within-class residual shuffle | 65536 | 194560 | 16384 | 13696 |
| residual without symbol id | 65536 | 1056768 | 16384 | 67584 |

Symbol-only, train-mean, shuffled assignment, and the residual arms each do one majority lookup per held-out row after the coordinate work. That is 64 row ops on condition 2 and 16 on condition 4. The shuffled-assignment control also permutes 64 or 16 symbol ids. Its coordinate cost is 0 because it does not compute a new distance.

Condition 2 symbol-plus-residual assignment plus decode is 260096 coordinate ops. The full-vector reference is 1048576. The ratio is 0.248046875. Condition 4 is 30080 against 65536, ratio 0.458984375. Those ratios are measurements. The predeclared advantage is the byte ratio 0.126953125, not these operation ratios and not the wall clock.

Learn and decode seconds below are one sample from this scoring process. They are not gates. A later process can record a different wall clock.

| stage | condition 2 seconds | condition 4 seconds |
|---|---|---|
| k-means learn | 0.0020497639998211525 | 0.000498088000313146 |
| symbol assignment | 0.000145804002386285 | 3.735099744517356e-05 |
| residual decode | 0.00151177499719779 | 0.0003629970015026629 |
| nearest-row assignment | 0.007061613996484084 | 0.00012672700177063234 |
| within-class shuffle decode | 0.001816350002627587 | 0.0006038190003891941 |
| payload-only decode | 0.002650132002600003 | 0.0004378769990580622 |

## Independent target

`INDEPENDENT_TARGET_UNAVAILABLE`.

No behavioral target was constructed that is not a function of the activation vector. Next-token logits of the hidden state were not stored and were not scored. Centroid cosine and reconstruction error were not substituted for a target. The slot probe is a label on the frozen strings. It is not an independent behavior.

The recommendation stays `NONE` for that reason. Condition 2's admission does not change the recommendation. The branch that would authorize a later runtime does not exist in this verdict order.

## What this does not authorize

- Replacing `word_sentence_svd`, the default character-trigram path, or the production k-means routine.
- Editing `clustering.py`, the certificate gates, the taus, the CLI, or the package version.
- Treating condition 2's 57/64 as a semantic reading of the hidden state, or as a definition of the symbol.
- Treating the payload-only 61/64, or the nearest-row 59/64, as a compressed code that this trial admitted.
- Admitting condition 4. Slot accuracy there stays 12/16. The reconstruction error fell, and that fall is not admission.
- Treating a nonzero residual, a balanced cluster count, or a fluent gloss as a result. No gloss was read.
- Shipping a hub runtime, an SAE, or a logit lens in this package.
- Fitting a new checkpoint, a new layer, or a new quantizer width and calling it this result.

The text-encoder trials stay frozen. The activation-boundary verdict stays `GEOMETRY_ONLY` with recommendation `NONE`.

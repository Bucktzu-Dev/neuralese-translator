# PR-1 activation boundary

Status: `PLAN_ONLY`. Parent commit `852b681`. Baseline commit `c6321d5`. Entering verdict: `UNRESOLVED`. Entering recommendation: `NONE`. Authority: `RUNTIME_NOT_AUTHORIZED`.

This note asks whether the final-layer activation boundary and a shared terminal token hide a controlled slot distinction from the existing farthest-first squared-Euclidean k-means. The vectors are a one-time forward of the same pinned checkpoint as [PR1_ACTIVATION_SOURCE.md](PR1_ACTIVATION_SOURCE.md). They are not character hashes, not TF-IDF, and not a truncated SVD of the prompt text. `word_sentence_svd`, `learn_pack`, `clustering.kmeans`, `certify`, `tau_residual`, `tau_kappa`, and the package version stay as they are. No runtime integration and no production admission.

The trial verdict is **GEOMETRY_ONLY**. The recommendation is **NONE**. Authority stays **RUNTIME_NOT_AUTHORIZED**. The independent behavioral target is **INDEPENDENT_TARGET_UNAVAILABLE**. The distinction gate did not pass, so the recommendation would have stayed `NONE` even if a target had been available. The text-encoder trials stay frozen.

## Frozen boundary

The prompts, the layer index, the token coordinate, and the admission constants were written down before any held-out probe accuracy was computed. Probe labels did not enter the activation matrix, the k-means fit, or the choice of layer.

| field | value |
|---|---|
| model | `EleutherAI/pythia-14m` |
| revision | `cf967c0a9a04383db6f7b1108d86b2962634b4ac` |
| weight sha256 | `116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1` |
| checkpoint dtype | float16 weights |
| forward dtype | float32 on CPU, one torch thread |
| stored dtype | float32 |
| capture stack | torch `2.14.1+cpu`, transformers `4.57.6` |
| tuple length | 7 |
| final layer | hidden_states index 6, the final layer norm |
| intermediate layer | hidden_states index 3, the tensor appended before block 3 (output of block 2) |
| condition 3 token coordinate | final unpadded position of the condition 2 prompts |
| distance | squared Euclidean |
| dispersion | sample standard deviation, divisor `n - 1` |
| codebook | production `kmeans`, `k = 8`, seed argument 0 |
| controlled property | which of two frozen strings sits in one subject slot |
| probe 0 | string `cat`, tokenizer id 5798, token index 1 |
| probe 1 | string `dog`, tokenizer id 4370, token index 1 |
| shared terminal | `.`, tokenizer id 15 |
| varied terminals | `!` `?` `;` `:`, ids 2, 32, 28, 27 |
| design fingerprint | `5fcef56e50adf6e5eab2ea643dcebefc02fee26839a8df9e49ddcd1bdfd45db3` |

Index 3 was chosen before scoring. It was not moved after the scores. Farthest-first initialization does not read the seed argument. This trial calls that routine. It does not edit it, and it does not call `learn_pack`. The research labels are k-means class ids. They are not pack codes. `min_cluster_size` quarantine is not applied to them. `certify` was not re-run. A certify pass remains structural, and this note does not add one.

The slot strings are probes. They are not features. Nothing in the verdict reads a gloss. Matching a probe label is not a claim that a coordinate means the word, and it is not semantic validity.

## Conditions

1. **Parent final layer, shared terminal.** The 32 train and 16 held-out prompts in `examples/activation_source`, last hidden state, final unpadded token. Every stored terminal id is 15. This condition is reproduced from that fixture. The prompts were not replaced.
2. **Final layer, varied terminal, controlled slot.** Sixteen train frames and eight held-out frames. Each frame is rendered with `cat` and with `dog`, then with each of the four varied terminals. A pair is the two slot strings in one frame with one terminal. Train rows: 128. Held-out rows: 64. Held-out frames are disjoint from train frames. Activation: hidden_states index 6 at the final unpadded position.
3. **Intermediate layer, same coordinates as condition 2.** Same strings, same final unpadded position, hidden_states index 3. Captured in the same forward as condition 2. The extra cost is the slice, not a second forward.
4. **Context-matched pairs, shared terminal.** The same frames and the same slot, with the terminal fixed as `.` (id 15). Train rows: 32. Held-out rows: 16. Eight held-out pairs. Final layer, final unpadded position.

A parent-prefix control, not a fifth condition, replaces only the final character of each parent prompt with the varied-terminal cycle. Prefix token ids match the parent fixture. It exists so a change in concentration can be compared without swapping the parent sentences for the pair frames.

Train and held-out strings were fixed before the forward. No centroid was fit on a held-out row. The symbol-to-probe map is the majority probe among training rows of that symbol. A tie takes the smaller probe id. An empty symbol would use the global training majority. This run assigned no held-out row to an empty training cluster.

## Gates

These constants sit above `run_activation_boundary`. They were not edited after the scores.

| constant | value | requirement |
|---|---|---|
| `DISTINCTION_MARGIN` | 0.10 | held-out probe accuracy is at least this far above each baseline |
| `DISTINCTION_GATE` | | above the train-mean assignment, the shuffled assignment (seed 0), the terminal-token id, and the nearest training row |
| `GEOMETRY_NN_FLOOR` | 0.75 | nearest training row clears this accuracy |
| `TERMINAL_CONCENTRATION_DROP` | 0.25 | minimum held-out max-centroid drop before a sentence may attribute the parent 11/16 mass to the final token |
| `SHUFFLE_ASSIGNMENT_SEED` | 0 | one permutation of the held-out symbol ids |
| `INDEPENDENT_TARGET_STATUS` | `INDEPENDENT_TARGET_UNAVAILABLE` | logits of the clustered state are disallowed and were not stored |

The distinction gate is the logical and of the four baselines on one condition. The trial-level gate is true if condition 2, 3, or 4 clears that bar. The geometry flag is true if any of those three has nearest-train accuracy at or above 0.75. Verdict order, also fixed before the scores:

1. A capture that is not genuine, or a parent boundary that does not reproduce, is `UNRESOLVED` / `NONE`.
2. A distinction gate with an independent target that also passes is `DISTINCTION_WITH_INDEPENDENT_TARGET` / `RESEARCH_NOTE`. This run cannot take that branch. The runner passes `independent_target_satisfied=False`.
3. A distinction gate without that target is `DISTINCTION_WITHOUT_INDEPENDENT_TARGET` / `NONE`.
4. Geometry at the floor, with the distinction gate missed, is `GEOMETRY_ONLY` / `NONE`.
5. Otherwise `RESEARCH_FAIL` / `NONE`.

`RESEARCH_NOTE` is not runtime authorization. Authority stays `RUNTIME_NOT_AUTHORIZED` on every branch.

## What the tests load

`examples/activation_boundary/activations.npz` and `capture_meta.json` are the new fixture. Tests call `numpy.load` and the existing library. They do not import torch, and CI does not run `examples/activation_boundary/capture_once.py`. The snapshot was already on disk. `download_seconds` is 0. The weight file was checked against the pinned sha256 before the forward.

Checksums of the float32 activation matrices, sha256 of the contiguous bytes:

| array | shape | sha256 |
|---|---|---|
| condition 2 train | `(128, 128)` | `f9140300c4308ea67da83fae727b67ce93cd26fb55c396d9f57139fcb3ede999` |
| condition 2 held-out | `(64, 128)` | `33fe6c348437a28c6496fc52c5cb6e0a6d8dad3c67ab885dfa1253f2f607d9c4` |
| condition 3 train | `(128, 128)` | `786b78db46f40f8314cd2d9587954c7999f988294ac973ef77548b97bd636b60` |
| condition 3 held-out | `(64, 128)` | `21b1e9364f44951da87e28fceb1723d7d9ad6021cccd38faeb4ad7b56836d180` |
| condition 4 train | `(32, 128)` | `dc17fbdffd8deb25cdb393239fba9fe26d6b27c0cf5280522a98406ffc629d64` |
| condition 4 held-out | `(16, 128)` | `fb23bebcfda1407a299c11ba48ebe04ebac68b319000f0b03e61a4ca3ce6f91c` |
| parent-terminal swap train | `(32, 128)` | `1312a45fc6a476d0fd5834f91ebe64bb98135ac66e1d10c1bb9e2a0b8309923a` |
| parent-terminal swap held-out | `(16, 128)` | `444bd4a7199c8e34158f9a99639142adac47f5f4ce9174d4d80d721269ab1c73` |

The npz is 238759 bytes. Token ids, lengths, probes, and slot token ids are in the same file. Probes are a side array. They are not columns of the matrix passed to `kmeans`. No logit matrix is stored.

## Condition 1 reproduction

Production `kmeans` on the parent training activations, then squared-Euclidean assignment of the parent held-out rows, matches the published parent geometry. Explicit integrity failures: none.

| field | value |
|---|---|
| train counts | 1, 2, 5, 1, 3, 3, 16, 1 |
| held-out counts | 0, 0, 4, 0, 0, 1, 11, 0 |
| held-out max | 11/16 = 0.6875, class 6 |
| occupied held-out centroids | 3 |
| held-out entropy | 1.121640762186858 bits |
| effective held-out symbols | 2.175942995894572 |
| train residual | 0.22755261731526205 |
| held-out prototype residual | 0.29343442149783966 |
| held-out residual to the train mean | 0.30512149677079187 |
| shuffled assignment, seed 0 | 0.33508908512206886 |
| terminal ids | 15 on all 48 rows |

The parent terminal distribution has entropy 0. One token id cannot, by itself, name which of the eight centroids receives the 11 rows.

## Distances

Squared Euclidean. Dispersion is the sample standard deviation. Within-condition figures use the upper triangle of the row-row matrix. A within-pair figure is the distance between the two slot strings of one frame and one terminal. A nonpair figure is the rest of that upper triangle. Condition 1 and the parent-prefix swap are not slot pairs, so they have no within-pair figure.

Index 6 and index 3 are not on a common scale. The mean squared distance between those two tensors on the same condition-2 row is 11472.302718528139 on train (std 811.68635546051, n = 128) and 11459.381722774397 on held-out (std 819.7809569244307, n = 64). That magnitude is the layer change. It is not a ranking of how well the slot is separated. Within one layer, the within-pair mean against the nonpair mean is the slot contrast.

| matrix | mean | std | n |
|---|---|---|---|
| condition 1 train, within | 2410.6353345276334 | 903.9171436797666 | 496 |
| condition 1 held-out, within | 2177.8451223707066 | 849.7528653780615 | 120 |
| condition 2 train, within | 4038.2944302608184 | 2976.4057032449746 | 8128 |
| condition 2 held-out, within | 4302.755472047234 | 3035.1831009016737 | 2016 |
| condition 2 train, within pair | 169.31065686436799 | 70.74343396554666 | 64 |
| condition 2 train, nonpair | 4069.0006506846 | 2968.0819373405493 | 8064 |
| condition 2 held-out, within pair | 178.10759597811648 | 78.21793798242732 | 32 |
| condition 2 held-out, nonpair | 4369.282050693509 | 3013.6251775468268 | 1984 |
| condition 3 train, within | 0.6555426362382194 | 0.2537873375700146 | 8128 |
| condition 3 held-out, within | 0.6335339552981316 | 0.2606513707440165 | 2016 |
| condition 3 train, within pair | 0.002247264033218228 | 0.0014964683276532974 | 64 |
| condition 3 train, nonpair | 0.6607275201446084 | 0.2480013431392772 | 8064 |
| condition 3 held-out, within pair | 0.0027677179942251794 | 0.0019174321423866803 | 32 |
| condition 3 held-out, nonpair | 0.6437076042869043 | 0.25002287694757136 | 1984 |
| condition 4 train, within | 1005.4665953668746 | 389.7972005765253 | 496 |
| condition 4 held-out, within | 1762.630980127786 | 1351.7512720925315 | 120 |
| condition 4 train, within pair | 248.65253867314732 | 35.05889181906088 | 16 |
| condition 4 train, nonpair | 1030.6937305899987 | 370.4170936376083 | 480 |
| condition 4 held-out, within pair | 288.74968795660504 | 72.16276942223688 | 8 |
| condition 4 held-out, nonpair | 1867.90821528287 | 1338.2257944770845 | 112 |
| swap train, within | 5195.798358788259 | 3137.3965736161385 | 496 |
| swap held-out, within | 4553.934329492813 | 2549.463343558548 | 120 |
| condition 1 vs swap, train, row aligned | 3895.9468667329375 | 1840.5985313073252 | 32 |
| condition 1 vs swap, held-out, row aligned | 3714.872117149182 | 1848.5229244146144 | 16 |
| condition 4 vs condition 2, train, same frame and word | 3584.4061817153624 | 1714.9065071561163 | 128 |
| condition 4 vs condition 2, held-out, same frame and word | 3552.274657119331 | 1483.6277310521161 | 64 |

On held-out rows, the within-pair mean divided by the nonpair mean is 0.0407636 in condition 2, 0.00429996 in condition 3, and 0.154584 in condition 4. Those quotients are not a gate. At index 3 the two slot strings land close to each other relative to other row pairs. The final layer does not show that same near-coincidence. That is a geometric description of these coordinates. It is not a semantic reading.

## Concentration

Train-only `kmeans`, held-out rows assigned to the frozen centroids. Max fraction is the largest count divided by the number of rows. Occupied centroids have at least one row. Entropy is Shannon entropy in bits over the occupied bins. Effective symbols are `2` to that power.

| split | counts | max | occupied | entropy bits | effective |
|---|---|---|---|---|---|
| condition 1 train | 1, 2, 5, 1, 3, 3, 16, 1 | 16/32 = 0.5 | 8 | 2.2775182662886326 | 4.848432054514796 |
| condition 1 held-out | 0, 0, 4, 0, 0, 1, 11, 0 | 11/16 = 0.6875 | 3 | 1.121640762186858 | 2.175942995894572 |
| condition 2 train | 32, 17, 2, 9, 32, 15, 14, 7 | 32/128 = 0.25 | 8 | 2.6908280811595775 | 6.456839120573595 |
| condition 2 held-out | 16, 9, 0, 3, 16, 7, 8, 5 | 16/64 = 0.25 | 7 | 2.616479102437745 | 6.132516040790386 |
| condition 3 train | 28, 4, 30, 12, 10, 18, 20, 6 | 30/128 = 0.234375 | 8 | 2.757355675223189 | 6.761557832375378 |
| condition 3 held-out | 14, 0, 16, 6, 2, 10, 14, 2 | 16/64 = 0.25 | 7 | 2.5103907223310395 | 5.697743682442577 |
| condition 4 train | 6, 2, 6, 3, 2, 2, 7, 4 | 7/32 = 0.21875 | 8 | 2.8304399385868573 | 7.1129101450570715 |
| condition 4 held-out | 3, 0, 7, 2, 0, 0, 3, 1 | 7/16 = 0.4375 | 5 | 2.0524212838293647 | 4.148015496294372 |
| swap train | 16, 1, 5, 1, 4, 2, 1, 2 | 16/32 = 0.5 | 8 | 2.2621987351738495 | 4.7972204398410465 |
| swap held-out | 8, 0, 2, 0, 4, 0, 0, 2 | 8/16 = 0.5 | 4 | 1.75 | 3.363585661014858 |

Held-out rows on an empty training cluster: 0 in every condition.

Terminal-token occupancy, which is not the k-means occupancy: condition 1 and condition 4 are one id (15), max fraction 1, entropy 0. Condition 2 and the swap use four ids, 2, 27, 28, and 32, with equal counts, max fraction 0.25, entropy 2.

The parent held-out max fraction is 0.6875. After changing only the final token, the swap held-out max fraction is 0.5. The drop is 0.1875. The predeclared bar for a terminal-attribution sentence is 0.25. The drop is short of that bar. **The parent 11/16 mass is not attributed to tokenizer id 15.** The controlled swap did move the held-out max fraction from 0.6875 to 0.5, and it raised held-out occupancy from 3 centroids to 4. The train max fraction stayed 0.5. That is the comparison. It does not license the causal sentence.

On the pair frames, the held-out max fraction is 0.4375 with the shared terminal (condition 4) and 0.25 with varied terminals (condition 2). Those prompts are not the parent prompts. The comparison does not explain the parent 11/16 count. Balanced counts are not the distinction gate. Condition 3 is also at a held-out max fraction of 0.25, and its symbols do not track the slot.

## Controlled distinction

Held-out probe accuracy after the train-only majority map. The required margin over every baseline is 0.10. Pair split is the number of held-out pairs whose two members received different k-means labels. It is reported. It is not the gate. A high pair split with chance accuracy would not pass, and a low pair split with chance accuracy does not pass.

| condition | symbols | train mean | shuffled | token id | nearest train row | gate |
|---|---|---|---|---|---|---|
| 2 | 38/64 = 0.59375 | 32/64 = 0.5 | 38/64 = 0.59375 | 32/64 = 0.5 | 59/64 = 0.921875 | miss |
| 3 | 32/64 = 0.5 | 32/64 = 0.5 | 32/64 = 0.5 | 32/64 = 0.5 | 44/64 = 0.6875 | miss |
| 4 | 12/16 = 0.75 | 8/16 = 0.5 | 8/16 = 0.5 | 8/16 = 0.5 | 14/16 = 0.875 | miss |

Margins, symbol accuracy minus the baseline:

| condition | train mean | shuffled | token id | nearest train row |
|---|---|---|---|---|
| 2 | 0.09375 | 0.0 | 0.09375 | -0.328125 |
| 3 | 0.0 | 0.0 | 0.0 | -0.1875 |
| 4 | 0.25 | 0.25 | 0.25 | -0.125 |

Explicit gate misses:

- Condition 2 is 0.09375 over the train-mean and token-id baselines, short of 0.10. It ties the shuffled assignment. It is below the nearest training row. The margin was not lowered.
- Condition 3 ties the three assignment baselines at 0.5. Nearest-train accuracy is 0.6875, below the 0.75 floor. Pair split is 0/32: every held-out pair received one shared label. The intermediate coordinate did not make the slot available to this codebook, and it did not clear the geometry floor.
- Condition 4 clears the three assignment baselines by 0.25, which is past 0.10, and its pair split is 5/8. It is 0.125 below the nearest training row, so the four-baseline gate fails. Symbols on this condition are not a restatement of the terminal id. They also do not beat trivial nearest-row geometry by the predeclared margin.

Pair splits for the assignment baselines: train-mean and token-id split 0 pairs in every condition, because the terminal matches inside a pair and the train-mean assignment is one symbol. Shuffled assignment splits 26/32, 27/32, and 6/8 pairs in conditions 2, 3, and 4. Nearest-train probe labels disagree inside 27/32, 12/32, and 6/8 pairs. Unseen token ids on held-out rows: 0.

Nearest-train accuracy clears 0.75 in condition 2 (0.921875) and condition 4 (0.875). It does not clear it in condition 3 (0.6875). The geometry flag is therefore true, and the distinction gate is false. The final-layer nearest training row carries the probe label more often than index 3 does. This trial does not find a slot distinction at index 3 that the final layer then hides from the codebook.

## Independent target

`INDEPENDENT_TARGET_UNAVAILABLE`.

No behavioral target was constructed that is not a function of the clustered vector. Next-token logits of the clustered hidden state are a linear readout of that state. They were not stored and they were not scored. Centroid cosine and reconstruction error were not substituted for a target. The parent trial's relative logit drop is not reused as evidence here.

The distinction gate also failed. Recommendation stays `NONE` for both reasons. The branch that would record `RESEARCH_NOTE` requires the distinction gate and an independent target together. It was not taken.

## Cost

Capture, symbol learning, and translation are separate. Capture for condition 1 is the parent fixture's tokenize-plus-forward time. Conditions 2 and 3 share one forward of 1.0315953350218479 seconds. Condition 2 then adds the index-6 slice, 0.0076977129792794585 seconds. Condition 3's capture figure is only the index-3 slice, 0.0030824560271867085 seconds. Condition 4 and the parent-prefix swap are their own forwards. Symbol learning is one production `kmeans` on the training matrix. Translation is held-out squared-Euclidean assignment to those centroids. It is not `translate_stream`, and no pack was sealed.

Learn and translate seconds below are one sample from the scoring process. They are not gates. A later process can record a different wall clock. Capture seconds are the fixture values.

| stage | capture seconds | learn seconds | translate seconds | rows in the capture |
|---|---|---|---|---|
| condition 1 | 0.24427694100086228 | 0.0003950909995182883 | 4.136600182391703e-05 | 48 |
| condition 2 | 1.0392930480011273 | 0.0007531189985456876 | 8.53829988045618e-05 | 192 |
| condition 3 | 0.0030824560271867085 | 0.001217375000123866 | 8.461399920633994e-05 | 192 |
| condition 4 | 0.24731432801127085 | 0.00034706500082393177 | 3.207500049029477e-05 | 48 |
| parent-terminal swap | 0.28465284099729615 | 0.000311218998831464 | 2.5008001102833077e-05 | 48 |

Model load, recorded separately, was 0.191399346000253 seconds. Peak `ru_maxrss` during the capture process was 437944 KB. Python tracemalloc during the forward loop peaked at 384441 bytes. Download time is 0.

## What this does not authorize

- Replacing `word_sentence_svd`, the default character-trigram path, or the production k-means routine.
- Editing `clustering.py`, the certify gates, or the taus.
- Treating a more even cluster count, a lower residual, or a nonzero vector as evidence that a symbol names the slot.
- Attributing the parent 11/16 held-out mass to the period token. The controlled final-token swap moved that fraction by 0.1875, short of the predeclared 0.25.
- Treating nearest-train probe accuracy as a semantic interpretation of the hidden state.
- Treating index 3 as a hidden distinction that the final layer removed. On this coordinate, index 3 did not clear the geometry floor and its symbols split none of the 32 held-out pairs.
- Shipping a hub runtime, an SAE, or a logit lens in this package.
- Fitting a new checkpoint or a new layer index and calling it this result.

The text-encoder trials stay frozen. Their verdicts remain `UNRESOLVED` with recommendation `NONE`.

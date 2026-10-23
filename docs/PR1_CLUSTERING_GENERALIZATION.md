# PR-1 clustering generalization

Status: research trial on four new fixtures. The alphabet algorithm is unchanged. Package version stays 0.7.0. `learn_pack` is not called. `certify()` stays the structural contract. It is not a purity gate.

The four candidates were frozen by the isolation trial at `c1179c9`. This note does not add a seed, drop a candidate, or pick a new winner. It does not reuse the varied-prose held-out scores. Production k-means stays farthest-first squared Euclidean. Its `rng` argument is not read.

Topic ids are a probe. They sit in a sidecar and are not an input to `word_sentence_svd`, the initializer, the Lloyd update, or the advantage rule's choice of method. There is no new CLI flag and no `neuralese report` field.

## Rule

These constants are fixed in `neuralese.clustering_generalization` before any eval score.

k is 8 on every fixture, including fixtures with fewer than 8 topics. The topic counts were set with the files. They were not chosen from a score.

Split, per fixture, in that file's own order: index modulo 5 equals 4. That is every 5th row, 1-based rows 5, 10, 15, and so on. It is not the varied-prose index list. Centroids are fit on the train rows only. Eval rows take the nearest centroid and do not move it.

The embedding is `word_sentence_svd` at rank 8 of that fixture only, one call on every row of the file. Eval text is inside that TF-IDF and that SVD. Topic labels are not.

Prototype residual is the mean squared Euclidean distance from each train row to its assigned centroid. For spherical k-means the centroid is the renormed mean, so on unit vectors the same number is mean `2 - 2 cos`.

A candidate has an **advantage** on a fixture only when all three hold:

1. Eval size-weighted purity is strictly greater than production on that same split.
2. Train prototype residual is at most `1.10` times the production residual on that fixture.
3. Median clustering runtime is at most `2` times the production clustering runtime, same harness.

A candidate may be recommended for an isolated implementation PR only if it has that advantage on every fixture and the controls below hold. Otherwise the recommendation is `NONE`.

Clustering time is the median of five samples. Each sample is 100 fits, reported per call, so a sub-millisecond fit is not the scheduler. Peak bytes are the median tracemalloc peak of five single calls. Learn-equivalent cost is embed plus cluster: `word_sentence_svd` on the fixture, then the train fit. It is recorded. It is not a fourth gate.

Negative controls, seeds fixed before the score: label permutation seed 0, row permutation seed 1. These are not candidate seeds. Shuffled labels keep the production train assignment and permute topics. Shuffled rows permute the embedding, leave topics on the file order, and refit production k-means on the train indices. Both must score a lower size-weighted purity than the real production assignment. The same comparison is required on the eval partition when that partition has a cluster of at least 2 rows. A singleton partition has purity 1 under every labeling, so it cannot show a collapse. If a counted control does not collapse, or a fixture has fewer than 40 rows, the verdict is `UNRESOLVED`.

Cause. The only matched pairs in the frozen four are Euclidean versus spherical random-subset, both seed 2, and Euclidean random-subset versus Euclidean k-means++, both seed 2. Spherical k-means++ is seed 3 and is not in either pair. A fixture is `ANGULAR_DISTANCE` only when the angular pair disagrees and the init pair agrees. It is `INIT_SENSITIVITY` only when the init pair disagrees and the angular pair agrees. Anything else on that fixture is `UNRESOLVED`, including the case where all four advantage bits agree. The trial names one cause only when every fixture names that same cause. Otherwise the cause is `UNRESOLVED`. `OBJECTIVE` is not claimed from these four rows: they are not a factorial that holds the objective fixed.

`RUNTIME_NOT_AUTHORIZED` either way. An advantage is not a runtime change.

## Uneven topic sizes

`examples/generalization/uneven`. 50 original sentences. Tram is 25 rows. Bindery, glasswork, semaphore, tannery, and foundry are 5 each. Six topics. k stays 8.

Train 40, eval 10. Eval indices begin 4, 9, 14.

Production train residual **0.227587**. `1.10 ×` that is 0.250345. Production eval purity **7/10**. Production train purity 26/40.

Cluster stability was not re-tuned. The adjusted Rand index is agreement with production on this fixture, train labels and eval labels separately. It does not use topic ids.

| candidate | seed | residual | / production | eval purity | purity | residual | runtime ratio | advantage | train ARI | eval ARI |
|---|---|---|---|---|---|---|---|---|---|---|
| production farthest-first | — | 0.227587 | 1 | 7/10 | baseline | baseline | 1 | baseline | 1 | 1 |
| euclidean random subset | 2 | 0.249518 | 1.096 | 9/10 | yes | yes | 0.887 | yes | 0.5846 | 0.6341 |
| euclidean k-means++ | 2 | 0.178037 | 0.782 | 8/10 | yes | yes | 0.843 | yes | 0.7414 | 0.7761 |
| spherical random subset | 2 | 0.279981 | 1.230 | 8/10 | yes | no | 0.871 | no | 0.5846 | 0.3902 |
| spherical k-means++ | 3 | 0.263660 | 1.159 | 7/10 | no | no | 0.799 | no | 0.6045 | 0.3713 |

Spherical random-subset is over the residual line. Spherical k-means++ ties production at 7/10 and is also over the residual line. The Euclidean seed-2 pair both clear. The matched angular pair does not. On this fixture alone that is `ANGULAR_DISTANCE`.

Eval misses, candidate cluster majority against the row topic. The sample is capped at 6. Counts are the full miss list.

| candidate | misses | sample |
|---|---|---|
| euclidean random subset | 1 | `uneven-040` semaphore, majority tram |
| euclidean k-means++ | 2 | `uneven-030` bindery, majority foundry; `uneven-040` semaphore, majority tram |
| spherical random subset | 2 | `uneven-030` bindery, majority tram; `uneven-040` semaphore, majority tram |
| spherical k-means++ | 3 | `uneven-030` bindery, majority tram; `uneven-040` semaphore, majority tram; `uneven-050` foundry, majority tram |

This run, clustering per call: production 0.544 ms, candidates 0.435–0.483 ms. Peak bytes about 6.0 KB to 68 KB. Embed plus cluster: about 2.68–2.82 ms, peak about 351 KB. Ratios are under `2×`. The test does not lock the milliseconds.

Controls. Train label shuffle 21/40 and train row shuffle 22/40, both under 26/40. Eval label shuffle 8/10, which is not under 7/10. Eval row shuffle 5/10, which is under. The eval partition is informative, so the label comparison counts. It did not collapse. `probe_ok` is false.

## Overlapping concepts

`examples/generalization/overlapping`. 48 original sentences, eight each of greenhouse, canal, lithography, puppetry, horology, and dyehouse. Six topics. k stays 8.

Greenhouse and canal share the ordinary words water, pump, valve, and pipe. Each word appears in at least three sentences of each topic and not in every sentence of either. The other four topics in this file do not use those four words. The sentences do not share a copied suffix. The common character suffix of the file is `.`

Train 39, eval 9. Eval indices begin 4, 9, 14.

Production train residual **0.133772**. `1.10 ×` that is 0.147149. Production eval purity **6/9**. Production train purity 27/39.

| candidate | seed | residual | / production | eval purity | purity | residual | runtime ratio | advantage | train ARI | eval ARI |
|---|---|---|---|---|---|---|---|---|---|---|
| production farthest-first | — | 0.133772 | 1 | 6/9 | baseline | baseline | 1 | baseline | 1 | 1 |
| euclidean random subset | 2 | 0.137488 | 1.028 | 7/9 | yes | yes | 1.051 | yes | 0.9019 | 0.7209 |
| euclidean k-means++ | 2 | 0.143358 | 1.072 | 7/9 | yes | yes | 1.196 | yes | 0.8441 | 0.7209 |
| spherical random subset | 2 | 0.135592 | 1.014 | 6/9 | no | yes | 1.453 | no | 0.9439 | 1.0000 |
| spherical k-means++ | 3 | 0.156481 | 1.170 | 7/9 | yes | no | 1.356 | no | 0.9008 | 0.7209 |

Spherical random-subset ties production at 6/9, so the purity clause fails. Its eval assignment agrees with production (adjusted Rand 1 on the nine eval rows). Spherical k-means++ is over the residual line. The Euclidean seed-2 pair both clear. Same angular split as the uneven fixture. On this fixture alone that is `ANGULAR_DISTANCE`.

| candidate | misses | sample |
|---|---|---|
| euclidean random subset | 2 | `overlapping-005` greenhouse, majority horology; `overlapping-045` dyehouse, majority lithography |
| euclidean k-means++ | 2 | the same two rows and majorities |
| spherical random subset | 3 | those two, plus `overlapping-015` canal, majority lithography |
| spherical k-means++ | 2 | `overlapping-005` greenhouse, majority horology; `overlapping-045` dyehouse, majority lithography |

This run, clustering per call: production 0.319 ms, candidates 0.335–0.463 ms. Peak bytes about 5.8 KB to 67 KB. Embed plus cluster: about 2.35–2.65 ms, peak about 330 KB.

Controls. Train label shuffle 19/39 and train row shuffle 14/39, both under 27/39. Eval label shuffle 5/9, under 6/9. Eval row shuffle 6/9, a tie, not strictly under. It counts. It did not collapse. `probe_ok` is false.

## Ambiguous rows

`examples/generalization/ambiguous`. 48 original sentences. Four topics: archive, clinic, radio, kitchen. Eight sentences are plainly one topic. Sixteen sentences are written so they fit two of those topics; the sidecar stores `secondary_topic` beside the primary `topic`. The scorer reads only the primary. Twelve rows carry each primary label. Four topics. k stays 8.

Train 39, eval 9. The eval rows are observation numbers 5, 10, 15, 20, 25, 30, 35, 40, and 45. The last three of those are double-labeled.

Production train residual **0.256768**. `1.10 ×` that is 0.282445. Production eval purity **8/9**. Production train purity 26/39. Every frozen residual is under the line. None of them beats 8/9.

| candidate | seed | residual | / production | eval purity | purity | residual | runtime ratio | advantage | train ARI | eval ARI |
|---|---|---|---|---|---|---|---|---|---|---|
| production farthest-first | — | 0.256768 | 1 | 8/9 | baseline | baseline | 1 | baseline | 1 | 1 |
| euclidean random subset | 2 | 0.246522 | 0.960 | 7/9 | no | yes | 1.037 | no | 0.7056 | 0.1563 |
| euclidean k-means++ | 2 | 0.257175 | 1.002 | 6/9 | no | yes | 1.559 | no | 0.7034 | 0.3077 |
| spherical random subset | 2 | 0.266365 | 1.037 | 7/9 | no | yes | 1.199 | no | 0.7056 | 0.4375 |
| spherical k-means++ | 3 | 0.275088 | 1.071 | 6/9 | no | yes | 1.415 | no | 0.7546 | 0.3077 |

All four advantage bits are false, so both matched pairs agree. On this fixture the cause is `UNRESOLVED`.

The miss sample is clear-topic rows. The three double-labeled eval rows are not in it: they matched their primary majority under every frozen candidate on this split. `secondary_topic` was not used to excuse a miss.

| candidate | misses | sample |
|---|---|---|
| euclidean random subset | 2 | `ambiguous-015` clinic, majority kitchen; `ambiguous-025` kitchen, majority radio |
| euclidean k-means++ | 3 | `ambiguous-015` clinic, majority radio; `ambiguous-020` radio, majority kitchen; `ambiguous-030` kitchen, majority radio |
| spherical random subset | 2 | `ambiguous-015` clinic, majority kitchen; `ambiguous-020` radio, majority kitchen |
| spherical k-means++ | 3 | `ambiguous-015` clinic, majority radio; `ambiguous-020` radio, majority kitchen; `ambiguous-030` kitchen, majority radio |

This run, clustering per call: production 0.247 ms, candidates 0.256–0.385 ms. Peak bytes about 5.9 KB to 66 KB. Embed plus cluster: about 2.09–2.33 ms, peak about 312 KB.

Controls. Train label shuffle 21/39 and train row shuffle 21/39, both under 26/39. Eval row shuffle 7/9, under 8/9. Eval label shuffle 8/9, a tie, not strictly under. It counts. It did not collapse. `probe_ok` is false.

## Balanced control

`examples/generalization/balanced`. 80 original sentences, ten each of cartography, luthier, volcanology, chess, neon, mycology, stenography, and subway. These are not the harbor, orchard, ledger, bakery, pottery, weather, joinery, or apiary blocks. Eight topics. k stays 8. The counts match because the control is one block per subject, set with the file, not taken from a score.

Blocks of ten start on a multiple of five, so each topic contributes two eval rows. Train 64, eval 16. Eval indices begin 4, 9, 14.

Production train residual **0.211101**. `1.10 ×` that is 0.232211. Production eval purity **9/16**. Production train purity 30/64.

| candidate | seed | residual | / production | eval purity | purity | residual | runtime ratio | advantage | train ARI | eval ARI |
|---|---|---|---|---|---|---|---|---|---|---|
| production farthest-first | — | 0.211101 | 1 | 9/16 | baseline | baseline | 1 | baseline | 1 | 1 |
| euclidean random subset | 2 | 0.265648 | 1.258 | 7/16 | no | no | 1.015 | no | 0.6408 | 0.6111 |
| euclidean k-means++ | 2 | 0.192682 | 0.913 | 10/16 | yes | yes | 1.192 | yes | 0.7404 | 0.8333 |
| spherical random subset | 2 | 0.296198 | 1.403 | 9/16 | no | no | 1.126 | no | 0.6579 | 0.5909 |
| spherical k-means++ | 3 | 0.227243 | 1.076 | 9/16 | no | yes | 1.045 | no | 0.7081 | 0.7766 |

Euclidean k-means++ seed 2 clears. Euclidean random-subset seed 2 does not: purity 7/16 and residual over the line. The angular pair agrees, both without an advantage. On this fixture alone that is `INIT_SENSITIVITY`. It is not the angular label from the first two fixtures.

Misses are capped at 6. Euclidean random-subset has 9 misses; the other three are at or just over the cap.

| candidate | misses | sample |
|---|---|---|
| euclidean random subset | 9 | `balanced-005` cartography, majority chess; `balanced-010` cartography, majority chess; `balanced-015` luthier, majority neon; `balanced-020` luthier, majority chess; `balanced-030` volcanology, majority subway; `balanced-050` neon, majority volcanology |
| euclidean k-means++ | 6 | `balanced-005` cartography, majority chess; `balanced-010` cartography, majority chess; `balanced-020` luthier, majority chess; `balanced-030` volcanology, majority subway; `balanced-050` neon, majority stenography; `balanced-060` mycology, majority neon |
| spherical random subset | 7 | `balanced-005` cartography, majority mycology; `balanced-010` cartography, majority chess; `balanced-020` luthier, majority chess; `balanced-030` volcanology, majority subway; `balanced-045` neon, majority volcanology; `balanced-050` neon, majority stenography |
| spherical k-means++ | 7 | `balanced-005` cartography, majority chess; `balanced-010` cartography, majority chess; `balanced-020` luthier, majority chess; `balanced-030` volcanology, majority subway; `balanced-050` neon, majority volcanology; `balanced-060` mycology, majority neon |

This run, clustering per call: production 0.344 ms, candidates 0.349–0.410 ms. Peak bytes about 6.8 KB to 107 KB. Embed plus cluster: about 5.65–5.80 ms, peak about 673 KB. The embedding is most of that time.

Controls. Train label shuffle 20/64 and train row shuffle 18/64, both under 30/64. Eval label shuffle 7/16 and eval row shuffle 8/16, both under 9/16. `probe_ok` is true. This is the one fixture whose counted controls collapse.

## Aggregate

Advantage on a fixture is the three-clause rule above. A pass on one fixture is not a recommendation.

| candidate | uneven | overlapping | ambiguous | balanced | every fixture |
|---|---|---|---|---|---|
| euclidean random subset, seed 2 | yes | yes | no | no | no |
| euclidean k-means++, seed 2 | yes | yes | no | yes | no |
| spherical random subset, seed 2 | no | no | no | no | no |
| spherical k-means++, seed 3 | no | no | no | no | no |

No frozen candidate has the advantage on every fixture. The advantage rule names `NONE`.

Per fixture, the matched-pair label is `ANGULAR_DISTANCE`, `ANGULAR_DISTANCE`, `UNRESOLVED`, `INIT_SENSITIVITY`. Those four do not agree. The cause is **UNRESOLVED**. This trial does not say whether the gap is initialization, angular distance, or the objective.

Counted controls collapse on the balanced fixture only. On uneven, the eval label shuffle is 8/10 against a production eval purity of 7/10. On overlapping, the eval row shuffle ties at 6/9. On ambiguous, the eval label shuffle ties at 8/9. Train shuffles are lower than production on all four fixtures. The eval partitions are informative, so those three failures count. The verdict is **UNRESOLVED**.

The fixtures are large enough to score: 50, 48, 48, and 80 rows, every eval set non-empty, k still 8. `UNRESOLVED` is the control result, not a minimum-row failure.

`RUNTIME_NOT_AUTHORIZED`. Recommendation **NONE**.

The advantage grid, read on its own, also has no candidate ahead on every fixture. That pattern is what the rule calls a failure to generalize. It is not promoted to `RESEARCH_FAIL` while a counted control fails to collapse.

## Receipts

| item | value |
|---|---|
| isolation commit | `c1179c90fb202aefc6979363847e9bfd33e1809c` |
| git revision | `git rev-parse HEAD` at the time the diagnostic runs |
| candidates | euclidean random subset seed 2; euclidean k-means++ seed 2; spherical random subset seed 2; spherical k-means++ seed 3 |
| split | file-order index % 5 == 4, each fixture on its own |
| encoder | `word_sentence_svd`, rank 8, of that fixture only |
| k | 8 |
| minimum rows | 40 |
| clustering timer | 5 samples, 100 fits each, median per call |
| label shuffle | seed 0 |
| row shuffle | seed 1 |
| NumPy | 1.26.4 |
| BLAS threads | 1 |
| production | farthest-first, squared Euclidean, `rng` unread |
| stability | not re-tuned |
| varied-prose split | not loaded |
| `learn_pack` | not called |

Embedding SHA-256 of the float64 rank-8 matrix:

| fixture | rows | train | eval | sha256 |
|---|---|---|---|---|
| uneven | 50 | 40 | 10 | `4cceffeb6047a39615385c6d647f4a5c8e84f7bcaf57816a4003701a313e56f9` |
| overlapping | 48 | 39 | 9 | `682d0a3e03ff04176a09b738989725d74be3737e149ecca2bcbbc4ec44887c51` |
| ambiguous | 48 | 39 | 9 | `a1caeef8574e10c82f10334d932580155d263bbd7f7b0d75480d5db02c8834cb` |
| balanced | 80 | 64 | 16 | `7544590a6a53794f64ce21163427a1e630038437b308c79b163b53f01d3fb893` |

Max absolute row-norm error on each matrix is about `2.2e-16`.

## Negative controls

Train assignment, production versus the two shuffles. Lower is the required direction.

| fixture | production | label shuffle | row shuffle | both lower |
|---|---|---|---|---|
| uneven | 26/40 | 21/40 | 22/40 | yes |
| overlapping | 27/39 | 19/39 | 14/39 | yes |
| ambiguous | 26/39 | 21/39 | 21/39 | yes |
| balanced | 30/64 | 20/64 | 18/64 | yes |

Eval assignment. A tie is not a collapse.

| fixture | production | label shuffle | row shuffle | both lower |
|---|---|---|---|---|
| uneven | 7/10 | 8/10 | 5/10 | no |
| overlapping | 6/9 | 5/9 | 6/9 | no |
| ambiguous | 8/9 | 8/9 | 7/9 | no |
| balanced | 9/16 | 7/16 | 8/16 | yes |

The balanced control holds. The other three eval controls do not. Verdict `UNRESOLVED`.

## Recommendation

**NONE.**

No frozen candidate has the advantage on every fixture, and three fixtures fail a counted negative control. Neither fact is a license to swap `clustering.kmeans`. An implementation PR is not recommended. `RUNTIME_NOT_AUTHORIZED` is unchanged.

## What the trial leaves in place

The product rank stays 8. `tau_residual` stays 0.55. `tau_kappa` stays 0.35. Checksum coverage is unchanged. The character-trigram unanchored reject stays locked. No reconstructor, logit lens, SAE, or hub runtime is added. The production clustering call is the same function.

## Cost

On this host, this run, clustering-only per-call medians ran from 0.247 ms to 0.544 ms. Ratios against production on the same fixture ran from 0.799 to 1.559, all under `2×`. Embed plus cluster ran from about 2.1 ms on the ambiguous file to about 5.8 ms on the balanced file. Clustering peaks were about 6 KB for the spherical fits and about 66 KB to 107 KB for the Euclidean fits. Embed-plus-cluster peaks were about 0.31 MB to 0.67 MB. The embedding is the long part. A clustering swap would not be an SVD cost. The test does not lock these milliseconds or byte counts.

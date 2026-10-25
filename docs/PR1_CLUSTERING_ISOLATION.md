# PR-1 clustering isolation

Status: diagnostic fixture on the frozen rank-8 word-sentence SVD. The alphabet algorithm is unchanged. Package version stays 0.7.0.

Audit decision: **RESEARCH_FAIL**. The table's isolated cause is **UNRESOLVED**. Four seeded candidates are **ADMISSIBLE** under the predeclared bar. **RUNTIME_NOT_AUTHORIZED** is unchanged. `certify()` stays the structural contract (`addressable`, `unfoldable`, `gloss_bound`, `residual_ok`). It is not a purity gate.

Production k-means is the farthest-first Euclidean routine in `neuralese.clustering.kmeans`. `learn_pack` still calls it. This trial does not replace that call. Clearing the bar is not a runtime change.

Topic ids are a probe. They are not an input to the embedding, the initializers, the Lloyd updates, or the seed choice. The module is `neuralese.clustering_isolation`. The lock is `tests/test_clustering_isolation.py`. There is no new CLI flag and no `neuralese report` field.

## Production k-means

`kmeans` initializes with `_farthest_first`: the maximum-norm row, then the row with the largest squared Euclidean distance to the nearest chosen row. Assignment is squared Euclidean. The update is the mean of the members. Empty clusters keep the previous centroid. Ties in `argmin` keep the lowest index. `max_iter` is 64, with `np.allclose` as the stop.

The `rng` argument is accepted and not read. Seeds 0 and 4 on this matrix return the same labels. The local Lloyd copy from the same farthest-first indices matches `kmeans` labels and centroids. Given the matrix order, the partition is fixed.

## Corpus and frozen matrix

`examples/varied_prose`, encoder `word_sentence_svd`, `n_symbols` 8, seed 0 for the product learn path. One call at rank 8. Rows are the L2-normalized coordinates that call returns. They are not truncated again. BLAS, OpenMP, and MKL are pinned to one thread inside that SVD. NumPy 1.26.4.

The singular-value gap is the rank-boundary pin, copied here so this matrix is the same cut:

| quantity | value |
|---|---|
| `s[7]` (component 8) | 1.577782436621879 |
| `s[8]` (component 9) | 1.556022686499294 |
| gap `s[7] - s[8]` | 0.021759750122584798 |
| ratio `s[8] / s[7]` | 0.9862086498001754 |
| neighbor gap `s[6] - s[7]` | 0.03810132918251674 |
| neighbor gap `s[8] - s[9]` | 0.0548368263330099 |

This run reproduces `s[7]`, `s[8]`, the gap, and the ratio. SHA-256 of the float64 matrix bytes: `7a218f13197a4c9335ee65e4597ad1e3c061f1911843a837290820525ab927e1`. Max absolute row-norm error is about `2.2e-16`.

The full-set production weather cluster is code 6, 29 of 64, purity **0.453125**. Contaminants: ledger 26, orchard 3, bakery 2, pottery 2, harbor 1, joinery 1. Live size-weighted purity on the pack's quarantine is 204/319, the rank-boundary receipt. The comparison below counts every non-empty cluster, so that singleton stays in the 205/320 figure and `tau_kappa` does not choose a candidate.

## Bar, seeds, and split

These constants are fixed in `neuralese.clustering_isolation` before any held-out score.

Seeds for the non-production initializers: `0, 1, 2, 3, 4`. k is 8.

A candidate is **ADMISSIBLE** only when all of the following hold:

1. Full-matrix prototype residual is at most `1.10` times the production residual. The residual is the mean squared Euclidean distance to the assigned centroid. For spherical k-means the centroid is the renormed mean, so on unit vectors the same number is mean `2 - 2 cos`.
2. Median clustering runtime is at most `2` times the production clustering runtime, measured in the same one-call harness. Learn-equivalent cost is `max(0, learn_pack median − production clustering median) + candidate median`, and it must stay strictly under `3 × 222.369` ms. `222.369` ms is the rank-boundary learn_pack median. Three times that is 667.107 ms.
3. Held-out weather purity and held-out size-weighted purity are both strictly greater than the production assignment of the same held-out rows.

Labels are used only to score (3). The seed that enters (3) is the lowest full-matrix residual inside its family, then the lowest seed integer. In-sample purity does not break a tie. The full-matrix residual sees held-out coordinates and does not see topic labels. Held-out centroids are fit on the train rows only.

Split, frozen before that score: file-order index modulo 5 equals 4. That is 1-based rows 5, 10, 15, and so on. 320 rows give 64 held out and 256 train. The rule does not read topics. Each topic is a contiguous block of 40, and 40 is divisible by 5, so the split is 8 rows of each topic. That balance is the file layout.

Negative-control seeds, also fixed before the score: label permutation seed 0, row permutation seed 1.

Isolation cutoffs, same block: a purity movement of `0.05`, and a weather-cluster size at least `2` times the smallest non-empty cluster. A missing weather-majority cluster counts as a purity span of 1 against runs that have one. Size explains the mixture only when, inside Euclidean runs, every balanced run is at least `0.05` purer than every imbalanced run. `OBJECTIVE` requires initialization, distance, and that size contrast to be quiet, both the baseline and the residual winner to sit under the size cutoff, both to stay mixed (purity under 0.5), and the winner to fail held-out clause (3). Two explanation flags, or a broken negative control, is `UNRESOLVED`.

## Procedures

| family | init | distance |
|---|---|---|
| production | farthest-first | squared Euclidean, mean update |
| euclidean random subset | partial Fisher-Yates, `Generator(seed)` | same Lloyd |
| euclidean k-means++ | Arthur and Vassilvitskii, first center uniform, later centers proportional to squared distance, inverse CDF of `Generator.random` | same Lloyd |
| spherical farthest-first | the production indices | cosine assignment, mean, L2 renorm |
| spherical random subset | the same indices as the Euclidean twin | same spherical Lloyd |
| spherical k-means++ | the same indices as the Euclidean twin | same spherical Lloyd |

Spherical farthest-first is the matched production initialization. It is there so Euclidean versus angular is not a different choice of centers. Every matched pair in this run shares its index list.

## Full-matrix comparison

Prototype residual, in-sample weather purity, and stability are unsupervised or diagnostic as marked. Purity is diagnostic. The admissible column is the bar, which uses held-out labels only at clause (3). Runtime is the median of five calls. Peak is the median tracemalloc peak of five calls, clustering only.

| family | seed | residual | ms | peak bytes | weather | non-weather | size-weighted | weather vs smallest | stability ARI | ARI vs production | admissible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| production farthest-first | — | 0.274516 | 1.129 | 355176 | 29/64 = 0.453125 | 35 | 205/320 | 64 vs 1 | seed-invariant | 1 | baseline |
| euclidean random subset | 2 | 0.196008 | 1.123 | 354728 | 29/41 = 0.707317 | 12 | 232/320 | 41 vs 26 | 0.8139 | 0.8168 | yes |
| euclidean k-means++ | 2 | 0.196202 | 1.340 | 356206 | 29/41 = 0.707317 | 12 | 231/320 | 41 vs 26 | 0.7478 | 0.8381 | yes |
| spherical farthest-first | — | 0.302867 | 0.709 | 52840 | 29/61 = 0.475410 | 32 | 205/320 | 61 vs 1 | single init | 0.9746 | no |
| spherical random subset | 2 | 0.207560 | 1.059 | 29584 | 29/41 = 0.707317 | 12 | 232/320 | 41 vs 27 | 0.8038 | 0.8289 | yes |
| spherical k-means++ | 3 | 0.207491 | 0.815 | 54382 | 29/41 = 0.707317 | 12 | 232/320 | 41 vs 27 | 0.9631 | 0.8225 | yes |

The residual winner is euclidean random-subset seed 2, residual 0.1960081180406189. Its weather cluster is 29 weather, orchard 4, bakery 2, pottery 2, and one each of apiary, harbor, joinery, and ledger.

Euclidean k-means++ seeds 3 and 4 have in-sample weather purity 29/39, higher than seed 2, and a higher residual (0.196387 against 0.196202). They do not enter the held-out score.

Other Euclidean weather results, diagnostic: random-subset seeds 0 and 3 are 29/40; seeds 1 and 4 are 29/41. k-means++ seed 0 is 29/40. k-means++ seed 1 has no weather-majority cluster: 30 weather rows sit in a bakery-majority cluster of 78. Its residual 0.258047 loses to seed 2.

`1.10 × 0.27451605149978975` is 0.30196765664976873. Spherical farthest-first residual 0.3028666187575211 is above that line. The four seeded rows are below it.

## Held-out

Train fit only. Held-out rows take the nearest centroid and do not move it.

| family | seed | weather | non-weather | contamination | size-weighted |
|---|---|---|---|---|---|
| production farthest-first | — | 5/14 = 0.357143 | 9 | 9/14 | 37/64 = 0.578125 |
| euclidean random subset | 2 | 5/9 = 0.555556 | 4 | 4/9 | 44/64 = 0.6875 |
| euclidean k-means++ | 2 | 5/9 = 0.555556 | 4 | 4/9 | 44/64 = 0.6875 |
| spherical farthest-first | — | 5/12 = 0.416667 | 7 | 7/12 | 37/64 = 0.578125 |
| spherical random subset | 2 | 5/9 = 0.555556 | 4 | 4/9 | 44/64 = 0.6875 |
| spherical k-means++ | 3 | 5/9 = 0.555556 | 4 | 4/9 | 44/64 = 0.6875 |

Production held-out contaminants in the weather-majority cluster: ledger 5, harbor 1, joinery 1, orchard 1, pottery 1. The four seeded rows that beat both purity numbers have harbor 1, ledger 1, orchard 1, pottery 1. Spherical farthest-first ties size-weighted purity at 37/64, so clause (3) fails on the strict inequality.

## Negative controls

Same split, same k.

Shuffled topic labels on the production full-set assignment, seed 0: size-weighted purity 63/320 = 0.196875. That is below the production size-weighted purity 205/320 and below the production weather purity 0.453125. The original 64-row weather cluster, under those shuffled labels, is joinery-majority 12/64 = 0.1875, below 0.453125.

Shuffled embedding rows, seed 1, topics left on the file order, production k-means fit on the train rows of the permuted matrix: held-out size-weighted purity 19/64 = 0.296875, below 37/64. Held-out weather purity on that control is 4/15. The probe holds. The verdict is not `UNRESOLVED` for a broken control.

## What the table isolates

**UNRESOLVED.** Initialization and angular distance both move the weather mixture under the predeclared cutoffs. The rule keeps a single label only when one of those flags is set.

Initialization. Production weather purity is 0.453125. Other Euclidean seeds that still have a weather majority land at 29/39, 29/40, or 29/41. k-means++ seed 1 has none. The recorded init span is 1. Among the runs that do have a weather majority, the span is `29/39 − 29/64` = 0.290465, which is also above 0.05. Pairwise adjusted Rand index is 0.8139 across random-subset seeds and 0.7478 across k-means++ seeds. Adjusted Rand index versus production on those seeds runs from about 0.60 to 0.84. Farthest-first is the initialization that leaves 26 ledger rows in the weather cluster.

Angular distance, matched init and matched seed. Farthest-first spherical versus production: adjusted Rand 0.9746, weather 29/61 versus 29/64, purity delta 0.0223, under 0.05. Random-subset seed 2 versus its spherical twin: adjusted Rand 0.9865, both 29/41, delta 0. k-means++ seed 1 uses one index list for both distances: Euclidean has no weather-majority cluster (30 of 78, bakery majority) and spherical keeps 29 of 41. The rule scores that pair as a span of 1, and the weather-mass fractions 30/78 and 29/41 differ by 0.323. Distance moves the mixture on that pair. It does not move the residual winner's pair.

Size. Production is 64 against a smallest cluster of 1. The residual winner is 41 against 26, ratio 1.5769, under the cutoff of 2. Random-subset seed 0 has ratio 4.44 and weather purity 0.725, so a large ratio is not the mixed solution. The within-Euclidean contrast is false.

The objective. The rule does not claim it. Initialization and distance both move the mixture, the production size ratio is above 2, and the residual winner passes held-out clause (3).

## Cost

`learn_pack` on this corpus, five runs, median at the middle of the sorted samples. Same thread pin. This host, this run:

| run (ms) |
|---|
| 205.574 |
| 201.907 |
| 200.442 |
| 203.985 |
| 204.002 |

Median: **203.985 ms**. The rank-boundary figure used in the bar stays **222.369 ms**. The test does not lock either millisecond value.

Clustering alone, five calls of production `kmeans` on the frozen matrix: median **1.115 ms** (samples 1.658, 1.156, 1.115, 1.104, 1.110). `word_sentence_svd` alone: median **196.449 ms**. The clustering step is about half a percent of `learn_pack`. A clustering swap is not an SVD cost.

Candidate medians in the table are 0.709 ms to 1.340 ms, ratios 0.63 to 1.19 against the production harness median 1.129 ms. All are under the `2×` line. Learn-equivalent cost is about 204 ms for each selected row, under `3 × 222.369` ms. Peak tracemalloc on the clustering call is about 30 KB to 360 KB.

## Observed failures

The cause label stays `UNRESOLVED` because the init flag and the angular flag are both set.

Spherical farthest-first fails the bar on two deterministic clauses. Its residual is above `1.10×` production, and its held-out size-weighted purity ties production at 37/64. Runtime and learn-equivalent cost pass.

The four seeded representatives pass residual, runtime, learn-equivalent cost, held-out weather purity, and held-out size-weighted purity. They are ADMISSIBLE. Production `kmeans` is still farthest-first Euclidean. Runtime authority is not granted.

`certify` on the production pack passes. `tau_residual` is 0.55. The certificate details have no `cluster_purity` field and no `failure_mode` field.

## What the fixture leaves in place

The product rank stays 8. `tau_residual` stays 0.55. `tau_kappa` stays 0.35. Checksum coverage is unchanged. The character-trigram unanchored reject stays locked. No reconstructor, logit lens, SAE, or hub runtime is added. Adaptive rank is not implemented. The production clustering call is the same function.

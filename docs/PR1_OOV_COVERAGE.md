# PR-1 OOV coverage

Status: `RESEARCH_ONLY`. Baseline commit `9040bb4`, [PR1_ENCODER_ISOLATION.md](PR1_ENCODER_ISOLATION.md). Entering verdict: `UNRESOLVED`. Entering recommendation: `NONE`. Authority: `RUNTIME_NOT_AUTHORIZED`.

This note asks whether inexpensive, train-independent text features can stop eval rows from collapsing to the zero vector and still keep distinctions that the train-only word-sentence encoder loses on unseen text. Clustering stays production farthest-first squared-Euclidean k-means, fit on training representations only. The cluster-to-topic map is fit on training labels only. Eval rows are assigned to those frozen centroids. `word_sentence_svd`, `learn_pack`, `certify`, `tau_residual`, `tau_kappa`, and the package version stay as they are. No reconstructor, logit lens, SAE, or hub runtime is added. A hashed arm that cleared the rule would still leave production in place.

The trial decision is **UNRESOLVED**. The recommendation is **NONE**. `RUNTIME_NOT_AUTHORIZED`. Production was not replaced. The decision is not a pass of the encoder-isolation purity rule.

## Arms

| arm | features | what is fit |
|---|---|---|
| 1 `tfidf_svd` | train-only TF-IDF and the rank-8 SVD from the encoder-isolation trial | vocabulary, document frequency, IDF, and the basis, on training rows only |
| 2 `char_hash` | `hashed_ngram_vector`, order 3, width 32, per-row L2 | nothing |
| 3 `word_hash` | signed SHA-256 buckets of the fixed regex `[a-z0-9']+`, width 32, per-row L2 | nothing |
| 4 `combined_hash` | per-row sum of arms 2 and 3, then per-row L2 | not scored |

Width 32 and character order 3 were fixed in `oov_coverage.py` before any hashed purity was computed. They are the width and order `hashed_ngram_vector` already uses. The word tokenizer is that regex on the lowercased row. It is not a vocabulary built from counts, and it does not drop the Glasgow stop list. No IDF and no SVD are fit for arms 2 or 3, on train or on eval. An eval token absent from the training vocabulary still contributes a bucket.

Arm 1 is the control. It was recomputed with `fit_train_only_encoder` and the same label seed 0 and row seed 1. The inductive hits, base rates, zero-row counts, control hits, and token-absence counts match the encoder-isolation table. The split is the frozen file-order rule, index modulo 5 equals 4. The fingerprint sha256 is `55f561c832ef47d453827b3fc9704cf4d98e66d0bda2e4ff4aa9603c5923c20e`. The run reproduced it.

## Admission rule

These constants sit above `run_oov_coverage`. The decision functions do not embed text. Eliminating zero vectors is not enough to recommend an arm.

An arm is `RECOMMENDED` for a later isolated encoder PR only when every generalization fixture meets every row below. Otherwise, when both controls fall, the arm is `EXPERIMENTAL`. When a control does not fall, the arm is `UNRESOLVED`.

| constant | requirement |
|---|---|
| `PURITY_ABOVE_ARM1_GATE` | eval size-weighted purity under the train-only map is strictly greater than arm 1's inductive eval purity on that fixture |
| `PURITY_ABOVE_BASE_GATE` | that purity is strictly greater than the fixture's eval majority base rate |
| `LABEL_CONTROL_GATE` | shuffled-train-label map purity is strictly below the real eval purity |
| `ROW_CONTROL_GATE` | eval-vector permutation purity is strictly below the real eval purity |
| `ZERO_RATE_GATE` | eval zero-vector rate is strictly below arm 1 |
| `DISTINCTION_GATE` | pairwise distinction retention on distinct eval texts is at least 0.95 |
| `RUNTIME_GATE` | median feature-extraction plus fit time is at most 5× arm 1's embed median on that fixture |

Pairwise distinction retention is the fraction of distinct eval texts whose vector equals no other distinct eval text. Shared zero vectors are one vector. On these eval sizes the 0.95 bar requires every distinct eval text to have its own vector: one shared vector leaves at most 8/10, 7/9, or 14/16, and each of those is under 0.95.

The label control permutes the training topic list with seed 0, fits the majority map on those shuffled training labels, and scores the real eval topics. The row control permutes eval vectors with seed 1, assigns them to the real training centroids, and scores the real eval topics with the real training map. Centroids are not refit. The map is not refit on eval labels. A tie is not a fall. Seeds are the generalization trial's seeds.

Trial decision, from arm statuses that are already computed:

- An arm that is `RECOMMENDED` is named, and the runtime authority stays `RUNTIME_NOT_AUTHORIZED`.
- A hashed arm whose controls do not fall makes the trial `UNRESOLVED`.
- When every hashed arm's controls fall and no arm clears the rule, the decision is `NO_ADMISSION`.
- When those arms disagree about the failure shape, the decision stays `UNRESOLVED`.
- `COVERAGE_WITHOUT_DISTINCTION` names only the shape where the zero-vector rate fell and pairwise distinction still missed 0.95. It is not a pass.

## Per-fixture results

Hits are size-weighted purity under the train-only map. The base rate is the eval majority count. Arm 1 is the recomputed inductive control.

| fixture | arm 1 | base | char hash | word hash |
|---|---|---|---|---|
| uneven | 5/10 | 5/10 | 5/10 | 5/10 |
| overlapping | 3/9 | 2/9 | 3/9 | 1/9 |
| ambiguous | 4/9 | 3/9 | 2/9 | 1/9 |
| balanced | 3/16 | 2/16 | 3/16 | 2/16 |

Char hash ties arm 1 on uneven, overlapping, and balanced, and falls below both arm 1 and the base rate on ambiguous (2/9 against 4/9 and 3/9). Word hash ties arm 1 on uneven and is below arm 1 on the other three fixtures. Word hash is also below the base rate on overlapping and ambiguous, and it ties the base rate on balanced. No hashed fixture is strictly above arm 1.

Zero-vector rate. A row is zero when every coordinate is 0.

| fixture | arm 1 | char hash | word hash |
|---|---|---|---|
| uneven | 3/10 | 0/10 | 0/10 |
| overlapping | 3/9 | 0/9 | 0/9 |
| ambiguous | 4/9 | 0/9 | 0/9 |
| balanced | 8/16 | 0/16 | 0/16 |

Both hash arms cut the zero-vector rate on every fixture.

OOV coverage. For arm 1 this is the fraction of eval content-token occurrences whose type is absent from the training vocabulary, recomputed with the train-only encoder. The absence counts match the encoder-isolation table: 56/72, 52/66, 44/58, and 74/94. Absent types are 55, 51, 43, and 72. Nonzero eval rows are 7/10, 6/9, 5/9, and 8/16. Hash arms fit no vocabulary. Every eval row is nonzero: 10/10, 9/9, 9/9, and 16/16 for each hash arm.

Pairwise distinction retention on distinct eval texts. The second number is the count of distinct vectors.

| fixture | arm 1 | char hash | word hash |
|---|---|---|---|
| uneven | 0.7000 (8 vectors / 10 texts) | 1.0000 (10/10) | 1.0000 (10/10) |
| overlapping | 0.6667 (7/9) | 1.0000 (9/9) | 1.0000 (9/9) |
| ambiguous | 0.5556 (6/9) | 1.0000 (9/9) | 1.0000 (9/9) |
| balanced | 0.5000 (9/16) | 1.0000 (16/16) | 1.0000 (16/16) |

Arm 1's shared zeros are the collisions. The seven, six, five, and eight nonzero eval rows each have their own vector. Both hash arms keep a distinct vector for every distinct eval text, so both clear 0.95 on every fixture.

Prototype residual is mean squared Euclidean distance to the assigned centroid. The eval residual uses the frozen training centroids.

| fixture | arm 1 train | arm 1 eval | char train | char eval | word train | word eval |
|---|---|---|---|---|---|---|
| uneven | 0.3334 | 0.3621 | 0.4587 | 0.7819 | 0.4923 | 0.6826 |
| overlapping | 0.1367 | 0.3549 | 0.4661 | 0.6525 | 0.4777 | 0.6626 |
| ambiguous | 0.3314 | 0.5619 | 0.4120 | 0.7327 | 0.4325 | 0.7240 |
| balanced | 0.2491 | 0.3756 | 0.5566 | 0.6790 | 0.5547 | 0.6702 |

Training purity under the same map: arm 1 is 26/40, 27/39, 22/39, 33/64. Char hash is 23/40, 18/39, 18/39, 17/64. Word hash is 22/40, 16/39, 17/39, 20/64.

## Controls

| fixture | char real | char label | char row | char both fall | word real | word label | word row | word both fall |
|---|---|---|---|---|---|---|---|---|
| uneven | 5/10 | 5/10 | 5/10 | no | 5/10 | 5/10 | 5/10 | no |
| overlapping | 3/9 | 1/9 | 2/9 | yes | 1/9 | 3/9 | 0/9 | no |
| ambiguous | 2/9 | 2/9 | 2/9 | no | 1/9 | 2/9 | 3/9 | no |
| balanced | 3/16 | 3/16 | 1/16 | no | 2/16 | 1/16 | 1/16 | yes |

Char hash ties its own purity on uneven (both controls), on ambiguous (both controls), and on balanced (the label control). Word hash ties on uneven. On overlapping the shuffled training map scores 3/9 against a real score of 1/9, and on ambiguous both controls score above 1/9. A tie is not a fall, and a control that scores above the real purity is not a fall. Neither hashed arm has both controls strictly below the real purity on every fixture. Each hashed arm is therefore `UNRESOLVED`. The purity misses would have made the arm `EXPERIMENTAL` only if the controls had fallen.

Arm 1's own controls are the published ones: uneven 5 and 4 against 5, overlapping 0 and 1 against 3, ambiguous 1 and 1 against 4, balanced 2 and 1 against 3. Uneven's label control still ties. That is the control arm, not a candidate.

## Synthetic fixtures

These strings are not rows of the generalization files.

Repeated input is `qxvblort mzzpfkye` twice. Char hash, word hash, and the combined helper each return bitwise-identical nonzero rows. The train-only projection of that string, fit on the one string and applied twice, is bitwise identical. `word_sentence_svd` on a two-row matrix of that same string differs by `1.11e-16` in one coordinate after the truncated SVD and the row L2 norm. That dust is left as it is. Production was not changed to force a bitwise match.

Unseen-token trains on four sentences whose content tokens are harbor, kiln, orchard, and apiary vocabulary, and evaluates four strings built from `qxvblort`, `mzzpfkye`, `jjwdnquv`, `htrgclox`, `bnvysaei`, `wplkdzuc`, `feyomqar`, and `tcuhjnsx`. No content token and no regex token is shared. Arm 1's train-only projection is the zero vector on all 4 eval rows. Those four distinct strings collapse to one vector, and the vector is zero, so this is a coverage failure and a distinction failure together. Char hash and word hash each return 4 nonzero rows and retention 1.0.

Nonsense-text is those eight tokens, each a separate string, with no shared token and no topic. Projected with the unseen-token training encoder, arm 1 maps all 8 to the zero vector. That is a total collapse. Char hash and word hash each keep 8 distinct nonzero vectors. Retention is 1.0. A nonzero vector that every distinct string shared would have been a distinction failure. That did not happen.

Hash collision walks `aaa`, `aab`, `aac`, … in product order, at most 4096 tokens, and stops at the first shared bucket. The word hash finds `aaa` and `aab` in bucket 24. The character hash finds `aaa` and `aab` in bucket 13. The vectors of that pair differ for both hashes, so a shared bucket was detected and the strings stayed distinct. `aaa` and `aca` share neither a word bucket nor a character bucket. A search that found no repeated bucket would have been recorded as impossible inside that bound. Both hashes found one. Arm 1 does not use these buckets. The same three strings, projected from the unseen-token training encoder, are three copies of the zero vector.

## Why arm 4 did not run

Arm 4 runs only after arms 2 and 3 have been scored, and only when one of them cuts the zero-vector rate on every generalization fixture while the other retains a synthetic distinction the first loses. Repeated input is not that distinction: identical vectors there are required.

Both hash arms cut the zero-vector rate on every generalization fixture. On the synthetic fixtures, both retain every distinction: nonsense retention 1.0, unseen-token retention 1.0, and the planted pair does not collapse to one vector. The precondition sentence is:

> Arm 4 is not justified. char_hash zero-rate cut on every fixture is true. word_hash zero-rate cut on every fixture is true. Distinctions char_hash loses and word_hash retains: none. Distinctions word_hash loses and char_hash retains: none.

That sentence does not justify a run. The note does not contain `ARM4_JUSTIFICATION_PRESENT`. `score_arm4` returns before it embeds a generalization row when that marker and this sentence are absent. Arm 4 was not scored. There is no combined-hash purity, residual, or cost on the frozen split.

## Costs

Timed on this host after the scores were fixed. Five samples. The arm 1 gate time is one train-only fit plus the eval projection, the same embed the encoder-isolation trial timed. Fit time is that fit alone. Feature-extraction time for arm 1 is the frozen eval projection. Hash arms fit no parameters, so fit time is 0 and the gate time is the median of hashing the training rows and the eval rows. Cluster time is separate and is not the 5× gate: five samples, each the mean of 100 production fits. Peak is the median `tracemalloc` peak, Python allocator traffic, not resident set size. Serialized size is the training parameter pack plus one eval matrix. The hash pack is 0 bytes because nothing was fit. The eval matrix is `n_eval × 32 × 8` bytes. Arm 1's pack is the vocabulary UTF-8, document-frequency counts, IDF, right singular vectors, and singular values.

| fixture | arm | extract ms | fit ms | extract+fit ms | cluster ms | extract+fit peak | cluster peak | pack bytes | eval bytes | serialized |
|---|---|---|---|---|---|---|---|---|---|---|
| uneven | arm 1 | 0.072 | 3.363 | 3.513 | 0.339 | 500151 | 68440 | 21192 | 640 | 21832 |
| uneven | char | 2.441 | 0 | 2.441 | 0.289 | 50488 | 222872 | 0 | 2560 | 2560 |
| uneven | word | 0.590 | 0 | 0.590 | 0.300 | 31872 | 222616 | 0 | 2560 | 2560 |
| overlapping | arm 1 | 0.064 | 3.070 | 3.202 | 0.249 | 477059 | 66704 | 20235 | 576 | 20811 |
| overlapping | char | 2.441 | 0 | 2.441 | 0.284 | 49176 | 221264 | 0 | 2304 | 2304 |
| overlapping | word | 0.600 | 0 | 0.600 | 0.297 | 31104 | 219728 | 0 | 2304 | 2304 |
| ambiguous | arm 1 | 0.069 | 3.087 | 3.068 | 0.336 | 458629 | 66640 | 19151 | 576 | 19727 |
| ambiguous | char | 2.464 | 0 | 2.464 | 0.375 | 49176 | 219216 | 0 | 2304 | 2304 |
| ambiguous | word | 0.643 | 0 | 0.643 | 0.293 | 31104 | 219216 | 0 | 2304 | 2304 |
| balanced | arm 1 | 0.127 | 7.611 | 7.807 | 0.686 | 942964 | 107352 | 30069 | 1024 | 31093 |
| balanced | char | 4.357 | 0 | 4.357 | 0.650 | 82168 | 277336 | 0 | 4096 | 4096 |
| balanced | word | 0.918 | 0 | 0.918 | 0.512 | 50880 | 273752 | 0 | 4096 | 4096 |

The extract-plus-fit column is the median of a combined call. The extract column and the fit column are medians of separate calls, so their sum is not the combined median. On ambiguous, arm 1's combined median is 3.068 ms and the fit-only median is 3.087 ms. Char-hash gate time over arm 1's combined median is 0.69, 0.76, 0.80, and 0.56. Word-hash is 0.17, 0.19, 0.21, and 0.12. Every ratio is under 5. The runtime gate is met. It is not the gate that blocks admission. The 32-dimensional cluster peak is larger than arm 1's because the assignment tensor is wider. Wall clock for this timed run was 3.442 s.

Numerical library, from `numpy.show_config` and `/proc/self/maps`: NumPy 1.26.4, system BLAS loaded as `libblas.so.3.12.0`, LAPACK loaded as `liblapack.so.3.12.0`. Threads were pinned to one. The measurement git revision is `9040bb4405721db2ffd9e85ec08c9f1f3474be1b`. A later commit records this note. The purity hits do not depend on that later hash. The millisecond table is this host's measurement.

## Verdict

Both hash arms remove every eval zero vector on the frozen split, and both give every distinct eval text its own vector. Char hash ties arm 1's purity on three fixtures and loses on ambiguous, including a loss to the base rate. Word hash does not beat arm 1 on any fixture and loses to the base rate on two. The shuffled-label and shuffled-row controls do not both fall on every fixture. Uneven ties for both arms. Ambiguous ties for char hash and rises for word hash. Word hash's shuffled training map on overlapping scores above the real purity.

The trial decision is **UNRESOLVED**.

`NO_ADMISSION` would require the controls to fall and no arm to clear the rule. The controls did not fall, so that decision is not available. `COVERAGE_WITHOUT_DISTINCTION` would describe a zero-vector drop whose distinct texts still shared vectors. Pairwise retention on the generalization fixtures is 1.0, and the synthetic nonsense strings do not collapse, so that is not the shape of this run. The two facts that do hold side by side are a real coverage gain and a real purity miss, with controls that do not fall. That mix stays `UNRESOLVED`. The recommendation is **NONE**.

`RUNTIME_NOT_AUTHORIZED`. An arm that had cleared the rule would have been named for a later isolated encoder PR and would still not have replaced production. No arm cleared it. Production was not replaced.

## What this leaves in place

The product rank stays 8. `tau_residual` stays 0.55. `tau_kappa` stays 0.35. `certify` stays the structural contract. `learn_pack` is not called. Package version stays 0.7.0. Production k-means stays farthest-first initialization, squared Euclidean assignment, and a mean update. `word_sentence_svd` is unchanged. The hash arms are a probe.

## Remaining uncertainties

Eval sets are 10, 9, 9, and 16 rows. Char hash's tie with arm 1 at 5/10, 3/9, and 3/16 has no room for a strict excess. One hit would change the comparison.

The planted bucket collision does not collapse the hash vectors. A different pair, chosen for equal signed buckets rather than the first shared index, could. This search stops at the first shared index in lexicographic order. It does not hunt for a total collapse after seeing purity.

`word_sentence_svd` on two copies of one string leaves a `1e-16` coordinate of SVD dust. The train-only projection of that string does not. The hash functions do not. The production function was not edited to remove the dust.

The singular spectrum of the arm 1 TF-IDF matrices is the same tight spectrum the encoder-isolation note describes. These purity hits matched that note on the single-thread reference BLAS/LAPACK 3.12 measurement.

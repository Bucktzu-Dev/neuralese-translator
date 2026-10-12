# PR-1 encoder isolation

Status: `RESEARCH_ONLY`. Baseline commit `12407d3`. Entering verdict of the evaluation-control audit: `UNRESOLVED`. Entering recommendation: `NONE`. Authority: `RUNTIME_NOT_AUTHORIZED`.

This note asks whether the symbolic distinctions on the frozen generalization fixtures survive when every corpus statistic in the word-sentence encoder is fit on training rows only. Clustering stays production farthest-first squared-Euclidean k-means. The four-candidate bake-off stays closed. `word_sentence_svd`, `learn_pack`, `certify`, `tau_residual`, `tau_kappa`, and the package version stay as they are. The 0.90 full-matrix agreement figure from the evaluation-control audit is not read and is not a gate.

The trial verdict is **UNRESOLVED**. The recommendation is **NONE**. `RUNTIME_NOT_AUTHORIZED`. Production was not replaced.

## What is fit, and what is per row

Deterministic features come from the row in hand. `content_tokens` lowercases that row, drops the fixed Glasgow stop list (`FUNCTION_WORDS`), and drops one-character tokens. Raw counts are the counts in that row. After IDF weights are applied, the row's own L2 norm scales that row. After the projection, the row's own L2 norm scales it again. Those two normalizations use no corpus moment.

Learned parameters are fit on the training rows and then frozen:

| parameter | training definition |
|---|---|
| vocabulary | sorted content tokens that occur in at least one training row |
| document frequency | training rows that contain the token |
| IDF | `ln(N_train / df) + 1` |
| SVD basis | truncated right singular vectors of the training TF-IDF |
| singular values | the values inside `U * S` |
| sign pivots | largest-magnitude coordinate of each training left singular vector, lowest index on a tie |

Centering is absent from `word_sentence_svd`. There is no centering statistic to fit.

An eval token absent from the training vocabulary adds nothing and does not create a column. Document frequency, IDF, and the basis are not rebuilt after an eval row is seen. The inductive arm calls `word_sentence_svd` on the training texts only, as a check that the frozen basis reproduces that function, and then projects eval rows with `X @ Vt.T`. On the training matrix that product equals `U * S`. A second multiply by the singular values would scale each coordinate again. Rows whose projected Euclidean norm is at or below `1e-8` stay the zero vector, so roundoff is not renormalized into a direction. That floor is part of the inductive projection. It is not applied inside `word_sentence_svd`. The matrix k-means sees for the training rows is the production function's own output on those texts.

The split is the frozen file-order rule, index modulo 5 equals 4. Eval ids, in fixture order:

| fixture | train | eval | eval observation ids |
|---|---|---|---|
| uneven | 40 | 10 | `uneven-005`, `uneven-010`, `uneven-015`, `uneven-020`, `uneven-025`, `uneven-030`, `uneven-035`, `uneven-040`, `uneven-045`, `uneven-050` |
| overlapping | 39 | 9 | `overlapping-005` through `overlapping-045`, step 5 |
| ambiguous | 39 | 9 | `ambiguous-005` through `ambiguous-045`, step 5 |
| balanced | 64 | 16 | `balanced-005` through `balanced-080`, step 5 |

The fingerprint payload is those ids, one fixture per line, `name:id,id,...`. Its sha256 is `55f561c832ef47d453827b3fc9704cf4d98e66d0bda2e4ff4aa9603c5923c20e`. The run reproduced that hash. Shared observation ids across the split: 0. Shared texts: 0. Duplicate ids inside each file: 0. Duplicate texts inside each file: 0.

## Gates

These constants are the admission rule. They are module-level text, written above `run_encoder_isolation`, and the verdict function does not compute a purity.

| constant | requirement |
|---|---|
| `COVERAGE_GATE` | every eval row has a finite train-only vector |
| `PURITY_GATE` | eval size-weighted purity under the train-only map is strictly greater than that fixture's eval majority-class base rate |
| `LABEL_CONTROL_GATE` | shuffled-train-label map purity is strictly below the real inductive eval purity |
| `ROW_CONTROL_GATE` | eval-embedding permutation purity is strictly below the real inductive eval purity |
| `LEAKAGE_GATE` | no shared ids or texts between train and eval |

`INDUCTIVE_PASS` requires every fixture to meet every one of those. `INDUCTIVE_FAIL` requires the controls to fall and at least one fixture to miss the base-rate gate. A control that does not fall, a fingerprint that does not reproduce, or partial coverage for a reason outside `complete` and `non_finite_coordinates` is `UNRESOLVED`. `INDUCTIVE_FAIL` is not issued in that case. A zero vector is finite. It counts as covered, and the zero-row count is reported beside the fraction.

The label control permutes the training topic list with seed 0, fits the majority map on those shuffled training labels, and scores the real eval topics. The row control permutes eval embedding rows with seed 1, assigns them to the real training centroids, and scores the real eval topics with the real training map. Seeds are the generalization trial's seeds. Centroids are not refit. The map is not refit on eval labels. The tie break is `clustering_isolation._majority`.

Analytical inequalities run before any fixture is embedded. Two isotropic blobs, data seed 11, dimension 8, gap 30, noise scale 1, 200 training rows and 100 eval rows in each class, production k-means with k = 8. The duplicate detector is checked on an empty pair and on a planted shared id and a planted shared text. A failure of those inequalities returns `UNRESOLVED` and does not embed the fixtures.

| score | hits |
|---|---|
| real eval, train map | 200/200 |
| eval majority base rate | 100/200 |
| shuffled train-label map | 95/200 |
| eval embedding permutation, real map | 110/200 |
| empty cross-split ids and texts | 0 and 0 |
| planted shared id, planted shared text | detected, detected |

200 > 100, 95 < 200, and 110 < 200. The embedding permutation hit count is the same draw the evaluation-control audit scored at 110/200. The label control here shuffles training labels, so 95/200 is this definition.

## Per-fixture results

Inductive purity uses the training majority map. The transductive column is the control in the next section. It is listed here so each fixture can be read before the verdict. Residuals are mean squared Euclidean distance to the assigned centroid. The inductive eval residual is distance to the frozen training centroids.

| fixture | inductive eval | base rate | transductive eval | inductive train | train residual | eval residual |
|---|---|---|---|---|---|---|
| uneven | 5/10 | 5/10 | 6/10 | 26/40 | 0.3334 | 0.3621 |
| overlapping | 3/9 | 2/9 | 4/9 | 27/39 | 0.1367 | 0.3549 |
| ambiguous | 4/9 | 3/9 | 7/9 | 22/39 | 0.3314 | 0.5619 |
| balanced | 3/16 | 2/16 | 3/16 | 33/64 | 0.2491 | 0.3756 |

Coverage and tokens. Absence is the fraction of eval content-token occurrences whose type is missing from the training vocabulary. The type fraction is beside it. Overlap is the number of eval rows with at least one training-vocabulary token.

| fixture | finite rows | zero rows | occurrence absence | type absence | rows with a train token |
|---|---|---|---|---|---|
| uneven | 10/10 | 3 | 56/72 | 55/69 | 9 |
| overlapping | 9/9 | 3 | 52/66 | 51/63 | 8 |
| ambiguous | 9/9 | 4 | 44/58 | 43/55 | 5 |
| balanced | 16/16 | 8 | 74/94 | 72/90 | 10 |

Every eval row is finite. Coverage is complete on every fixture. Zero rows are part of that complete coverage. On ambiguous the four zero rows are the four rows with no training-vocabulary token. On uneven, overlapping, and balanced, the zero-row count is higher than the count of rows with no training token, so some rows that do contain a training token fall under the `1e-8` projection floor.

Assignment stability is adjusted Rand between the transductive labels and the inductive labels. The correlation is Pearson correlation of pairwise squared distances on the training rows.

| fixture | train ARI | eval ARI | distance correlation |
|---|---|---|---|
| uneven | 0.2971 | 0.1000 | 0.4083 |
| overlapping | 0.8415 | 0.2500 | 0.8080 |
| ambiguous | 0.4508 | 0.0471 | 0.5170 |
| balanced | 0.3009 | 0.0238 | 0.4131 |

Uneven's inductive eval purity equals the base rate, 5/10. The gate requires a strict excess, so that fixture misses it. Overlapping, ambiguous, and balanced each clear the base rate by one hit: 3 > 2, 4 > 3, and 3 > 2.

Training rows whose pre-renorm projection norm is at or below `1e-8`: uneven 11 of 40, overlapping 3 of 39, ambiguous 8 of 39, balanced 18 of 64. `word_sentence_svd` still L2-normalizes those training rows. The inductive k-means uses that production output. The floor is applied on the inductive projection used for eval rows.

## Transductive control

Label: `TRANSDUCTIVE`. Encoder: `word_sentence_svd` on every row of the fixture. Clustering: production farthest-first k-means on the training rows of that full matrix. Eval rows change the vocabulary, the IDF, and the basis. They do not enter the centroid update. The eval purity below uses the majority map of those training labels. This block is not an inductive result and it is not a pass gate.

| fixture | transductive train | transductive eval, train map | self-majority on the eval slice | train residual |
|---|---|---|---|---|
| uneven | 26/40 | 6/10 | 7/10 | 0.2275865734 |
| overlapping | 27/39 | 4/9 | 6/9 | 0.1337722038 |
| ambiguous | 26/39 | 7/9 | 8/9 | 0.2567684914 |
| balanced | 30/64 | 3/16 | 9/16 | 0.2111005807 |

The train residuals and the train-map eval hits match the production column of the evaluation-control audit. The self-majority column matches that audit's original eval scorer. Self-majority lets the eval labels name the eval clusters. It is recorded so the two scorers stay distinct. It is not the control score.

On uneven and balanced, the inductive training embedding matches `word_sentence_svd` bitwise with the evaluation-control helper's training embedding, and the published train adjusted Rand and distance correlation are reproduced (uneven train ARI 0.2971403038, correlation 0.4082952119; balanced train ARI 0.3008570540, correlation 0.4130501262). On overlapping and ambiguous the helper's training embedding differs from `word_sentence_svd` by more than 1 in coordinate. The singular values of those TF-IDF matrices sit close together, so a last-bit difference in the helper's rebuilt matrix moves the basis. This run's training rows are `word_sentence_svd`.

## Negative controls

| fixture | real inductive eval | shuffled train-label map | eval embedding permutation | both strictly below |
|---|---|---|---|---|
| uneven | 5/10 | 5/10 | 4/10 | no |
| overlapping | 3/9 | 0/9 | 1/9 | yes |
| ambiguous | 4/9 | 1/9 | 1/9 | yes |
| balanced | 3/16 | 2/16 | 1/16 | yes |

Overlapping, ambiguous, and balanced drop under both controls. Uneven's label control ties the real score at 5/10. A tie is not a fall. The row control on uneven drops to 4/10. Cross-split duplicate ids and texts are empty on every fixture, and empty inside each file.

The label control did not fall on every fixture. The verdict is therefore `UNRESOLVED`. It is not `INDUCTIVE_FAIL`. Uneven also misses the base-rate gate, and that miss would be the fail condition only together with controls that fall.

## Diagnostic

The diagnostic runs only as an explanation when inductive eval purity misses the base-rate gate. It is not a gate. Cutoffs, fixed with the gates: largest principal angle above 15 degrees, or orthogonal Procrustes relative residual above 0.50, counts as a moved training subspace. Mean matched centroid displacement above 0.50 counts as moved centroids. An eval row is far when its squared distance to the nearest training centroid exceeds the farthest training row's squared distance to its own centroid, and the fixture flag fires when that fraction exceeds 0.50. A single cause is named only when exactly one of those three flags fires. Otherwise the cause is `UNRESOLVED`.

Uneven is the only fixture that misses the base-rate gate.

| measurement | uneven |
|---|---|
| largest principal angle | 12.36° |
| unaligned mean cosine | 0.314 |
| aligned mean cosine | 0.716 |
| Procrustes relative residual | 0.753 |
| mean matched centroid L2 | 0.593 |
| eval rows outside the train radius | 0.10 |
| subspace flag | yes, from the Procrustes residual |
| centroid flag | yes |
| far flag | no |
| cause | `UNRESOLVED` |

The principal angle stays under 15°. The aligned clouds still disagree: relative residual 0.753, aligned mean cosine 0.716, pairwise distance correlation 0.408. Centroids also move, by 0.593. One eval row in ten lies outside the training radius. Two of the three flags fire. The numbers do not isolate a single cause.

Overlapping, ambiguous, and balanced clear the base-rate gate, so their cause is `NOT_APPLICABLE`. Their geometry is still in the tables above. Ambiguous and balanced have Procrustes residuals 0.665 and 0.792. Overlapping's is 0.385, with train adjusted Rand 0.8415.

## Costs

Timed on this host after the scores were fixed. Clustering median: 5 samples, each the mean of 100 production fits, median at the middle sorted sample. Embed median: 5 samples of one train-only fit plus eval projection, including the check against `word_sentence_svd` on the training texts. Peak is the median `tracemalloc` peak of those calls. That is Python allocator traffic, not resident set size. None of these numbers is a gate.

| fixture | embed median | cluster median | embed peak | cluster peak |
|---|---|---|---|---|
| uneven | 3.27 ms | 0.320 ms | 500247 bytes | 68440 bytes |
| overlapping | 3.15 ms | 0.250 ms | 477251 bytes | 66704 bytes |
| ambiguous | 3.03 ms | 0.318 ms | 458725 bytes | 66640 bytes |
| balanced | 7.02 ms | 0.652 ms | 943060 bytes | 107352 bytes |

The sum of the four embed medians is 16.48 ms. The sum of the four cluster medians is 1.54 ms. Wall clock for the timed run, analytical controls included, was 1.33 s. The largest tracemalloc peak was the balanced embed, 943060 bytes.

Numerical library, from `numpy.show_config` and `/proc/self/maps`: NumPy 1.26.4, system BLAS loaded as `libblas.so.3.12.0`, LAPACK loaded as `liblapack.so.3.12.0`. `show_config` names the BLAS library `blas` and the LAPACK build 1.26.4. Threads were pinned to one. The measurement git revision is `12407d36e8d7972cf922910a8148c12e73a0d772`. A later run records whatever `HEAD` is; the scores above do not depend on that hash.

## Verdict

Analytical inequalities passed. The split fingerprint reproduced. Coverage is complete on every fixture. Three fixtures beat their eval majority base rate by one hit, and their controls fall. Uneven ties its base rate at 5/10, and its shuffled training map also scores 5/10.

The trial verdict is **UNRESOLVED**.

`INDUCTIVE_PASS` would mean every fixture had a finite eval vector, inductive eval purity strictly above that fixture's majority base rate, both controls strictly below that purity, and an empty cross-split of ids and texts. Uneven misses the strict base-rate excess, and its label control does not fall. `INDUCTIVE_FAIL` would mean the controls fell and at least one fixture missed the base-rate gate. The label control did not fall, so that verdict is not available. The recommendation is **NONE**.

`RUNTIME_NOT_AUTHORIZED`. A pass would not have replaced production. This run did not pass. Production was not replaced.

## What this leaves in place

The product rank stays 8. `tau_residual` stays 0.55. `tau_kappa` stays 0.35. `certify` stays the structural contract. `learn_pack` is not called. Package version stays 0.7.0. Production k-means stays farthest-first initialization, squared Euclidean assignment, and a mean update. `word_sentence_svd` is unchanged. The four frozen clustering candidates are not rerun. The inductive purity numbers are a probe.

## Remaining uncertainties

Eval sets are 10, 9, 9, and 16 rows. One hit changes the comparison to the base rate. Uneven's real score of 5/10 has no room for a control to tie and still count as a fall.

Most eval content-token occurrences are words the training split never sees: 56/72, 52/66, 44/58, and 74/94. An eval row in the train-only basis is the overlap that remains. Several eval rows are the zero vector. A zero vector is a finite vector under the coverage gate and a weak observation under the purity gate.

The singular spectrum of these TF-IDF matrices is tight. Another BLAS, or a last-bit change in the matrix, can rotate the truncated basis. The numbers above are the single-thread reference BLAS/LAPACK 3.12 measurement.

The diagnostic cutoffs are a reporting rule. They do not certify a mechanism. On the one fixture that misses the base rate, the aligned training cloud and the centroids both move, and the cause stays `UNRESOLVED`.

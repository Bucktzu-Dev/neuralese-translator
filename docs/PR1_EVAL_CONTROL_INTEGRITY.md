# PR-1 evaluation-control integrity

Status: integrity audit of the clustering-generalization trial. Baseline commit `75f26cb`. Entering verdict of that trial: `UNRESOLVED`. This note does not change production clustering, the SVD, `certify`, `learn_pack`, the package version, or the four frozen candidates. It does not retune their seeds. The admission thresholds stay where that trial froze them: eval size-weighted purity strictly above production, train prototype residual at most `1.10×`, clustering time at most `2×`, and the advantage required on every fixture.

The audit verdict is about the evaluation method. It is not a claim that semantic clustering quality is established. Structural certification, numerical fidelity, semantic quality, and runtime authority stay separate. `RUNTIME_NOT_AUTHORIZED` in every verdict this note can return.

## What is permuted

The controls live in `neuralese.clustering_generalization._negative_controls`. Scoring of a partition lives in `_score_assignment`. Both are quoted here as they stand. This audit does not edit them. The generalization tests still lock the hit counts those functions produced.

`_score_assignment` builds the cluster-to-label map from the topics it was just given:

```python
def _score_assignment(labels: np.ndarray, topics: Sequence[str]) -> dict:
    """Size-weighted majority purity. Tie break matches the isolation probe."""
    labels = np.asarray(labels)
    if int(labels.size) != len(topics):
        raise ValueError("labels and topics differ in length")
    majority_of: Dict[int, Optional[str]] = {}
    hits = 0
    for code in np.unique(labels):
        members = np.flatnonzero(labels == code)
        member_topics = [topics[int(index)] for index in members]
        majority, count = _majority(member_topics)
        majority_of[int(code)] = majority
        hits += int(count)
```

The majority topic of each cluster is the mode of `topics` inside that cluster. Hits are the rows whose topic equals that mode. The same array is the map and the score.

`_negative_controls` then does two permutations.

Label shuffle, seed 0. The permuted array is the topic sidecar, over every row of the file. Observations, embeddings, row indices of the assignment, and cluster ids stay put. Production train labels and production eval labels are reused.

```python
label_rng = np.random.default_rng(LABEL_SHUFFLE_SEED)
permutation = label_rng.permutation(len(topics))
shuffled_topics = [topics[int(index)] for index in permutation]
train_topics = [topics[int(index)] for index in train]
train_topics_shuffled = [shuffled_topics[int(index)] for index in train]
eval_topics = [topics[int(index)] for index in test]
eval_topics_shuffled = [shuffled_topics[int(index)] for index in test]

real_train = _score_assignment(production["train_labels"], train_topics)
label_train = _score_assignment(production["train_labels"], train_topics_shuffled)
real_eval = _score_assignment(production["eval_labels"], eval_topics)
label_eval = _score_assignment(production["eval_labels"], eval_topics_shuffled)
```

`shuffled_topics[i]` is `topics[permutation[i]]`. The eval slice is those strings at the eval indices. `label_eval` refits the majority map on the shuffled eval topics. The geometry of the production eval partition is intact. The map is chosen after the shuffle, from the shuffled labels themselves.

Row shuffle, seed 1. The permuted array is the embedding matrix. Topic strings stay on file order. Production k-means is refit on the shuffled train rows. Eval rows of the shuffled matrix are assigned to those new centroids. Purity is again `_score_assignment` on the eval labels of that new partition, so the map is the majority topic inside each new eval cluster.

```python
row_rng = np.random.default_rng(ROW_SHUFFLE_SEED)
order = row_rng.permutation(matrix.shape[0])
shuffled_matrix = matrix[order]
spec = _production_spec()
shuffled_train_labels, centroids, _ = _fit_algorithm(shuffled_matrix[train], spec)
row_train = _score_assignment(shuffled_train_labels, train_topics)
shuffled_eval_labels = _assign(shuffled_matrix[test], centroids, "euclidean")
row_eval = _score_assignment(shuffled_eval_labels, eval_topics)
```

`shuffled_matrix[i]` is the embedding that used to sit at `order[i]`. The label scored at index `i` is still `topics[i]`. The embedding and the label are different rows.

Seed 1 on these four files has 1, 0, 0, and 1 fixed points. Train positions whose source row happens to carry the same topic string are 13/40, 7/39, 9/39, and 2/64. The permutation does not carry labels along with embeddings.

## Which of the three failure modes happened

Three descriptions were in play. Two of them are what the eval controls did. One of them is not.

The label shuffle is scored with a cluster-to-label map fit on the shuffled eval labels. That happened. It is not a disruption of a fixed map. Uneven's eval label shuffle rose from production 7/10 to 8/10. Ambiguous's eval label shuffle tied at 8/9. Both comparisons call `_score_assignment` on the shuffled eval topics and the unchanged production eval cluster ids.

The row shuffle does not leave labels aligned to the new order. Embeddings move, labels stay, and the fixed-point counts above are the measurement. The overlapping eval row shuffle tied at 6/9 because purity after the refit is the self-majority of whatever clusters nine rows fell into. A nine-row partition can tie a real self-majority of 6/9 by chance. The tie is that eval-label refit.

A tiny eval set can tie or rise under that refit. Eval lengths are 10, 9, 9, and 16. Production eval cluster sizes are:

| fixture | eval cluster sizes | singletons |
|---|---|---|
| uneven | 1, 2, 2, 2, 3 | 1 |
| overlapping | 1, 1, 2, 2, 3 | 2 |
| ambiguous | 1, 1, 1, 1, 2, 3 | 4 |
| balanced | 1, 1, 1, 2, 2, 3, 6 | 3 |

A singleton is a hit under every labeling once the majority is taken from that one row. Ambiguous has four singletons out of nine eval rows, so four hits are automatic before the pair and the triple are labeled. Production self-majority there is 8/9. A shuffle that also scores 8/9 has not kept a topic signal. It has kept small clusters. Uneven's production self-majority is 7/10 on clusters of size at most 3, and the shuffle scored 8/10. Balanced, with 16 rows, was the one file whose self-majority shuffles both fell. That fall is the same scorer on a slightly larger set. It is not evidence that the train map transfers. Under the train-only map, balanced production eval purity is 3/16.

Train shuffles fell on all four files (see the copied table below). They use the same refit. Train lengths are 40, 39, 39, and 64, so a random labeling's optimal purity sits further under the real train purity. A train drop does not validate the eval comparison. The eval comparison is the one that failed to collapse, and it failed because the map was refit on the eval labels.

## Class counts

The split is file order, index modulo 5 equals 4. Counts below are that split.

Uneven. Train: tram 20, and bindery, foundry, glasswork, semaphore, tannery at 4 each. Eval: tram 5, and the other five topics at 1 each. Eval majority base rate 5/10. Tram is half the eval split. A self-majority score on this split has a high floor from the dominant class and from the small clusters. Production self-majority 7/10 is two hits above a score that gives each of the five clusters a single hit. The label shuffle can clear 7/10 without a preserved map.

Overlapping. Train: dyehouse, greenhouse, and lithography at 7; canal, horology, and puppetry at 6. Eval: canal 2, dyehouse 1, greenhouse 1, horology 2, lithography 1, puppetry 2. Eval majority base rate 2/9. No class dominates. The row-shuffle tie at 6/9 is the small partition, not a dominant class.

Ambiguous. Train: archive 10, clinic 10, radio 10, kitchen 9. Eval: archive 2, clinic 2, kitchen 3, radio 2. Eval majority base rate 3/9. Kitchen is a mild plurality. The label-shuffle tie at 8/9 lines up with four singleton clusters, not with that plurality.

Balanced. Train: 8 each of cartography, chess, luthier, mycology, neon, stenography, subway, and volcanology. Eval: 2 each. Eval majority base rate 2/16. No dominant class. Both original eval shuffles fell under self-majority. The corrected train-only score is 3/16, and the expected hit count of a label permutation against that fixed map is 2.0. The real score is one hit above that null.

## Leakage

Train and eval are a partition of one file. Duplicate observation ids inside each file: 0. Duplicate texts inside each file: 0. Observation ids shared by the train and eval index sets: 0. Texts shared by those sets: 0. There is no copied row.

The embedding is one `word_sentence_svd` call on every row of the fixture. Document frequency uses eval rows. The truncated basis is fit on eval rows. Topic labels are not an input. That is leakage of eval text into the coordinates of every row, including the train rows the centroids are fit on.

It is material. A comparison embedding uses train texts only: train document frequency, train vocabulary, rank-8 SVD of the train TF-IDF, eval rows projected onto that basis, tokens that never occur in train dropped. Production k-means on that basis disagrees with production k-means on the full-matrix basis.

| fixture | eval token occurrences | occurrences absent from train | absent types | train ARI | eval ARI | train pairwise distance correlation |
|---|---|---|---|---|---|---|
| uneven | 72 | 56 | 55 | 0.2971 | 0.0120 | 0.4083 |
| overlapping | 66 | 52 | 51 | 0.7222 | 0.2500 | 0.7212 |
| ambiguous | 58 | 44 | 43 | 0.3232 | 0.0471 | 0.3894 |
| balanced | 94 | 74 | 72 | 0.3009 | 0.1000 | 0.4131 |

The materiality bar frozen with the revised definition is train adjusted Rand below `0.90`. Every fixture is below it. Overlapping is the closest, at 0.7222, and its eval assignment agreement is 0.25. Most eval content-token occurrences are words the train split never sees, so an eval row in the joint SVD is not a coordinate in a train-defined basis, and the train rows move as well.

### Overlapping train ARI on GitHub Actions

The locked overlapping train ARI is `0.7222354745271766`. It is
`leakage["train_ari_full_vs_train_only"]` in `audit_eval_controls`: the
adjusted Rand index between production k-means on the full-matrix train rows
and production k-means on the train-only embedding of the same rows. The test
`tests/test_eval_control_integrity.py::test_corrected_scores_and_original_controls`
reads that field. `LEAKAGE["overlapping"]["train_ari"]` is the published
AVX512 reference-BLAS value of that field. The CI value is that same field.

Run `39549106578` (head `1d91340596d644c8e762856b6e001bc4387fa22f`, 21 October
2026, the latest completed run on `cursor/unfold-reservoir-0d9f`) reported
`1 failed, 219 passed`. The failure is that one assertion. Obtained
`0.8414648893519251`. Expected `0.7222354745271766`. The expected number stays
`0.7222354745271766`.

The Install step of that run printed NumPy `1.26.4` from
`/usr/lib/python3/dist-packages/numpy/__init__.py` and the loaded libraries
`/usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.12.0` and
`/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.12.0`. Those are the Ubuntu 24.04
packages `python3-numpy 1:1.26.4+ds-6ubuntu1`, `libblas3 3.12.0-3build1.1`, and
`liblapack3 3.12.0-3build1.1`. The reference BLAS selection in the workflow
held for that process.

A local run with those same three packages returns `0.7222354745271766` for
overlapping, and the other three fixtures match their locks as well
(uneven `0.2971403038427167`, ambiguous `0.3232177014940806`, balanced
`0.3008570540432524`). `pin_blas_threads(1)` sets `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to `1` before either SVD. The
local CPU is an Intel Xeon, model 173, and `/proc/cpuinfo` includes `avx512f`.
NumPy 1.26.4's AVX512 log of `39/5` is `0x1.06ed86c91473cp+1`
(`2.054123733695546`). `math.log(39/5)` is `0x1.06ed86c91473dp+1`
(`2.0541237336955462`).

`_train_only_embedding` builds IDF with `np.log`. `word_sentence_tfidf` and
`word_sentence_svd` use `math.log`. The helper then accepts the train block
when `np.allclose` matches `word_sentence_tfidf`. On this AVX512 host the
maximum absolute gap is `5.55e-17`. Three overlapping train tokens, `clock`,
`pipe`, and `pump`, have document frequency 5 in 39 train rows, so their IDF
uses `log(39/5)`. The rank-8 singular values of that train matrix sit near
`1.127137143471583` and `1.126081482704378`. With the AVX512 log, overlapping's
train ARI is `0.7222354745271766`. With AVX512 disabled in
`NPY_DISABLE_CPU_FEATURES`, `np.log(39/5)` matches `math.log(39/5)` bit for
bit, the train TF-IDF matches `word_sentence_tfidf` bit for bit, and the same
field is `0.8414648893519251`. That is the value run `39549106578` obtained.
The full-matrix `word_sentence_svd` hash is unchanged across the two
dispatches. Uneven and balanced stay on their locks in both dispatches.
Ambiguous's train ARI moves with the same log kernel, from the locked
`0.3232177014940806` to `0.45078056237550373`, and overlapping's eval ARI moves
from `0.24999999999999994` to `0.3661971830985915`. Run `39549106578` stopped
at the overlapping train comparison, so those later rows were not the
reported failure of that run.

GitHub-hosted `ubuntu-24.04` does not offer a selector for an AVX512 log
kernel. Forcing the feature on a runner whose CPUID omits it faults. The
workflow keeps reference BLAS and prints, in the Install step and in the Test
process, the NumPy version, the NumPy file, the loaded BLAS and LAPACK paths,
`cpu_count`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
whether `avx512f` is in `/proc/cpuinfo`, and the hex of `np.log(39/5)` beside
`math.log(39/5)`. The local reference-BLAS run still produces
`0.7222354745271766`. That remains the published value. The recorded trial
value is that AVX512 measurement.

The train-ARI assertion is kernel-conditioned. After `pin_blas_threads(1)`, the test
reads `float(np.log(39/5)).hex()`. Hex `0x1.06ed86c91473cp+1` requires the
published AVX512 reference-BLAS value `0.7222354745271766`. Hex
`0x1.06ed86c91473dp+1` requires `0.8414648893519251`. That second number is
what this statistic returns when `np.log(39/5)` matches libm. It was recorded
because it was measured. Any other hex fails the test. The message names the
observed kernel and includes both hex values and both ARIs. The published
value remains the AVX512 one. Neither value is a claim-verifier result.
`coordinate_chain_rule_v1` did not cause the split. No other locked residual,
purity, ARI, or admission byte is retuned for this branch.

Run `39888775181` (head `dff58c5501602b9f29575aff791bb22c2d13aae8`, 23 October
2026) is the latest completed run on `cursor/unfold-reservoir-0d9f`. It
reported `1 failed, 220 passed`. The Install step and the Test process both
printed `avx512f False` and `np.log(39/5)` `0x1.06ed86c91473dp+1`, the libm
hex, beside the same `math.log` hex. NumPy was `1.26.4` from
`/usr/lib/python3/dist-packages/numpy/__init__.py`, with
`/usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.12.0` and
`/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.12.0`. The overlapping train
comparison therefore required `0.8414648893519251` and passed. The failure is
the next assertion in
`tests/test_eval_control_integrity.py::test_corrected_scores_and_original_controls`:
`leakage["eval_ari_full_vs_train_only"]` for overlapping. Obtained
`0.3661971830985915`. Expected the published AVX512 value
`0.24999999999999994`. That obtained number is the libm image already measured
above. It is not a third kernel and not a third train ARI.

The published overlapping eval ARI stays `0.24999999999999994`. The eval-ARI
assertion is now kernel-conditioned on the same hex. `0x1.06ed86c91473cp+1`
requires `0.24999999999999994`. `0x1.06ed86c91473dp+1` requires
`0.3661971830985915`. Any other hex fails. On this AVX512 host,
`NPY_DISABLE_CPU_FEATURES=AVX512F` makes `np.log(39/5)` the libm hex and
returns that same eval ARI. The same reproduction also moves overlapping's
train pairwise distance correlation from `0.7212291143117482` to
`0.808035700074246`, ambiguous's train ARI from `0.3232177014940806` to
`0.45078056237550373`, and ambiguous's correlation from
`0.38942027980065236` to `0.517007726774036`. Those three locks were not
retuned. Uneven and balanced stay on their locks under that hex. The
published overlapping train ARI stays `0.7222354745271766`.

This audit does not change `word_sentence_svd`. The corrected candidate scores below still use the full-matrix embedding. They are a rescoring of the same leaky vectors. They are not an out-of-sample semantic measurement. `SVD_MATERIAL_TRAIN_ARI` is not an admission threshold and it was not used to drop a candidate.

## Map used at eval

Majority-topic maps in the generalization trial are learned on the rows being scored. For an eval score that is `_score_assignment(eval_labels, eval_topics)`, the majority inside each eval cluster is computed from the eval labels. Those labels are used twice: once to name the cluster, and once to count hits. A label shuffle of the eval topics still gets a fresh majority. That inflates purity. The singleton analytical control below is the extreme: after any shuffle, self-majority on eight singletons stays 8/8, and the train-only map falls to 2/8.

On the fixtures, production eval self-majority and the train-only map disagree:

| fixture | self-majority | train-only map |
|---|---|---|
| uneven | 7/10 | 6/10 |
| overlapping | 6/9 | 4/9 |
| ambiguous | 8/9 | 7/9 |
| balanced | 9/16 | 3/16 |

Balanced is the large gap. The 9/16 figure is the eval rows naming their own majorities. The train map predicts 3 of the 16.

Train purity is unaffected by this distinction. The map for a train score is the train majority, which is the ordinary cluster purity. The bug is using that recipe on eval.

## Frozen revised definition

These constants are in `neuralese.eval_control_integrity` and they are fixed before `audit_eval_controls` scores a candidate. `run_analytical_controls` runs first. If it fails, the function returns `FAIL` and does not embed the fixtures.

| constant | value |
|---|---|
| `REVISED_DEFINITION` | `train_only_majority_map` |
| `MAP_SOURCE` | `train_rows_only` |
| `LABEL_CONTROL` | `permute_eval_labels` |
| `EMBEDDING_CONTROL` | `permute_eval_embeddings_relative_to_eval_labels` |
| label shuffle seed | 0, the trial's `LABEL_SHUFFLE_SEED` |
| embedding shuffle seed | 1, the trial's `ROW_SHUFFLE_SEED` |
| `BLOB_REAL_MIN_PURITY` | 0.99 |
| `SHUFFLE_MAX_PURITY` | 0.60 |
| `GAUSSIAN_ABS_TOLERANCE` | 0.02 |
| `SVD_MATERIAL_TRAIN_ARI` | 0.90 |
| `RESIDUAL_WORSE_FACTOR` | 1.10, unchanged |
| `CLUSTER_RUNTIME_FACTOR` | 2.0, unchanged |

The map is the majority topic of each cluster on train rows only. The tie break is the isolation probe: higher count, then the topic string in ascending order. An eval hit is an eval row whose topic equals that train label. A cluster id absent from the train map is a miss. The map is not refit on eval topics, shuffled or otherwise.

The label control permutes eval topics only, seed 0, and scores those strings with the unchanged train map and the unchanged eval cluster ids.

The embedding control permutes eval embedding rows only, seed 1, assigns them to the production centroids fit on the real train rows, and scores the unpermuted eval topics with the same train map. Centroids are not refit on the permutation.

Both must be strictly below the unshuffled train-map hit count. A tie is not a collapse. The admission rule for a candidate is the generalization rule, applied to this purity: strictly greater than production on that fixture, residual at most `1.10×`, clustering median at most `2×`, on every fixture. Otherwise the recommendation is `NONE`.

## Analytical controls

Data seed 11. Dimension 8. Production farthest-first k-means, k = 8, the library constant. These cases do not read the fixtures.

Two isotropic blobs. 200 train rows and 100 eval rows in each class, gap 30 on the first axis, noise scale 1. A correct train map scores every eval row.

| score | hits | purity |
|---|---|---|
| real eval, train-only map | 200/200 | 1.00 |
| eval label shuffle, same map | 92/200 | 0.46 |
| eval embedding permutation, same map | 110/200 | 0.55 |

Equal classes put chance at 0.50. Both shuffles are at or under 0.60 and strictly under the real purity. The label-shuffle hit count is the class match of seed 0 against a perfect prediction, so it does not depend on how k-means splits a pure blob.

One standard Gaussian. 2000 train rows, 2000 eval rows. Labels are an independent draw at rate 0.75, so they carry no position. Every train cluster's majority is `maj`. Eval purity equals the eval majority count.

| score | hits | rate |
|---|---|---|
| eval majority base rate | 1476/2000 | 0.738 |
| eval purity, train-only map | 1476/2000 | 0.738 |

The absolute gap is 0, inside the tolerance 0.02.

Singletons. Eight eval clusters of one row, train map matching the real label, four train rows per cluster. After the seed-0 label shuffle, self-majority is 8/8. The train-only map scores 2/8. The test fails the build if that shuffle does not drop. A scorer that refits the majority on the shuffled singletons stays at 8/8 and fails the same test.

All eight analytical checks passed. The candidate rerun was allowed.

## Original outcomes

Copied from [PR1_CLUSTERING_GENERALIZATION.md](PR1_CLUSTERING_GENERALIZATION.md). This audit recomputes them by calling `_negative_controls` and they match. Self-majority on the candidate eval assignments matches that trial's eval hit table. Residuals are that trial's residuals.

Train assignment, production versus the two shuffles.

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

The three eval controls that did not collapse are uneven's label shuffle (8/10, above 7/10), overlapping's row shuffle (6/9, a tie), and ambiguous's label shuffle (8/9, a tie). The trial verdict stayed `UNRESOLVED`. Recommendation `NONE`.

## Corrected outcomes

Same embeddings, same centroids, train-only map. Production eval hits fall relative to self-majority, as in the map table above. Candidate eval hits under the train-only map:

| candidate | uneven | overlapping | ambiguous | balanced |
|---|---|---|---|---|
| production farthest-first | 6/10 | 4/9 | 7/9 | 3/16 |
| euclidean random subset, seed 2 | 5/10 | 5/9 | 4/9 | 4/16 |
| euclidean k-means++, seed 2 | 5/10 | 5/9 | 3/9 | 4/16 |
| spherical random subset, seed 2 | 5/10 | 4/9 | 4/9 | 4/16 |
| spherical k-means++, seed 3 | 6/10 | 5/9 | 3/9 | 4/16 |

Corrected controls on the production assignment. The expectation is the mean hit count of a uniform permutation of the eval labels against the fixed train map.

| fixture | train-map real | label shuffle | embedding shuffle | both lower | expected label-shuffle hits |
|---|---|---|---|---|---|
| uneven | 6/10 | 5/10 | 5/10 | yes | 4.6 |
| overlapping | 4/9 | 0/9 | 2/9 | yes | 1.667 |
| ambiguous | 7/9 | 2/9 | 1/9 | yes | 2.222 |
| balanced | 3/16 | 3/16 | 5/16 | no | 2.0 |

Uneven, overlapping, and ambiguous drop under both revised controls. Those are the three files whose original eval controls did not collapse. The drop is what a fixed map does once the shuffle can no longer rename the clusters.

Balanced does not drop. The real train-map score is 3/16. The seed-0 label shuffle ties at 3/16. The seed-1 embedding shuffle rises to 5/16. A control that must fall has nothing to fall from when the real score is one hit above an expectation of 2. The original balanced collapse was self-majority, 9/16 against shuffles of 7/16 and 8/16.

## Candidate rerun

The analytical checks passed, so the four frozen candidates were rerun on the existing fixtures. Seeds are unchanged. The purity clause uses the train-only eval hits. Residual and runtime clauses are the original ones. This run's clustering ratios, median of five samples of 100 fits, were all under `2×`. The largest sample on this host was about 1.97× on ambiguous spherical random-subset, which already fails the purity clause. The test locks the advantage bits and does not lock the milliseconds.

Advantage on a fixture is still all three clauses. A pass on one fixture is not a recommendation.

| candidate | uneven | overlapping | ambiguous | balanced | every fixture |
|---|---|---|---|---|---|
| euclidean random subset, seed 2 | no | yes | no | no | no |
| euclidean k-means++, seed 2 | no | yes | no | yes | no |
| spherical random subset, seed 2 | no | no | no | no | no |
| spherical k-means++, seed 3 | no | no | no | yes | no |

No frozen candidate has the advantage on every fixture. The recommendation stays `NONE`. The thresholds were not moved to make one of them pass.

Overlapping is the only file where both Euclidean seed-2 candidates clear the corrected purity clause and the residual clause. Ambiguous is a loss for all four on purity: production's train-only eval score is 7/9 and every candidate is at 4/9 or 3/9. Uneven ties or loses on purity. Balanced's corrected purity edge is 4/16 against 3/16, one hit on a null whose label-shuffle expectation is 2, and two of the four candidates are over the residual line.

## Verdict

Analytical controls passed. Candidates were rerun. Corrected fixture controls do not all strictly drop, because balanced ties and then rises. The full-matrix SVD moves the train clustering on every fixture, so the fixture scores are not a clean holdout even under the corrected map.

The audit verdict is **UNRESOLVED**.

`PASS` would mean the revised method is internally valid on these fixtures: analytical checks passed, both corrected controls strictly dropped on every fixture, and the train clustering agreed with a train-only basis at adjusted Rand at least 0.90. The analytical checks passed. The fixture controls and the SVD comparison did not. `FAIL` is reserved for an analytical control that does not behave. That did not happen, so the candidates were scored.

Recommendation **NONE**. `RUNTIME_NOT_AUTHORIZED`.

An unresolved audit is not a certificate pass, not a semantic-quality result, and not a runtime change. `certify` stays the structural contract. Numerical fidelity of the encoder is untouched. The purity numbers, old and corrected, are a probe.

## What the audit leaves in place

The product rank stays 8. `tau_residual` stays 0.55. `tau_kappa` stays 0.35. Checksum coverage is unchanged. The character-trigram unanchored reject stays locked. No reconstructor, logit lens, SAE, or hub runtime is added. The production clustering call is the same function. `learn_pack` is not called. Package version stays 0.7.0.

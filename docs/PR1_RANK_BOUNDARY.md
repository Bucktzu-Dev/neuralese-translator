# PR-1 rank boundary

Status: diagnostic fixture on the frozen rank-8 word-sentence SVD. The alphabet algorithm is unchanged. Package version stays 0.7.0.

Audit decision: **RESEARCH_FAIL**. **RUNTIME_NOT_AUTHORIZED** is unchanged.

`certify()` passing is the existing structural contract (`addressable`, `unfoldable`, `gloss_bound`, `residual_ok`). It is evidence of that seal. It is not numerical success and it is not semantic success. `tau_residual` stays 0.55. `tau_kappa` stays 0.35. Checksum coverage is unchanged. The character-trigram unanchored reject on this file stays locked.

Topic ids are a probe. They are not an input to the embedding, the codebook, or `certify`. The module is `neuralese.rank_boundary`. The lock is `tests/test_rank_boundary.py`. There is no new CLI flag and no `neuralese report` field.

## Corpus and product settings

`examples/varied_prose`, encoder `word_sentence_svd`, seed 0, `n_symbols` 8. Reference BLAS/LAPACK 3.12. BLAS, OpenMP, and MKL are pinned to one thread inside the same SVD the encoder uses.

The varied-prose lock on this host stays:

| quantity | value |
|---|---|
| residual | 0.5239427941099951 |
| mean kappa | 0.8814174642164703 |
| live / quarantined | 7 / 1 |
| size-weighted purity | 0.6394984326018809 (204/319) |
| worst live cluster | weather, 29 of 64, purity 0.453125, mixed |

## Singular values at the rank cut

The spectrum is the full SVD of the same row-L2 TF-IDF matrix, with `compute_uv` left on so the values match the factorization inside `word_sentence_svd`. Index 7 is the last kept component (1-based 8). Index 8 is the first discarded component (1-based 9).

| index | 1-based | value |
|---|---|---|
| 5 | 6 | 1.6381173195284298 |
| 6 | 7 | 1.6158837658043956 |
| 7 | 8 | 1.577782436621879 |
| 8 | 9 | 1.556022686499294 |
| 9 | 10 | 1.5011858601662842 |
| 10 | 11 | 1.494124134967109 |

| quantity | value |
|---|---|
| gap `s[7] - s[8]` | 0.021759750122584798 |
| ratio `s[8] / s[7]` | 0.9862086498001754 |
| neighbor gap `s[6] - s[7]` | 0.03810132918251674 |
| neighbor gap `s[8] - s[9]` | 0.0548368263330099 |

The cut is smaller than either neighbor. The trailing component is not a spectral cliff. The test locks the window, the gap, and the ratio within `1e-9` absolute.

## Decision rule

Distances are Euclidean in the frozen rank-8 embedding that `learn_pack` stores. Rows are the ones the current k-means (seed 0, k 8) placed in the weather-majority cluster. Those rows are split by the probe topic into weather and other. Within-group distance is the mean pairwise Euclidean distance. The between-group distance is the Euclidean distance between the two group centroids.

`closer_fraction` is the fraction of weather rows in that cluster whose nearest other-topic centroid, also taken inside the cluster, is strictly closer than the weather centroid.

Let `ratio = between_centroid / within_weather`.

1. Overlap by spread when `ratio < 1`. The between-centroid distance is then smaller than the within-weather spread.
2. Overlap by assignment when `closer_fraction > 0.5`.
3. Separated when `ratio >= 2`. The between-centroid distance is then at least twice the within-weather spread.
4. If separated and overlap by assignment are both true, the mode is `UNRESOLVED`. The conflict string names both numbers.
5. Otherwise, if either overlap clause is true, the mode is `RANK_LOSS`.
6. Otherwise, if separated is true and the weather-majority cluster is mixed (purity under 0.5, the same cutoff `cluster_majority_diagnostic` already uses), the mode is `CLUSTERING_LOSS`.
7. Otherwise the mode is `UNRESOLVED`, and the conflict string says which clause failed. A ratio in `[1, 2)` with a closer-fraction at or below 0.5 is this case.

The two readings are not averaged. The thresholds are `OVERLAP_RATIO = 1`, `SEPARATED_RATIO = 2`, and `CLOSER_MAJORITY = 0.5` in `neuralese.rank_boundary`, and the test writes `2.0` and `0.5` against the locked distances.

## Rank-8 weather cluster

Code 6. 64 rows. Majority weather, 29. Purity 0.453125. Mixed.

The other 35 rows are ledger 26, orchard 3, bakery 2, pottery 2, harbor 1, joinery 1. The within-other distance is wide because that side is several topics.

| quantity | value |
|---|---|
| within-weather mean pairwise | 0.3781671594255758 |
| within-other mean pairwise | 0.7700430881162342 |
| between-group centroid | 0.979159660507253 |
| `between / within-weather` | 2.5892244635799844 |
| weather rows closer to another topic centroid | 8 of 29 (0.27586206896551724) |

`2.589 >= 2`, the closer fraction is under 0.5, and purity is under 0.5. Clause 6 applies. Clause 4 does not. The mode is **CLUSTERING_LOSS**.

The same row partition is produced when `learn_pack` runs on the file that still carries `metadata.topic`. The probe builds its own rows with empty metadata and index ids. The partitions match, so the topic strings are not what placed the rows.

## Wider truncations of the same encoder

Same k-means: seed 0, k 8, and the same quarantine (`min_cluster_size` 2, `tau_kappa` 0.35). The width is `word_sentence_svd` at that rank. Rank 8 is the product width, listed so the other rows have a baseline. Ranks 9, 12, and 16 are not a new default.

| rank | live clusters | weather-majority cluster | purity | mixed |
|---|---|---|---|---|
| 8 | 7 | 29 of 64 | 0.453125 | yes |
| 9 | 7 | 31 of 107 | 0.2897196261682243 | yes |
| 12 | 8 | 29 of 37 | 0.7837837837837838 | no |
| 16 | 8 | 28 of 54 | 0.5185185185185185 | no |

These three wider truncations do not move together. Rank 9 mixes weather further (31 of 107). Rank 12 raises the weather-majority purity to 29 of 37. Rank 16 leaves it at 28 of 54, just above the mixed cutoff. That table is not folded into the mode. It does not establish rank loss. It also does not show that every extra component leaves weather mixed, so it is not a claim that clustering is the only thing that could be wrong in a wider subspace. The locked label remains the rank-8 rule.

## Cost

`learn_pack` on this corpus, which is the SVD, the k-means, and the rest of the learn path. Five runs on this host, median of the sorted samples:

| run (ms) |
|---|
| 217.586 |
| 218.876 |
| 222.369 |
| 224.329 |
| 225.030 |

Median: **222.369 ms**. This is the cheap-path baseline. It is not a certify gate. The test does not lock the milliseconds.

## What the fixture leaves in place

The product rank stays 8. `certify` is untouched. The character-trigram reject stays locked. No reconstructor, logit lens, SAE, or hub runtime is added. Adaptive rank and selective refinement are not implemented. The singular-value gap is recorded above; the distance rule, not the gap size, is what sets `CLUSTERING_LOSS`.

The next action is a clustering check on these frozen rank-8 vectors: same embedding, same k, same seed family, aimed at the weather-majority mixture whose centroid gap is already 2.59 times the within-weather spread. Raising the SVD rank is not that action. The wider truncations do not agree with each other.

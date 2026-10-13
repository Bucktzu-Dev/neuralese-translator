# PR-1 activation source

Status: `RESEARCH_ONLY`. Baseline commit `c6321d5`. Entering verdict: `UNRESOLVED`. Entering recommendation: `NONE`. Authority: `RUNTIME_NOT_AUTHORIZED`.

This note asks whether the existing symbolic pack can be learned from genuine model activation vectors. The vectors are a one-time forward pass of a pinned public checkpoint. They are not character hashes, not TF-IDF, and not a truncated SVD of the prompt text. `word_sentence_svd`, `learn_pack`'s default encoder, `clustering.kmeans`, `certify`, `tau_residual`, `tau_kappa`, and the package version stay as they are. No runtime integration and no production admission.

The trial verdict is **PASS_FOR_LATER_PR**. The recommendation is **LATER_ISOLATED_PR**. Authority stays **RUNTIME_NOT_AUTHORIZED**. The text-encoder trials stay frozen. A later isolated PR would still have to land on its own; this note does not change the production encoder.

## Frozen boundary

The prompts, the checkpoint hashes, the coordinate rule, and the admission constants were written down before any held-out residual was computed.

| field | value |
|---|---|
| model | `EleutherAI/pythia-14m` |
| revision | `cf967c0a9a04383db6f7b1108d86b2962634b4ac` |
| url | https://huggingface.co/EleutherAI/pythia-14m/commit/cf967c0a9a04383db6f7b1108d86b2962634b4ac |
| architecture | `GPTNeoXForCausalLM`, 6 blocks, hidden size 128, vocab 50304 |
| weight file | `model.safetensors`, 28143920 bytes |
| weight sha256 | `116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1` |
| config sha256 | `f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17` |
| tokenizer.json sha256 | `870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2` |
| checkpoint dtype | float16 weights |
| forward dtype | float32 on CPU |
| stored dtype | float32 |
| capture stack | torch `2.14.1+cpu`, transformers `4.57.6` |
| layer rule | last tensor in `outputs.hidden_states` |
| recorded index | 6, tuple length 7 |
| token rule | final position of an unpadded prompt, `add_special_tokens` false, no padding |
| prompts | 32 train, 16 held-out, disjoint, fingerprint `1780cae1dc80350e1e10c4437f7d8b7d4a2cd5c818a0e0d846febb6a84c7bfba` |

In transformers 4.57.6 the tuple is the embedding after dropout, then the input of each later block, then the final layer norm. Index 6 is that final layer norm. It is the tensor the model returns as `last_hidden_state`. The next-token logits at the same position are the LM head applied to this vector. Those logits are stored beside the activations. They are not a column of the matrix passed to `kmeans` or `learn_pack`.

Every prompt ends with a period. The pinned tokenizer maps that final token to id 15 on all 48 rows. The stored activation is the 128-d state at that position, which still depends on the tokens before it. The token id is not a clustering feature.

Train and held-out strings were fixed in `activation_source.py` before the forward pass. No string is shared, and neither split contains the other's sentence. Nothing was fit on the held-out activations: no centroid, no SVD basis, no symbol-conditional logit mean.

## What the tests load

`examples/activation_source/activations.npz` and `capture_meta.json` are the fixture. Tests call `numpy.load` and the existing library. They do not import torch, and CI does not run `examples/activation_source/capture_once.py`. That script is the one-time capture. This run verified a snapshot already on disk, so `download_seconds` in the meta is 0. The weight fetch is not part of the cluster, lookup, or certify times.

Checksums of the float32 arrays, sha256 of the contiguous bytes:

| array | shape | sha256 |
|---|---|---|
| train activations | `(32, 128)` | `487e55dfd8a8712a572fcbeed4226ec50dfcfe40fd19fbf321c118a8de87cc84` |
| held-out activations | `(16, 128)` | `c2fe5cf25f106fea4f64ccc33e67b95b3a5538e3e1824ea2d8aac7b5a4ee00ec` |
| train logits | `(32, 50304)` | `bc5159f6880205733ed898102f2c5dd67898673bc7290dcb8010a8d36428101f` |
| held-out logits | `(16, 50304)` | `f7605e74e32638647548327cf515cc3e4f9eec5f93be57da9f3d00f5d0d76131` |

The npz is 8968670 bytes. Token ids and lengths are in the same file so the final-position rule can be checked without the tokenizer.

## Gates

These constants sit above `run_activation_source`. The verdict function does not read a gloss.

| constant | requirement |
|---|---|
| `GEOMETRY_GATE` | held-out prototype residual is strictly below the residual to the train mean and strictly below the shuffled-assignment residual |
| `SHUFFLE_ASSIGNMENT_SEED` | 0, one permutation of the held-out assignments to the frozen train centroids |
| `DOWNSTREAM_RELATIVE_MARGIN` | 0.05 |
| `DOWNSTREAM_GATE` | `(shuffled_logit_residual - true_logit_residual) / shuffled_logit_residual >= 0.05`, with the shuffled residual positive |
| `SHUFFLE_SYMBOL_SEED` | 1, one permutation of the held-out symbol ids |

Residuals are `reconstruction_error`: the normalized Frobenius norm. The train-mean baseline replaces every held-out row with the mean of the training activations. The shuffled-assignment control keeps the real centroids and permutes which held-out row receives which centroid. The downstream means are the per-symbol mean of the training logits only. An empty symbol would fall back to the global training-logit mean. This run had no empty symbol.

Verdict order, also fixed before the scores:

1. A capture that is not genuine, or that the test cannot reload from the fixture, is `UNRESOLVED` / `NONE`.
2. A train pack that fails structural certify is `STRUCTURAL_FAILURE` / `NONE`.
3. Geometry that does not beat both baselines is `UNRESOLVED` / `NONE`.
4. Geometry that beats both baselines with no downstream quantity is `GEOMETRY_ONLY` / `NONE`.
5. A downstream quantity whose shuffled control misses the margin is `UNRESOLVED` / `NONE`.
6. All of the above passing is `PASS_FOR_LATER_PR` / `LATER_ISOLATED_PR`, and the authority line stays `RUNTIME_NOT_AUTHORIZED`.

Structural certify is `certify(pack)` at the library default `tau_residual` 0.55, plus the reservoir rows still holding the activation vectors, plus `translate_stream` on code `1000000000` returning state `unknown`. `LearnConfig()` is the default: encoder `char_trigram`, `n_symbols` 8, `tau_kappa` 0.35, `tau_residual` 0.55, seed 0. Rows already carry embeddings, so `ensure_embedding` does not hash the text.

## Structural result

`learn_pack` on the 32 training activations accepts. `certify` passes. Failures: none. Explicit trial failures: none.

| field | value |
|---|---|
| codebook size | 8 |
| live | 5 |
| quarantined | 3 |
| empty clusters | 0 |
| sealed residual | 0.22755261731526205 |
| cluster residual | the same value; the `0.25 * svd` term does not raise it |
| mean kappa | 0.9832387137375892 |
| decision | `accept` |
| receipts | `factorization`, `clustering`, `finalize` |
| unknown code `1000000000` | state `unknown` |

The three quarantined symbols are the three size-1 clusters. `min_cluster_size` is the library default 2. Their kappas are 1 because a singleton matches its own mean. Kappa was not the cause, and the taus were not loosened.

Train membership, in class order:

| class | rows | kappa | state |
|---|---|---|---|
| 0 | 1 | 1.000000 | quarantined |
| 1 | 2 | 0.976016 | live |
| 2 | 5 | 0.973248 | live |
| 3 | 1 | 1.000000 | quarantined |
| 4 | 3 | 0.971348 | live |
| 5 | 3 | 0.976586 | live |
| 6 | 16 | 0.968712 | live |
| 7 | 1 | 1.000000 | quarantined |

Class 6 holds 16 of the 32 training rows. The heuristic gloss on that class is the keyword sentence `Symbol for he, until, end, she.` That sentence is not evidence. The pack's other glosses are the same keyword template.

Production `kmeans` labels match the pack's membership. Non-empty prototypes match the centroids `kmeans` returned. Held-out ids are absent from the reservoir. The reservoir vectors match the float64 view of the stored float32 activations, and their width is 128, so they are not the 32-d character trigram.

## Geometry and the downstream quantity

Held-out rows were assigned by the squared-Euclidean argmin that `kmeans` uses after farthest-first seeding. Centroids were not refit.

| residual | value |
|---|---|
| nearest train centroid | 0.29343442149783966 |
| train mean | 0.30512149677079187 |
| shuffled assignment, seed 0 | 0.33508908512206886 |

The prototype residual is lower than both baselines. The gap against the train mean is 0.011687. The gap against the shuffled assignment is 0.041655. The gate asked only for a strict inequality. It holds. It is a small separation from the mean, on 16 held-out rows, with 11 of those rows assigned to class 6, 4 to class 2, and 1 to class 5. All 16 land on live symbols, so `translate_stream` returns state `ok` for them. The three quarantined centroids receive no held-out row.

Downstream quantity: the next-token logit vector at the same final position, captured in the forward pass and unused in the fit. The symbol-conditional mean was fit on training logits only.

| score | value |
|---|---|
| true symbol id, held-out logit residual | 0.2974424885101723 |
| shuffled symbol id, seed 1 | 0.34492301344576626 |
| relative drop | 0.13765542768882114 |
| required margin | 0.05 |
| empty-symbol fallbacks | 0 |

The relative drop clears 0.05. Because index 6 is the final layer norm, this logit vector is a linear readout of the clustered coordinate. The control is still a real one: a shuffled symbol id, using the same training means, predicts the held-out logits worse by the margin above. It is not a test of an earlier layer, and it is not a claim that the keyword gloss names the next token.

## Assignment stability and cost

| measure | value |
|---|---|
| train rows whose nearest centroid is their k-means label | 1.0 (32/32) |
| agreement of seed 0 and seed 1 | 1.0 |
| agreement after refitting on 80% of the train rows, seed 2, labels rematched | 0.8125 (26/32) |

Seed agreement is 1 because production farthest-first does not read the `rng` argument. The subsample figure is the stability measurement that actually changes the fit. The subsample is training rows only.

Wall clock on this machine. Capture is tokenize plus forward, 48 prompts, and it is stored in the fixture meta. Cluster is one `learn_pack` after a discarded warmup. Lookup is held-out assignment plus `translate_stream`. Certify is one `certify` call. Cluster, lookup, and certify are single samples; they are not gates.

| stage | seconds | per observation |
|---|---|---|
| capture | 0.244276941 | 0.005089103 over 48 prompts |
| cluster | 0.004641353 | 0.000145042 over 32 train rows |
| lookup | 0.000093046 | 0.000005815 over 16 held-out rows |
| certify | 0.002378617 | 0.000074332 over 32 train rows |

Model load, recorded separately, was 0.164121247 seconds. Peak `ru_maxrss` during the capture process, including the loaded weights, was 499580 KB. Python tracemalloc during the forward loop peaked at 9727638 bytes. Tracemalloc during one later learn, lookup, and certify peaked at 530124 bytes.

## What this does not authorize

- Replacing `word_sentence_svd` or the default character-trigram path.
- Editing `clustering.py`, the certify gates, or the taus.
- Shipping a hub runtime, an SAE, or a logit lens in this package.
- Treating the keyword gloss as a definition of the symbol.
- Treating the geometry gap of 0.011687 against the train mean as a large separation.
- Treating the logit result as evidence about a layer the LM head does not read.
- Fitting a new checkpoint, a new layer, or a new prompt split and calling it the same result.

The text-encoder trials stay frozen. Their verdicts remain `UNRESOLVED` with recommendation `NONE`.

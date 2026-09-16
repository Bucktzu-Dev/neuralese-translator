# Ingest dumped activations

`neuralese ingest` turns a hidden-state dump into observation JSONL. The translator still does not import HuggingFace, run a model, or pool token axes for you. After ingest, learn and certify as usual; translation residual stays the caller `--tau-residual`.

## CLI

```bash
neuralese ingest states.npy -o observations.jsonl --texts prompts.jsonl --layer 12
neuralese learn observations.jsonl -o pack.json
neuralese certify pack.json --observations observations.jsonl --fail-on-undecodable
```

Accepted activation files:

- `.npy` — 2-D real array `(n_observations, hidden_dim)`
- `.npz` — named `hidden_states` / `activations` / `embeddings`, or exactly one matrix array. Optional 1-D `texts` and `observation_ids` arrays are aligned to rows. `--texts` replaces both.
- `.jsonl` — one object per row with `hidden_state` or `embedding` (not both)

`--texts` is only valid with `.npy` / `.npz`. It may be JSONL objects (`text`, optional `observation_id`) or a JSON array of strings. `--layer` is stored on each observation's metadata; on JSONL it overwrites a record `layer`.

## Fail closed

- Rank-3+ dumps, empty axes, bool/complex/object/unicode dtypes, non-finite values
- Corrupt ZIP-backed `.npy` / `.npz`, pickle payloads (`allow_pickle=False`), mislabeled `.npz` bytes served as `.npy`
- Overflowing JSON numbers, recursive JSON, cyclic metadata, non-object metadata
- Duplicate ids, blank ids, non-string ids (JSON `null` means missing and gets `obs-{line}`)
- String/bytes used as a texts or ids sequence, `--texts` on JSONL activations

Library helpers: `load_activation_matrix`, `load_activations`, `load_alignment_texts`, `observations_from_activations`, `save_observations_jsonl`.

## Dump from HuggingFace (user-side)

This snippet is not part of the package. Pool to two dimensions before ingest.

```python
import numpy as np
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModel.from_pretrained("gpt2")
prompts = ["hello there friend", "audit the trail please"]
encoded = tokenizer(prompts, return_tensors="pt", padding=True)
states = model(**encoded).last_hidden_state
mask = encoded["attention_mask"].unsqueeze(-1)
pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
np.save("states.npy", pooled.detach().cpu().numpy())
```

Then write `prompts.jsonl` with one `{"text": ...}` per row, in the same order.

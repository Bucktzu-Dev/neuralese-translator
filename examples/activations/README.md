# Ingest dumped activations

`neuralese ingest` turns a hidden-state dump into observation JSONL. The translator still does not import HuggingFace, run a model, or pool token axes for you.

## CLI

```bash
neuralese ingest states.npy -o observations.jsonl --texts prompts.jsonl --layer 12
neuralese learn observations.jsonl -o pack.json
neuralese certify pack.json --observations observations.jsonl --fail-on-undecodable
```

Accepted activation files:

- `.npy` — 2-D `float` array `(n_observations, hidden_dim)`
- `.npz` — one array, or a named `hidden_states` / `activations` / `embeddings` array
- `.jsonl` — one object per row with `hidden_state` or `embedding`

`--texts` is only valid with `.npy` / `.npz`. It may be JSONL objects (`text`, optional `observation_id`) or a JSON array of strings. Rank-3+ dumps fail closed.

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

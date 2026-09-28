# Public pack

Twelve original sentences about a harbor, an orchard, and a ledger. Each row carries an 8-d activation-style vector. Nothing here is private.

`observations.jsonl` is ready for `neuralese learn`. `hidden_states.jsonl` is the same rows as a hidden-state dump for `neuralese adapt`.

```bash
python -m neuralese adapt examples/public_domain/hidden_states.jsonl -o /tmp/public-obs.jsonl --layout vectors
python -m neuralese learn /tmp/public-obs.jsonl -o /tmp/public-pack.json --n-symbols 3 --seed 0
python -m neuralese certify /tmp/public-pack.json --fail-on-undecodable
python -m neuralese unfold /tmp/public-pack.json --code 0
```

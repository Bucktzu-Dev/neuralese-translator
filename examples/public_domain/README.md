# Public pack

Toy fixture, not the text path. Twelve original sentences about a harbor, an orchard, and a ledger, each with a hand-placed 8-d near-orthogonal vector. Nothing here is private. The text-learned corpus is [examples/prose_corpus](../prose_corpus). See [docs/ROADMAP.md](../../docs/ROADMAP.md).

`observations.jsonl` is ready for `neuralese learn`. `hidden_states.jsonl` is the same rows as a hidden-state dump for `neuralese adapt`.

```bash
python -m neuralese adapt examples/public_domain/hidden_states.jsonl -o /tmp/public-obs.jsonl --layout vectors
python -m neuralese learn /tmp/public-obs.jsonl -o /tmp/public-pack.json --n-symbols 3 --seed 0
python -m neuralese certify /tmp/public-pack.json --fail-on-undecodable
python -m neuralese unfold /tmp/public-pack.json --code 0
```

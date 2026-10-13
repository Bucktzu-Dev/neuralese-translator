# Toy stream

Toy fixture. Twelve short lines with hand-placed near-orthogonal embeddings. This is not the text-learned path. See [examples/prose_corpus](../prose_corpus) and [docs/ROADMAP.md](../../docs/ROADMAP.md).

Run from the package root:

    neuralese learn examples/toy_stream/observations.jsonl -o pack.json --n-symbols 3
    neuralese translate pack.json examples/toy_stream/stream.json
    neuralese unfold pack.json --code 0
    neuralese certify pack.json --fail-on-undecodable

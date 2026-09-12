# Neuralese to English Translator
Run from the package root:

    neuralese learn examples/toy_stream/observations.jsonl -o pack.json --n-symbols 3
    neuralese translate pack.json examples/toy_stream/stream.json
    neuralese certify pack.json --fail-on-undecodable

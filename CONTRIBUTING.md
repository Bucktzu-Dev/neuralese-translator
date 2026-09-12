# Contributing

This tree is a public audit tool. Please keep it honest.

- Do not import Eris internals (event bus, identity, SLAR orchestrator, secrets).
- Do not contribute a decoder that invents English for missing codes. `unknown` and `quarantined` are required states.
- Gloss text must remain sealed in the pack checksum. Tests should fail if a definition can be mutated without `certify()` noticing.
- Tests should be deterministic (fixed seeds). Label heuristic glosses as heuristic.
- Docs that widen a capability claim must also update LIMITATIONS.md.

Useful work: HuggingFace/activation adapters, parent-pack ΔMDL fixtures, stronger unfold round-trips, and CI examples on real (non-secret) packs.

# SLAR mapping (appendix)

You do **not** need this file to use the translator. Public names are enough.

This appendix exists so people who already know the originating Eris SLAR engine can see the correspondence. The two namespaces are not identified as one product.

| Public (this package) | SLAR origin | Notes |
|---|---|---|
| Observation | SLM/event feature row | JSONL adapter, no event bus |
| Symbol | `SymbolClass` + assigned code | `GlyphAtom` members collapse to `observation_ids` |
| SymbolPack | `LRP` (Lexicon Reconstruction Pack) | Same job: sealed state-transfer bundle |
| Receipt | `XRRCReceipt` | Public receipts do not require Eris XRRC machinery |
| GuardSnapshot | `NumericGuards` + `PersistenceResult` + `CompatResult` | Combined snapshot |
| `learn_pack` | `SLAROrchestrator.run_pipeline` | No `ResourceMonitor`, no bus |
| Subjective symbol dynamics | `SubjectiveSymbolDynamics` | Same Boltzmann tilt \(P(z_t\|z_{t-1},c,s)\) |
| `rewrite_stream` | `eris.slar.aliases.rewrite_stream` | Lossless integer remap |
| `certify` | `LRP.guards_pass` + unfold checks | Adds gloss-binding and reservoir checks |
| `Gloss` | `SemanticLearner` definition fields | States are new: ok/aliased/quarantined/unknown |
| `mdl_bits` Δ vs parent | intended ΔMDL | Implemented here against `parent.mdl_bits`; mothership stubs are not copied |

Intentionally **not** mapped:

- ParameterMorpheon / RelationalMorpheon (CSLP / Dalton). This tool is alphabet + audit, not adapter geometry.
- Narrative Morphogenesis Engine (Dream Cycle). Narrative ops are not a decoder.
- Eris event bus, Prometheus, SLM facade, encryption key paths.

If you are extracting further leaves from a mothership tree, keep this package import-free of `eris.*`.

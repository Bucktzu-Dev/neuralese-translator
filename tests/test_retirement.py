from neuralese.adapters import load_observations_jsonl
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify
from neuralese.contracts import Observation
from neuralese.translator import translate_stream
from neuralese.unfold import unfold_code

from test_mdl import FIXTURE


def test_unmatched_parent_code_stays_quarantined():
    obs = load_observations_jsonl(FIXTURE)
    parent = learn_pack(obs, config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0))
    by_code = {symbol.code: symbol for symbol in parent.symbols}
    north_code = next(code for code, symbol in by_code.items() if "north" in (symbol.definition or ""))
    south_code = next(code for code, symbol in by_code.items() if "south" in (symbol.definition or ""))
    north = [row for row in obs if row.observation_id.startswith("north")]

    child = learn_pack(north, config=LearnConfig(n_symbols=1, min_cluster_size=2, seed=0), previous=parent)
    glosses = {gloss.code: gloss for gloss in translate_stream(child, [north_code, south_code])}

    assert glosses[north_code].state == "ok"
    assert "north" in glosses[north_code].english
    assert glosses[south_code].state == "quarantined"
    assert glosses[south_code].english.startswith("[quarantined:")
    retired = next(symbol for symbol in child.symbols if symbol.code == south_code)
    assert retired.quarantined
    assert retired.metadata["retired_from"] == parent.pack_id
    assert "south-1" in child.observations
    report = unfold_code(child, south_code)
    assert report.missing_ids == []
    assert report.prototype_l2 is not None and report.prototype_l2 < 1e-5
    cert = certify(child)
    assert cert.passed, cert.failures


def test_relearn_keeps_codes_and_zero_delta():
    obs = load_observations_jsonl(FIXTURE)
    parent = learn_pack(obs, config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0))
    child = learn_pack(obs, config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0), previous=parent)
    assert child.guards.delta_mdl == 0.0
    assert [symbol.code for symbol in child.symbols if symbol.quarantined] == []
    parent_gloss = {symbol.code: symbol.definition for symbol in parent.symbols}
    child_gloss = {symbol.code: symbol.definition for symbol in child.symbols}
    assert child_gloss == parent_gloss


def test_parent_alias_still_resolves_after_relearn():
    obs = load_observations_jsonl(FIXTURE)
    parent = learn_pack(obs, config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0))
    kept = parent.symbols[0].code
    parent.aliases[9] = kept
    parent.seal()
    child = learn_pack(obs, config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0), previous=parent)
    gloss = translate_stream(child, [9])[0]
    assert gloss.state == "aliased"
    assert gloss.resolved_code == kept
    assert gloss.english == parent.symbols[0].definition


def test_missing_text_seals_an_unglossed_marker():
    rows = [
        Observation("e1", embedding=[1.0, 0.0, 0.0, 0.0]),
        Observation("e2", embedding=[0.98, 0.02, 0.0, 0.0]),
    ]
    pack = learn_pack(rows, config=LearnConfig(n_symbols=1, min_cluster_size=1, seed=0))
    definition = pack.symbols[0].definition or ""
    assert definition.startswith("[unglossed:")
    gloss = translate_stream(pack, [pack.symbols[0].code])[0]
    assert gloss.english == definition
    assert certify(pack).passed

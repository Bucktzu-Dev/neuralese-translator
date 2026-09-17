from neuralese.aliases import rewrite_stream
from neuralese.contracts import Symbol
from neuralese.translator import translate_stream

from packutil import make_pack


def test_ok_and_unknown_states():
    pack = make_pack()
    glosses = translate_stream(pack, [0, 99])
    assert glosses[0].state == "ok"
    assert "hello" in glosses[0].english.lower()
    assert glosses[0].observation_ids == ["obs-hello"]
    assert glosses[0].pack_checksum == pack.checksum
    assert glosses[1].state == "unknown"
    assert glosses[1].english.startswith("[undecodable:")
    assert glosses[1].confidence == 0.0


def test_aliased_state_uses_resolved_definition():
    pack = make_pack(aliases={7: 0})
    glosses = translate_stream(pack, [7])
    assert glosses[0].state == "aliased"
    assert glosses[0].resolved_code == 0
    assert "hello" in glosses[0].english.lower()


def test_explicit_source_pack_uses_only_that_table():
    pack = make_pack(aliases={"aaaa" * 16: {7: 0}, "bbbb" * 16: {7: 1}})
    glosses = translate_stream(pack, [7], source_pack_checksum="aaaa" * 16)
    assert glosses[0].state == "aliased"
    assert glosses[0].resolved_code == 0


def test_quarantined_state_does_not_use_definition_as_ok():
    import pytest

    from neuralese.contracts import UncertifiedPackError

    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0],
                observation_ids=["obs-1"],
                definition="should not appear as ok",
                quarantined=True,
            )
        ]
    )
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])
    glosses = translate_stream(pack, [0], require_certified=False)
    assert glosses[0].state == "quarantined"
    assert glosses[0].english.startswith("[quarantined:")


def test_unglossed_live_symbol_is_explicit_gap():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0],
                observation_ids=["obs-1"],
                definition=None,
            )
        ]
    )
    glosses = translate_stream(pack, [0], require_certified=False)
    assert glosses[0].state == "ok"
    assert glosses[0].english.startswith("[unglossed:")
    assert glosses[0].confidence == 0.0


def test_unglossed_sentinel_translates_at_zero_confidence():
    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0],
                observation_ids=["obs-1"],
                definition="[unglossed]",
                confidence=0.9,
            )
        ]
    )
    glosses = translate_stream(pack, [0], require_certified=False)
    assert glosses[0].english.startswith("[unglossed:")
    assert glosses[0].confidence == 0.0
    glosses = translate_stream(pack, [0], require_gloss=False)
    assert glosses[0].english.startswith("[unglossed:")
    assert glosses[0].confidence == 0.0


def test_rewrite_stream_matches_alias_map():
    assert rewrite_stream([7, 1, 7], {7: 0}) == [0, 1, 0]


def test_rewrite_stream_follows_multiple_hops():
    assert rewrite_stream([7], {7: 8, 8: 0}) == [0]


def test_rewrite_stream_follows_chain_longer_than_64():
    table = {i: i + 1 for i in range(70)}
    assert rewrite_stream([0], table) == [70]
    assert len(table) > 64


def test_rewrite_stream_rejects_alias_cycles():
    import pytest

    from neuralese.aliases import follow_aliases

    with pytest.raises(ValueError, match="cycle"):
        rewrite_stream([0], {0: 1, 1: 0})
    with pytest.raises(ValueError, match="cycle"):
        follow_aliases(0, {0: 0})
    with pytest.raises(ValueError, match="cycle"):
        follow_aliases(0, {0: 1, 1: 0}, max_hops=2)
    with pytest.raises(ValueError, match="cycle"):
        rewrite_stream([7], {7: 0, 0: 7})
    with pytest.raises(ValueError, match="historical mapping is ambiguous"):
        follow_aliases(7, {7: 0, 0: 1}, current_codes={0, 1})
    with pytest.raises(ValueError, match="historical mapping is ambiguous"):
        follow_aliases(7, {7: 0, 0: 7}, current_codes={0})
    assert follow_aliases(7, {7: 0}, current_codes={0, 1}) == 0


def test_rewrite_stream_versioned_requires_source():
    import pytest

    tables = {"a" * 64: {7: 0}, "b" * 64: {7: 1}}
    with pytest.raises(ValueError, match="source_pack_checksum"):
        rewrite_stream([7], tables)
    assert rewrite_stream([7], tables, source_pack_checksum="a" * 64) == [0]


def test_rewrite_stream_mixed_tables_use_legacy_without_source():
    tables = {"legacy": {7: 0}, "a" * 64: {7: 1}}
    assert rewrite_stream([7], tables) == [0]
    assert rewrite_stream([7], tables, source_pack_checksum="a" * 64) == [1]


def test_rewrite_stream_flat_map_ignores_unrelated_source():
    assert rewrite_stream([7], {7: 0}, source_pack_checksum="a" * 64) == [7]
    assert rewrite_stream([7], {7: 0}) == [0]


def test_rewrite_stream_legacy_source_is_not_a_checksum():
    assert rewrite_stream([7], {7: 0}, source_pack_checksum="legacy") == [7]
    tables = {"legacy": {7: 0}, "a" * 64: {7: 1}}
    assert rewrite_stream([7], tables, source_pack_checksum="legacy") == [7]


def test_rewrite_stream_empty_source_does_not_select_empty_key():
    import pytest

    from neuralese.contracts import select_alias_table

    with pytest.raises(ValueError, match="non-empty"):
        rewrite_stream([7], {"": {7: 0}}, source_pack_checksum="")
    assert select_alias_table({"": {7: 0}}, "") == {}
    assert rewrite_stream([7], {"a" * 64: {7: 0}}, source_pack_checksum="") == [7]


def test_rewrite_stream_rejects_falsey_non_mapping_aliases():
    import pytest

    for aliases in ([], False, ""):
        with pytest.raises(ValueError, match="aliases must be a mapping"):
            rewrite_stream([7], aliases)
    assert rewrite_stream([7], {}) == [7]


def test_rewrite_stream_reuses_shared_alias_suffixes():
    table = {0: 2, 1: 2, 2: 4}
    assert rewrite_stream([0, 1, 2, 9], table) == [4, 4, 4, 9]


def test_translate_stream_resolves_alias_table_once():
    from unittest.mock import patch

    from neuralese.aliases import resolve_alias_table

    # Live codebook codes are 0 and 1; start the historical chain at 7.
    table = {i: i + 1 for i in range(7, 41)}
    table[40] = 0
    pack = make_pack(aliases=table)
    with patch(
        "neuralese.translator.resolve_alias_table", wraps=resolve_alias_table
    ) as spy:
        glosses = translate_stream(pack, [7] * 25, require_certified=False)
    assert spy.call_count == 1
    assert all(g.resolved_code == 0 for g in glosses)
    assert all(g.state == "aliased" for g in glosses)


def test_live_alias_source_is_ambiguous():
    import pytest

    from neuralese.audit import certify
    from neuralese.contracts import UncertifiedPackError

    pack = make_pack(aliases={0: 1})
    cert = certify(pack)
    assert cert.addressable is False
    assert any("historical mapping is ambiguous" in f for f in cert.failures)
    assert cert.passed is False
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])
    with pytest.raises(ValueError, match="historical mapping is ambiguous"):
        pack.resolve_code(0)
    chained = make_pack(aliases={7: 0, 0: 1})
    assert certify(chained).addressable is False
    with pytest.raises(ValueError, match="historical mapping is ambiguous"):
        translate_stream(chained, [7], require_certified=False)


def test_python_api_stream_codes_match_cli():
    import numpy as np
    import pytest
    from fractions import Fraction

    from neuralese.contracts import as_stream_code

    pack = make_pack()
    with pytest.raises(ValueError, match="stream codes must be integers"):
        translate_stream(pack, [True])
    with pytest.raises(ValueError, match="stream codes must be integers"):
        translate_stream(pack, [np.bool_(True)])
    with pytest.raises(ValueError, match="stream codes must be integers"):
        translate_stream(pack, [0.9])
    with pytest.raises(ValueError, match="stream codes must be integers"):
        translate_stream(pack, ["1"])
    with pytest.raises(ValueError, match="stream codes must be integers"):
        translate_stream(pack, [np.array([1])])
    with pytest.raises(ValueError, match="stream codes must be integers"):
        translate_stream(pack, [Fraction(9007199254740993, 2)])
    with pytest.raises(ValueError, match="stream codes must be integers"):
        pack.resolve_code(True)
    with pytest.raises(ValueError, match="stream codes must be integers"):
        pack.resolve_code(np.bool_(True))
    with pytest.raises(ValueError, match="stream codes must be integers"):
        pack.resolve_code(0.9)
    with pytest.raises(ValueError, match="stream codes must be integers"):
        pack.resolve_code("1")
    with pytest.raises(ValueError, match="stream codes must be integers"):
        pack.resolve_code(np.array([1]))
    with pytest.raises(ValueError, match="stream codes must be integers"):
        pack.resolve_code(Fraction(9007199254740993, 2))
    assert pack.resolve_code(1.0) == (1, False)
    assert pack.resolve_code(Fraction(1, 1)) == (1, False)
    assert pack.resolve_code(Fraction(9007199254740993, 1)) == (
        9007199254740993,
        False,
    )
    assert as_stream_code(Fraction(1, 1)) == 1
    assert as_stream_code(Fraction(9007199254740993, 1)) == 9007199254740993
    glosses = translate_stream(pack, [Fraction(10**400, 1)])
    assert glosses[0].code == 10**400
    assert glosses[0].state == "unknown"
    glosses = translate_stream(pack, [1.0])
    assert glosses[0].code == 1
    assert glosses[0].state == "ok"
    glosses = translate_stream(pack, [np.int64(1)])
    assert glosses[0].code == 1
    assert glosses[0].state == "ok"
    glosses = translate_stream(pack, [np.float64(1.0)])
    assert glosses[0].code == 1


def test_all_quarantined_pack_is_not_certified():
    from neuralese.audit import certify
    from neuralese.contracts import UncertifiedPackError

    pack = make_pack(
        symbols=[
            Symbol(
                class_id=0,
                code=0,
                proto_embedding=[1.0],
                observation_ids=["obs-1"],
                definition="[quarantined class 0]",
                quarantined=True,
            )
        ]
    )
    cert = certify(pack)
    assert cert.passed is False
    assert cert.admission_valid is False
    assert any("no admitted symbols" in f for f in cert.failures)
    integrity = certify(pack, policy="integrity")
    assert integrity.passed
    assert integrity.admission_valid is False
    assert any("no admitted symbols" in f for f in integrity.failures)
    import pytest

    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_certify_reuses_one_current_code_set():
    from unittest.mock import patch

    from neuralese.audit import certify
    from neuralese.contracts import SymbolPack

    aliases = {i: i + 1 for i in range(7, 12)}
    aliases[12] = 0
    pack = make_pack(aliases={"a" * 64: aliases, "b" * 64: {13: 0}})
    with patch.object(SymbolPack, "current_codes", wraps=pack.current_codes) as spy:
        cert = certify(pack)
    assert cert.addressable, cert.failures
    assert spy.call_count == 1

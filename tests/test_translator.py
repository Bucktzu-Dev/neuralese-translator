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
    glosses = translate_stream(pack, [0])
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

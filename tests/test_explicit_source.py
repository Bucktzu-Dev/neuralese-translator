import pytest

from neuralese.aliases import normalize_aliases
from neuralese.audit import certify
from neuralese.contracts import select_alias_table
from neuralese.translator import translate_stream

from packutil import make_pack


def test_unknown_source_pack_does_not_merge_unrelated_aliases():
    pack = make_pack(aliases={"cccc" * 16: {7: 0}})
    with pytest.raises(ValueError, match="unresolved source pack checksum"):
        translate_stream(pack, [7], source_pack_checksum="dddd" * 16)
    with pytest.raises(ValueError, match="unresolved source pack checksum"):
        pack.resolve_code(0, source_pack_checksum="dddd" * 16)


def test_empty_source_pack_checksum_is_explicit():
    pack = make_pack(aliases={"cccc" * 16: {7: 0}})
    with pytest.raises(ValueError, match="unresolved source pack checksum"):
        translate_stream(pack, [7], source_pack_checksum="")
    with pytest.raises(ValueError, match="unresolved source pack checksum"):
        translate_stream(pack, [0], source_pack_checksum="")


def test_unresolved_explicit_source_rejected_before_current_lookup():
    pack = make_pack(aliases={"cccc" * 16: {7: 0}})
    for source in ("", "legacy", "not-a-checksum", "dddd" * 16):
        with pytest.raises(ValueError, match="unresolved source pack checksum"):
            translate_stream(pack, [0], source_pack_checksum=source)
        with pytest.raises(ValueError, match="unresolved source pack checksum"):
            pack.resolve_code(0, source_pack_checksum=source)
    glosses = translate_stream(pack, [0], source_pack_checksum="cccc" * 16)
    assert glosses[0].state == "ok"
    parented = make_pack(parent_checksum="eeee" * 16)
    glosses = translate_stream(parented, [0], source_pack_checksum="eeee" * 16)
    assert glosses[0].state == "ok"
    assert parented.resolve_code(0, source_pack_checksum="eeee" * 16) == (0, False)


def test_legacy_source_pack_checksum_does_not_select_legacy_table():
    pack = make_pack(aliases={7: 0})
    with pytest.raises(ValueError, match="unresolved source pack checksum"):
        translate_stream(pack, [7], source_pack_checksum="legacy")
    with pytest.raises(ValueError, match="unresolved source pack checksum"):
        pack.resolve_code(0, source_pack_checksum="legacy")
    assert pack.alias_table("legacy") == {}
    assert pack.alias_table()[7] == 0


def test_empty_alias_source_key_is_not_selectable():
    from neuralese.contracts import select_alias_table

    pack = make_pack()
    data = pack.to_dict()
    data["aliases"] = {"": {"7": 0}}
    with pytest.raises(ValueError, match="non-empty"):
        pack.from_dict(data)
    assert select_alias_table({"": {7: 0}}, "") == {}
    with pytest.raises(ValueError, match="unresolved source pack checksum"):
        translate_stream(pack, [7], source_pack_checksum="")
    with pytest.raises(ValueError, match="unresolved source pack checksum"):
        translate_stream(pack, [0], source_pack_checksum="")


def test_non_checksum_alias_source_is_not_selectable():
    from neuralese.contracts import select_alias_table

    with pytest.raises(ValueError, match="legacy or SHA-256"):
        normalize_aliases({"not-a-checksum": {7: 0}})
    with pytest.raises(ValueError, match="legacy or SHA-256"):
        make_pack(aliases={"not-a-checksum": {7: 0}})
    pack = make_pack()
    data = pack.to_dict()
    data["aliases"] = {"not-a-checksum": {"7": 0}}
    with pytest.raises(ValueError, match="legacy or SHA-256"):
        pack.from_dict(data)
    pack.aliases = {"not-a-checksum": {7: 0}}
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("alias source key is not a pack checksum" in f for f in cert.failures)
    assert select_alias_table(pack.aliases, "not-a-checksum") == {}
    with pytest.raises(ValueError, match="unresolved source pack checksum"):
        translate_stream(
            pack, [7], source_pack_checksum="not-a-checksum", require_certified=False
        )
    with pytest.raises(ValueError, match="unresolved source pack checksum"):
        translate_stream(
            pack, [0], source_pack_checksum="not-a-checksum", require_certified=False
        )
    with pytest.raises(TypeError, match="legacy or SHA-256"):
        pack.to_dict()

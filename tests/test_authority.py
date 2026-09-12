from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuralese.alphabet import learn_pack
from neuralese.audit import certify, check_pack
from neuralese.cli import main
from neuralese.contracts import Observation, TranslationPack
from neuralese.translator import translate_stream
from tests.packutil import example_pack, write_observations


def test_translate_refuses_uncertified_pack() -> None:
    pack = example_pack()
    with pytest.raises(ValueError, match="uncertified"):
        list(translate_stream(["hello"], pack))


def test_translate_allow_uncertified_still_runs() -> None:
    pack = example_pack()
    assert list(translate_stream(["hello"], pack, allow_uncertified=True)) == ["hello"]


def test_certify_fails_closed_on_tampered_parent_checksum() -> None:
    pack = example_pack()
    pack.parent_checksum = "b" * 64
    ok, errors = certify(pack)
    assert ok is False
    assert any("parent_checksum" in error for error in errors)
    assert pack.decision == "rejected"


def test_integrity_policy_does_not_authorize_translation() -> None:
    pack = example_pack()
    pack.checksum = "deadbeef"
    pack.decision = "needs_review"
    ok, errors = check_pack(pack, policy="integrity")
    assert ok is True
    assert errors == ()
    with pytest.raises(ValueError, match="uncertified"):
        list(translate_stream(["hello"], pack))


def test_integrity_policy_still_requires_full_sha256() -> None:
    pack = example_pack()
    pack.checksum = "deadbeef"
    pack.decision = "certified"
    ok, errors = check_pack(pack, policy="integrity")
    assert ok is False
    assert any("64-character SHA-256" in error for error in errors)


def test_cli_audit_rejects_unknown_policy(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(example_pack().dumps(), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["audit", "--pack", str(pack_path), "--policy", "translate-anyway"])


def test_cli_certify_integrity_does_not_mark_certified(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    observations = tmp_path / "obs.jsonl"
    write_observations(observations, [{"observation_id": "o1", "text": "hello world"}])
    pack_path = tmp_path / "pack.json"
    main(["learn", "--observations", str(observations), "--pack", str(pack_path)])
    main(["certify", "--pack", str(pack_path), "--policy", "integrity"])
    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    assert payload["decision"] == "needs_review"
    captured = capsys.readouterr()
    assert "certified=False" in captured.out


def test_cli_translate_refuses_integrity_policy(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(example_pack().dumps(), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["translate", "--pack", str(pack_path), "--text", "hello", "--policy", "integrity"])


def test_cli_translate_refuses_uncertified_pack(tmp_path: Path) -> None:
    observations = tmp_path / "obs.jsonl"
    pack_path = tmp_path / "pack.json"
    write_observations(observations, [{"observation_id": "o1", "text": "hello world"}])
    main(["learn", "--observations", str(observations), "--pack", str(pack_path)])
    with pytest.raises(SystemExit):
        main(["translate", "--pack", str(pack_path), "--text", "hello"])


def test_learn_fails_closed_on_duplicate_observation_ids(tmp_path: Path) -> None:
    observations = tmp_path / "obs.jsonl"
    write_observations(
        observations,
        [
            {"observation_id": "dup", "text": "hello world"},
            {"observation_id": "dup", "text": "other text"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate observation_id"):
        learn_pack(observations, tmp_path / "pack.json")


def test_certify_fails_closed_on_duplicate_observation_ids() -> None:
    pack = example_pack()
    pack.observations = pack.observations + (
        Observation(observation_id="obs-1", text="another copy of the same id"),
    )
    ok, errors = certify(pack)
    assert ok is False
    assert any("duplicate observation_id" in error for error in errors)


def test_int_and_str_observation_ids_are_duplicates() -> None:
    pack = example_pack()
    pack.observations = (
        Observation(observation_id="1", text="first copy"),
        Observation(observation_id=1, text="second copy"),  # type: ignore[arg-type]
    )
    ok, errors = certify(pack)
    assert ok is False
    assert any("duplicate observation_id: 1" in error for error in errors)


def test_certify_fails_closed_on_empty_observation_text() -> None:
    pack = example_pack()
    pack.observations = pack.observations + (
        Observation(observation_id="blank", text="   "),
    )
    ok, errors = certify(pack)
    assert ok is False
    assert any("empty text" in error for error in errors)


def test_from_dict_does_not_coerce_empty_decoder_version() -> None:
    payload = example_pack().to_dict()
    payload["decoder_version"] = ""
    pack = TranslationPack.from_dict(payload)
    assert pack.decoder_version == ""
    ok, errors = check_pack(pack, policy="integrity")
    assert ok is False
    assert any("decoder_version must be 0.1.1" in error for error in errors)


def test_from_dict_preserves_null_decision_as_empty() -> None:
    payload = example_pack().to_dict()
    payload["decision"] = None
    pack = TranslationPack.from_dict(payload)
    assert pack.decision == ""
    ok, errors = check_pack(pack, policy="integrity")
    assert ok is False
    assert any("decision must be one of" in error for error in errors)


def test_from_dict_coerces_null_metadata_to_empty_mapping() -> None:
    payload = example_pack().to_dict()
    payload["metadata"] = None
    payload["atoms"][0]["metadata"] = None
    payload["observations"][0]["metadata"] = None
    pack = TranslationPack.from_dict(payload)
    assert pack.metadata == {}
    assert pack.atoms[0].metadata == {}
    assert pack.observations[0].metadata == {}


def test_integrity_policy_rejects_unknown_decision() -> None:
    pack = example_pack()
    pack.decision = "maybe"
    ok, errors = check_pack(pack, policy="integrity")
    assert ok is False
    assert any("decision must be one of" in error for error in errors)

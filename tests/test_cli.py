import json
from pathlib import Path

import pytest

from neuralese.cli import main

TOY_DIR = Path(__file__).resolve().parents[1] / "examples" / "toy_stream"


def test_cli_learn_translate_certify(tmp_path, capsys):
    pack_path = tmp_path / "pack.json"
    rc = main(
        [
            "learn",
            str(TOY_DIR / "observations.jsonl"),
            "-o",
            str(pack_path),
            "--n-symbols",
            "3",
        ]
    )
    assert rc == 0
    assert pack_path.exists()
    learned = json.loads(capsys.readouterr().out)
    assert "checksum" in learned

    rc = main(["translate", str(pack_path), str(TOY_DIR / "stream.json")])
    assert rc == 0
    glosses = json.loads(capsys.readouterr().out)
    assert isinstance(glosses, list)
    assert glosses[-1]["state"] == "unknown"

    rc = main(["certify", str(pack_path), "--fail-on-undecodable"])
    assert rc == 0
    cert = json.loads(capsys.readouterr().out)
    assert cert["passed"] is True

    rc = main(
        [
            "certify",
            str(pack_path),
            "--fail-on-undecodable",
            "--observations",
            str(TOY_DIR / "observations.jsonl"),
        ]
    )
    assert rc == 0
    cert = json.loads(capsys.readouterr().out)
    assert cert["passed"] is True
    assert cert["details"]["observations_checked"] is True


def test_cli_translate_rejects_integrity_policy(tmp_path, capsys):
    pack_path = tmp_path / "pack.json"
    main(
        [
            "learn",
            str(TOY_DIR / "observations.jsonl"),
            "-o",
            str(pack_path),
            "--n-symbols",
            "3",
        ]
    )
    capsys.readouterr()
    rc = main(
        ["translate", str(pack_path), str(TOY_DIR / "stream.json"), "--policy", "integrity"]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "does not authorize translation" in err


def test_cli_integrity_rejected_even_with_allow_uncertified(tmp_path, capsys):
    pack_path = tmp_path / "pack.json"
    main(
        [
            "learn",
            str(TOY_DIR / "observations.jsonl"),
            "-o",
            str(pack_path),
            "--n-symbols",
            "3",
        ]
    )
    capsys.readouterr()
    rc = main(
        [
            "translate",
            str(pack_path),
            str(TOY_DIR / "stream.json"),
            "--policy",
            "integrity",
            "--allow-uncertified",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "does not authorize translation" in err


def test_cli_certify_fails_on_undecodable(tmp_path, capsys):
    pack_path = tmp_path / "pack.json"
    main(
        [
            "learn",
            str(TOY_DIR / "observations.jsonl"),
            "-o",
            str(pack_path),
            "--n-symbols",
            "3",
        ]
    )
    capsys.readouterr()
    data = json.loads(pack_path.read_text())
    data["symbols"][0]["observation_ids"] = []
    data["symbols"][0]["quarantined"] = False
    pack_path.write_text(json.dumps(data))
    rc = main(["certify", str(pack_path), "--fail-on-undecodable"])
    assert rc == 1
    cert = json.loads(capsys.readouterr().out)
    assert cert["passed"] is False


def test_cli_audit_unknown_policy_is_clean_error(capsys):
    with pytest.raises(SystemExit) as err:
        main(["audit", "pack.json", "--policy", "typo"])
    assert err.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_certify_unknown_policy_is_clean_error(capsys):
    with pytest.raises(SystemExit) as err:
        main(["certify", "pack.json", "--policy", "typo"])
    assert err.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_learn_duplicate_ids_is_clean_error(tmp_path, capsys):
    obs = tmp_path / "obs.jsonl"
    obs.write_text(
        '{"observation_id":"x","text":"hello there friend"}\n'
        '{"observation_id":"x","text":"hello there pal"}\n'
    )
    rc = main(["learn", str(obs), "-o", str(tmp_path / "pack.json"), "--n-symbols", "1"])
    assert rc == 1
    assert "duplicate observation_id" in capsys.readouterr().err
    assert not (tmp_path / "pack.json").exists()

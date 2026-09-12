import json
from pathlib import Path

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

import json

import pytest

from neuralese.audit import certify
from neuralese.cli import main
from neuralese.contracts import UncertifiedPackError
from neuralese.translator import translate_stream

from packutil import make_pack


def test_none_evidence_key_is_not_stringified():
    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    digest = pack.evidence[obs_id]
    data = pack.to_dict()
    data["evidence"] = {None: digest, obs_id: digest}
    loaded = pack.from_dict(data)
    assert None in loaded.evidence
    assert "None" not in loaded.evidence
    cert = certify(loaded)
    assert not cert.evidence_valid
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("not a non-blank string" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])


def test_non_mapping_evidence_survives_to_dict_round_trip():
    pack = make_pack()
    data = pack.to_dict()
    data["evidence"] = []
    loaded = pack.from_dict(data)
    dumped = loaded.to_dict()
    assert dumped["evidence"] == []
    again = pack.from_dict(dumped)
    assert again.evidence == []
    cert = certify(again)
    assert not cert.passed
    assert not cert.evidence_valid
    data["evidence"] = None
    loaded = pack.from_dict(data)
    assert loaded.to_dict()["evidence"] is None
    cert = certify(loaded)
    assert not cert.passed


def test_non_numeric_guard_metric_fails_schema():
    pack = make_pack()
    pack.guards.kappa_avg = "0.8"
    pack.seal()
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("guards.kappa_avg is not a number" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_loaded_string_guard_metric_fails_schema():
    pack = make_pack()
    data = pack.to_dict()
    data["guards"]["min_survival"] = "1.0"
    loaded = pack.from_dict(data)
    assert loaded.guards.min_survival == "1.0"
    loaded.seal()
    cert = certify(loaded)
    assert not cert.passed
    assert any("guards.min_survival is not a number" in f for f in cert.failures)


def test_translate_uses_caller_tau_residual_not_pack_metadata():
    pack = make_pack(reconstruction_error=0.6)
    pack.metadata["tau_residual"] = 0.9
    pack.seal()
    cert = certify(pack)
    assert not cert.passed
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])
    assert certify(pack, tau_residual=0.9).passed
    glosses = translate_stream(pack, [0], tau_residual=0.9)
    assert glosses[0].state == "ok"


def test_cli_translate_tau_residual_is_operator_gate(tmp_path, capsys):
    from pathlib import Path

    toy = Path(__file__).resolve().parents[1] / "examples" / "toy_stream"
    pack_path = tmp_path / "pack.json"
    stream = tmp_path / "stream.json"
    rc = main(
        [
            "learn",
            str(toy / "observations.jsonl"),
            "-o",
            str(pack_path),
            "--n-symbols",
            "3",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    data = json.loads(pack_path.read_text())
    data["reconstruction_error"] = 0.6
    data["metadata"]["tau_residual"] = 0.9
    if data.get("guards"):
        data["guards"]["reconstruction_error"] = 0.6
    from neuralese.contracts import SymbolPack

    pack = SymbolPack.from_dict(data)
    pack.seal()
    pack_path.write_text(json.dumps(pack.to_dict(), indent=2) + "\n")
    stream.write_text("[0]\n")
    rc = main(["translate", str(pack_path), str(stream)])
    assert rc == 1
    failed = capsys.readouterr()
    assert failed.out == ""
    cert = json.loads(failed.err)
    assert cert["passed"] is False
    rc = main(["translate", str(pack_path), str(stream), "--tau-residual", "0.9"])
    assert rc == 0
    glosses = json.loads(capsys.readouterr().out)
    assert glosses[0]["state"] in {"ok", "unknown", "aliased"}

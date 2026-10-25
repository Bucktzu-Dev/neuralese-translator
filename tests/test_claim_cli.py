"""The claim CLI prints an interpretation and a computed verdict."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from neuralese.claims.cli import main
from test_claim_verifier import POSITIVE

ROOT = Path(__file__).resolve().parents[1]
ONE_INDEX = ROOT / "examples" / "claims" / "one_index_coordinate_change.txt"
TWO_INDEX = ROOT / "examples" / "claims" / "two_index_coordinate_change.txt"


def test_one_index_example_matches_the_reference_fixture():
    assert ONE_INDEX.read_text(encoding="utf-8") == POSITIVE


def test_cli_prints_the_one_index_derivation(capsys):
    code = main([str(ONE_INDEX)])
    assert code == 0
    out = capsys.readouterr().out
    assert "interpreted claim" in out
    assert "direction new_to_old" in out
    assert "log_branch log(1) = 0" in out
    assert "assumptions:" in out
    assert "span:" in out
    assert "ast:" in out
    assert "outcome: PROVED" in out
    assert "residual: 0" in out
    assert "D0.t: (D Psi) compose Phi :: Y -> 1 + t" in out
    assert "D1.u: (D Psi) compose Phi :: 1 -> 1" in out
    assert "Phi o Psi Y: Y" in out


def test_cli_prints_the_two_index_derivation(capsys):
    code = main([str(TWO_INDEX)])
    assert code == 0
    out = capsys.readouterr().out
    assert "coordinates Y, X1, X2" in out
    assert "direction new_to_old" in out
    assert "log_branch log(1) = 0" in out
    assert "Compose(MultiplyCoefficient(Y), Partial(Y))" in out
    assert "Partial(X1)" in out
    assert "Partial(X2)" in out
    assert "outcome: PROVED" in out
    assert "D0.t: (D Psi) compose Phi :: Y -> 1 + t" in out
    assert "D1.u1: (D Psi) compose Phi :: 1 -> 1" in out
    assert "D2.u2: (D Psi) compose Phi :: 1 -> 1" in out


def test_cli_prints_a_counterexample(capsys):
    path = ROOT / "examples" / "claims" / "eval" / "e02_wrong_coefficient.txt"
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "outcome: DISPROVED" in out
    assert "function: p" in out
    assert "actual: 1" in out
    assert "claimed: 2" in out
    assert "residual: -1" in out


def test_cli_asks_for_clarification_without_filling_a_direction(capsys):
    path = ROOT / "examples" / "claims" / "eval" / "e09_missing_direction.txt"
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "direction (absent)" in out
    assert "outcome: UNRESOLVED" in out
    assert "reason: missing_pullback_direction" in out
    assert "clarification:" in out
    assert "none was inferred" in out
    assert "direction new_to_old" not in out


def test_cli_prints_invalid_span(capsys):
    path = ROOT / "examples" / "claims" / "eval" / "e04_malformed.txt"
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "outcome: INVALID_INPUT" in out
    assert "reason: malformed_claim" in out
    assert "span: 0.." in out


def test_module_entrypoint_matches_main():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "neuralese.claims", str(ONE_INDEX)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "outcome: PROVED" in proc.stdout
    assert proc.stderr == ""


def test_missing_file_does_not_invent_a_claim(capsys):
    code = main(["examples/claims/does-not-exist.txt"])
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not found" in captured.err

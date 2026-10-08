"""User-facing errors from the actual training and prediction processes."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from tennpot.cli._errors import report_input_errors


def _command(command, arguments):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [sys.executable, "-m", f"tennpot.cli.{command}", *arguments],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("command", ["train", "predict"])
def test_invalid_input_is_concise_unless_debug_requested(tmp_path, command, debug):
    source = tmp_path / "input.extxyz"
    output = tmp_path / "output"
    if command == "train":
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms.calc = SinglePointCalculator(atoms, energy=-1, forces=np.zeros((2, 3)))
        expected = "missing required per-structure value 'electronic_temperature'"
        arguments = [str(source), "--output", str(output)]
    else:
        atoms = Atoms("Ge", positions=[[0, 0, 0]])
        expected = "this checkpoint supports only Si; received ['Ge']"
        arguments = [str(source), str(output)]
    write(source, atoms, format="extxyz")
    arguments += ["--device", "cpu"]
    if debug:
        arguments += ["--debug"]

    result = _command(command, arguments)

    assert result.returncode == (1 if debug else 2)
    assert expected in result.stderr
    assert ("Traceback (most recent call last)" in result.stderr) == debug
    assert not output.exists()
    if not debug:
        assert len(result.stderr.splitlines()) == 1
        assert ": error: " in result.stderr


@pytest.mark.parametrize("command", ["train", "predict"])
def test_missing_input_file_reports_error_without_traceback(tmp_path, command):
    source = tmp_path / "missing.extxyz"
    output = tmp_path / "output"
    arguments = [str(source)]
    arguments += ["--output", str(output)] if command == "train" else [str(output)]
    result = _command(command, arguments + ["--device", "cpu"])
    assert result.returncode == 2
    assert "missing.extxyz" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_unexpected_errors_are_not_suppressed():
    parser = argparse.ArgumentParser()
    with pytest.raises(RuntimeError, match="unexpected failure"):
        with report_input_errors(parser):
            raise RuntimeError("unexpected failure")

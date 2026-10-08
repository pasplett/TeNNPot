from dataclasses import replace
import json
import subprocess
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian, FixCom
from ase.io import read, write
from ase.md.verlet import VelocityVerlet

from tennpot import _md_checkpoint
from tennpot.ase_calculator import TeNNPotCalculator
from tennpot.checkpoint import save_checkpoint
from tennpot.cli import plot_properties, reproduce_fig3
from tennpot.config import build_model, load_model_config
from tennpot.properties import (
    MDInitialState,
    MDInitialStateEnsemble,
    rmsd_trajectories,
    run_md_ensemble,
    run_md_trajectory,
    write_property_table,
)

matplotlib.use("Agg")


def _model():
    torch.manual_seed(19)
    return build_model(replace(load_model_config("silicon_tiny"), dtype="float64"))


def _state(constraint=None, velocity_scale=1):
    atoms = Atoms(
        "Si3", positions=[[0, 0, 0], [1.1, 0, 0], [0.2, 1, 0]], cell=[10, 10, 10]
    )
    atoms.set_masses([28, 29, 30])
    if constraint is not None:
        atoms.set_constraint(constraint)
    atoms.set_velocities(
        np.array([[0.01, 0, 0], [0, 0.02, 0], [0, 0, 0.01]]) * velocity_scale
    )
    return MDInitialState(atoms, atoms.positions.copy() - 0.05)


def _run(state, checkpoint, steps=6, **kwargs):
    return run_md_trajectory(
        _model(),
        state,
        12631,
        steps=steps,
        checkpoint_path=checkpoint,
        checkpoint_interval=2,
        **kwargs,
    )


def _count_steps(monkeypatch):
    calls = []
    original = VelocityVerlet.step

    def step(self, *args, **kwargs):
        calls.append(self.nsteps)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(VelocityVerlet, "step", step)
    return calls


@pytest.mark.parametrize(
    "constraint",
    [None, FixAtoms(indices=[0]), FixCartesian(0, mask=[True, False, False])],
)
def test_disk_restart_matches_continuous_md_and_preserves_state(tmp_path, constraint):
    state = _state(constraint)
    original_positions = state.atoms.positions.copy()
    original_momenta = state.atoms.get_momenta().copy()
    full_path = tmp_path / "full.traj"
    split_path = tmp_path / "split.traj"
    full = _run(state, full_path)
    _run(state, split_path, steps=2)
    resumed = _run(state, split_path)
    np.testing.assert_array_equal(resumed, full)
    expected, actual = read(full_path), read(split_path)
    np.testing.assert_array_equal(actual.positions, expected.positions)
    np.testing.assert_array_equal(actual.get_momenta(), expected.get_momenta())
    np.testing.assert_array_equal(actual.get_masses(), state.atoms.get_masses())
    assert actual.info["tennpot_md"]["completed_steps"] == 6
    assert len(actual.constraints) == len(state.atoms.constraints)
    np.testing.assert_array_equal(state.atoms.positions, original_positions)
    np.testing.assert_array_equal(state.atoms.get_momenta(), original_momenta)


def test_interruption_during_force_calculation_resumes_in_fresh_process(
    tmp_path, monkeypatch
):
    state = _state(FixAtoms(indices=[0]))
    baseline_path = tmp_path / "baseline.traj"
    expected = _run(state, baseline_path)
    checkpoint = tmp_path / "interrupted.traj"
    step_number = 0
    original_step = VelocityVerlet.step
    original_calculate = TeNNPotCalculator.calculate

    def step(self, *args, **kwargs):
        nonlocal step_number
        step_number += 1
        return original_step(self, *args, **kwargs)

    def calculate(self, *args, **kwargs):
        if step_number == 3:
            raise KeyboardInterrupt("interrupted during a force evaluation")
        return original_calculate(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(VelocityVerlet, "step", step)
        patch.setattr(TeNNPotCalculator, "calculate", calculate)
        with pytest.raises(KeyboardInterrupt):
            _run(state, checkpoint)
    assert read(checkpoint).info["tennpot_md"]["completed_steps"] == 2
    state.atoms.info["reference"] = state.reference_positions_angstrom
    initial_path = tmp_path / "initial.traj"
    write(initial_path, state.atoms)
    model = _model()
    model_path = tmp_path / "model.pt"
    save_checkpoint(model_path, model, model.config)
    result_path = tmp_path / "resumed.npy"
    code = (
        "import sys,numpy as np,torch; from ase.io import read; "
        "from tennpot.checkpoint import load_checkpoint; "
        "from tennpot.properties import MDInitialState,run_md_trajectory; "
        f"torch.set_num_threads({torch.get_num_threads()}); "
        "torch.use_deterministic_algorithms("
        f"{torch.are_deterministic_algorithms_enabled()}); "
        "model,_=load_checkpoint(sys.argv[1]); atoms=read(sys.argv[2]); "
        "state=MDInitialState(atoms,atoms.info['reference']); "
        "rmsd=run_md_trajectory(model,state,12631,steps=6,"
        "checkpoint_path=sys.argv[3],checkpoint_interval=2); np.save(sys.argv[4],rmsd)"
    )
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(model_path),
            str(initial_path),
            str(checkpoint),
            str(result_path),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    np.testing.assert_array_equal(np.load(result_path), expected)
    actual, baseline = read(checkpoint), read(baseline_path)
    np.testing.assert_array_equal(actual.positions, baseline.positions)
    np.testing.assert_array_equal(actual.get_momenta(), baseline.get_momenta())


def test_completed_and_shorter_requests_do_no_integration(tmp_path, monkeypatch):
    state = _state()
    checkpoint = tmp_path / "run.traj"
    expected = _run(state, checkpoint)
    original = checkpoint.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("completed MD must not evaluate forces or integrate")

    monkeypatch.setattr(VelocityVerlet, "step", forbidden)
    monkeypatch.setattr(TeNNPotCalculator, "calculate", forbidden)
    for target in (6, 3, 0):
        np.testing.assert_array_equal(
            _run(state, checkpoint, steps=target), expected[: target + 1]
        )
    assert checkpoint.read_bytes() == original


def test_zero_step_checkpoint_can_be_extended(tmp_path):
    state = _state()
    checkpoint = tmp_path / "run.traj"
    assert _run(state, checkpoint, steps=0).shape == (1,)
    np.testing.assert_array_equal(
        _run(state, checkpoint, steps=2), _run(state, None, steps=2)
    )


def test_ensemble_recovers_interrupted_member_and_skips_completed_members(
    tmp_path, monkeypatch
):
    states = (
        _state(velocity_scale=1),
        _state(velocity_scale=2),
        _state(velocity_scale=3),
    )
    expected = run_md_ensemble(_model(), states, 12631, steps=4)
    cache = tmp_path / "cache"
    original = VelocityVerlet.step
    calls = []

    def interrupt(self, *args, **kwargs):
        calls.append(self.nsteps)
        if len(calls) == 7:
            raise RuntimeError("stop in the second member")
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(VelocityVerlet, "step", interrupt)
        with pytest.raises(RuntimeError, match="second member"):
            run_md_ensemble(
                _model(),
                states,
                12631,
                steps=4,
                checkpoint_dir=cache,
                checkpoint_interval=2,
            )
    calls = _count_steps(monkeypatch)
    messages = []
    progress = []
    actual = run_md_ensemble(
        _model(),
        states,
        12631,
        steps=4,
        checkpoint_dir=cache,
        checkpoint_interval=2,
        status=messages.append,
        progress=lambda number, total: progress.append((number, total)),
    )
    assert len(calls) == 6  # two remaining steps in member 2, four in member 3
    assert any("run 1/3: reusing" in message for message in messages)
    assert any("run 2/3: resuming at step 2/4" in message for message in messages)
    assert any("run 3/3: starting" in message for message in messages)
    assert progress == [(1, 3), (2, 3), (3, 3)]
    for name in (
        "time_fs",
        "individual_angstrom",
        "mean_angstrom",
        "standard_error_angstrom",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_seeded_ensemble_extends_steps_and_adds_members(tmp_path, monkeypatch):
    atoms = _state().atoms
    del atoms.arrays["momenta"]
    cache = tmp_path / "cache"
    first = rmsd_trajectories(
        _model(), 12631, atoms=atoms, steps=2, runs=1, seed=17, checkpoint_dir=cache
    )
    calls = _count_steps(monkeypatch)
    extended = rmsd_trajectories(
        _model(), 12631, atoms=atoms, steps=4, runs=2, seed=17, checkpoint_dir=cache
    )
    assert len(calls) == 6
    assert len(list(cache.glob("*.traj"))) == 2
    expected = rmsd_trajectories(_model(), 12631, atoms=atoms, steps=4, runs=2, seed=17)
    np.testing.assert_array_equal(
        extended.individual_angstrom, expected.individual_angstrom
    )
    np.testing.assert_array_equal(
        extended.individual_angstrom[0, :3], first.individual_angstrom[0]
    )
    assert not atoms.has("momenta")


@pytest.mark.parametrize(
    "change",
    [
        "temperature",
        "timestep",
        "weights",
        "positions",
        "momenta",
        "masses",
        "cell",
        "pbc",
        "reference",
        "constraints",
        "software",
    ],
)
def test_incompatible_explicit_checkpoint_is_rejected(tmp_path, monkeypatch, change):
    state = _state()
    checkpoint = tmp_path / "run.traj"
    _run(state, checkpoint, steps=2)
    model = _model()
    kwargs = {"electronic_temperature_kelvin": 12631, "timestep_fs": 2.0}
    if change == "temperature":
        kwargs["electronic_temperature_kelvin"] = 22104
    elif change == "timestep":
        kwargs["timestep_fs"] = 1.0
    elif change == "weights":
        with torch.no_grad():
            next(model.parameters()).add_(0.1)
    elif change == "positions":
        state.atoms.positions[0, 0] += 0.1
    elif change == "momenta":
        state.atoms.set_momenta(state.atoms.get_momenta() * 2)
    elif change == "masses":
        state.atoms.set_masses([27, 28, 29])
    elif change == "cell":
        state.atoms.set_cell([11, 10, 10])
    elif change == "pbc":
        state.atoms.set_pbc(True)
    elif change == "reference":
        state.reference_positions_angstrom[0, 0] += 0.1
    elif change == "constraints":
        state.atoms.set_constraint(FixAtoms(indices=[0]))
    else:
        monkeypatch.setattr(_md_checkpoint.ase, "__version__", "different-version")
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="incompatible MD checkpoint"):
        run_md_trajectory(model, state, steps=3, checkpoint_path=checkpoint, **kwargs)
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize(
    "damage",
    [
        "schema",
        "step",
        "length",
        "nan_history",
        "negative_history",
        "last_rmsd",
        "reference",
        "missing_momenta",
        "nan_positions",
        "nan_momenta",
        "masses",
        "cell",
        "missing_metadata",
    ],
)
def test_corrupt_checkpoints_fail_without_being_overwritten(tmp_path, damage):
    state = _state()
    checkpoint = tmp_path / "run.traj"
    _run(state, checkpoint, steps=2)
    atoms = read(checkpoint)
    metadata = atoms.info["tennpot_md"]
    if damage == "schema":
        metadata["schema_version"] = 99
    elif damage == "step":
        metadata["completed_steps"] = -1
    elif damage == "length":
        metadata["rmsd_angstrom"] = metadata["rmsd_angstrom"][:-1]
    elif damage == "nan_history":
        metadata["rmsd_angstrom"][1] = np.nan
    elif damage == "negative_history":
        metadata["rmsd_angstrom"][1] = -0.1
    elif damage == "last_rmsd":
        metadata["rmsd_angstrom"][-1] += 0.1
    elif damage == "reference":
        metadata["reference_positions_angstrom"][0, 0] += 0.1
    elif damage == "missing_momenta":
        del atoms.arrays["momenta"]
    elif damage == "nan_positions":
        atoms.positions[0, 0] = np.nan
    elif damage == "nan_momenta":
        atoms.arrays["momenta"][0, 0] = np.nan
    elif damage == "masses":
        atoms.set_masses([27, 28, 29])
    elif damage == "cell":
        atoms.set_cell([11, 10, 10])
    else:
        atoms.info.clear()
    write(checkpoint, atoms)
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="invalid or incompatible MD checkpoint"):
        _run(state, checkpoint)
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_checkpoint_write_keeps_prior_snapshot(tmp_path, monkeypatch, failure):
    state = _state()
    checkpoint = tmp_path / "run.traj"
    _run(state, checkpoint, steps=2)
    before = checkpoint.read_bytes()
    original = _md_checkpoint.Trajectory

    class BrokenWriter:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, atoms):
            raise OSError("simulated checkpoint write failure")

    with monkeypatch.context() as patch:
        if failure == "write":
            patch.setattr(
                _md_checkpoint,
                "Trajectory",
                lambda path, mode, **kwargs: (
                    original(path, mode, **kwargs) if mode == "r" else BrokenWriter()
                ),
            )
        else:

            def replace(*args):
                raise OSError("simulated checkpoint replace failure")

            patch.setattr(_md_checkpoint.os, "replace", replace)
        with pytest.raises(OSError, match="simulated checkpoint"):
            _run(state, checkpoint)
    assert checkpoint.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
    np.testing.assert_array_equal(_run(state, checkpoint), _run(state, None))


def test_changed_calculations_use_separate_cache_files(tmp_path):
    state = _state()
    cache = tmp_path / "cache"
    run_md_ensemble(_model(), [state], 12631, steps=2, checkpoint_dir=cache)
    original = next(cache.glob("*.traj"))
    before = original.read_bytes()
    for temperature, timestep in ((22104, 2.0), (12631, 1.0)):
        run_md_ensemble(
            _model(),
            [state],
            temperature,
            timestep_fs=timestep,
            steps=2,
            checkpoint_dir=cache,
        )
    run_md_ensemble(
        _model(), [_state(velocity_scale=2)], 12631, steps=2, checkpoint_dir=cache
    )
    assert len(list(cache.glob("*.traj"))) == 4
    assert original.read_bytes() == before


def test_other_constraints_can_run_without_checkpointing(tmp_path):
    state = _state(FixCom())
    with pytest.raises(ValueError, match="FixAtoms and FixCartesian"):
        _run(state, tmp_path / "run.traj", steps=1)
    assert _run(state, None, steps=1).shape == (2,)


@pytest.mark.parametrize(
    "options",
    [
        {"steps": -1},
        {"steps": 1.5},
        {"steps": True},
        {"timestep_fs": 0},
        {"timestep_fs": np.nan},
        {"electronic_temperature_kelvin": np.inf},
        {"checkpoint_interval": 0},
        {"checkpoint_interval": 1.5},
    ],
)
def test_invalid_md_settings_fail_before_writing(tmp_path, options):
    checkpoint = tmp_path / "run.traj"
    kwargs = {"steps": 1, "electronic_temperature_kelvin": 12631, **options}
    with pytest.raises(ValueError):
        run_md_trajectory(_model(), _state(), checkpoint_path=checkpoint, **kwargs)
    assert not checkpoint.exists()


def _prepared_ensemble(directory):
    states = (_state(), _state(velocity_scale=2))
    reference = states[0].reference_positions_angstrom
    write_property_table(
        directory / "reference.csv",
        {
            f"position_angstrom_{axis}": reference[:, index]
            for index, axis in enumerate("xyz")
        },
    )
    files = []
    for number, state in enumerate(states):
        name = f"run_{number}.csv"
        files.append(name)
        velocities = state.atoms.get_velocities()
        write_property_table(
            directory / name,
            {
                **{
                    f"position_angstrom_{axis}": state.atoms.positions[:, index]
                    for index, axis in enumerate("xyz")
                },
                **{
                    f"velocity_angstrom_per_ase_time_{axis}": velocities[:, index]
                    for index, axis in enumerate("xyz")
                },
            },
        )
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "electronic_temperature_kelvin": 12631,
                "timestep_fs": 2,
                "cell_angstrom": states[0].atoms.cell.array.tolist(),
                "pbc": [False] * 3,
                "reference_positions_file": "reference.csv",
                "run_files": files,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("prepared", [False, True])
def test_md_command_resumes_extends_and_can_disable_checkpointing(
    tmp_path, monkeypatch, capsys, prepared
):
    monkeypatch.setattr(
        plot_properties, "_load_selected_model", lambda *args: (_model(), "test")
    )
    output = tmp_path / "output"
    if prepared:
        directory = tmp_path / "ensemble"
        _prepared_ensemble(directory)
        inputs = ["--initial-state-ensemble", str(directory)]
    else:
        structure = tmp_path / "structure.extxyz"
        write(structure, _state().atoms)
        inputs = [str(structure), "--temperature", "12631"]
    common = inputs + ["--output", str(output), "--checkpoint-interval", "1"]
    calls = _count_steps(monkeypatch)
    plot_properties.tennpot_md_main(common + ["--steps", "2", "--runs", "1"])
    plot_properties.tennpot_md_main(common + ["--steps", "2", "--runs", "1"])
    assert len(calls) == 2
    assert "reusing at step 2/2" in capsys.readouterr().out
    plot_properties.tennpot_md_main(common + ["--steps", "4", "--runs", "2"])
    assert len(calls) == 8
    snapshots = {p: p.read_bytes() for p in (output / "md_cache").glob("*.traj")}
    assert len(snapshots) == 2
    plot_properties.tennpot_md_main(
        common + ["--steps", "2", "--runs", "1", "--no-resume"]
    )
    assert len(calls) == 10
    assert all(path.read_bytes() == data for path, data in snapshots.items())
    assert (output / "md.csv").is_file()


@pytest.mark.parametrize("prepared", [False, True])
def test_figure3_command_checkpoints_each_temperature(tmp_path, monkeypatch, prepared):
    monkeypatch.setattr(
        reproduce_fig3, "_load_selected_model", lambda *args: (_model(), "test")
    )
    monkeypatch.setattr(
        reproduce_fig3,
        "_paper_cohesive_curves",
        lambda *args, **kwargs: {
            "lattice_constant_angstrom": [4, 5],
            "diamond": [-1, -2],
        },
    )
    monkeypatch.setattr(
        reproduce_fig3,
        "diamond_thermodynamics",
        lambda *args, **kwargs: SimpleNamespace(
            temperature_kelvin=torch.tensor([1000, 2000]),
            free_energy_ev=torch.tensor([1, 2]),
            internal_energy_ev=torch.tensor([1, 2]),
            heat_capacity_ev_per_kelvin=torch.tensor([1, 2]),
        ),
    )
    monkeypatch.setattr(reproduce_fig3, "_plot_figure", lambda *args, **kwargs: None)
    output = tmp_path / "output"
    common = [
        "--output",
        str(output),
        "--skip-phonons",
        "--md-checkpoint-interval",
        "1",
    ]
    if prepared:
        monkeypatch.setattr(
            reproduce_fig3,
            "find_md_initial_state_ensemble",
            lambda directory, temperature, runs: MDInitialStateEnsemble(
                (_state(), _state(velocity_scale=2))[:runs], temperature, 2
            ),
        )
    else:
        structure = tmp_path / "structure.extxyz"
        write(structure, _state().atoms)
        common += ["--md-structure", str(structure)]
    calls = _count_steps(monkeypatch)
    short = common + ["--md-steps", "2", "--md-runs", "1"]
    reproduce_fig3.main(short)
    reproduce_fig3.main(short)
    assert len(calls) == 4
    reproduce_fig3.main(common + ["--md-steps", "4", "--md-runs", "2"])
    assert len(calls) == 16
    snapshots = {p: p.read_bytes() for p in (output / "md_cache").glob("*.traj")}
    assert len(snapshots) == 4
    reproduce_fig3.main(short + ["--no-resume"])
    assert len(calls) == 20
    assert all(path.read_bytes() == data for path, data in snapshots.items())
    assert (output / "rmsd_12631.csv").is_file()
    assert (output / "rmsd_22104.csv").is_file()


def test_float32_model_cache_survives_calculator_conversion(tmp_path, monkeypatch):
    torch.manual_seed(19)
    model = build_model(load_model_config("silicon_tiny"))
    assert next(model.parameters()).dtype == torch.float32
    cache = tmp_path / "cache"
    state = _state()
    expected = run_md_ensemble(model, [state], 12631, steps=2, checkpoint_dir=cache)
    assert next(model.parameters()).dtype == torch.float64
    calls = _count_steps(monkeypatch)
    for candidate in [model, None]:
        if candidate is None:
            torch.manual_seed(19)
            candidate = build_model(load_model_config("silicon_tiny"))
        actual = run_md_ensemble(
            candidate, [state], 12631, steps=2, checkpoint_dir=cache
        )
        np.testing.assert_array_equal(
            actual.individual_angstrom, expected.individual_angstrom
        )
    assert not calls
    assert len(list(cache.glob("*.traj"))) == 1

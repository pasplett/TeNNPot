"""Physical-property calculations used for Figure 3 of the paper."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase import Atoms
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.phonons import Phonons

from tennpot._md_checkpoint import (
    _validate_atoms,
    load_md_checkpoint,
    md_checkpoint_path,
    md_checkpoint_settings,
    save_md_checkpoint,
    validate_md_options,
)
from tennpot.ase_calculator import TeNNPotCalculator
from tennpot.physics import (
    BOHR_TO_ANGSTROM,
    electronic_thermodynamics,
)

CRYSTAL_STRUCTURES = ("diamond", "fcc", "bcc", "sc")
PLANCK_EV_SECOND = 4.135667696e-15
PAPER_DIAMOND_LATTICE_ANGSTROM = 5.398718823353496


def silicon_crystal(structure, lattice_constant, *, repetitions=(1, 1, 1)):
    """Build a periodic conventional cell for one of the structures in Fig. 3a."""
    if structure not in CRYSTAL_STRUCTURES:
        raise ValueError(f"unknown silicon structure: {structure}")
    atoms = bulk("Si", structure, a=lattice_constant, cubic=True)
    return atoms.repeat(repetitions)


def cohesive_energy_curves(
    model,
    temperature_kelvin,
    lattice_constants,
    *,
    structures=CRYSTAL_STRUCTURES,
    device="cpu",
):
    """Calculate cohesive free energy per atom versus lattice constant."""
    lattice_constants = np.asarray(lattice_constants, dtype=float)
    curves = {}
    for structure in structures:
        energies = []
        for lattice_constant in lattice_constants:
            atoms = silicon_crystal(structure, lattice_constant)
            atoms.calc = TeNNPotCalculator(
                model,
                electronic_temperature=temperature_kelvin,
                energy_reference="cohesive",
                device=device,
            )
            energies.append(atoms.get_potential_energy() / len(atoms))
        curves[structure] = np.asarray(energies)
    return curves


def diamond_thermodynamics(
    model,
    temperatures_kelvin,
    *,
    lattice_constant=PAPER_DIAMOND_LATTICE_ANGSTROM,
    repetitions=(2, 2, 2),
    device="cpu",
):
    """Evaluate the bulk-diamond electronic properties in Fig. 3c and 3d."""
    atoms = silicon_crystal("diamond", lattice_constant, repetitions=repetitions)
    parameter = next(model.parameters())
    dtype = parameter.dtype
    device = torch.device(device)
    positions = torch.as_tensor(
        atoms.positions / BOHR_TO_ANGSTROM, dtype=dtype, device=device
    )
    cell = torch.as_tensor(
        atoms.cell.array / BOHR_TO_ANGSTROM, dtype=dtype, device=device
    )
    temperatures = torch.as_tensor(temperatures_kelvin, dtype=dtype, device=device)
    return electronic_thermodynamics(
        model,
        positions,
        temperatures,
        cell=cell,
        periodic=(True, True, True),
        include_isolated_atom=True,
    )


@dataclass(frozen=True)
class PhononBands:
    """Phonon dispersion with frequencies in THz."""

    path_coordinate: np.ndarray
    frequencies_thz: np.ndarray
    label_coordinates: np.ndarray
    labels: list[str]


@dataclass(frozen=True)
class RMSDTrajectories:
    """Mean RMSD and its standard error across independent MD runs."""

    time_fs: np.ndarray
    mean_angstrom: np.ndarray
    standard_error_angstrom: np.ndarray
    individual_angstrom: np.ndarray


@dataclass(frozen=True)
class MDInitialState:
    """One fully specified initial condition for an ASE MD trajectory."""

    atoms: Atoms
    reference_positions_angstrom: np.ndarray | None = None


@dataclass(frozen=True)
class MDInitialStateEnsemble:
    """Reusable collection of MD initial conditions and shared settings."""

    states: tuple[MDInitialState, ...]
    electronic_temperature_kelvin: float
    timestep_fs: float


def phonon_band_structure(
    model,
    temperature_kelvin,
    output_dir,
    *,
    lattice_constant=PAPER_DIAMOND_LATTICE_ANGSTROM,
    supercell=(3, 3, 3),
    path="GKXGLXWL",
    npoints=200,
    displacement=0.01,
    device="cpu",
):
    """Calculate bulk-diamond phonons by finite displacement using ASE.

    Displaced-force calculations are cached under ``output_dir`` and reused on a
    subsequent invocation, which makes this calculation safely resumable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atoms = bulk("Si", "diamond", a=lattice_constant)
    atoms.calc = TeNNPotCalculator(
        model,
        electronic_temperature=temperature_kelvin,
        energy_reference="cohesive",
        device=device,
    )
    phonons = Phonons(
        atoms,
        atoms.calc,
        supercell=supercell,
        delta=displacement,
        name=str(output_dir / "displacements"),
    )
    phonons.run()
    phonons.read(acoustic=True)
    band_path = atoms.cell.bandpath(path, npoints=npoints)
    bands = phonons.get_band_structure(band_path, verbose=False)
    x_values, label_coordinates, labels = bands.get_labels()
    frequencies = np.asarray(bands.energies[0]) / PLANCK_EV_SECOND / 1e12
    result = PhononBands(
        path_coordinate=np.asarray(x_values),
        frequencies_thz=frequencies,
        label_coordinates=np.asarray(label_coordinates),
        labels=list(labels),
    )
    columns = {"path_coordinate": result.path_coordinate}
    columns.update(
        {
            f"mode_{index + 1}_thz": result.frequencies_thz[:, index]
            for index in range(result.frequencies_thz.shape[1])
        }
    )
    write_property_table(output_dir / "phonon_bands.csv", columns)
    return result


def root_mean_square_displacement(positions, reference_positions):
    """Calculate RMSD over atoms without fitting away physical displacements."""
    displacement = np.asarray(positions) - np.asarray(reference_positions)
    if displacement.ndim != 2 or displacement.shape[1] != 3:
        raise ValueError("positions must have shape (atoms, 3)")
    return float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))))


def run_md_trajectory(
    model,
    initial_state,
    electronic_temperature_kelvin,
    *,
    timestep_fs=2.0,
    steps=500,
    device="cpu",
    checkpoint_path=None,
    checkpoint_interval=10,
    status=None,
):
    """Run or resume NVE MD; steps is the total target including saved steps.

    A checkpoint stores a completed Verlet step and the original RMSD reference.
    Without checkpoint_path, the calculation runs entirely in memory.
    """
    validate_md_options(
        steps, timestep_fs, electronic_temperature_kelvin, checkpoint_interval
    )
    initial = initial_state.atoms.copy()
    if not initial.has("momenta"):
        initial.set_momenta(initial.get_momenta(), apply_constraint=False)
    _validate_atoms(initial)
    reference = initial_state.reference_positions_angstrom
    if reference is None:
        reference = initial.positions.copy()
    reference = np.asarray(reference, dtype=float)
    if reference.shape != initial.positions.shape or not np.isfinite(reference).all():
        raise ValueError("RMSD reference positions must match the atom positions")
    calculator = TeNNPotCalculator(
        model,
        electronic_temperature=electronic_temperature_kelvin,
        energy_reference="cohesive",
        device=device,
    )
    current = initial.copy()
    rmsd = [root_mean_square_displacement(current.positions, reference)]
    completed = 0
    action = "starting"
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        settings = md_checkpoint_settings(
            model, electronic_temperature_kelvin, timestep_fs, device
        )
        if checkpoint_path.exists():
            current, rmsd, completed = load_md_checkpoint(
                checkpoint_path, settings, initial, reference
            )
            action = "reusing" if completed >= steps else "resuming"
        else:
            save_md_checkpoint(
                checkpoint_path, current, settings, initial, reference, rmsd
            )
    if status is not None:
        status(f"{action} at step {min(completed, steps)}/{steps}")
    if completed >= steps:
        return np.asarray(rmsd[: steps + 1])
    current.calc = calculator
    dynamics = VelocityVerlet(current, timestep_fs * units.fs)
    dynamics.nsteps = completed
    for step in range(completed + 1, steps + 1):
        dynamics.run(1)
        value = root_mean_square_displacement(current.positions, reference)
        if not np.isfinite(value):
            raise ValueError("MD produced a non-finite displacement")
        rmsd.append(value)
        if checkpoint_path is not None and (
            step % checkpoint_interval == 0 or step == steps
        ):
            save_md_checkpoint(
                checkpoint_path, current, settings, initial, reference, rmsd
            )
    return np.asarray(rmsd)


def run_md_ensemble(
    model,
    initial_states,
    electronic_temperature_kelvin,
    *,
    timestep_fs=2.0,
    steps=500,
    device="cpu",
    progress=None,
    checkpoint_dir=None,
    checkpoint_interval=10,
    status=None,
):
    """Run prepared initial conditions, reusing or resuming each saved member."""
    states = tuple(initial_states)
    if not states:
        raise ValueError("at least one MD initial state is required")
    validate_md_options(
        steps, timestep_fs, electronic_temperature_kelvin, checkpoint_interval
    )
    settings = (
        md_checkpoint_settings(
            model, electronic_temperature_kelvin, timestep_fs, device
        )
        if checkpoint_dir is not None
        else None
    )
    trajectories = []
    for number, state in enumerate(states, start=1):
        checkpoint = None
        if checkpoint_dir is not None:
            reference = state.reference_positions_angstrom
            if reference is None:
                reference = state.atoms.positions
            checkpoint = md_checkpoint_path(
                checkpoint_dir,
                settings,
                state.atoms,
                np.asarray(reference, dtype=float),
                number,
            )
        trajectories.append(
            run_md_trajectory(
                model,
                state,
                electronic_temperature_kelvin,
                timestep_fs=timestep_fs,
                steps=steps,
                device=device,
                checkpoint_path=checkpoint,
                checkpoint_interval=checkpoint_interval,
                status=(
                    (
                        lambda message: status(
                            f"MD run {number}/{len(states)}: {message}"
                        )
                    )
                    if status
                    else None
                ),
            )
        )
        if progress is not None:
            progress(number, len(states))
    individual = np.asarray(trajectories)
    standard_error = (
        individual.std(axis=0, ddof=1) / np.sqrt(len(states))
        if len(states) > 1
        else np.zeros(steps + 1)
    )
    return RMSDTrajectories(
        time_fs=np.arange(steps + 1) * timestep_fs,
        mean_angstrom=individual.mean(axis=0),
        standard_error_angstrom=standard_error,
        individual_angstrom=individual,
    )


def _read_xyz_csv(path, prefix):
    table = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
    names = tuple(f"{prefix}_{axis}" for axis in "xyz")
    if table.dtype.names != names:
        raise ValueError(f"{path} must contain columns {names}")
    values = np.column_stack([table[name] for name in names])
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite coordinates")
    return values


def load_md_initial_state_ensemble(path, *, runs=None):
    """Load a prepared, portable MD ensemble from a directory of CSV files."""
    path = Path(path)
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"MD ensemble metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        temperature = float(metadata["electronic_temperature_kelvin"])
        timestep_fs = float(metadata["timestep_fs"])
        cell = np.asarray(metadata["cell_angstrom"], dtype=float)
        pbc = np.asarray(metadata["pbc"], dtype=bool)
        reference_file = metadata["reference_positions_file"]
        run_files = tuple(metadata["run_files"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid MD ensemble metadata: {metadata_path}") from error
    if (
        temperature <= 0
        or timestep_fs <= 0
        or cell.shape != (3, 3)
        or pbc.shape != (3,)
    ):
        raise ValueError(f"invalid MD ensemble metadata: {metadata_path}")
    if runs is not None:
        if not 1 <= runs <= len(run_files):
            raise ValueError(f"--runs must be between 1 and {len(run_files)}")
        run_files = run_files[:runs]
    reference = _read_xyz_csv(path / reference_file, "position_angstrom")
    states = []
    for run_file in run_files:
        table = np.genfromtxt(path / run_file, delimiter=",", names=True, ndmin=1)
        expected = tuple(
            f"{prefix}_{axis}"
            for prefix in ("position_angstrom", "velocity_angstrom_per_ase_time")
            for axis in "xyz"
        )
        if table.dtype.names != expected:
            raise ValueError(f"{path / run_file} must contain columns {expected}")
        positions = np.column_stack(
            [table[f"position_angstrom_{axis}"] for axis in "xyz"]
        )
        velocities = np.column_stack(
            [table[f"velocity_angstrom_per_ase_time_{axis}"] for axis in "xyz"]
        )
        if positions.shape != reference.shape or not np.isfinite(velocities).all():
            raise ValueError(f"{path / run_file} is incompatible with its reference")
        atoms = Atoms("Si" * len(positions), positions=positions, cell=cell, pbc=pbc)
        atoms.set_velocities(velocities)
        states.append(MDInitialState(atoms, reference))
    return MDInitialStateEnsemble(tuple(states), temperature, timestep_fs)


def find_md_initial_state_ensemble(reference_db_dir, temperature_kelvin, *, runs=None):
    """Find a prepared ensemble by its electronic temperature."""
    candidates = []
    for metadata_path in Path(reference_db_dir).glob("rmsd/*/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if np.isclose(
            float(metadata.get("electronic_temperature_kelvin", np.nan)),
            temperature_kelvin,
            atol=5,
        ):
            candidates.append(metadata_path.parent)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"no unique prepared MD ensemble for {temperature_kelvin} K "
            f"in {reference_db_dir}"
        )
    return load_md_initial_state_ensemble(candidates[0], runs=runs)


def rmsd_trajectories(
    model,
    electronic_temperature_kelvin,
    *,
    atoms=None,
    ionic_temperature_kelvin=300.0,
    lattice_constant=PAPER_DIAMOND_LATTICE_ANGSTROM,
    repetitions=(2, 2, 2),
    timestep_fs=2.0,
    steps=500,
    runs=4,
    seed=42,
    device="cpu",
    checkpoint_dir=None,
    checkpoint_interval=10,
    status=None,
):
    """Create generic initial states, then run them through the ASE MD engine."""
    validate_md_options(
        steps, timestep_fs, electronic_temperature_kelvin, checkpoint_interval
    )
    if not isinstance(runs, int) or isinstance(runs, bool) or runs < 1:
        raise ValueError("runs must be a positive integer")
    if not np.isfinite(ionic_temperature_kelvin) or ionic_temperature_kelvin <= 0:
        raise ValueError("ionic_temperature_kelvin must be finite and positive")
    states = []
    for run_index in range(runs):
        current = (
            silicon_crystal("diamond", lattice_constant, repetitions=repetitions)
            if atoms is None
            else atoms.copy()
        )
        if not current.has("momenta"):
            MaxwellBoltzmannDistribution(
                current,
                temperature_K=ionic_temperature_kelvin,
                rng=np.random.default_rng(seed + run_index),
            )
            if len(current) > 1:
                Stationary(current)
        states.append(MDInitialState(current, current.positions.copy()))
    return run_md_ensemble(
        model,
        states,
        electronic_temperature_kelvin,
        timestep_fs=timestep_fs,
        steps=steps,
        device=device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        status=status,
    )


def write_property_table(path, columns):
    """Write named one-dimensional property arrays to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    values = [np.asarray(columns[name]) for name in names]
    if not values or any(value.ndim != 1 for value in values):
        raise ValueError("all property columns must be one-dimensional")
    if len({len(value) for value in values}) != 1:
        raise ValueError("all property columns must have the same length")
    np.savetxt(
        path,
        np.column_stack(values),
        delimiter=",",
        header=",".join(names),
        comments="",
    )

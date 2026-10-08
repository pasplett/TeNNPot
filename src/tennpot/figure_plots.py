"""Reusable property plots and optional CHIVES reference lookup.

Each plotting function accepts a Matplotlib axis and returns it, so the same
functions can create a single plot or a multi-panel paper figure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tennpot.physics import millihartree_to_kelvin

CRYSTAL_COLORS = {
    "sc": "blue",
    "bcc": "magenta",
    "fcc": "limegreen",
    "diamond": "orange",
}


def read_reference(path):
    """Read a named-column CSV, or return None when no path is given."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"reference data not found: {path}")
    return np.genfromtxt(path, delimiter=",", names=True, ndmin=1)


def resolve_reference(
    kind,
    reference=None,
    *,
    reference_dir="data/ab_initio_reference",
    electronic_temperature_kelvin=None,
    atom_count=None,
):
    """Find a matching plot-ready CSV in the local reference database."""
    if reference is None or reference is False:
        return None
    if reference is not True:
        return Path(reference)
    if kind not in {"cohesive", "phonon_bands", "thermodynamics", "rmsd"}:
        raise ValueError(f"unknown reference kind: {kind}")
    root = Path(reference_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"reference database not found: {root}. Supply a prepared "
            "reference database or pass --reference with a matching CSV"
        )
    if kind == "thermodynamics":
        candidates = [root / "thermodynamics" / "diamond.csv"]
    else:
        candidates = sorted((root / kind).glob("*.csv"))
        if kind == "rmsd" and atom_count is not None:
            candidates = [
                path for path in candidates if path.stem.startswith(f"N{atom_count}_")
            ]
        if electronic_temperature_kelvin is None:
            raise ValueError(f"electronic temperature is required for {kind} lookup")
        candidates = [
            path
            for path in candidates
            if np.isclose(
                electronic_temperature_kelvin,
                millihartree_to_kelvin(
                    int(path.stem.removesuffix("mHa").split("_")[-1])
                ),
                atol=5,
                rtol=0,
            )
        ]
    if len(candidates) != 1 or not candidates[0].is_file():
        raise FileNotFoundError(
            f"no unique {kind} reference for {electronic_temperature_kelvin} K "
            f"in {root}; supply a matching CSV with --reference"
        )
    return candidates[0]


def cohesive_columns(table, structure):
    """Return lattice constants in angstrom and energies in eV per atom."""
    lattice_name = f"{structure}_lattice_constant_angstrom"
    energy_name = f"{structure}_energy_ev_per_atom"
    names = table.dtype.names if getattr(table, "dtype", None) is not None else table
    if lattice_name in names:
        return table[lattice_name], table[energy_name]
    return table["lattice_constant_angstrom"], table[structure]


def plot_cohesive(axis, curves, structures, reference=None):
    """Plot cohesive energy in the Figure 3a units."""
    for structure in structures:
        lattice, energy = cohesive_columns(curves, structure)
        axis.plot(
            lattice / 10,
            energy,
            color=CRYSTAL_COLORS[structure],
            label="dia" if structure == "diamond" else structure,
        )
    reference = read_reference(reference)
    if reference is not None:
        for structure in structures:
            lattice, energy = cohesive_columns(reference, structure)
            axis.plot(lattice / 10, energy, color="black", ls="--", lw=0.8)
    axis.axhline(0, color="black", lw=0.7)
    axis.set(xlabel="lattice constant (nm)", ylabel="cohesive free energy (eV/atom)")
    axis.legend()
    return axis


def plot_phonons(axis, bands, reference=None):
    """Plot phonon frequencies in THz against the reciprocal-space path."""
    if bands is not None:
        for mode in bands.frequencies_thz.T:
            axis.plot(bands.path_coordinate, mode, color="C3")
        axis.set_xticks(bands.label_coordinates, bands.labels)
        axis.set_xlim(bands.path_coordinate[0], bands.path_coordinate[-1])
    else:
        axis.text(0.5, 0.5, "phonon calculation skipped", ha="center")
    reference = read_reference(reference)
    if reference is not None:
        for name in reference.dtype.names[1:]:
            axis.plot(
                reference["path_coordinate"], reference[name], color="black", lw=0.8
            )
    axis.set(ylabel="frequency (THz)")
    return axis


def plot_internal_energy(axis, thermodynamics, reference=None):
    """Plot electronic internal-energy change in eV per atom."""
    temperature = thermodynamics["temperature_kelvin"]
    internal = thermodynamics["internal_energy_ev"]
    axis.plot(temperature, internal - internal[0], color="C3")
    reference = read_reference(reference)
    if reference is not None:
        axis.plot(
            reference["temperature_kelvin"],
            reference["internal_energy_ev"],
            color="black",
            lw=0.8,
        )
    axis.set(xlabel=r"$T_e$ (K)", ylabel=r"$U_e-U_e(T_{min})$ (eV/atom)")
    return axis


def plot_heat_capacity(axis, thermodynamics, reference=None):
    """Plot electronic heat capacity in meV per kelvin per atom."""
    axis.plot(
        thermodynamics["temperature_kelvin"],
        1000 * thermodynamics["heat_capacity_ev_per_kelvin"],
        color="C3",
    )
    reference = read_reference(reference)
    if reference is not None:
        axis.plot(
            reference["temperature_kelvin"],
            1000 * reference["heat_capacity_ev_per_kelvin"],
            color="black",
            lw=0.8,
        )
    axis.set(xlabel=r"$T_e$ (K)", ylabel=r"$C_e$ (meV/K/atom)")
    return axis


def plot_rmsd(axis, trajectory, reference=None):
    """Plot mean MD displacement and standard-error bands in picometers."""
    if trajectory is None:
        axis.text(0.5, 0.5, "MD calculation skipped", ha="center")
    else:
        axis.plot(trajectory.time_fs / 1000, trajectory.mean_angstrom * 100, color="C3")
        error = trajectory.standard_error_angstrom
        axis.fill_between(
            trajectory.time_fs / 1000,
            (trajectory.mean_angstrom - error) * 100,
            (trajectory.mean_angstrom + error) * 100,
            color="C3",
            alpha=0.25,
        )
    reference = read_reference(reference)
    if reference is not None:
        axis.plot(
            reference["time_fs"] / 1000,
            reference["mean_angstrom"] * 100,
            color="black",
            lw=0.8,
        )
        error = reference["standard_error_angstrom"]
        axis.fill_between(
            reference["time_fs"] / 1000,
            (reference["mean_angstrom"] - error) * 100,
            (reference["mean_angstrom"] + error) * 100,
            color="black",
            alpha=0.15,
        )
    axis.set(xlabel="time (ps)", ylabel="RMSD (pm)")
    return axis

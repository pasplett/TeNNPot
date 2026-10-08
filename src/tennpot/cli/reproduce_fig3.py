"""Generate Figure 3 with the published model or another TeNNPot checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tennpot.checkpoint import _load_selected_model
from tennpot.physics import millihartree_to_kelvin
from tennpot.figure_plots import (
    plot_cohesive,
    plot_phonons,
    plot_internal_energy,
    plot_heat_capacity,
    plot_rmsd,
    read_reference,
    resolve_reference,
)
from tennpot.properties import (
    CRYSTAL_STRUCTURES,
    cohesive_energy_curves,
    diamond_thermodynamics,
    find_md_initial_state_ensemble,
    phonon_band_structure,
    run_md_ensemble,
    rmsd_trajectories,
    write_property_table,
)

PAPER_TEMPERATURE_K = 12631.0
PAPER_COHESIVE_XLIM_NM = (0.15, 1.30)
PAPER_COHESIVE_YLIM_EV = (-2.75, 0.25)
PAPER_MD_XLIM_PS = (0.0, 1.0)
PAPER_MD_YLIM_PM = {12631: (12.5, 22.0), 22104: (0.0, 300.0)}
PAPER_CRYSTAL_PLOT_ORDER = ("sc", "bcc", "fcc", "diamond")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="compatible TeNNPot checkpoint (default: bundled published model)",
    )
    parser.add_argument("--output", type=Path, default=Path("results/figure3"))
    parser.add_argument(
        "--ab-initio-reference",
        action="store_true",
        help="load matching CSVs from the local reference database",
    )
    parser.add_argument(
        "--reference-db-dir",
        type=Path,
        default=Path("data/ab_initio_reference"),
    )
    parser.add_argument(
        "--cohesive-temperature", type=float, default=PAPER_TEMPERATURE_K
    )
    parser.add_argument("--phonon-temperature", type=float, default=PAPER_TEMPERATURE_K)
    parser.add_argument(
        "--md-temperatures", type=float, nargs=2, default=(12631, 22104)
    )
    parser.add_argument(
        "--md-structure",
        type=Path,
        help="ASE-readable structure for MD at arbitrary temperatures",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-phonons", action="store_true")
    parser.add_argument("--skip-md", action="store_true")
    parser.add_argument(
        "--md-runs",
        type=int,
        help="limit the archived runs (paper defaults: 10 and 40)",
    )
    parser.add_argument(
        "--md-steps",
        type=int,
        default=500,
        help="total MD steps, including any completed steps",
    )
    parser.add_argument(
        "--md-checkpoint-interval",
        type=int,
        default=10,
        help="save MD state every N completed steps (default: 10)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="run MD without reading or writing restart checkpoints",
    )
    parser.add_argument("--phonon-supercell", type=int, default=3)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small smoke calculation, not publication-resolution output",
    )
    return parser


def _paper_cohesive_curves(
    model, reference, *, quick, device, temperature=PAPER_TEMPERATURE_K
):
    if reference is None:
        lattice_constants = np.linspace(2.0, 13.0, 12 if quick else 111)
        curves = cohesive_energy_curves(
            model, temperature, lattice_constants, device=device
        )
        return {"lattice_constant_angstrom": lattice_constants, **curves}

    columns = {}
    for structure in CRYSTAL_STRUCTURES:
        lattice_name = f"{structure}_lattice_constant_angstrom"
        energy_name = f"{structure}_energy_ev_per_atom"
        lattice_constants = reference[lattice_name]
        reference_energy = reference[energy_name]
        selected = (
            np.isfinite(lattice_constants)
            & np.isfinite(reference_energy)
            & (reference_energy < 0)
            & (lattice_constants <= 10 * PAPER_COHESIVE_XLIM_NM[1])
        )
        lattice_constants = lattice_constants[selected]
        if quick and lattice_constants.size > 12:
            indices = np.linspace(0, lattice_constants.size - 1, 12, dtype=int)
            lattice_constants = lattice_constants[indices]
        energy = cohesive_energy_curves(
            model,
            temperature,
            lattice_constants,
            structures=(structure,),
            device=device,
        )[structure]
        columns[lattice_name] = lattice_constants
        columns[energy_name] = energy

    size = max(len(values) for values in columns.values())
    for name, values in columns.items():
        padded = np.full(size, np.nan)
        padded[: len(values)] = values
        columns[name] = padded
    return columns


def _plot_figure(
    output_dir,
    cohesive,
    phonons,
    thermodynamics,
    trajectories,
    reference_paths=None,
    md_temperatures=(12631, 22104),
):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(12, 13))
    axes = axes.flatten()
    reference_paths = reference_paths or {}

    plot_cohesive(
        axes[0],
        cohesive,
        PAPER_CRYSTAL_PLOT_ORDER,
        reference_paths.get("cohesive"),
    )
    axes[0].set(
        xlabel="lattice constant (nm)",
        ylabel="cohesive free energy (eV/atom)",
        xlim=PAPER_COHESIVE_XLIM_NM,
        ylim=PAPER_COHESIVE_YLIM_EV,
        xticks=(0.3, 0.6, 0.9, 1.2),
        yticks=(-2, -1, 0),
    )
    plot_phonons(
        axes[1],
        phonons,
        reference_paths.get("phonon_bands"),
    )
    thermo_reference = reference_paths.get("thermodynamics")
    plot_internal_energy(axes[2], thermodynamics, thermo_reference)
    plot_heat_capacity(axes[3], thermodynamics, thermo_reference)

    for axis, temperature_value in zip(axes[4:], md_temperatures):
        trajectory = trajectories.get(temperature_value)
        plot_rmsd(
            axis,
            trajectory,
            reference_paths.get(("rmsd", temperature_value)),
        )
        axis.set(
            xlim=PAPER_MD_XLIM_PS,
            ylim=PAPER_MD_YLIM_PM.get(temperature_value, None),
            title=rf"$T_e={temperature_value:g}$ K",
        )

    for label, axis in zip("abcdef", axes):
        axis.text(0.02, 0.96, label, transform=axis.transAxes, va="top")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output_dir = Path(output_dir)
    figure.savefig(output_dir / "figure3.pdf")
    figure.savefig(output_dir / "figure3.png", dpi=200)
    plt.close(figure)


def main(argv=None):
    """Assemble the paper's Figure 3 panels from model and optional references."""
    args = _parser().parse_args(argv)
    if args.ab_initio_reference and args.md_structure is not None:
        raise ValueError(
            "automatic CHIVES MD references describe the archived paper "
            "initial states; use standalone plots with matching CSVs"
        )
    if args.md_structure is None and not args.skip_md:
        unsupported = set(args.md_temperatures) - {12631, 22104}
        if unsupported:
            raise ValueError("custom MD temperatures require --md-structure")
    args.output.mkdir(parents=True, exist_ok=True)
    model, model_cache_key = _load_selected_model(args.checkpoint, args.device)

    reference_paths = {}
    if args.ab_initio_reference:
        for kind, temperature in (
            ("cohesive", args.cohesive_temperature),
            ("phonon_bands", args.phonon_temperature),
            ("thermodynamics", None),
        ):
            reference_paths[kind] = resolve_reference(
                kind,
                True,
                reference_dir=args.reference_db_dir,
                electronic_temperature_kelvin=temperature,
            )
        for temperature in args.md_temperatures:
            reference_paths[("rmsd", temperature)] = resolve_reference(
                "rmsd",
                True,
                reference_dir=args.reference_db_dir,
                electronic_temperature_kelvin=temperature,
                atom_count=640 if temperature == 12631 else 288,
            )
    reference_cohesive = read_reference(reference_paths.get("cohesive"))
    cohesive = _paper_cohesive_curves(
        model,
        reference_cohesive,
        quick=args.quick,
        device=args.device,
        temperature=args.cohesive_temperature,
    )
    write_property_table(args.output / "cohesive.csv", cohesive)

    temperature_millihartree = np.linspace(1, 100, 8 if args.quick else 100)
    temperatures_kelvin = millihartree_to_kelvin(temperature_millihartree)
    thermo = diamond_thermodynamics(model, temperatures_kelvin, device=args.device)
    thermodynamics = {
        "temperature_kelvin": thermo.temperature_kelvin.cpu().numpy(),
        "free_energy_ev": thermo.free_energy_ev.cpu().numpy(),
        "internal_energy_ev": thermo.internal_energy_ev.cpu().numpy(),
        "heat_capacity_ev_per_kelvin": thermo.heat_capacity_ev_per_kelvin.cpu().numpy(),
    }
    write_property_table(args.output / "thermodynamics.csv", thermodynamics)

    phonons = None
    if not args.skip_phonons:
        size = 1 if args.quick else args.phonon_supercell
        phonons = phonon_band_structure(
            model,
            args.phonon_temperature,
            args.output
            / "phonons"
            / f"{model_cache_key}_{args.phonon_temperature:g}_{size}",
            supercell=(size, size, size),
            npoints=20 if args.quick else 200,
            device=args.device,
        )

    trajectories = {}
    if not args.skip_md:
        if args.md_structure is not None:
            from ase.io import read

            md_atoms = read(args.md_structure)
        for temperature in args.md_temperatures:
            if args.md_structure is None:
                ensemble = find_md_initial_state_ensemble(
                    args.reference_db_dir,
                    temperature,
                    runs=1 if args.quick else args.md_runs,
                )
                trajectory = run_md_ensemble(
                    model,
                    ensemble.states,
                    ensemble.electronic_temperature_kelvin,
                    timestep_fs=ensemble.timestep_fs,
                    steps=2 if args.quick else args.md_steps,
                    device=args.device,
                    checkpoint_dir=None if args.no_resume else args.output / "md_cache",
                    checkpoint_interval=args.md_checkpoint_interval,
                    status=lambda message: print(
                        f"Figure 3 {temperature:g} K: {message}", flush=True
                    ),
                )
            else:
                trajectory = rmsd_trajectories(
                    model,
                    temperature,
                    atoms=md_atoms,
                    steps=2 if args.quick else args.md_steps,
                    runs=1 if args.quick else (args.md_runs or 1),
                    device=args.device,
                    checkpoint_dir=None if args.no_resume else args.output / "md_cache",
                    checkpoint_interval=args.md_checkpoint_interval,
                    status=lambda message: print(
                        f"Figure 3 {temperature:g} K: {message}", flush=True
                    ),
                )
            trajectories[temperature] = trajectory
            write_property_table(
                args.output / f"rmsd_{temperature:g}.csv",
                {
                    "time_fs": trajectory.time_fs,
                    "mean_angstrom": trajectory.mean_angstrom,
                    "standard_error_angstrom": trajectory.standard_error_angstrom,
                },
            )

    _plot_figure(
        args.output,
        cohesive,
        phonons,
        thermodynamics,
        trajectories,
        reference_paths,
        md_temperatures=args.md_temperatures,
    )
    print(f"Wrote {args.output / 'figure3.pdf'}")
    print(f"Wrote {args.output / 'figure3.png'}")


if __name__ == "__main__":  # pragma: no cover
    main()

"""Calculate and plot one TeNNPot property or MD RMSD trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read

from tennpot.checkpoint import _load_selected_model
from tennpot.figure_plots import (
    plot_cohesive,
    plot_heat_capacity,
    plot_internal_energy,
    plot_phonons,
    plot_rmsd,
    resolve_reference,
)
from tennpot.properties import (
    CRYSTAL_STRUCTURES,
    PAPER_DIAMOND_LATTICE_ANGSTROM,
    cohesive_energy_curves,
    diamond_thermodynamics,
    load_md_initial_state_ensemble,
    phonon_band_structure,
    run_md_ensemble,
    rmsd_trajectories,
    write_property_table,
)


def _parser(property_name):
    parser = argparse.ArgumentParser(
        prog="tennpot-md" if property_name == "md" else property_name,
        description=f"Calculate and plot TeNNPot {property_name} results.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="trained checkpoint (default: bundled model)"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/properties"))
    references = parser.add_mutually_exclusive_group()
    references.add_argument("--reference", type=Path, help="explicit reference CSV")
    references.add_argument(
        "--ab-initio-reference",
        action="store_true",
        help="load matching CSV from the local reference database",
    )
    parser.add_argument(
        "--reference-db-dir",
        type=Path,
        default=Path("data/ab_initio_reference"),
    )
    if property_name == "cohesive":
        parser.add_argument(
            "--temperature",
            type=float,
            required=True,
            help="electronic temperature in K",
        )
        parser.add_argument(
            "--structures",
            nargs="+",
            choices=CRYSTAL_STRUCTURES,
            default=list(CRYSTAL_STRUCTURES),
        )
        parser.add_argument("--lattice-min", type=float, default=2.0, help="angstrom")
        parser.add_argument("--lattice-max", type=float, default=13.0, help="angstrom")
        parser.add_argument("--points", type=int, default=111)
    elif property_name == "phonons":
        parser.add_argument(
            "--temperature",
            type=float,
            required=True,
            help="electronic temperature in K",
        )
        parser.add_argument(
            "--lattice-constant", type=float, default=PAPER_DIAMOND_LATTICE_ANGSTROM
        )
        parser.add_argument("--supercell", type=int, default=3)
        parser.add_argument("--points", type=int, default=200)
    elif property_name in ("internal-energy", "specific-heat"):
        parser.add_argument("--temperature-min", type=float, default=315.8, help="K")
        parser.add_argument("--temperature-max", type=float, default=31577.0, help="K")
        parser.add_argument("--points", type=int, default=100)
        parser.add_argument(
            "--lattice-constant", type=float, default=PAPER_DIAMOND_LATTICE_ANGSTROM
        )
    elif property_name == "md":
        parser.add_argument(
            "structure", type=Path, nargs="?", help="ASE-readable structure file"
        )
        parser.add_argument(
            "--initial-state-ensemble",
            type=Path,
            help="directory containing metadata.json and prepared initial-state CSVs",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            help="electronic temperature in K (read from an ensemble when omitted)",
        )
        parser.add_argument(
            "--ionic-temperature",
            type=float,
            default=300.0,
            help="K, used if input has no velocities",
        )
        parser.add_argument(
            "--steps",
            type=int,
            default=500,
            help="total steps, including any completed steps",
        )
        parser.add_argument(
            "--checkpoint-interval",
            type=int,
            default=10,
            help="save MD state every N completed steps (default: 10)",
        )
        parser.add_argument(
            "--no-resume",
            action="store_true",
            help="run MD without reading or writing restart checkpoints",
        )
        parser.add_argument("--timestep-fs", type=float, default=2.0)
        parser.add_argument(
            "--runs",
            type=int,
            help="number of trajectories (default: 1, or all ensemble states)",
        )
        parser.add_argument("--seed", type=int, default=42)
    else:
        raise ValueError(f"unknown property command: {property_name}")
    return parser


def _run(property_name, argv=None):
    """Run a standalone property or MD plot from command-line arguments."""
    args = _parser(property_name).parse_args(argv)
    md_ensemble = None
    if property_name == "md":
        if bool(args.structure) == bool(args.initial_state_ensemble):
            raise ValueError(
                "tennpot-md requires exactly one of STRUCTURE or "
                "--initial-state-ensemble"
            )
        if args.initial_state_ensemble:
            md_ensemble = load_md_initial_state_ensemble(
                args.initial_state_ensemble, runs=args.runs
            )
            if args.temperature is not None and not np.isclose(
                args.temperature, md_ensemble.electronic_temperature_kelvin, atol=5
            ):
                raise ValueError("--temperature does not match the prepared ensemble")
            args.temperature = md_ensemble.electronic_temperature_kelvin
            if args.timestep_fs != 2.0 and not np.isclose(
                args.timestep_fs, md_ensemble.timestep_fs
            ):
                raise ValueError("--timestep-fs does not match the prepared ensemble")
            args.timestep_fs = md_ensemble.timestep_fs
        if args.temperature is None:
            raise ValueError("--temperature is required when using a structure")
    if getattr(args, "points", 2) < 2:
        raise ValueError("--points must be at least 2")
    if getattr(args, "temperature", 1) <= 0:
        raise ValueError("electronic temperature must be positive")
    model, model_key = _load_selected_model(args.checkpoint, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    if property_name == "md" and args.ab_initio_reference and md_ensemble is None:
        raise ValueError(
            "the archived CHIVES RMSD curve is specific to the paper's MD "
            "initial states; provide a matching CSV with --reference for "
            "a custom structure"
        )
    if (
        args.ab_initio_reference
        and property_name in {"phonons", "internal-energy", "specific-heat"}
        and args.lattice_constant != PAPER_DIAMOND_LATTICE_ANGSTROM
    ):
        raise ValueError(
            "automatic CHIVES reference uses the paper's diamond lattice "
            "constant; use the default lattice constant or a matching --reference CSV"
        )
    reference_input = args.reference if args.reference else args.ab_initio_reference
    reference_kind = {
        "cohesive": "cohesive",
        "phonons": "phonon_bands",
        "internal-energy": "thermodynamics",
        "specific-heat": "thermodynamics",
        "md": "rmsd",
    }[property_name]
    reference = resolve_reference(
        reference_kind,
        reference_input,
        reference_dir=args.reference_db_dir,
        electronic_temperature_kelvin=getattr(args, "temperature", None),
        atom_count=len(md_ensemble.states[0].atoms) if md_ensemble else None,
    )
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    if property_name == "cohesive":
        if args.lattice_min <= 0 or args.lattice_max <= args.lattice_min:
            raise ValueError("invalid lattice constant range")
        lattice = np.linspace(args.lattice_min, args.lattice_max, args.points)
        curves = cohesive_energy_curves(
            model,
            args.temperature,
            lattice,
            structures=args.structures,
            device=args.device,
        )
        table = {"lattice_constant_angstrom": lattice, **curves}
        plot_cohesive(axis, table, args.structures, reference)
    elif property_name == "phonons":
        cache_name = (
            f"{model_key}_{args.temperature:g}_"
            f"{args.lattice_constant:g}_{args.supercell}"
        )
        bands = phonon_band_structure(
            model,
            args.temperature,
            args.output / "phonon_cache" / cache_name,
            lattice_constant=args.lattice_constant,
            supercell=(args.supercell,) * 3,
            npoints=args.points,
            device=args.device,
        )
        table = {"path_coordinate": bands.path_coordinate}
        table.update(
            {
                f"mode_{index + 1}_thz": mode
                for index, mode in enumerate(bands.frequencies_thz.T)
            }
        )
        plot_phonons(axis, bands, reference)
    elif property_name in ("internal-energy", "specific-heat"):
        if args.temperature_min <= 0 or args.temperature_max <= args.temperature_min:
            raise ValueError("invalid temperature range")
        temperatures = np.linspace(
            args.temperature_min, args.temperature_max, args.points
        )
        thermo = diamond_thermodynamics(
            model,
            temperatures,
            lattice_constant=args.lattice_constant,
            device=args.device,
        )
        table = {
            "temperature_kelvin": thermo.temperature_kelvin.cpu().numpy(),
            "free_energy_ev": thermo.free_energy_ev.cpu().numpy(),
            "internal_energy_ev": thermo.internal_energy_ev.cpu().numpy(),
            "heat_capacity_ev_per_kelvin": (
                thermo.heat_capacity_ev_per_kelvin.cpu().numpy()
            ),
        }
        if property_name == "internal-energy":
            plot_internal_energy(axis, table, reference)
        else:
            plot_heat_capacity(axis, table, reference)
    else:
        if md_ensemble is not None:
            trajectory = run_md_ensemble(
                model,
                md_ensemble.states,
                args.temperature,
                steps=args.steps,
                timestep_fs=args.timestep_fs,
                device=args.device,
                checkpoint_dir=None if args.no_resume else args.output / "md_cache",
                checkpoint_interval=args.checkpoint_interval,
                status=lambda message: print(message, flush=True),
            )
        else:
            atoms = read(args.structure)
            trajectory = rmsd_trajectories(
                model,
                args.temperature,
                atoms=atoms,
                ionic_temperature_kelvin=args.ionic_temperature,
                steps=args.steps,
                timestep_fs=args.timestep_fs,
                runs=args.runs or 1,
                seed=args.seed,
                device=args.device,
                checkpoint_dir=None if args.no_resume else args.output / "md_cache",
                checkpoint_interval=args.checkpoint_interval,
                status=lambda message: print(message, flush=True),
            )
        table = {
            "time_fs": trajectory.time_fs,
            "mean_angstrom": trajectory.mean_angstrom,
            "standard_error_angstrom": trajectory.standard_error_angstrom,
        }
        plot_rmsd(axis, trajectory, reference)
        axis.set_title(rf"$T_e={args.temperature:g}$ K")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    stem = property_name.replace("-", "_")
    write_property_table(args.output / f"{stem}.csv", table)
    figure.savefig(args.output / f"{stem}.pdf")
    figure.savefig(args.output / f"{stem}.png", dpi=200)
    plt.close(figure)
    print(f"Wrote {args.output / (stem + '.pdf')}")


def cohesive_main(argv=None):
    """Run the ``cohesive`` command."""
    return _run("cohesive", argv)


def phonons_main(argv=None):
    """Run the ``phonons`` command."""
    return _run("phonons", argv)


def internal_energy_main(argv=None):
    """Run the ``internal-energy`` command."""
    return _run("internal-energy", argv)


def specific_heat_main(argv=None):
    """Run the ``specific-heat`` command."""
    return _run("specific-heat", argv)


def tennpot_md_main(argv=None):
    """Run the ``tennpot-md`` command."""
    return _run("md", argv)

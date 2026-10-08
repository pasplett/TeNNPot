"""Evaluate ASE-readable structures with a TeNNPot checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from tennpot.checkpoint import load_checkpoint, load_published_model
from tennpot.cli._errors import report_input_errors
from tennpot.inference import predict


def main(argv=None):
    """Predict energies and forces for an ASE-readable structure collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="ASE-readable structure file")
    parser.add_argument("output", type=Path, help="predicted Extended XYZ output")
    parser.add_argument("--checkpoint", type=Path, help="default: published model")
    parser.add_argument("--temperature", type=float, default=300.0, help="kelvin")
    parser.add_argument(
        "--energy-reference", choices=("cohesive", "total"), default="total"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--debug", action="store_true", help="show error tracebacks")
    args = parser.parse_args(argv)
    with report_input_errors(parser, debug=args.debug):
        return _run(args)


def _run(args):
    """Execute predictions after parsing command-line options."""

    dtype = getattr(torch, args.dtype)
    if args.checkpoint is None:
        model = load_published_model(device=args.device, dtype=dtype)
    else:
        model, _ = load_checkpoint(args.checkpoint, device=args.device, dtype=dtype)

    structures = read(args.input, index=":")
    predictions = predict(
        structures,
        model=model,
        electronic_temperature=args.temperature,
        energy_reference=args.energy_reference,
        device=args.device,
        dtype=dtype,
    )
    for atoms, result in zip(structures, predictions):
        atoms.info["electronic_temperature"] = args.temperature
        atoms.info["energy_reference"] = args.energy_reference
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=result.energy_ev,
            free_energy=result.energy_ev,
            forces=result.forces_ev_per_angstrom,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, structures, format="extxyz")


if __name__ == "__main__":
    main()

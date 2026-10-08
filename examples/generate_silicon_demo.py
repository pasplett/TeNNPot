"""Generate a tiny synthetic silicon dataset labeled by the bundled model.

The geometries are ASE diamond crystals with small deterministic displacements.
Energies and forces are model predictions, not ab-initio reference labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write

from tennpot import TeNNPotCalculator


def main():
    """Write five labeled silicon frames for a training workflow smoke test."""
    rng = np.random.default_rng(42)
    frames = []
    for index in range(5):
        atoms = bulk("Si", "diamond", a=5.43, cubic=True)
        atoms.positions += rng.normal(
            scale=0.02 * (index + 1), size=atoms.positions.shape
        )
        temperature = (40 + 5 * index) * 315.775
        atoms.calc = TeNNPotCalculator(
            electronic_temperature=temperature,
            energy_reference="cohesive",
        )
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        atoms.calc = None
        atoms.info["energy"] = energy
        atoms.info["electronic_temperature"] = temperature
        atoms.info["split"] = "validation" if index == 4 else "train"
        atoms.arrays["forces"] = forces
        frames.append(atoms)
    output = Path(__file__).with_name("silicon_demo.extxyz")
    write(output, frames, format="extxyz")


if __name__ == "__main__":
    main()

"""Atomic Simulation Environment interface for TeNNPot."""

from __future__ import annotations

import math

import numpy as np
import torch
from ase.data import atomic_numbers
from ase.calculators.calculator import Calculator, all_changes

from tennpot.checkpoint import load_published_model
from tennpot.physics import (
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    isolated_atom_free_energy,
    kelvin_to_model_temperature,
)


class TeNNPotCalculator(Calculator):
    """Expose TeNNPot energies and forces through ASE.

    ASE supplies positions in angstrom and expects energies in eV and forces in
    eV/angstrom. The neural network uses bohr and hartree internally.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        model=None,
        *,
        electronic_temperature=300.0,
        energy_reference="total",
        device="cpu",
        dtype=torch.float64,
        **kwargs,
    ):
        if "include_isolated_atom" in kwargs:
            raise TypeError("include_isolated_atom was removed; use energy_reference")
        super().__init__(**kwargs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = (
            load_published_model(device=self.device, dtype=dtype)
            if model is None
            else model
        )
        self.model = self.model.to(device=self.device, dtype=dtype).eval()
        if energy_reference not in {"cohesive", "total"}:
            raise ValueError("energy_reference must be 'cohesive' or 'total'")
        element = getattr(getattr(self.model, "config", None), "element", "Si")
        if energy_reference == "total" and element != "Si":
            raise ValueError("the isolated-atom reference is available only for Si")
        self.energy_reference = energy_reference
        self.set_electronic_temperature(electronic_temperature)

    def set_electronic_temperature(self, temperature_kelvin):
        """Change the electronic temperature and invalidate cached results."""
        temperature_kelvin = float(temperature_kelvin)
        if not math.isfinite(temperature_kelvin) or temperature_kelvin <= 0:
            raise ValueError("electronic_temperature must be finite and positive")
        self.electronic_temperature = temperature_kelvin
        self.reset()

    def _validate_atoms(self):
        if len(self.atoms) == 0:
            raise ValueError("TeNNPot cannot evaluate an empty structure")
        element = getattr(getattr(self.model, "config", None), "element", "Si")
        expected_number = atomic_numbers.get(element)
        if expected_number is None:
            raise ValueError(f"unknown model element: {element!r}")
        if np.any(self.atoms.numbers != expected_number):
            raise ValueError(
                f"this checkpoint supports only {element}; received "
                f"{sorted(set(self.atoms.get_chemical_symbols()))}"
            )
        if not np.isfinite(self.atoms.positions).all():
            raise ValueError("atomic positions must be finite")
        if not np.isfinite(self.atoms.cell.array).all():
            raise ValueError("cell vectors must be finite")
        if self.atoms.pbc.any() and self.atoms.cell.rank < 3:
            raise ValueError("periodic structures require a three-dimensional cell")

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        """Evaluate cohesive or total energy in eV and forces in eV/angstrom."""
        super().calculate(atoms, properties, system_changes)
        self._validate_atoms()
        positions = torch.as_tensor(
            self.atoms.positions / BOHR_TO_ANGSTROM,
            dtype=self.dtype,
            device=self.device,
        )[None, ...]
        cell = torch.as_tensor(
            self.atoms.cell.array / BOHR_TO_ANGSTROM,
            dtype=self.dtype,
            device=self.device,
        )
        temperature = positions.new_tensor([[self.electronic_temperature]])
        compute_forces = "forces" in properties
        prediction = self.model(
            positions,
            kelvin_to_model_temperature(temperature),
            cell=cell,
            periodic=self.atoms.pbc,
            compute_forces=compute_forces,
        )
        if compute_forces:
            energy, forces = prediction
        else:
            energy = prediction
        if self.energy_reference == "total":
            energy = energy + len(self.atoms) * isolated_atom_free_energy(
                temperature
            ).to(positions)

        energy_ev = float(energy.detach().cpu().item() * HARTREE_TO_EV)
        self.results = {
            "energy": energy_ev,
            "free_energy": energy_ev,
        }
        if compute_forces:
            self.results["forces"] = (
                forces.detach().cpu().numpy()[0] * HARTREE_TO_EV / BOHR_TO_ANGSTROM
            )

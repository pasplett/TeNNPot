"""Convenience inference helpers built on the ASE calculator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from tennpot.ase_calculator import TeNNPotCalculator


@dataclass(frozen=True)
class Prediction:
    """Energy and forces for one structure in ASE units."""

    energy_ev: float
    forces_ev_per_angstrom: np.ndarray
    electronic_temperature_kelvin: float
    energy_reference: str


def predict(
    structures,
    *,
    model=None,
    electronic_temperature=300.0,
    energy_reference="total",
    device="cpu",
    dtype=None,
):
    """Predict energy and forces for one or more ASE structures.

    A single :class:`ase.Atoms` input returns one :class:`Prediction`; an
    iterable returns a list in the same order.
    """
    single = isinstance(structures, Atoms)
    items = [structures] if single else list(structures)
    kwargs = {
        "model": model,
        "electronic_temperature": electronic_temperature,
        "energy_reference": energy_reference,
        "device": device,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    calculator = TeNNPotCalculator(**kwargs)
    predictions = []
    for atoms in items:
        calculator.calculate(atoms, properties=("energy", "forces"))
        predictions.append(
            Prediction(
                energy_ev=calculator.results["energy"],
                forces_ev_per_angstrom=calculator.results["forces"].copy(),
                electronic_temperature_kelvin=float(electronic_temperature),
                energy_reference=energy_reference,
            )
        )
    return predictions[0] if single else predictions

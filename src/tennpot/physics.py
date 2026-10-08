"""Unit conversions and electronic thermodynamics for the silicon potential."""

from __future__ import annotations

from dataclasses import dataclass

import torch

HARTREE_TO_EV = 27.211396641308
BOHR_TO_ANGSTROM = 0.529177210903
BOLTZMANN_EV_PER_K = 8.617333262145e-5
BOLTZMANN_HARTREE_PER_K = BOLTZMANN_EV_PER_K / HARTREE_TO_EV

_ISOLATED_ATOM_COEFFICIENTS = (
    -102.905307363449,
    -10.3921123486934,
    -1.06389593948069,
    12.4244554652269,
    -183.607940058980,
    816.889082583121,
    -1946.46133504029,
    2899.84763931852,
    -2817.24323376719,
    1750.31164692968,
    -635.201199559016,
    102.801048351004,
)


def kelvin_to_millihartree(temperature):
    """Convert a temperature in kelvin to ``k_B T`` in millihartree."""
    return temperature * (1000 * BOLTZMANN_HARTREE_PER_K)


def millihartree_to_kelvin(temperature):
    """Convert ``k_B T`` in millihartree to kelvin."""
    return temperature / (1000 * BOLTZMANN_HARTREE_PER_K)


def kelvin_to_model_temperature(temperature):
    """Convert kelvin to the temperature convention used by the neural network."""
    return kelvin_to_millihartree(temperature) / 100


def model_temperature_to_kelvin(temperature):
    """Convert the neural-network temperature coordinate to kelvin."""
    return millihartree_to_kelvin(temperature * 100)


def isolated_atom_free_energy(temperature_kelvin):
    """Return the fitted isolated-Si-atom free energy in hartree.

    The piecewise polynomial is the reference subtraction used to build the
    published CHIVES data set. It is restored here so cohesive predictions can be
    converted back to absolute Mermin free energies.
    """
    temperature = torch.as_tensor(temperature_kelvin)
    if not temperature.is_floating_point():
        temperature = temperature.to(torch.get_default_dtype())
    x = (temperature - 4500.0) / 31577.0
    low_temperature = (
        _ISOLATED_ATOM_COEFFICIENTS[0] + _ISOLATED_ATOM_COEFFICIENTS[1] * x
    )
    high_temperature = low_temperature
    for power, coefficient in enumerate(_ISOLATED_ATOM_COEFFICIENTS[2:], start=4):
        high_temperature = high_temperature + coefficient * x**power
    free_energy_ev = torch.where(
        temperature > 4500.0, high_temperature, low_temperature
    )
    return free_energy_ev / HARTREE_TO_EV


def total_free_energy(
    model,
    positions,
    temperature_kelvin,
    *,
    cell=None,
    periodic=None,
    include_isolated_atom=True,
):
    """Evaluate total free energy in hartree for one or more configurations."""
    temperature = torch.as_tensor(
        temperature_kelvin, dtype=positions.dtype, device=positions.device
    )
    if temperature.ndim == 0:
        temperature = temperature.expand(positions.shape[0])
    if temperature.shape != (positions.shape[0],):
        raise ValueError("temperature must be scalar or have one value per structure")
    model_kwargs = {"cell": cell, "compute_forces": False}
    if periodic is not None:
        model_kwargs["periodic"] = periodic
    free_energy = model(
        positions,
        kelvin_to_model_temperature(temperature)[:, None],
        **model_kwargs,
    )
    if include_isolated_atom:
        atom_reference = isolated_atom_free_energy(temperature).to(positions)
        free_energy = free_energy + positions.shape[1] * atom_reference[:, None]
    return free_energy


@dataclass(frozen=True)
class ElectronicThermodynamics:
    """Per-atom electronic thermodynamic properties in practical units."""

    temperature_kelvin: torch.Tensor
    free_energy_ev: torch.Tensor
    internal_energy_ev: torch.Tensor
    heat_capacity_ev_per_kelvin: torch.Tensor


def electronic_thermodynamics(
    model,
    positions,
    temperature_kelvin,
    *,
    cell=None,
    periodic=None,
    include_isolated_atom=True,
):
    """Calculate ``F``, ``U=F-T dF/dT``, and ``C=-T d²F/dT²`` per atom."""
    if positions.ndim == 2:
        positions = positions[None, ...]
    if positions.ndim != 3 or positions.shape[0] != 1:
        raise ValueError("positions must describe exactly one structure")

    temperature = torch.as_tensor(
        temperature_kelvin, dtype=positions.dtype, device=positions.device
    ).flatten()
    if not temperature.requires_grad:
        temperature = temperature.clone().requires_grad_(True)
    batched_positions = positions.expand(temperature.numel(), -1, -1)
    batched_cell = cell
    if cell is not None:
        cell_tensor = torch.as_tensor(
            cell, dtype=positions.dtype, device=positions.device
        )
        if cell_tensor.ndim == 2:
            batched_cell = cell_tensor.expand(temperature.numel(), -1, -1)

    free_energy = total_free_energy(
        model,
        batched_positions,
        temperature,
        cell=batched_cell,
        periodic=periodic,
        include_isolated_atom=include_isolated_atom,
    ).flatten()
    first_derivative = torch.autograd.grad(
        free_energy.sum(), temperature, create_graph=True
    )[0]
    second_derivative = torch.autograd.grad(
        first_derivative.sum(), temperature, create_graph=False
    )[0]
    atom_count = positions.shape[1]
    unit_factor = HARTREE_TO_EV / atom_count
    return ElectronicThermodynamics(
        temperature_kelvin=temperature.detach(),
        free_energy_ev=(free_energy * unit_factor).detach(),
        internal_energy_ev=(
            (free_energy - temperature * first_derivative) * unit_factor
        ).detach(),
        heat_capacity_ev_per_kelvin=(
            -temperature * second_derivative * unit_factor
        ).detach(),
    )

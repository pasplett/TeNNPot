import torch
from pytest import approx

from tennpot.physics import (
    BOLTZMANN_HARTREE_PER_K,
    HARTREE_TO_EV,
    electronic_thermodynamics,
    isolated_atom_free_energy,
    kelvin_to_millihartree,
    kelvin_to_model_temperature,
    millihartree_to_kelvin,
    model_temperature_to_kelvin,
)


def test_temperature_conversions_are_inverses():
    temperature = torch.tensor([300.0, 10000.0], dtype=torch.float64)
    assert millihartree_to_kelvin(kelvin_to_millihartree(temperature)) == approx(
        temperature
    )
    assert model_temperature_to_kelvin(
        kelvin_to_model_temperature(temperature)
    ) == approx(temperature)


def test_isolated_atom_polynomial_matches_low_temperature_definition():
    temperature = torch.tensor(300.0, dtype=torch.float64)
    x = (temperature.item() - 4500.0) / 31577.0
    expected_ev = -102.905307363449 - 10.3921123486934 * x
    assert isolated_atom_free_energy(temperature).item() == approx(
        expected_ev / HARTREE_TO_EV
    )


class QuadraticTemperatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, positions, el_temp, cell=None, compute_forces=False):
        del cell, compute_forces
        return el_temp.square() + self.anchor * positions.sum(dim=(1, 2))[:, None]


def test_electronic_thermodynamics_uses_analytic_temperature_derivatives():
    model = QuadraticTemperatureModel()
    positions = torch.zeros((1, 2, 3), dtype=torch.float64)
    temperature = torch.tensor([1000.0, 2000.0], dtype=torch.float64)
    result = electronic_thermodynamics(
        model,
        positions,
        temperature,
        include_isolated_atom=False,
    )

    scale = BOLTZMANN_HARTREE_PER_K / 0.1
    expected_free = HARTREE_TO_EV * (scale * temperature) ** 2 / 2
    assert result.free_energy_ev == approx(expected_free)
    assert result.internal_energy_ev == approx(-expected_free)
    expected_heat = -HARTREE_TO_EV * scale**2 * temperature
    assert result.heat_capacity_ev_per_kelvin == approx(expected_heat)

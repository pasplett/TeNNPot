import pytest
import torch

from tennpot.model import TeNNPot


def _model(*, periodic=(False, False, False), device=None):
    torch.manual_seed(19)
    return TeNNPot(
        rad_cutoff_radii=torch.tensor([2.5, 3.0]),
        ang_cutoff_radii=torch.tensor([2.5, 3.0]),
        lambda_=torch.tensor([1.0, -1.0]),
        zeta=torch.tensor([1.0, 2.0]),
        periodic=periodic,
        box_length=(5.0, 5.0, 5.0),
        nodes_hid=(6, 4),
        activations=("tanh", "gelu-speed"),
        dtype=torch.float64,
        device=device,
    )


def _inputs(device=None):
    positions = torch.tensor(
        [[[0.2, 0.3, 0.4], [1.1, 0.2, 0.3], [0.1, 1.2, 0.4], [0.3, 0.4, 1.3]]],
        dtype=torch.float64,
        device=device,
    )
    temperature = torch.tensor([[0.4]], dtype=torch.float64, device=device)
    return positions, temperature


def test_energy_and_forces_are_translation_rotation_and_permutation_equivariant():
    model = _model()
    positions, temperature = _inputs()
    energy, forces = model(positions, temperature)

    translated_energy, translated_forces = model(
        positions + torch.tensor([2.3, -1.7, 0.9]), temperature
    )
    assert torch.allclose(translated_energy, energy, atol=1e-12)
    assert torch.allclose(translated_forces, forces, atol=1e-12)

    angle = torch.tensor(0.7, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=torch.float64,
    )
    rotated_energy, rotated_forces = model(positions @ rotation.T, temperature)
    assert torch.allclose(rotated_energy, energy, atol=1e-11)
    assert torch.allclose(rotated_forces, forces @ rotation.T, atol=1e-11)

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_energy, permuted_forces = model(positions[:, permutation], temperature)
    assert torch.allclose(permuted_energy, energy, atol=1e-12)
    assert torch.allclose(permuted_forces, forces[:, permutation], atol=1e-12)


def test_periodic_wrapping_does_not_change_predictions():
    model = _model(periodic=(True, True, True))
    positions, temperature = _inputs()
    cell = torch.diag(torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64))
    energy, forces = model(positions, temperature, cell=cell)
    wrapped = positions.clone()
    wrapped[0, 1] += cell[0]
    wrapped_energy, wrapped_forces = model(wrapped, temperature, cell=cell)
    assert torch.allclose(wrapped_energy, energy, atol=1e-11)
    assert torch.allclose(wrapped_forces, forces, atol=1e-11)


def test_batched_cells_match_individual_evaluation():
    model = _model(periodic=(True, True, True))
    positions, temperature = _inputs()
    positions = torch.cat((positions, positions * 1.1))
    temperature = torch.cat((temperature, temperature + 0.1))
    cells = torch.stack(
        (
            torch.diag(torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)),
            torch.diag(torch.tensor([5.5, 5.5, 5.5], dtype=torch.float64)),
        )
    )
    batch_energy, batch_forces = model(positions, temperature, cell=cells)
    for index in range(2):
        energy, forces = model(
            positions[index : index + 1],
            temperature[index : index + 1],
            cell=cells[index],
        )
        assert torch.allclose(batch_energy[index], energy[0], atol=1e-12)
        assert torch.allclose(batch_forces[index], forces[0], atol=1e-12)


def test_energy_has_first_and_second_temperature_derivatives():
    model = _model()
    positions, temperature = _inputs()
    temperature.requires_grad_(True)
    energy = model(positions, temperature, compute_forces=False)
    first = torch.autograd.grad(energy.sum(), temperature, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), temperature)[0]
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_model_energy_and_forces_run_on_cuda():
    model = _model(device="cuda")
    positions, temperature = _inputs(device="cuda")
    energy, forces = model(positions, temperature)
    assert energy.is_cuda
    assert forces.is_cuda
    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()


@pytest.mark.parametrize("skew", [0.0, 0.7])
def test_three_structure_batch_with_shared_cell_matches_single_predictions(skew):
    model = _model(periodic=(True, True, True))
    positions, temperature = _inputs()
    positions = torch.cat([positions, positions * 1.1, positions * 0.9])
    temperature = torch.tensor([[0.2], [0.4], [0.6]], dtype=torch.float64)
    cell = torch.tensor(
        [[3.5, 0.0, 0.0], [skew, 3.7, 0.0], [0.0, 0.0, 4.0]],
        dtype=torch.float64,
    )
    batch_energy, batch_forces = model(positions, temperature, cell=cell)
    for index in range(3):
        energy, forces = model(
            positions[index : index + 1], temperature[index : index + 1], cell=cell
        )
        torch.testing.assert_close(batch_energy[index], energy[0], atol=1e-12, rtol=0)
        torch.testing.assert_close(batch_forces[index], forces[0], atol=1e-12, rtol=0)

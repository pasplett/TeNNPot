import numpy as np
import pytest
import torch

from tennpot.acsf import Angular, Radial, cutoff
from tennpot.config import load_model_config


def test_cutoff():
    batch_size = 8
    n_atoms = 20
    n_sym_fun = 10
    distance = torch.rand((batch_size, n_atoms, n_atoms))
    cutoff_radii = torch.rand((n_sym_fun))

    assert cutoff(distance, cutoff_radii).shape == (
        batch_size,
        n_sym_fun,
        n_atoms,
        n_atoms,
    )


def test_radial():
    test_data = np.load("tests/test_tennpot.npz")
    config = load_model_config()
    positions = torch.from_numpy(test_data["positions"]).requires_grad_()
    radial_functions = Radial(
        cutoff_radii=torch.tensor(config.radial_cutoff_radii, dtype=torch.float64)
    )
    radials = radial_functions(
        positions,
        cell=torch.from_numpy(test_data["cell"]),
        periodic=test_data["periodic"],
    )
    radials.backward(gradient=torch.ones_like(radials))
    np.testing.assert_allclose(
        radials.detach().numpy(), test_data["radial"], atol=1e-10
    )
    np.testing.assert_allclose(
        positions.grad.numpy(), test_data["radial_grad"], atol=1e-9
    )


def test_radial_includes_multiple_periodic_images_inside_cutoff():
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    radial = Radial(
        cutoff_radii=torch.tensor([2.5]),
        periodic=[True, False, False],
        box_length=[3.0, 10.0, 10.0],
    )(positions)

    expected = (1 - 1.0 / 2.5) ** 2 + (1 - 2.0 / 2.5) ** 2
    assert radial[0, 0, 0] == pytest.approx(expected)
    assert radial[0, 0, 1] == pytest.approx(expected)


def test_angular():
    test_data = np.load("tests/test_tennpot.npz")
    config = load_model_config()
    positions = torch.from_numpy(test_data["positions"]).requires_grad_()
    angular_functions = Angular(
        cutoff_radii=torch.tensor(config.angular_cutoff_radii, dtype=torch.float64),
        lambda_=torch.tensor(config.angular_lambda, dtype=torch.float64),
        zeta=torch.tensor(config.angular_zeta, dtype=torch.float64),
    )
    angulars = angular_functions(
        positions,
        cell=torch.from_numpy(test_data["cell"]),
        periodic=test_data["periodic"],
    )
    angulars.backward(gradient=torch.ones_like(angulars))
    np.testing.assert_allclose(
        angulars.detach().numpy(), test_data["angular"], atol=1e-10
    )
    np.testing.assert_allclose(
        positions.grad.numpy(), test_data["angular_grad"], atol=1e-9
    )


def test_single_angular_function():
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    angular = Angular(
        cutoff_radii=torch.tensor([2.0], dtype=torch.float64),
        lambda_=torch.tensor([1.0], dtype=torch.float64),
        zeta=torch.tensor([2.0], dtype=torch.float64),
    )(positions)

    assert angular.shape == (1, 1, 3)
    assert torch.isfinite(torch.autograd.grad(angular.sum(), positions)[0]).all()


@pytest.mark.parametrize("module", [Radial, Angular])
def test_descriptor_buffers_follow_dtype(module):
    if module is Radial:
        descriptor = module(torch.tensor([2.0]))
    else:
        descriptor = module(
            torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([1.0])
        )

    descriptor = descriptor.to(dtype=torch.float64)
    assert all(
        buffer.dtype in (torch.float64, torch.bool) for buffer in descriptor.buffers()
    )

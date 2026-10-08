import numpy as np
import pytest
import torch
import torch.nn as nn

import tennpot.acsf as acsf_module
from tennpot.checkpoint import load_published_model, published_model_info
from tennpot.config import build_model, load_model_config
from tennpot.model import AtomicNN, TeNNPot


@pytest.mark.parametrize(
    "n_sym_fun, nodes_hid", [(19, [10, 10]), (49, (10,)), (99, [20, 10, 5]), (20, [])]
)
def test_atomic_nn(n_sym_fun, nodes_hid):
    model = AtomicNN(n_sym_fun, nodes_hid)

    assert model.nodes_in == n_sym_fun + 1
    assert model.nodes_hid == list(nodes_hid)

    if len(model.layers) > 0:
        assert model.layers[0].weight.shape[1] == n_sym_fun + 1
        for i, layer in enumerate(model.layers):
            assert layer.weight.shape[0] == nodes_hid[i]
    assert model.out_layer.weight.shape[0] == 1

    dummy_output = model(sym_fun=torch.zeros(10, n_sym_fun), el_temp=torch.zeros(10, 1))
    assert dummy_output.shape == (10, 1)


@pytest.mark.parametrize(
    "activations, expected",
    [
        (["tanh", "tanh"], [nn.Tanh(), nn.Tanh()]),
        (["tanh", "gelu"], [nn.Tanh(), nn.GELU()]),
        (["relu", "relu", "elu"], [nn.ReLU(), nn.ReLU(), nn.ELU()]),
    ],
)
def test_atomic_nn_activations(activations, expected):
    model = AtomicNN(20, [10] * len(activations), activations=activations)

    for i, exp in enumerate(expected):
        assert model.activations[i].__class__ == exp.__class__


def test_te_bias_correction():
    model = AtomicNN(19, [10], activations=["gelu"])

    sym_fun = torch.zeros((10, 19))
    el_temp = torch.rand((10, 1))

    y = model(sym_fun=sym_fun, el_temp=el_temp)
    assert torch.all(y == torch.zeros_like(y))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_precision(dtype):
    model = AtomicNN(19, [10], activations=["gelu"], dtype=dtype)
    if len(model.layers) > 0:
        for layer in model.layers:
            assert layer.weight.dtype == dtype
    assert model.out_layer.weight.dtype == dtype

    sym_fun = torch.rand((10, 19), dtype=dtype)
    el_temp = torch.rand((10, 1), dtype=dtype)

    y = model(sym_fun=sym_fun, el_temp=el_temp)
    assert y.dtype == dtype


def _small_model(dtype=torch.float64):
    return TeNNPot(
        rad_cutoff_radii=torch.tensor([2.0], dtype=dtype),
        ang_cutoff_radii=torch.tensor([2.0, 1.5], dtype=dtype),
        lambda_=torch.tensor([1.0, -1.0], dtype=dtype),
        zeta=torch.tensor([1.0, 2.0], dtype=dtype),
        nodes_hid=[4, 3],
        activations=["tanh", "gelu-speed"],
        dtype=dtype,
    )


def _small_inputs(dtype=torch.float64):
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=dtype,
    )
    return positions, torch.tensor([[0.5]], dtype=dtype)


def test_tiny_radial_only_model_produces_energy_and_forces():
    config = load_model_config("silicon_tiny")
    model = build_model(config)
    assert model.angular.n_ang_fun == 0
    assert model.atomic_nn.nodes_in == 4
    positions = torch.tensor([[[0.0, 0.0, 0.0], [4.4, 0.0, 0.0]]], dtype=torch.float32)
    energy, forces = model(positions, torch.tensor([[0.4]]))
    assert energy.shape == (1, 1)
    assert forces.shape == positions.shape
    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()


def test_all_layers_are_registered():
    model = AtomicNN(73, [50, 50], activations=["tanh", "gelu-speed"])
    assert sum(parameter.numel() for parameter in model.parameters()) == 6250
    assert set(model.state_dict()) == {
        "layers.0.weight",
        "layers.1.weight",
        "out_layer.weight",
    }


def test_default_float32_model_runs():
    model = _small_model(dtype=torch.float32)
    positions, el_temp = _small_inputs(dtype=torch.float32)
    energy, forces = model(positions, el_temp)
    assert energy.dtype == torch.float32
    assert forces.dtype == torch.float32


def test_forces_match_finite_difference_energy_gradient():
    torch.manual_seed(7)
    model = _small_model()
    positions, el_temp = _small_inputs()
    _, forces = model(positions, el_temp)

    step = 1e-5
    plus = positions.clone()
    minus = positions.clone()
    plus[0, 0, 0] += step
    minus[0, 0, 0] -= step
    finite_difference = -(
        model(plus, el_temp, compute_forces=False)
        - model(minus, el_temp, compute_forces=False)
    ) / (2 * step)

    assert forces[0, 0, 0] == pytest.approx(finite_difference.item(), abs=1e-7)


def test_energy_and_force_loss_backpropagates_to_every_layer():
    model = _small_model()
    positions, el_temp = _small_inputs()
    energy, forces = model(positions, el_temp, create_graph=True)
    loss = energy.square().mean() + forces.square().mean()
    loss.backward()

    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_repeated_force_calls_do_not_accumulate_gradients():
    model = _small_model()
    positions, el_temp = _small_inputs()
    _, first = model(positions, el_temp)
    _, second = model(positions, el_temp)
    assert torch.equal(first, second)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_published_model_on_archived_structure():
    test_data = np.load("tests/test_tennpot.npz")
    assert (
        test_data["model_sha256"].item() == published_model_info()["checkpoint_sha256"]
    )
    assert "energy" not in test_data and "forces" not in test_data
    model = load_published_model(dtype=torch.float64)
    positions = torch.from_numpy(test_data["positions"])
    energy, forces = model(
        positions,
        torch.from_numpy(test_data["electronic_temperature"]),
        cell=torch.from_numpy(test_data["cell"]),
        periodic=test_data["periodic"],
    )
    np.testing.assert_allclose(
        energy.detach().numpy(), test_data["model_energy"], atol=1e-10
    )
    np.testing.assert_allclose(
        forces.detach().numpy(), test_data["model_forces"], atol=1e-9
    )


def test_radial_and_angular_descriptors_share_neighbor_geometry(monkeypatch):
    model = _small_model()
    positions, _ = _small_inputs()
    original = acsf_module.primitive_neighbor_list
    calls = []

    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(acsf_module, "primitive_neighbor_list", counted)
    model.symmetry_functions(positions)
    assert len(calls) == positions.shape[0]

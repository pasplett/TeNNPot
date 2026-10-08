import numpy as np
import torch

from tennpot.checkpoint import (
    load_checkpoint,
    load_published_model,
    published_model_info,
    save_checkpoint,
)
from tennpot.config import build_model, load_model_config


def test_published_configuration():
    config = load_model_config()
    assert len(config.radial_cutoff_radii) == 13
    assert len(config.angular_cutoff_radii) == 60
    assert len(config.scale_factors) == 73
    assert config.periodic == (True, True, False)
    assert config.hidden_nodes == (50, 50)
    assert config.activations == ("tanh", "gelu-speed")
    assert config.element == "Si"
    assert config.energy_reference == "cohesive"
    assert config.training_temperature_range_mha == (1.0, 100.0)


def test_published_model_uses_safe_registered_state():
    model = load_published_model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 6250
    assert set(model.atomic_nn.state_dict()) == {
        "layers.0.weight",
        "layers.1.weight",
        "out_layer.weight",
    }
    assert model.training is False
    assert model.config.element == "Si"


def test_published_model_manifest_and_checksum():
    info = published_model_info()
    assert info["model_id"] == "tennpot-silicon-2023"
    assert info["element"] == "Si"
    assert info["checkpoint_sha256"] == (
        "937494642d2fd1c58a285ec88978f89b697f09fecb262363a8e772ea0335e5d9"
    )
    model = load_published_model()
    assert model.model_info == info


def test_published_model_golden_energy_and_forces():
    """Protect the converted weights and descriptor behavior from drift."""
    model = load_published_model(dtype=torch.float64)
    positions_angstrom = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.35, 0.0, 0.0], [0.5, 2.25, 0.2], [1.2, 0.8, 2.1]]],
        dtype=torch.float64,
    )
    positions = positions_angstrom / 0.529177210903
    energy, forces = model(
        positions,
        torch.tensor([[0.4]], dtype=torch.float64),
        cell=torch.diag(torch.tensor([30.0, 30.0, 30.0], dtype=torch.float64)),
        periodic=(False, False, False),
    )
    np.testing.assert_allclose(
        energy.detach().numpy(), [[-0.20780957689469207]], atol=1e-13, rtol=0
    )
    np.testing.assert_allclose(
        forces.detach().numpy(),
        [
            [
                [-0.003381772966129841, -0.006882073894974708, 0.004934222051416769],
                [-0.008527201447939406, 0.013080938122095466, 0.011131259798913058],
                [0.01245013951757849, -0.006111877596235382, 0.007894715887005174],
                [
                    -0.0005411651035092477,
                    -0.00008698663088537329,
                    -0.023960197737335008,
                ],
            ]
        ],
        atol=1e-12,
        rtol=0,
    )


def test_safe_checkpoint_roundtrip(tmp_path):
    config = load_model_config()
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        config,
        optimizer=optimizer,
        epoch=4,
        metrics={"validation_loss": 0.25},
    )

    payload = torch.load(checkpoint, weights_only=True)
    restored, restored_payload = load_checkpoint(checkpoint)

    assert payload["format_version"] == 1
    assert restored_payload["epoch"] == 4
    assert restored_payload["metrics"]["validation_loss"] == 0.25
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_invalid_scale_factors_are_rejected_before_inference():
    from dataclasses import replace
    import pytest

    config = load_model_config("silicon_tiny")
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            replace(config, scale_factors=(value,) * len(config.scale_factors))

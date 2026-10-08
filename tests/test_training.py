import json

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from torch.utils.data import DataLoader, TensorDataset

from tennpot.checkpoint import load_checkpoint
from tennpot.cli.train import main as train_main
from tennpot.cli.train import _limit_dataset
from tennpot.config import ModelConfig, build_model
from tennpot.training import (
    RelativeEnergyForceLoss,
    TrainingConfig,
    atomistic_collate,
    evaluate_model,
    fit_scale_factors,
    train_model,
)


def _tiny_config():
    return ModelConfig(
        schema_version=1,
        name="test model",
        radial_cutoff_radii=(2.5,),
        angular_cutoff_radii=(2.5, 2.0),
        angular_lambda=(1.0, -1.0),
        angular_zeta=(1.0, 2.0),
        scale_factors=(1.0, 1.0, 1.0),
        periodic=(False, False, False),
        cell=(10.0, 10.0, 10.0),
        hidden_nodes=(4,),
        activations=("tanh",),
    )


def _tiny_loader():
    torch.manual_seed(11)
    config = _tiny_config()
    teacher = build_model(config)
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.0, 1.0, 0.2]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 0.9, 0.0]],
        ],
        dtype=torch.float64,
    )
    el_temp = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    energy, forces = teacher(positions, el_temp)
    dataset = TensorDataset(positions, el_temp, energy.detach(), forces.detach())
    return config, DataLoader(dataset, batch_size=1, shuffle=False)


def test_relative_energy_force_loss():
    criterion = RelativeEnergyForceLoss()
    energy = torch.tensor([[2.0]])
    reference_energy = torch.tensor([[1.0]])
    forces = torch.tensor([[[2.0, 0.0, 0.0]]])
    reference_forces = torch.tensor([[[1.0, 0.0, 0.0]]])
    total, energy_loss, force_loss = criterion.components(
        energy, reference_energy, forces, reference_forces
    )
    assert energy_loss == 1.0
    assert force_loss == 1.0
    assert total == 1.0


def test_evaluate_model_reports_relative_errors():
    config, loader = _tiny_loader()
    model = build_model(config)
    metrics = evaluate_model(model, loader)
    assert set(metrics) == {
        "loss",
        "relative_rmse",
        "energy_relative_rmse",
        "force_relative_rmse",
    }
    assert all(value >= 0 for value in metrics.values())


def test_training_writes_resumable_checkpoints_and_stops_early(tmp_path, caplog):
    caplog.set_level("INFO", logger="tennpot.training")
    config, loader = _tiny_loader()
    model = build_model(config)
    training_config = TrainingConfig(
        epochs=5,
        learning_rate=0.0,
        early_stopping_patience=1,
        mlflow_enabled=False,
    )
    best, history = train_model(
        model,
        loader,
        loader,
        model_config=config,
        output_dir=tmp_path,
        config=training_config,
    )

    assert best == tmp_path / "best.pt"
    assert best.is_file()
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "history.json").is_file()
    assert len(history) == 2
    restored, payload = load_checkpoint(best)
    assert payload["epoch"] == 0
    assert "optimizer_state_dict" in payload
    assert payload["training_state"]["best_epoch"] == 0
    assert "rng_state" in payload
    assert sum(parameter.numel() for parameter in restored.parameters()) == 20
    assert "Epoch 1/5 | train loss" in caplog.text
    assert "Epoch 2/5 | train loss" in caplog.text
    assert "Early stopping after epoch 2" in caplog.text
    assert "Training complete | best epoch 1" in caplog.text


def test_scale_factors_are_fitted_from_descriptors():
    config, loader = _tiny_loader()
    model = build_model(config)
    fitted = fit_scale_factors(model, loader)
    assert len(fitted) == 3
    assert all(value > 0 for value in fitted)


def test_resumed_training_matches_uninterrupted_training(tmp_path):
    config, loader = _tiny_loader()
    torch.manual_seed(23)
    uninterrupted = build_model(config)
    train_model(
        uninterrupted,
        loader,
        loader,
        model_config=config,
        output_dir=tmp_path / "uninterrupted",
        config=TrainingConfig(epochs=2, seed=91, mlflow_enabled=False),
    )

    _, first_loader = _tiny_loader()
    torch.manual_seed(23)
    interrupted = build_model(config)
    train_model(
        interrupted,
        first_loader,
        first_loader,
        model_config=config,
        output_dir=tmp_path / "resumed",
        config=TrainingConfig(epochs=1, seed=91, mlflow_enabled=False),
    )
    resumed, payload = load_checkpoint(tmp_path / "resumed" / "latest.pt")
    _, resume_loader = _tiny_loader()
    _, history = train_model(
        resumed,
        resume_loader,
        resume_loader,
        model_config=config,
        output_dir=tmp_path / "resumed",
        config=TrainingConfig(epochs=2, seed=91, mlflow_enabled=False),
        resume_payload=payload,
    )

    assert len(history) == 2
    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


@pytest.mark.mlflow
def test_training_logs_metrics_and_checkpoints_to_mlflow(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    config, loader = _tiny_loader()
    model = build_model(config)
    database_path = (tmp_path / "mlflow.db").resolve().as_posix()
    tracking_uri = f"sqlite:///{database_path}"
    experiment_name = "tennpot-test"
    training_config = TrainingConfig(
        epochs=1,
        mlflow_enabled=True,
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment=experiment_name,
        mlflow_run_name="integration-test",
    )

    train_model(
        model,
        loader,
        loader,
        model_config=config,
        output_dir=tmp_path / "checkpoints",
        config=training_config,
    )

    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.validation_loss"] >= 0
    run_id = runs.iloc[0]["run_id"]
    artifacts = mlflow.artifacts.list_artifacts(
        run_id=run_id, artifact_path="checkpoints"
    )
    assert {artifact.path for artifact in artifacts} == {
        "checkpoints/best.pt",
        "checkpoints/latest.pt",
    }


def test_training_cli_accepts_extended_xyz(tmp_path):
    config = _tiny_config()
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    dataset_path = tmp_path / "training.extxyz"
    structures = []
    for index in range(4):
        atoms = Atoms(
            "Si3",
            positions=[[0, 0, 0], [1 + index / 20, 0, 0], [0, 1, 0]],
            cell=[10, 10, 10],
        )
        atoms.info["energy"] = -1.0 - index / 10
        atoms.info["electronic_temperature"] = 5000.0
        atoms.info["split"] = "train" if index < 3 else "validation"
        atoms.arrays["forces"] = np.full((3, 3), 0.05 + index / 100)
        structures.append(atoms)
    write(dataset_path, structures, format="extxyz")

    output = tmp_path / "run"
    train_main(
        [
            str(dataset_path),
            "--model-config",
            str(config_path),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ]
    )

    assert (output / "best.pt").is_file()
    assert (output / "latest.pt").is_file()
    assert (output / "model_config.json").is_file()
    assert (output / "training_config.json").is_file()


def test_training_sample_cap_is_seeded_and_spans_dataset():
    dataset = list(range(100))
    first = _limit_dataset(dataset, 8, 42)
    second = _limit_dataset(dataset, 8, 42)
    assert first.indices == second.indices
    assert len(set(first.indices)) == 8
    assert max(first.indices) >= 8
    with pytest.raises(ValueError, match="positive"):
        _limit_dataset(dataset, 0, 42)


@pytest.mark.parametrize("resume_learning_rate", [None, 0.0])
def test_cli_separate_validation_files_use_all_frames_and_resume_settings(
    tmp_path, resume_learning_rate
):
    config = _tiny_config()
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    paths = [tmp_path / "train.extxyz", tmp_path / "validation.extxyz"]
    for file_index, path in enumerate(paths):
        frames = []
        for index in range(3 if file_index == 0 else 2):
            atoms = Atoms(
                "Si3",
                positions=[[0, 0, 0], [1 + index / 20, 0, 0], [0, 1, 0]],
                cell=[10, 10, 10],
            )
            atoms.info.update(energy=-1.0, electronic_temperature=5000.0)
            atoms.arrays["forces"] = np.full((3, 3), 0.05)
            frames.append(atoms)
        write(path, frames, format="extxyz")
    output = tmp_path / "run"
    common = [
        str(paths[0]),
        "--validation-data",
        str(paths[1]),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]
    train_main(
        common
        + [
            "--model-config",
            str(config_path),
            "--epochs",
            "1",
            "--batch-size",
            "3",
            "--dtype",
            "float64",
            "--seed",
            "7",
            "--learning-rate",
            "0.003",
        ]
    )
    initial, payload = load_checkpoint(output / "latest.pt")
    previous_parameters = [p.detach().clone() for p in initial.parameters()]
    assert payload["metadata"]["dataset"]["use_all_frames"] is True
    assert payload["metadata"]["dataset"]["structures"] == 3
    resume_args = ["--resume", str(output / "latest.pt"), "--epochs", "2"]
    if resume_learning_rate is not None:
        resume_args += ["--learning-rate", str(resume_learning_rate)]
    train_main(common + resume_args)
    restored, payload = load_checkpoint(output / "latest.pt")
    assert len(payload["training_state"]["history"]) == 2
    assert payload["training_config"]["seed"] == 7
    assert payload["training_config"]["batch_size"] == 3
    assert restored.config.dtype == "float64"
    expected_lr = 0.003 if resume_learning_rate is None else resume_learning_rate
    assert payload["training_config"]["learning_rate"] == expected_lr
    assert payload["optimizer_state_dict"]["param_groups"][0]["lr"] == expected_lr
    if expected_lr == 0:
        for before, after in zip(previous_parameters, restored.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize(
    "settings",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": float("nan")},
        {"energy_weight": -1},
        {"energy_weight": 0, "force_weight": 0},
        {"early_stopping_patience": 0},
        {"adam_beta1": 1.0},
    ],
)
def test_invalid_training_settings_fail_before_writing_checkpoints(settings):
    with pytest.raises(ValueError):
        TrainingConfig(**settings)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_resume_restores_rng_tensors_loaded_on_cuda():
    from tennpot.training import capture_rng_state, restore_rng_state

    generator = torch.Generator().manual_seed(31)
    loader = DataLoader(list(range(4)), generator=generator)
    state = capture_rng_state(loader)
    state["torch_cpu"] = state["torch_cpu"].cuda()
    state["data_loader"] = state["data_loader"].cuda()
    state["torch_cuda"] = [value.cuda() for value in state["torch_cuda"]]
    restore_rng_state(state, loader)
    assert torch.equal(torch.get_rng_state(), state["torch_cpu"].cpu())
    assert torch.equal(generator.get_state(), state["data_loader"].cpu())


def test_resumed_optimizer_honors_overrides_and_preserves_state(tmp_path):
    config, loader = _tiny_loader()
    model = build_model(config)
    train_model(
        model,
        loader,
        loader,
        model_config=config,
        output_dir=tmp_path,
        config=TrainingConfig(
            epochs=1, learning_rate=0.003, adam_beta1=0.8, adam_beta2=0.95
        ),
    )
    resumed, payload = load_checkpoint(tmp_path / "latest.pt")
    previous_parameters = [p.detach().clone() for p in resumed.parameters()]
    previous_steps = {
        key: state["step"].item()
        for key, state in payload["optimizer_state_dict"]["state"].items()
    }
    train_model(
        resumed,
        loader,
        loader,
        model_config=config,
        output_dir=tmp_path,
        config=TrainingConfig(
            epochs=2, learning_rate=0.0, adam_beta1=0.7, adam_beta2=0.9
        ),
        resume_payload=payload,
    )
    _, updated = load_checkpoint(tmp_path / "latest.pt")
    group = updated["optimizer_state_dict"]["param_groups"][0]
    assert group["lr"] == updated["training_config"]["learning_rate"] == 0.0
    assert group["betas"] == (0.7, 0.9)
    for expected, actual in zip(previous_parameters, resumed.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for key, state in updated["optimizer_state_dict"]["state"].items():
        assert state["step"].item() == previous_steps[key] + len(loader)


def test_resume_without_config_uses_saved_training_settings(tmp_path):
    config, loader = _tiny_loader()
    model = build_model(config)
    train_model(
        model,
        loader,
        loader,
        model_config=config,
        output_dir=tmp_path,
        config=TrainingConfig(epochs=1, learning_rate=0.003),
    )
    resumed, payload = load_checkpoint(tmp_path / "latest.pt")
    _, history = train_model(
        resumed,
        loader,
        loader,
        model_config=config,
        output_dir=tmp_path,
        resume_payload=payload,
    )
    assert history == payload["training_state"]["history"]


def test_relative_loss_aggregates_structures_before_normalizing(monkeypatch):
    model = build_model(_tiny_config())

    def prediction(positions, temperature, **kwargs):
        return positions.new_ones((positions.shape[0], 1)), torch.zeros_like(positions)

    monkeypatch.setattr(model, "forward", prediction)
    samples = [
        {
            "positions": torch.zeros((count, 3), dtype=torch.float64),
            "electronic_temperature": torch.tensor([0.4]),
            "energy": torch.tensor([energy]),
            "forces": torch.ones((count, 3)),
            "pbc": torch.tensor([False, False, False]),
        }
        for count, energy in [(2, 1.0), (3, 2.0)]
    ]
    loader = DataLoader(samples, batch_size=2, collate_fn=atomistic_collate)
    metrics = evaluate_model(model, loader, energy_weight=0.25, force_weight=0.75)
    assert metrics["energy_relative_rmse"] == pytest.approx(np.sqrt(1 / 5))
    assert metrics["force_relative_rmse"] == pytest.approx(1.0)
    assert metrics["loss"] == pytest.approx(0.25 * (1 / 5) + 0.75)

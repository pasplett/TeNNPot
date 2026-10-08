"""Energy-and-force training with optional MLflow experiment tracking."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from ase.data import atomic_numbers
from torch import nn

from tennpot.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Published-protocol training hyperparameters."""

    epochs: int = 50
    batch_size: int = 20
    learning_rate: float = 0.001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    energy_weight: float = 0.5
    force_weight: float = 0.5
    early_stopping_patience: int | None = None
    seed: int = 42
    deterministic: bool = False
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "TeNNPot"
    mlflow_run_name: str | None = None

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning_rate must be finite and non-negative")
        if not all(0 <= beta < 1 for beta in (self.adam_beta1, self.adam_beta2)):
            raise ValueError("Adam betas must be in [0, 1)")
        weights = (self.energy_weight, self.force_weight)
        if not all(math.isfinite(weight) and weight >= 0 for weight in weights):
            raise ValueError("loss weights must be finite and non-negative")
        if sum(weights) == 0:
            raise ValueError("at least one loss weight must be positive")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be positive")

    @classmethod
    def from_dict(cls, values):
        """Create training settings from a versioned JSON-compatible mapping."""
        values = dict(values)
        schema_version = values.pop("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"unsupported training-config schema: {schema_version}")
        return cls(**values)


def load_training_config(path=None):
    """Load the bundled paper protocol or a user-provided JSON file."""
    if path is None:
        from importlib.resources import files

        resource = files("tennpot").joinpath("configs", "paper_protocol.json")
        text = resource.read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return TrainingConfig.from_dict(json.loads(text))


def _energy_force_sums(energy, reference_energy, forces, reference_forces):
    numerators = torch.stack(
        (
            (energy - reference_energy).square().sum(),
            (forces - reference_forces).square().sum(),
        )
    )
    denominators = torch.stack(
        (reference_energy.square().sum(), reference_forces.square().sum())
    )
    return numerators, denominators


def _relative_loss_components(numerators, denominators, weights, eps=1e-20):
    scales = denominators.clamp_min(eps)
    errors = numerators / scales
    total = (weights * numerators / scales).sum()
    return total, errors[0], errors[1]


class RelativeEnergyForceLoss(nn.Module):
    """Relative squared energy-and-force objective from the paper."""

    def __init__(self, energy_weight=0.5, force_weight=0.5, eps=1e-20):
        super().__init__()
        if energy_weight < 0 or force_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if energy_weight + force_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.eps = eps

    def components(self, energy, reference_energy, forces, reference_forces):
        """Return weighted loss and separate relative energy and force losses."""
        numerators, denominators = _energy_force_sums(
            energy, reference_energy, forces, reference_forces
        )
        weights = numerators.new_tensor([self.energy_weight, self.force_weight])
        return _relative_loss_components(numerators, denominators, weights, self.eps)

    def forward(self, energy, reference_energy, forces, reference_forces):
        """Return the weighted relative energy-and-force training loss."""
        return self.components(energy, reference_energy, forces, reference_forces)[0]


def seed_everything(seed, *, deterministic=False):
    """Seed Python, NumPy, and PyTorch before constructing a model or loader."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def capture_rng_state(data_loader=None):
    """Capture serializable random state required for faithful resumption."""
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    generator = getattr(data_loader, "generator", None)
    if generator is not None:
        state["data_loader"] = generator.get_state()
    return state


def restore_rng_state(state, data_loader=None):
    """Restore state captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    generator = getattr(data_loader, "generator", None)
    if generator is not None and "data_loader" in state:
        generator.set_state(state["data_loader"].cpu())


def atomistic_collate(samples):
    """Group mapping-style samples by atom count and periodicity."""
    groups = {}
    for sample in samples:
        key = (
            int(sample["positions"].shape[0]),
            tuple(bool(value) for value in sample["pbc"]),
        )
        groups.setdefault(key, []).append(sample)
    batched = []
    for group in groups.values():
        batched.append(
            {name: torch.stack([sample[name] for sample in group]) for name in group[0]}
        )
    return batched


def _batch_groups(batch):
    if isinstance(batch, dict):
        return [batch]
    if isinstance(batch, list) and (not batch or isinstance(batch[0], dict)):
        return batch
    positions, temperature, energy, forces = batch[:4]
    return [
        {
            "positions": positions,
            "electronic_temperature": temperature,
            "energy": energy,
            "forces": forces,
        }
    ]


def _prepare_group(group, model):
    parameter = next(model.parameters())
    kwargs = {"device": parameter.device, "dtype": parameter.dtype}
    prepared = {
        name: group[name].to(**kwargs)
        for name in ("positions", "electronic_temperature", "energy", "forces")
    }
    if "cell" in group:
        prepared["cell"] = group["cell"].to(**kwargs)
    else:
        prepared["cell"] = None
    if "pbc" in group:
        pbc = group["pbc"]
        if pbc.ndim == 2:
            if not torch.all(pbc == pbc[0]):
                raise ValueError("each collated group must share periodicity")
            pbc = pbc[0]
        prepared["pbc"] = pbc
    else:
        prepared["pbc"] = None
    if "atomic_numbers" in group:
        expected = atomic_numbers[model.config.element]
        if torch.any(group["atomic_numbers"] != expected):
            raise ValueError(
                f"model configuration supports only {model.config.element}"
            )
    return prepared


def _predict_group(model, group, *, create_graph=False):
    kwargs = {"cell": group["cell"], "create_graph": create_graph}
    if group["pbc"] is not None:
        kwargs["periodic"] = group["pbc"]
    return model(group["positions"], group["electronic_temperature"], **kwargs)


def _error_tensors(model, batch, *, create_graph=False):
    numerators = None
    denominators = None
    for raw_group in _batch_groups(batch):
        group = _prepare_group(raw_group, model)
        energy, forces = _predict_group(model, group, create_graph=create_graph)
        current_num, current_den = _energy_force_sums(
            energy, group["energy"], forces, group["forces"]
        )
        numerators = current_num if numerators is None else numerators + current_num
        denominators = (
            current_den if denominators is None else denominators + current_den
        )
    return numerators, denominators


def _metrics(numerators, denominators, energy_weight=0.5, force_weight=0.5):
    numerators = torch.tensor(numerators, dtype=torch.float64)
    denominators = numerators.new_tensor(denominators)
    weights = numerators.new_tensor([energy_weight, force_weight])
    loss, energy_rse, force_rse = (
        value.item()
        for value in _relative_loss_components(numerators, denominators, weights)
    )
    return {
        "loss": loss,
        "relative_rmse": math.sqrt(loss),
        "energy_relative_rmse": math.sqrt(energy_rse),
        "force_relative_rmse": math.sqrt(force_rse),
    }


def evaluate_model(model, data_loader, *, energy_weight=0.5, force_weight=0.5):
    """Evaluate full-dataset relative errors without changing parameter gradients."""
    model.eval()
    numerators = [0.0, 0.0]
    denominators = [0.0, 0.0]
    for batch in data_loader:
        current_num, current_den = _error_tensors(model, batch)
        for index in range(2):
            numerators[index] += current_num[index].detach().item()
            denominators[index] += current_den[index].detach().item()
    return _metrics(numerators, denominators, energy_weight, force_weight)


def fit_scale_factors(model, data_loader, *, minimum=1e-12):
    """Return maximum descriptor values over the training partition only."""
    maxima = None
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            for raw_group in _batch_groups(batch):
                group = _prepare_group(raw_group, model)
                kwargs = {"cell": group["cell"], "scale": False}
                if group["pbc"] is not None:
                    kwargs["periodic"] = group["pbc"]
                descriptors = model.symmetry_functions(group["positions"], **kwargs)
                values = descriptors.amax(dim=(0, 2))
                maxima = values if maxima is None else torch.maximum(maxima, values)
    if maxima is None:
        raise ValueError("cannot fit scale factors on an empty dataset")
    if torch.any(~torch.isfinite(maxima)) or torch.any(maxima <= minimum):
        raise ValueError(
            "all fitted symmetry-function scales must be finite and positive"
        )
    return tuple(float(value) for value in maxima.detach().cpu())


def train_model(
    model,
    train_loader,
    validation_loader,
    *,
    model_config,
    output_dir,
    config=None,
    optimizer=None,
    resume_payload=None,
    dataset_metadata=None,
):
    """Train TeNNPot and return the best checkpoint and complete history."""
    if config is None:
        saved_config = (resume_payload or {}).get("training_config", {})
        config = TrainingConfig.from_dict(saved_config)
    if resume_payload is None:
        seed_everything(config.seed, deterministic=config.deterministic)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = optimizer or torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    start_epoch = 0
    best_loss = math.inf
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    if resume_payload is not None:
        # Loading optimizer state also restores options; preserve requested overrides.
        options = [
            {name: group[name] for name in ("lr", "betas") if name in group}
            for group in optimizer.param_groups
        ]
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        for group, requested in zip(optimizer.param_groups, options):
            group.update(requested)
        state = resume_payload.get("training_state", {})
        saved_epoch = resume_payload.get("epoch")
        start_epoch = 0 if saved_epoch is None else int(saved_epoch) + 1
        best_loss = float(state.get("best_loss", math.inf))
        best_epoch = state.get("best_epoch")
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        history = list(state.get("history", []))
        if "rng_state" in resume_payload:
            restore_rng_state(resume_payload["rng_state"], train_loader)

    best_checkpoint = output_dir / "best.pt"
    latest_checkpoint = output_dir / "latest.pt"
    weights = torch.tensor(
        [config.energy_weight, config.force_weight],
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )

    from tennpot.tracking import tracking_run

    with tracking_run(config, model_config, dataset_metadata) as tracking:
        for epoch in range(start_epoch, config.epochs):
            epoch_started = time.perf_counter()
            model.train()
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                numerators, denominators = _error_tensors(
                    model, batch, create_graph=True
                )
                loss = _relative_loss_components(numerators, denominators, weights)[0]
                loss.backward()
                optimizer.step()

            train_metrics = evaluate_model(
                model,
                train_loader,
                energy_weight=config.energy_weight,
                force_weight=config.force_weight,
            )
            validation_metrics = evaluate_model(
                model,
                validation_loader,
                energy_weight=config.energy_weight,
                force_weight=config.force_weight,
            )
            epoch_metrics = {
                **{f"train_{name}": value for name, value in train_metrics.items()},
                **{
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                },
            }
            history.append(epoch_metrics)
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            improved = validation_metrics["loss"] < best_loss
            if improved:
                best_loss = validation_metrics["loss"]
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            training_state = {
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            }
            save_kwargs = {
                "optimizer": optimizer,
                "epoch": epoch,
                "metrics": epoch_metrics,
                "training_state": training_state,
                "training_config": asdict(config),
                "rng_state": capture_rng_state(train_loader),
                "metadata": {"dataset": dict(dataset_metadata or {})},
            }
            save_checkpoint(latest_checkpoint, model, model_config, **save_kwargs)
            if improved:
                save_checkpoint(best_checkpoint, model, model_config, **save_kwargs)

            logger.info(
                "Epoch %d/%d | train loss %.5g (energy RRMSE %.4g, force RRMSE %.4g) "
                "| validation loss %.5g (energy RRMSE %.4g, force RRMSE %.4g) "
                "| %.1f s%s",
                epoch + 1,
                config.epochs,
                train_metrics["loss"],
                train_metrics["energy_relative_rmse"],
                train_metrics["force_relative_rmse"],
                validation_metrics["loss"],
                validation_metrics["energy_relative_rmse"],
                validation_metrics["force_relative_rmse"],
                time.perf_counter() - epoch_started,
                " | new best" if improved else "",
            )

            if tracking is not None:
                tracking.log_metrics(epoch_metrics, step=epoch)
            patience = config.early_stopping_patience
            if patience is not None and epochs_without_improvement >= patience:
                logger.info("Early stopping after epoch %d", epoch + 1)
                break

        if tracking is not None:
            tracking.log_artifact(str(best_checkpoint), artifact_path="checkpoints")
            tracking.log_artifact(str(latest_checkpoint), artifact_path="checkpoints")
            tracking.log_artifact(str(output_dir / "history.json"))
    if best_epoch is not None:
        logger.info(
            "Training complete | best epoch %d | validation loss %.5g | %s",
            best_epoch + 1,
            best_loss,
            best_checkpoint,
        )
    return best_checkpoint, history

from importlib.metadata import PackageNotFoundError, version

try:
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from tennpot.checkpoint import (
    load_checkpoint,
    load_published_model,
    published_model_info,
    save_checkpoint,
)
from tennpot.config import ModelConfig, build_model, load_model_config
from tennpot.dataset import (
    AtomisticDataset,
    CHIVESDataset,
    ExtendedXYZDataset,
    load_dataset,
)
from tennpot.ase_calculator import TeNNPotCalculator
from tennpot.inference import Prediction, predict
from tennpot.model import AtomicNN, TeNNPot
from tennpot.physics import electronic_thermodynamics, isolated_atom_free_energy
from tennpot.training import (
    RelativeEnergyForceLoss,
    TrainingConfig,
    atomistic_collate,
    evaluate_model,
    fit_scale_factors,
    load_training_config,
    seed_everything,
    train_model,
)

__all__ = [
    "AtomicNN",
    "AtomisticDataset",
    "CHIVESDataset",
    "ExtendedXYZDataset",
    "ModelConfig",
    "Prediction",
    "RelativeEnergyForceLoss",
    "TeNNPot",
    "TeNNPotCalculator",
    "TrainingConfig",
    "build_model",
    "atomistic_collate",
    "electronic_thermodynamics",
    "evaluate_model",
    "fit_scale_factors",
    "isolated_atom_free_energy",
    "load_checkpoint",
    "load_dataset",
    "load_model_config",
    "load_training_config",
    "load_published_model",
    "published_model_info",
    "predict",
    "save_checkpoint",
    "seed_everything",
    "train_model",
]

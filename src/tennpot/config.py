"""Validated model configurations and constructors."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

import torch

from tennpot.model import TeNNPot


@dataclass(frozen=True)
class ModelConfig:
    """Serializable configuration for a :class:`~tennpot.model.TeNNPot`."""

    schema_version: int
    name: str
    radial_cutoff_radii: tuple[float, ...]
    angular_cutoff_radii: tuple[float, ...]
    angular_lambda: tuple[float, ...]
    angular_zeta: tuple[float, ...]
    scale_factors: tuple[float, ...]
    periodic: tuple[bool, bool, bool]
    cell: tuple[float, float, float]
    hidden_nodes: tuple[int, ...]
    activations: tuple[str, ...]
    dtype: str = "float64"
    electronic_temperature_unit: str = "0.1 Ha"
    energy_unit: str = "Ha"
    length_unit: str = "Bohr"
    force_unit: str = "Ha/Bohr"
    element: str = "Si"
    energy_reference: str = "cohesive"
    training_temperature_range_mha: tuple[float, float] | None = (1.0, 100.0)

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError(f"unsupported model-config schema: {self.schema_version}")
        if len(self.radial_cutoff_radii) + len(self.angular_cutoff_radii) != len(
            self.scale_factors
        ):
            raise ValueError("one scale factor is required for every symmetry function")
        if not all(math.isfinite(value) and value > 0 for value in self.scale_factors):
            raise ValueError("scale factors must be finite and positive")
        if not (
            len(self.angular_cutoff_radii)
            == len(self.angular_lambda)
            == len(self.angular_zeta)
        ):
            raise ValueError("all angular parameter arrays must have equal length")
        if len(self.hidden_nodes) != len(self.activations):
            raise ValueError("one activation is required for every hidden layer")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if not self.element:
            raise ValueError("element must not be empty")
        if self.energy_reference not in {"cohesive", "total"}:
            raise ValueError("energy_reference must be 'cohesive' or 'total'")
        if self.training_temperature_range_mha is not None:
            low, high = self.training_temperature_range_mha
            if low < 0 or high <= low:
                raise ValueError("invalid training temperature range")

    @classmethod
    def from_dict(cls, values):
        """Create a validated configuration from JSON-compatible values."""
        tuple_fields = {
            "radial_cutoff_radii",
            "angular_cutoff_radii",
            "angular_lambda",
            "angular_zeta",
            "scale_factors",
            "periodic",
            "cell",
            "hidden_nodes",
            "activations",
            "training_temperature_range_mha",
        }
        values = dict(values)
        for name in tuple_fields:
            if name in values and values[name] is not None:
                values[name] = tuple(values[name])
        return cls(**values)

    def to_dict(self):
        """Return a JSON-compatible mapping of model settings."""
        return asdict(self)


def load_model_config(name="silicon_2023"):
    """Load a bundled, human-readable model configuration."""
    path = Path(name)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        resource = files("tennpot").joinpath("configs", f"{name}.json")
        text = resource.read_text(encoding="utf-8")
    return ModelConfig.from_dict(json.loads(text))


def build_model(config, *, device=None, dtype=None):
    """Construct an untrained TeNNPot from a validated configuration."""
    if not isinstance(config, ModelConfig):
        config = ModelConfig.from_dict(config)
    if dtype is None:
        dtype = getattr(torch, config.dtype)
    tensor_kwargs = {"device": device, "dtype": dtype}
    model = TeNNPot(
        rad_cutoff_radii=torch.tensor(config.radial_cutoff_radii, **tensor_kwargs),
        ang_cutoff_radii=torch.tensor(config.angular_cutoff_radii, **tensor_kwargs),
        lambda_=torch.tensor(config.angular_lambda, **tensor_kwargs),
        zeta=torch.tensor(config.angular_zeta, **tensor_kwargs),
        periodic=config.periodic,
        box_length=config.cell,
        nodes_hid=config.hidden_nodes,
        activations=config.activations,
        scale_factors=torch.tensor(config.scale_factors, **tensor_kwargs),
        device=device,
        dtype=dtype,
    )
    model.config = config
    return model

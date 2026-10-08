"""Optional MLflow integration kept outside the core training module."""

from __future__ import annotations

import contextlib
import json
import platform
from dataclasses import asdict

import numpy as np
import torch


@contextlib.contextmanager
def tracking_run(config, model_config, dataset_metadata=None):
    """Open and populate an optional MLflow run."""
    if not config.mlflow_enabled:
        yield None
        return
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow logging requires the 'mlflow' extra: "
            "python -m pip install .[mlflow]"
        ) from error

    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run(run_name=config.mlflow_run_name):
        training_values = {
            f"training.{name}": value
            for name, value in asdict(config).items()
            if not name.startswith("mlflow_") and value is not None
        }
        model_values = {
            "model.name": model_config.name,
            "model.element": model_config.element,
            "model.dtype": model_config.dtype,
            "model.hidden_nodes": json.dumps(model_config.hidden_nodes),
            "model.activations": json.dumps(model_config.activations),
        }
        environment = {
            "environment.python": platform.python_version(),
            "environment.pytorch": torch.__version__,
            "environment.numpy": np.__version__,
        }
        mlflow.log_params({**training_values, **model_values, **environment})
        if dataset_metadata:
            mlflow.log_dict(dataset_metadata, "dataset.json")
        mlflow.log_dict(model_config.to_dict(), "model_config.json")
        mlflow.log_dict(asdict(config), "training_config.json")
        yield mlflow

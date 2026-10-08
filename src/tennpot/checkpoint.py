"""Safe, versioned TeNNPot checkpoint utilities."""

from __future__ import annotations

import json
from importlib.resources import as_file, files
from pathlib import Path

import torch

from tennpot._utils import file_sha256
from tennpot.config import ModelConfig, build_model

CHECKPOINT_FORMAT_VERSION = 1
PUBLISHED_MODEL_NAME = "silicon_2023"


def published_model_info(*, verify=True):
    """Return provenance metadata for the model distributed with TeNNPot.

    Parameters
    ----------
    verify : bool
        Verify the checkpoint SHA-256 checksum before returning metadata.
    """
    package = files("tennpot").joinpath("saved_models")
    manifest = json.loads(
        package.joinpath(f"{PUBLISHED_MODEL_NAME}.json").read_text(encoding="utf-8")
    )
    if verify:
        resource = package.joinpath(f"{PUBLISHED_MODEL_NAME}.pt")
        with as_file(resource) as checkpoint_path:
            actual = file_sha256(checkpoint_path)
        expected = manifest["checkpoint_sha256"]
        if actual != expected:
            raise RuntimeError(
                "published checkpoint checksum mismatch: "
                f"expected {expected}, got {actual}"
            )
    return manifest


def checkpoint_payload(
    model,
    config,
    *,
    optimizer=None,
    epoch=None,
    metrics=None,
    metadata=None,
    training_state=None,
    training_config=None,
    rng_state=None,
):
    """Create a weights-only-compatible checkpoint payload."""
    if not isinstance(config, ModelConfig):
        config = ModelConfig.from_dict(config)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": dict(metrics or {}),
        "metadata": dict(metadata or {}),
    }
    if training_state is not None:
        payload["training_state"] = training_state
    if training_config is not None:
        payload["training_config"] = training_config
    if rng_state is not None:
        payload["rng_state"] = rng_state
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def save_checkpoint(path, model, config, **kwargs):
    """Save model state without pickling model classes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, config, **kwargs), path)


def load_checkpoint(path, *, device="cpu", dtype=None):
    """Load a versioned checkpoint using PyTorch's restricted unpickler."""
    payload = torch.load(path, map_location=device, weights_only=True)
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format: {version}")
    config = ModelConfig.from_dict(payload["config"])
    model = build_model(config, device=device, dtype=dtype)
    model.load_state_dict(payload["model_state_dict"])
    model.checkpoint_metadata = dict(payload.get("metadata", {}))
    model.eval()
    return model, payload


def load_published_model(*, device="cpu", dtype=None):
    """Load the converted model used in the 2023 paper."""
    info = published_model_info(verify=True)
    resource = files("tennpot").joinpath("saved_models", f"{PUBLISHED_MODEL_NAME}.pt")
    with as_file(resource) as checkpoint_path:
        model, _ = load_checkpoint(checkpoint_path, device=device, dtype=dtype)
    model.model_info = info
    return model


def _load_selected_model(checkpoint, device):
    """Load the bundled or selected checkpoint and return a short cache key."""
    if checkpoint is None:
        model = load_published_model(device=device)
        return model, model.model_info["checkpoint_sha256"][:12]
    checkpoint = Path(checkpoint)
    digest = file_sha256(checkpoint)[:12]
    model, _ = load_checkpoint(checkpoint, device=device)
    return model, digest

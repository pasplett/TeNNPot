"""Train a single-element TeNNPot using the published protocol."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from tennpot.checkpoint import load_checkpoint
from tennpot.cli._errors import report_input_errors
from tennpot.config import build_model, load_model_config
from tennpot.dataset import load_dataset
from tennpot.training import (
    TrainingConfig,
    atomistic_collate,
    fit_scale_factors,
    load_training_config,
    seed_everything,
    train_model,
)

logger = logging.getLogger(__name__)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/training"))
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="deterministically sample at most this many training structures",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        help="deterministically sample at most this many validation structures",
    )
    parser.add_argument(
        "--energy-reference", choices=("cohesive", "total"), default="cohesive"
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--use-config-scale-factors", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--mlflow-experiment")
    parser.add_argument("--mlflow-run-name")
    parser.add_argument("--debug", action="store_true", help="show error tracebacks")
    return parser


def _override_training_config(config, args):
    updates = {}
    for argument, field in (
        (args.epochs, "epochs"),
        (args.batch_size, "batch_size"),
        (args.learning_rate, "learning_rate"),
        (args.patience, "early_stopping_patience"),
        (args.seed, "seed"),
        (args.mlflow_tracking_uri, "mlflow_tracking_uri"),
        (args.mlflow_experiment, "mlflow_experiment"),
        (args.mlflow_run_name, "mlflow_run_name"),
    ):
        if argument is not None:
            updates[field] = argument
    if args.deterministic:
        updates["deterministic"] = True
    if args.mlflow or args.mlflow_tracking_uri:
        updates["mlflow_enabled"] = True
    return replace(config, **updates)


def _dataset(path, split, args, dtype, seed):
    return load_dataset(
        path,
        split=split,
        validation_fraction=args.validation_fraction,
        seed=seed,
        energy_reference=args.energy_reference,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        use_all_frames=args.validation_data is not None,
        dtype=dtype,
    )


def _limit_dataset(dataset, maximum, seed):
    if maximum is None:
        return dataset
    if maximum < 1:
        raise ValueError("sample caps must be positive")
    if maximum >= len(dataset):
        return dataset
    selected = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )
    return Subset(dataset, selected[:maximum].tolist())


def main(argv=None):
    """Train or resume a single-element model from command-line arguments."""
    parser = _parser()
    args = parser.parse_args(argv)
    with report_input_errors(parser, debug=args.debug):
        return _run(args)


def _run(args):
    """Execute training after parsing command-line options."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    resume_payload = None
    if args.resume is not None:
        model, resume_payload = load_checkpoint(
            args.resume,
            device=args.device,
            dtype=None if args.dtype is None else getattr(torch, args.dtype),
        )
    if (
        resume_payload is not None
        and args.training_config is None
        and "training_config" in resume_payload
    ):
        base_training_config = TrainingConfig.from_dict(
            resume_payload["training_config"]
        )
    else:
        base_training_config = load_training_config(args.training_config)
    training_config = _override_training_config(base_training_config, args)
    if args.dtype is None:
        args.dtype = model.config.dtype if resume_payload is not None else "float32"
    if (
        args.validation_data is not None
        and args.dataset.resolve() == args.validation_data.resolve()
    ):
        raise ValueError("--validation-data must be a separate file")
    dtype = getattr(torch, args.dtype)
    seed_everything(training_config.seed, deterministic=training_config.deterministic)

    train_dataset = _dataset(args.dataset, "train", args, dtype, training_config.seed)
    validation_path = args.validation_data or args.dataset
    validation_dataset = _dataset(
        validation_path, "validation", args, dtype, training_config.seed
    )
    train_dataset = _limit_dataset(
        train_dataset, args.max_train_samples, training_config.seed
    )
    validation_dataset = _limit_dataset(
        validation_dataset, args.max_validation_samples, training_config.seed + 1
    )
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("training and validation partitions must both be non-empty")
    logger.info(
        "Dataset: %d training and %d validation structures | device %s | dtype %s",
        len(train_dataset),
        len(validation_dataset),
        args.device,
        args.dtype,
    )

    generator = torch.Generator().manual_seed(training_config.seed)
    loader_kwargs = {
        "batch_size": training_config.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": atomistic_collate,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    if resume_payload is not None:
        model_config = replace(model.config, dtype=args.dtype)
        model.config = model_config
    else:
        model_config = load_model_config(args.model_config or "silicon_2023")
        model_config = replace(model_config, dtype=args.dtype)
        if not args.use_config_scale_factors:
            logger.info("Fitting descriptor scales from the training partition")
            unscaled = replace(
                model_config,
                scale_factors=tuple(1.0 for _ in model_config.scale_factors),
            )
            descriptor_model = build_model(unscaled, device=args.device, dtype=dtype)
            fitted = fit_scale_factors(descriptor_model, train_loader)
            model_config = replace(model_config, scale_factors=fitted)
        seed_everything(
            training_config.seed, deterministic=training_config.deterministic
        )
        model = build_model(model_config, device=args.device, dtype=dtype)
    logger.info(
        "Model: %s | epochs %d | batch size %d",
        model_config.name,
        training_config.epochs,
        training_config.batch_size,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "model_config.json").write_text(
        json.dumps(model_config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "training_config.json").write_text(
        json.dumps(asdict(training_config), indent=2) + "\n", encoding="utf-8"
    )
    dataset_metadata = getattr(train_dataset, "manifest", None)
    best_checkpoint, _ = train_model(
        model,
        train_loader,
        validation_loader,
        model_config=model_config,
        output_dir=args.output,
        config=training_config,
        resume_payload=resume_payload,
        dataset_metadata=dataset_metadata,
    )
    print(best_checkpoint)


if __name__ == "__main__":
    main()

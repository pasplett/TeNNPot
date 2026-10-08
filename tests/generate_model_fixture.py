"""Regenerate the public regression fixture from one private Si structure.

Only positions and electronic temperature are read from the source archive.
All saved descriptors, gradients, energy, and forces are computed by the
bundled silicon_2023 model. No source energy or force labels are copied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tennpot import load_published_model, published_model_info


def main(argv=None):
    """Write a model-generated regression fixture for one source structure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/si_thin_film_chives.npz")
    )
    parser.add_argument("--index", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("tests/test_tennpot.npz"))
    args = parser.parse_args(argv)

    with np.load(args.dataset, allow_pickle=False) as source:
        positions = np.array(source["positions_train"][args.index : args.index + 1])
        temperature = np.array(source["el_temp_train"][args.index : args.index + 1])
    if positions.shape != (1, 320, 3) or temperature.shape != (1, 1):
        raise ValueError("the selected source sample has an unexpected shape")

    model = load_published_model(dtype=torch.float64)
    cell = np.diag(np.asarray(model.config.cell, dtype=np.float64))
    periodic = np.asarray(model.config.periodic, dtype=np.bool_)
    positions_tensor = torch.from_numpy(positions).requires_grad_()
    temperature_tensor = torch.from_numpy(temperature)
    cell_tensor = torch.from_numpy(cell)

    radial = model.radial(positions_tensor, cell=cell_tensor, periodic=periodic)
    angular = model.angular(positions_tensor, cell=cell_tensor, periodic=periodic)
    radial_grad = torch.autograd.grad(
        radial.sum(), positions_tensor, retain_graph=True
    )[0]
    angular_grad = torch.autograd.grad(angular.sum(), positions_tensor)[0]
    energy, forces = model(
        positions_tensor.detach(),
        temperature_tensor,
        cell=cell_tensor,
        periodic=periodic,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positions=positions,
        electronic_temperature=temperature,
        cell=cell,
        periodic=periodic,
        radial=radial.detach().numpy(),
        angular=angular.detach().numpy(),
        radial_grad=radial_grad.detach().numpy(),
        angular_grad=angular_grad.detach().numpy(),
        model_energy=energy.detach().numpy(),
        model_forces=forces.detach().numpy(),
        model_sha256=np.asarray(published_model_info()["checkpoint_sha256"]),
    )


if __name__ == "__main__":
    main()

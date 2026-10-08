"""Validated, atomic snapshots for deterministic ASE velocity-Verlet MD."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
import tempfile

import ase
import numpy as np
import torch
from ase.constraints import FixAtoms, FixCartesian
from ase.io.jsonio import encode
from ase.io.trajectory import Trajectory

from tennpot import __version__

_METADATA_KEY = "tennpot_md"
_SCHEMA_VERSION = 1


def validate_md_options(steps, timestep_fs, temperature, checkpoint_interval):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if not all(
        math.isfinite(value) and value > 0 for value in (timestep_fs, temperature)
    ):
        raise ValueError("temperatures/timestep must be finite and positive")
    if (
        not isinstance(checkpoint_interval, int)
        or isinstance(checkpoint_interval, bool)
        or checkpoint_interval < 1
    ):
        raise ValueError("checkpoint_interval must be a positive integer")


def _digest(value):
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def _constraints(atoms):
    # These constraints have no evolving internal state to restore.
    if any(type(c) not in (FixAtoms, FixCartesian) for c in atoms.constraints):
        raise ValueError(
            "MD checkpointing supports only FixAtoms and FixCartesian constraints; "
            "disable checkpointing for other constraints"
        )
    return [constraint.todict() for constraint in atoms.constraints]


def _static_atoms(atoms):
    return {
        "numbers": atoms.numbers,
        "masses": atoms.get_masses(),
        "cell": atoms.cell.array,
        "pbc": atoms.pbc,
        "constraints": _constraints(atoms),
    }


def _initial_identity(atoms, reference):
    return _digest(
        {
            **_static_atoms(atoms),
            "positions": atoms.positions,
            "momenta": atoms.get_momenta(),
            "reference": reference,
        }
    )


def _validate_atoms(atoms):
    if len(atoms) == 0:
        raise ValueError("MD requires at least one atom")
    for value in (atoms.positions, atoms.get_momenta()):
        if value.shape != (len(atoms), 3) or not np.isfinite(value).all():
            raise ValueError(
                "MD positions and momenta must be finite (atoms, 3) arrays"
            )
    masses = atoms.get_masses()
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("MD masses must be finite and positive")
    if not np.isfinite(atoms.cell.array).all():
        raise ValueError("MD cell vectors must be finite")
    if atoms.pbc.any() and atoms.cell.rank < 3:
        raise ValueError("periodic MD requires a three-dimensional cell")


def md_checkpoint_settings(model, temperature, timestep_fs, device):
    """Identify the effective calculator, integrator and numerical environment."""
    config = getattr(model, "config", None)
    architecture = [
        (name, type(module).__module__ + "." + type(module).__qualname__)
        for name, module in model.named_modules()
    ]
    digest = hashlib.sha256(
        encode(
            {
                "architecture": architecture,
                "config": config.to_dict() if config else None,
            }
        ).encode("utf-8")
    )
    for name, tensor in sorted(model.state_dict().items()):
        # The ASE calculator evaluates floating tensors in float64.
        tensor = tensor.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.to(torch.float64)
        digest.update(
            json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        )
        digest.update(tensor.contiguous().numpy().tobytes())
    device = torch.device(device)
    backend = {
        "device": str(device),
        "platform": platform.system(),
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        backend.update(
            gpu=torch.cuda.get_device_name(device),
            cuda=torch.version.cuda,
            tf32=torch.backends.cuda.matmul.allow_tf32,
        )
    return {
        "model_sha256": digest.hexdigest(),
        "integrator": "ase.md.verlet.VelocityVerlet",
        # Increment this when calculator or MD semantics change.
        "calculation_version": 1,
        "tennpot_version": __version__,
        "electronic_temperature_kelvin": float(temperature),
        "timestep_fs": float(timestep_fs),
        "energy_reference": "cohesive",
        "dtype": "float64",
        "ase_version": ase.__version__,
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "backend": backend,
    }


def md_checkpoint_path(directory, settings, atoms, reference, number):
    """Separate changed calculations while allowing longer or larger ensembles."""
    identity = _digest(
        {"settings": settings, "initial_sha256": _initial_identity(atoms, reference)}
    )
    return Path(directory) / f"run_{number:04d}_{identity}.traj"


def load_md_checkpoint(path, settings, initial_atoms, reference):
    """Read a compatible single-frame snapshot and its complete RMSD history."""
    try:
        with Trajectory(str(path), "r") as trajectory:
            if len(trajectory) != 1:
                raise ValueError("expected one restart snapshot")
            atoms = trajectory[0]
        metadata = atoms.info[_METADATA_KEY]
        if (
            type(metadata["schema_version"]) is not int
            or metadata["schema_version"] != _SCHEMA_VERSION
        ):
            raise ValueError("unsupported checkpoint schema")
        if metadata["settings"] != settings:
            raise ValueError("model, MD settings or numerical environment changed")
        if metadata["initial_sha256"] != _initial_identity(initial_atoms, reference):
            raise ValueError("initial conditions or RMSD reference changed")
        if not np.array_equal(metadata["reference_positions_angstrom"], reference):
            raise ValueError("RMSD reference changed")
        _validate_atoms(atoms)
        if not atoms.has("momenta"):
            raise ValueError("missing checkpoint momenta")
        if _digest(_static_atoms(atoms)) != _digest(_static_atoms(initial_atoms)):
            raise ValueError("atom identities, masses, cell or constraints changed")
        completed = metadata["completed_steps"]
        rmsd = np.asarray(metadata["rmsd_angstrom"], dtype=float)
        if (
            type(completed) is not int
            or completed < 0
            or rmsd.shape != (completed + 1,)
            or not np.isfinite(rmsd).all()
            or np.any(rmsd < 0)
        ):
            raise ValueError("invalid step count or RMSD history")
        expected_initial = np.sqrt(
            np.mean(np.sum((initial_atoms.positions - reference) ** 2, axis=1))
        )
        expected_last = np.sqrt(
            np.mean(np.sum((atoms.positions - reference) ** 2, axis=1))
        )
        if not np.allclose(
            [rmsd[0], rmsd[-1]],
            [expected_initial, expected_last],
            rtol=1e-12,
            atol=1e-14,
        ):
            raise ValueError("RMSD history does not match the saved positions")
    except Exception as error:
        raise ValueError(
            f"invalid or incompatible MD checkpoint {path}: {error}"
        ) from error
    atoms.info.pop(_METADATA_KEY)
    return atoms, rmsd.tolist(), completed


def save_md_checkpoint(path, atoms, settings, initial_atoms, reference, rmsd):
    """Replace a checkpoint only after a complete, flushed snapshot is written."""
    _validate_atoms(atoms)
    rmsd = np.asarray(rmsd, dtype=float)
    if (
        rmsd.ndim != 1
        or not rmsd.size
        or not np.isfinite(rmsd).all()
        or np.any(rmsd < 0)
    ):
        raise ValueError("MD checkpoint RMSD history must be finite and non-negative")
    snapshot = atoms.copy()
    snapshot.info = {
        _METADATA_KEY: {
            "schema_version": _SCHEMA_VERSION,
            "settings": settings,
            "initial_sha256": _initial_identity(initial_atoms, reference),
            "reference_positions_angstrom": np.asarray(reference),
            "completed_steps": len(rmsd) - 1,
            "rmsd_angstrom": np.asarray(rmsd),
        }
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with Trajectory(str(temporary), "w", properties=[]) as trajectory:
            trajectory.write(snapshot)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

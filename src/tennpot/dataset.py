"""Datasets for public Extended XYZ and the original CHIVES archive."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from ase.data import atomic_numbers
from ase.io import read
from torch.utils.data import Dataset

from tennpot._utils import file_sha256
from tennpot.config import load_model_config
from tennpot.physics import (
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    isolated_atom_free_energy,
    kelvin_to_model_temperature,
)

DATASET_SCHEMA_VERSION = 1


def _normalize_split(value):
    value = str(value).lower()
    if value == "test":
        value = "validation"
    if value not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    return value


class AtomisticDataset(Dataset):
    """Common mapping-style dataset consumed by TeNNPot training."""

    split: str


class CHIVESDataset(AtomisticDataset):
    """Read the original fixed-size CHIVES NPZ training archive directly."""

    def __init__(self, file, *, split="train", dtype=torch.float64):
        self.split = _normalize_split(split)
        self.train = self.split == "train"
        partition = "train" if self.train else "test"
        self.dtype = dtype
        config = load_model_config("silicon_2023")

        required = ("positions", "el_temp", "energy", "forces")
        with np.load(file, allow_pickle=False) as archive:
            missing = [
                f"{name}_{partition}"
                for name in required
                if f"{name}_{partition}" not in archive
            ]
            if missing:
                raise ValueError(f"dataset is missing arrays: {missing}")
            self.positions = self._tensor(archive[f"positions_{partition}"])
            self.el_temp = self._tensor(archive[f"el_temp_{partition}"])
            self.energy = self._tensor(archive[f"energy_{partition}"])
            self.forces = self._tensor(archive[f"forces_{partition}"])

        self.cell = torch.tensor(config.cell, dtype=dtype).diag()
        self.pbc = torch.tensor(config.periodic, dtype=torch.bool)
        self.atomic_number = atomic_numbers[config.element]
        self._validate()

    def _tensor(self, values):
        return torch.from_numpy(np.array(values, copy=True)).to(self.dtype)

    def _validate(self):
        count = self.positions.shape[0]
        if self.positions.ndim != 3 or self.positions.shape[-1] != 3:
            raise ValueError("positions must have shape (samples, atoms, 3)")
        if self.forces.shape != self.positions.shape:
            raise ValueError("forces and positions must have identical shapes")
        for name in ("el_temp", "energy"):
            if getattr(self, name).shape != (count, 1):
                raise ValueError(f"{name} must have shape (samples, 1)")
        physical = (self.positions, self.el_temp, self.energy, self.forces)
        if not all(torch.isfinite(values).all() for values in physical):
            raise ValueError("dataset contains non-finite physical values")

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, index):
        atom_count = self.positions.shape[1]
        return {
            "positions": self.positions[index],
            "electronic_temperature": self.el_temp[index],
            "energy": self.energy[index],
            "forces": self.forces[index],
            "cell": self.cell,
            "pbc": self.pbc,
            "atomic_numbers": torch.full(
                (atom_count,), self.atomic_number, dtype=torch.int64
            ),
            "source_index": torch.tensor(index, dtype=torch.int64),
        }


class ExtendedXYZDataset(AtomisticDataset):
    """Read Extended XYZ through a reusable, compressed native cache."""

    def __init__(
        self,
        file,
        *,
        split="train",
        validation_fraction=0.1,
        seed=42,
        energy_reference="cohesive",
        cache_dir=None,
        rebuild_cache=False,
        use_all_frames=False,
        dtype=torch.float64,
    ):
        self.use_all_frames = use_all_frames
        self.source = Path(file)
        if not self.source.is_file():
            raise FileNotFoundError(self.source)
        self.split = _normalize_split(split)
        self.train = self.split == "train"
        self.dtype = dtype
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if energy_reference not in {"cohesive", "total"}:
            raise ValueError("energy_reference must be 'cohesive' or 'total'")

        source_hash = file_sha256(self.source)
        cache_root = Path(cache_dir or self.source.parent / ".tennpot-cache")
        options = json.dumps(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "source_sha256": source_hash,
                "validation_fraction": validation_fraction,
                "seed": seed,
                "energy_reference": energy_reference,
                "use_all_frames": use_all_frames,
            },
            sort_keys=True,
        )
        cache_key = hashlib.sha256(options.encode()).hexdigest()[:12]
        self.cache_path = cache_root / f"{self.source.stem}-{cache_key}"
        manifest_path = self.cache_path / "manifest.json"
        if rebuild_cache or not manifest_path.is_file():
            self._build_cache(
                source_hash=source_hash,
                validation_fraction=validation_fraction,
                seed=seed,
                energy_reference=energy_reference,
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._load_groups()

    @staticmethod
    def _frame_value(atoms, name):
        if name in atoms.info:
            return atoms.info[name]
        if atoms.calc is not None and name in atoms.calc.results:
            return atoms.calc.results[name]
        raise ValueError(f"missing required per-structure value {name!r}")

    @classmethod
    def _extract_frame(cls, atoms, index, energy_reference):
        if len(atoms) == 0:
            raise ValueError(f"structure {index} is empty")
        symbols = set(atoms.get_chemical_symbols())
        if len(symbols) != 1:
            raise ValueError(f"structure {index} is not single-element")
        positions = np.asarray(atoms.positions, dtype=np.float64)
        forces = np.asarray(cls._frame_value(atoms, "forces"), dtype=np.float64)
        energy = float(cls._frame_value(atoms, "energy"))
        temperature = float(cls._frame_value(atoms, "electronic_temperature"))
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        if forces.shape != positions.shape:
            raise ValueError(f"structure {index} has invalid force shape")
        arrays = (positions, forces, cell, np.array([energy, temperature]))
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError(f"structure {index} contains non-finite values")
        if temperature <= 0:
            raise ValueError(f"structure {index} has non-positive temperature")
        if atoms.pbc.any() and atoms.cell.rank < 3:
            raise ValueError(f"periodic structure {index} requires a full cell")

        energy_hartree = energy / HARTREE_TO_EV
        if energy_reference == "total":
            if symbols != {"Si"}:
                raise ValueError("the isolated-atom reference is available only for Si")
            isolated = isolated_atom_free_energy(temperature).item()
            energy_hartree -= len(atoms) * isolated
        split = atoms.info.get("split")
        return {
            "positions": positions / BOHR_TO_ANGSTROM,
            "forces": forces * BOHR_TO_ANGSTROM / HARTREE_TO_EV,
            "energy": energy_hartree,
            "electronic_temperature": kelvin_to_model_temperature(temperature),
            "cell": cell / BOHR_TO_ANGSTROM,
            "pbc": np.asarray(atoms.pbc, dtype=np.bool_),
            "atomic_number": int(atoms.numbers[0]),
            "split": None if split is None else _normalize_split(split),
            "source_index": index,
        }

    def _build_cache(self, *, source_hash, validation_fraction, seed, energy_reference):
        frames = read(self.source, index=":")
        if not frames:
            raise ValueError("Extended XYZ dataset contains no structures")
        records = [
            self._extract_frame(atoms, index, energy_reference)
            for index, atoms in enumerate(frames)
        ]
        elements = {record["atomic_number"] for record in records}
        if len(elements) != 1:
            raise ValueError(
                "the initial TeNNPot release supports one element per dataset"
            )

        declared = [record["split"] for record in records]
        if self.use_all_frames:
            for record in records:
                record["split"] = "train"
        elif any(value is not None for value in declared):
            if not all(value is not None for value in declared):
                raise ValueError("split must be set on every structure or none")
        else:
            order = np.random.default_rng(seed).permutation(len(records))
            validation_count = int(round(len(records) * validation_fraction))
            if validation_fraction > 0 and len(records) > 1:
                validation_count = max(1, validation_count)
            validation_indices = set(order[:validation_count].tolist())
            for index, record in enumerate(records):
                record["split"] = (
                    "validation" if index in validation_indices else "train"
                )

        self.cache_path.mkdir(parents=True, exist_ok=True)
        groups = {}
        for record in records:
            key = (record["split"], len(record["positions"]))
            groups.setdefault(key, []).append(record)

        shard_metadata = []
        for (split, atom_count), group in sorted(groups.items()):
            filename = f"{split}-atoms-{atom_count}.npz"
            np.savez_compressed(
                self.cache_path / filename,
                positions=np.stack([item["positions"] for item in group]),
                forces=np.stack([item["forces"] for item in group]),
                energy=np.array([[item["energy"]] for item in group]),
                electronic_temperature=np.array(
                    [[item["electronic_temperature"]] for item in group]
                ),
                cell=np.stack([item["cell"] for item in group]),
                pbc=np.stack([item["pbc"] for item in group]),
                source_index=np.array(
                    [item["source_index"] for item in group], dtype=np.int64
                ),
            )
            shard_metadata.append(
                {
                    "file": filename,
                    "split": split,
                    "atom_count": atom_count,
                    "structures": len(group),
                }
            )

        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "source_file": self.source.name,
            "source_sha256": source_hash,
            "source_units": {
                "length": "Angstrom",
                "energy": "eV",
                "force": "eV/Angstrom",
                "electronic_temperature": "K",
            },
            "internal_units": {
                "length": "Bohr",
                "energy": "Ha",
                "force": "Ha/Bohr",
                "electronic_temperature": "0.1 Ha",
            },
            "source_energy_reference": energy_reference,
            "internal_energy_reference": "cohesive",
            "atomic_number": next(iter(elements)),
            "structures": len(records),
            "validation_fraction": validation_fraction,
            "use_all_frames": self.use_all_frames,
            "seed": seed,
            "shards": shard_metadata,
        }
        (self.cache_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def _load_groups(self):
        self._groups = []
        self._ends = []
        total = 0
        for shard in self.manifest["shards"]:
            if not self.use_all_frames and shard["split"] != self.split:
                continue
            with np.load(self.cache_path / shard["file"], allow_pickle=False) as data:
                group = {
                    name: torch.from_numpy(np.array(data[name], copy=True))
                    for name in data.files
                }
            floating = (
                "positions",
                "forces",
                "energy",
                "electronic_temperature",
                "cell",
            )
            for name in floating:
                group[name] = group[name].to(self.dtype)
            self._groups.append(group)
            total += len(group["energy"])
            self._ends.append(total)

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        group_index = bisect_right(self._ends, index)
        start = 0 if group_index == 0 else self._ends[group_index - 1]
        group = self._groups[group_index]
        local_index = index - start
        atom_count = group["positions"].shape[1]
        return {
            "positions": group["positions"][local_index],
            "electronic_temperature": group["electronic_temperature"][local_index],
            "energy": group["energy"][local_index],
            "forces": group["forces"][local_index],
            "cell": group["cell"][local_index],
            "pbc": group["pbc"][local_index].to(torch.bool),
            "atomic_numbers": torch.full(
                (atom_count,), self.manifest["atomic_number"], dtype=torch.int64
            ),
            "source_index": group["source_index"][local_index],
        }


def load_dataset(path, **kwargs):
    """Load CHIVES NPZ or public Extended XYZ data by filename extension."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        supported = {"split", "dtype"}
        chives_kwargs = {
            key: value for key, value in kwargs.items() if key in supported
        }
        return CHIVESDataset(path, **chives_kwargs)
    return ExtendedXYZDataset(path, **kwargs)

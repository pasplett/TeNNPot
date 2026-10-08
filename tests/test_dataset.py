from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from pytest import approx

from tennpot.dataset import CHIVESDataset, ExtendedXYZDataset
from tennpot.physics import BOHR_TO_ANGSTROM, HARTREE_TO_EV


@pytest.mark.data
def test_chives_train_dataset(si_thin_film_train):
    assert si_thin_film_train.train
    assert len(si_thin_film_train) == 9727

    sample = si_thin_film_train[42]
    positions = sample["positions"]
    el_temp = sample["electronic_temperature"]
    energy = sample["energy"]
    forces = sample["forces"]
    assert positions.shape == (320, 3)
    assert el_temp.shape == (1,)
    assert energy.shape == (1,)
    assert forces.shape == (320, 3)

    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()


@pytest.mark.data
def test_chives_test_dataset(si_thin_film_test):
    assert not si_thin_film_test.train
    assert len(si_thin_film_test) == 8767

    sample = si_thin_film_test[42]
    positions = sample["positions"]
    el_temp = sample["electronic_temperature"]
    energy = sample["energy"]
    forces = sample["forces"]
    assert positions.shape == (320, 3)
    assert el_temp.shape == (1,)
    assert energy.shape == (1,)
    assert forces.shape == (320, 3)

    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()
    assert si_thin_film_test.split == "validation"


@pytest.fixture
def si_thin_film_train():
    path = Path("data/si_thin_film_chives.npz")
    if not path.is_file():
        pytest.skip("full untracked CHIVES archive is unavailable")
    return CHIVESDataset(file=path, split="train")


@pytest.fixture
def si_thin_film_test():
    path = Path("data/si_thin_film_chives.npz")
    if not path.is_file():
        pytest.skip("full untracked CHIVES archive is unavailable")
    return CHIVESDataset(file=path, split="validation")


def _training_atoms(symbols, positions, *, split=None):
    atoms = Atoms(symbols, positions=positions, cell=[8, 8, 8], pbc=True)
    atoms.info["energy"] = -2.5
    atoms.info["electronic_temperature"] = 5000.0
    if split is not None:
        atoms.info["split"] = split
    atoms.arrays["forces"] = np.full((len(atoms), 3), 0.25)
    return atoms


def test_extended_xyz_builds_and_reuses_native_cache(tmp_path):
    source = tmp_path / "training.extxyz"
    frames = [
        _training_atoms("Si2", [[0, 0, 0], [2, 0, 0]], split="train"),
        _training_atoms("Si3", [[0, 0, 0], [2, 0, 0], [0, 2, 0]], split="train"),
        _training_atoms("Si2", [[0, 0, 0], [2.1, 0, 0]], split="validation"),
    ]
    write(source, frames, format="extxyz")

    train = ExtendedXYZDataset(source, split="train", cache_dir=tmp_path / "cache")
    validation = ExtendedXYZDataset(
        source, split="validation", cache_dir=tmp_path / "cache"
    )
    assert len(train) == 2
    assert len(validation) == 1
    assert train.manifest["structures"] == 3
    assert len(train.manifest["shards"]) == 3
    assert (train.cache_path / "manifest.json").is_file()

    sample = train[0]
    assert set(sample) == {
        "positions",
        "electronic_temperature",
        "energy",
        "forces",
        "cell",
        "pbc",
        "atomic_numbers",
        "source_index",
    }
    assert sample["energy"].item() == approx(-2.5 / HARTREE_TO_EV)
    assert sample["positions"][1, 0].item() == approx(2 / BOHR_TO_ANGSTROM)
    assert sample["forces"][0, 0].item() == approx(
        0.25 * BOHR_TO_ANGSTROM / HARTREE_TO_EV
    )


def test_extended_xyz_constructs_a_deterministic_split(tmp_path):
    source = tmp_path / "unsplit.extxyz"
    frames = [
        _training_atoms("Si2", [[0, 0, 0], [2 + index / 10, 0, 0]])
        for index in range(10)
    ]
    write(source, frames, format="extxyz")
    first = ExtendedXYZDataset(
        source,
        split="validation",
        validation_fraction=0.2,
        seed=7,
        cache_dir=tmp_path / "cache",
    )
    second = ExtendedXYZDataset(
        source,
        split="validation",
        validation_fraction=0.2,
        seed=7,
        cache_dir=tmp_path / "cache",
    )
    assert len(first) == 2
    assert [first[i]["source_index"].item() for i in range(2)] == [
        second[i]["source_index"].item() for i in range(2)
    ]


def test_extended_xyz_rejects_invalid_public_frames():
    atoms = _training_atoms("Si2", [[0, 0, 0], [2, 0, 0]])
    del atoms.arrays["forces"]
    with pytest.raises(ValueError, match="missing required.*forces"):
        ExtendedXYZDataset._extract_frame(atoms, 0, "cohesive")

    atoms = _training_atoms("Si2", [[0, 0, 0], [2, 0, 0]])
    atoms.info["forces"] = atoms.arrays["forces"]
    atoms.info["electronic_temperature"] = -1
    with pytest.raises(ValueError, match="non-positive temperature"):
        ExtendedXYZDataset._extract_frame(atoms, 0, "cohesive")

    atoms = _training_atoms("Si2", [[0, 0, 0], [2, 0, 0]])
    atoms.info["forces"] = atoms.arrays["forces"]
    atoms.cell = [8, 8, 0]
    with pytest.raises(ValueError, match="three-dimensional cell|full cell"):
        ExtendedXYZDataset._extract_frame(atoms, 0, "cohesive")

    atoms = _training_atoms("SiGe", [[0, 0, 0], [2, 0, 0]])
    with pytest.raises(ValueError, match="not single-element"):
        ExtendedXYZDataset._extract_frame(atoms, 0, "cohesive")

    atoms = _training_atoms("Si2", [[0, 0, 0], [2, 0, 0]])
    atoms.info["forces"] = np.zeros((1, 3))
    with pytest.raises(ValueError, match="invalid force shape"):
        ExtendedXYZDataset._extract_frame(atoms, 0, "cohesive")

    atoms = _training_atoms("Si2", [[0, 0, 0], [2, 0, 0]])
    atoms.info["forces"] = atoms.arrays["forces"]
    atoms.info["energy"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ExtendedXYZDataset._extract_frame(atoms, 0, "cohesive")


def test_distributable_training_example_has_declared_splits():
    source = Path("examples/silicon_demo.extxyz")
    train = ExtendedXYZDataset(source, split="train")
    validation = ExtendedXYZDataset(source, split="validation")
    assert len(train) == 4
    assert len(validation) == 1
    assert train[0]["positions"].shape == (8, 3)


def test_chives_common_interface(tmp_path):
    path = tmp_path / "small.npz"
    arrays = {}
    for partition in ("train", "test"):
        arrays[f"positions_{partition}"] = np.zeros((2, 3, 3))
        arrays[f"forces_{partition}"] = np.zeros((2, 3, 3))
        arrays[f"energy_{partition}"] = np.zeros((2, 1))
        arrays[f"el_temp_{partition}"] = np.ones((2, 1)) * 0.4
    np.savez_compressed(path, **arrays)

    dataset = CHIVESDataset(path)
    sample = dataset[0]
    assert sample["positions"].shape == (3, 3)
    assert torch.equal(sample["pbc"], torch.tensor([True, True, False]))
    assert sample["atomic_numbers"].tolist() == [14, 14, 14]


@pytest.mark.parametrize("declared_splits", [False, True])
def test_explicit_file_mode_uses_every_frame(tmp_path, declared_splits):
    source = tmp_path / "explicit.extxyz"
    frames = [
        _training_atoms(
            "Si2",
            [[0, 0, 0], [2 + index / 10, 0, 0]],
            split=(
                ("train" if index == 0 else "validation") if declared_splits else None
            ),
        )
        for index in range(3)
    ]
    write(source, frames, format="extxyz")
    for split in ("train", "validation"):
        dataset = ExtendedXYZDataset(source, split=split, use_all_frames=True)
        assert len(dataset) == 3
        assert {sample["source_index"].item() for sample in dataset} == {0, 1, 2}
        partitioned = ExtendedXYZDataset(source, split=split)
        assert len(partitioned) < len(dataset)
        assert partitioned.cache_path != dataset.cache_path


def test_total_energy_dataset_rejects_non_silicon_reference():
    atoms = _training_atoms("C2", [[0, 0, 0], [2, 0, 0]])
    atoms.info["forces"] = atoms.arrays["forces"]
    with pytest.raises(ValueError, match="only for Si"):
        ExtendedXYZDataset._extract_frame(atoms, 0, "total")

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read, write
from pytest import approx

from tennpot import predict
from tennpot.ase_calculator import TeNNPotCalculator
from tennpot.checkpoint import save_checkpoint
from tennpot.cli.predict import main as predict_main
from tennpot.config import ModelConfig, build_model
from tennpot.physics import (
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    isolated_atom_free_energy,
    kelvin_to_model_temperature,
)


def _model():
    config = ModelConfig(
        schema_version=1,
        name="calculator test",
        radial_cutoff_radii=(3.0,),
        angular_cutoff_radii=(3.0,),
        angular_lambda=(1.0,),
        angular_zeta=(1.0,),
        scale_factors=(1.0, 1.0),
        periodic=(False, False, False),
        cell=(10.0, 10.0, 10.0),
        hidden_nodes=(3,),
        activations=("tanh",),
    )
    torch.manual_seed(3)
    return build_model(config)


def test_ase_calculator_converts_energy_and_force_units():
    model = _model()
    atoms = Atoms(
        "Si3",
        positions=[[0, 0, 0], [1.1, 0, 0], [0, 1.2, 0]],
        cell=[10, 10, 10],
        pbc=False,
    )
    calculator = TeNNPotCalculator(
        model,
        electronic_temperature=5000,
        energy_reference="cohesive",
    )
    atoms.calc = calculator

    positions = torch.tensor(atoms.positions / BOHR_TO_ANGSTROM, dtype=torch.float64)[
        None
    ]
    temperature = torch.tensor([[5000.0]], dtype=torch.float64)
    reference_energy, reference_forces = model(
        positions,
        kelvin_to_model_temperature(temperature),
        cell=torch.tensor(atoms.cell.array / BOHR_TO_ANGSTROM),
    )
    assert atoms.get_potential_energy() == approx(
        reference_energy.item() * HARTREE_TO_EV
    )
    assert atoms.get_forces() == approx(
        reference_forces.detach().numpy()[0] * HARTREE_TO_EV / BOHR_TO_ANGSTROM
    )


def test_ase_calculator_uses_the_atoms_periodicity():
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    atoms.calc = TeNNPotCalculator(_model(), energy_reference="cohesive")
    assert np.isfinite(atoms.get_potential_energy())


def test_ase_calculator_loads_published_model_by_default():
    calculator = TeNNPotCalculator()
    assert calculator.model.atomic_nn.nodes_in == 74


def test_ase_calculator_rejects_unsupported_elements():
    atoms = Atoms("C", positions=[[0, 0, 0]], cell=[10, 10, 10])
    atoms.calc = TeNNPotCalculator(_model())
    with pytest.raises(ValueError, match="supports only Si"):
        atoms.get_potential_energy()


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_ase_calculator_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="finite and positive"):
        TeNNPotCalculator(_model(), electronic_temperature=temperature)


def test_energy_reference_controls_offset_and_preserves_forces():
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[10, 10, 10])
    atoms.calc = TeNNPotCalculator(
        _model(), electronic_temperature=5000, energy_reference="cohesive"
    )
    cohesive_energy = atoms.get_potential_energy()
    cohesive_forces = atoms.get_forces()
    atoms.calc = TeNNPotCalculator(
        _model(), electronic_temperature=5000, energy_reference="total"
    )
    total_energy = atoms.get_potential_energy()
    offset = len(atoms) * isolated_atom_free_energy(5000).item() * HARTREE_TO_EV
    assert total_energy - cohesive_energy == approx(offset)
    assert atoms.get_forces() == approx(cohesive_forces)
    atoms.calc = TeNNPotCalculator(_model(), electronic_temperature=5000)
    assert atoms.get_potential_energy() == approx(total_energy)


@pytest.mark.parametrize("include", [False, True])
def test_removed_reference_argument_cannot_be_silently_ignored(include):
    with pytest.raises(TypeError, match="use energy_reference"):
        TeNNPotCalculator(_model(), include_isolated_atom=include)


def test_energy_only_request_does_not_return_forces():
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[10, 10, 10])
    calculator = TeNNPotCalculator(_model(), energy_reference="cohesive")
    calculator.calculate(atoms, properties=("energy",))
    assert set(calculator.results) == {"energy", "free_energy"}


def test_predict_accepts_one_or_multiple_structures():
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[10, 10, 10])
    single = predict(
        atoms,
        model=_model(),
        electronic_temperature=5000,
        energy_reference="cohesive",
    )
    multiple = predict(
        [atoms, atoms.copy()],
        model=_model(),
        electronic_temperature=5000,
        energy_reference="cohesive",
    )
    assert np.isfinite(single.energy_ev)
    assert single.forces_ev_per_angstrom.shape == (2, 3)
    assert len(multiple) == 2


def test_prediction_cli_writes_extxyz(tmp_path):
    model = _model()
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, model.config)
    input_path = tmp_path / "input.extxyz"
    output_path = tmp_path / "output.extxyz"
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[10, 10, 10])
    write(input_path, atoms)

    predict_main(
        [
            str(input_path),
            str(output_path),
            "--checkpoint",
            str(checkpoint),
            "--temperature",
            "5000",
            "--energy-reference",
            "cohesive",
        ]
    )

    predicted = read(output_path)
    assert predicted.info["electronic_temperature"] == 5000
    assert np.isfinite(predicted.get_potential_energy())
    assert predicted.get_forces().shape == (2, 3)


def test_non_silicon_model_requires_cohesive_reference():
    from dataclasses import replace

    model = _model()
    model.config = replace(model.config, element="C")
    with pytest.raises(ValueError, match="only for Si"):
        TeNNPotCalculator(model, energy_reference="total")
    atoms = Atoms("C2", positions=[[0, 0, 0], [1.2, 0, 0]])
    atoms.calc = TeNNPotCalculator(model, energy_reference="cohesive")
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()

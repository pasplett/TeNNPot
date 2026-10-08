import json

import numpy as np
import torch
from ase import Atoms
from pytest import approx, raises

from tennpot.config import ModelConfig, build_model
from tennpot.properties import (
    PAPER_DIAMOND_LATTICE_ANGSTROM,
    MDInitialState,
    cohesive_energy_curves,
    load_md_initial_state_ensemble,
    phonon_band_structure,
    rmsd_trajectories,
    root_mean_square_displacement,
    run_md_ensemble,
    run_md_trajectory,
    silicon_crystal,
    write_property_table,
)


class _ZeroForceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("marker", torch.nn.Parameter(torch.tensor(0.0)))
        self.temperatures = []

    def forward(self, positions, el_temp, **kwargs):
        self.temperatures.append(float(el_temp.item()))
        return positions.sum(dim=(1, 2))[:, None] * 0, torch.zeros_like(positions)


def test_paper_diamond_lattice_matches_archived_figure3_cell():
    bohr_to_angstrom = 0.529177210903
    assert PAPER_DIAMOND_LATTICE_ANGSTROM == approx(30.6063 / 3 * bohr_to_angstrom)
    assert PAPER_DIAMOND_LATTICE_ANGSTROM == approx(40.8084 / 4 * bohr_to_angstrom)


def _periodic_model():
    config = ModelConfig(
        schema_version=1,
        name="properties test",
        radial_cutoff_radii=(2.5,),
        angular_cutoff_radii=(2.5,),
        angular_lambda=(1.0,),
        angular_zeta=(1.0,),
        scale_factors=(1.0, 1.0),
        periodic=(True, True, True),
        cell=(10.0, 10.0, 10.0),
        hidden_nodes=(2,),
        activations=("tanh",),
    )
    torch.manual_seed(4)
    return build_model(config)


def test_silicon_crystal_builds_conventional_periodic_cells():
    expected_atoms = {"diamond": 8, "fcc": 4, "bcc": 2, "sc": 1}
    for structure, atom_count in expected_atoms.items():
        atoms = silicon_crystal(structure, 5.0)
        assert len(atoms) == atom_count
        assert atoms.pbc.all()
    with raises(ValueError, match="unknown silicon structure"):
        silicon_crystal("hcp", 5.0)


def test_cohesive_energy_curves_have_one_value_per_lattice_constant():
    curves = cohesive_energy_curves(
        _periodic_model(),
        1000,
        [4.0, 4.5],
        structures=("diamond", "sc"),
    )
    assert set(curves) == {"diamond", "sc"}
    assert all(values.shape == (2,) for values in curves.values())
    assert all(np.isfinite(values).all() for values in curves.values())


def test_write_property_table_validates_columns(tmp_path):
    output = tmp_path / "values.csv"
    write_property_table(output, {"x": [1, 2], "y": [3, 4]})
    assert output.read_text().startswith("x,y")
    with raises(ValueError, match="same length"):
        write_property_table(output, {"x": [1], "y": [2, 3]})


def test_phonon_calculation_writes_six_diamond_modes(tmp_path):
    bands = phonon_band_structure(
        _periodic_model(),
        1000,
        tmp_path / "phonons",
        supercell=(1, 1, 1),
        path="GX",
        npoints=4,
    )
    assert bands.frequencies_thz.shape == (4, 6)
    assert bands.path_coordinate.shape == (4,)
    assert (tmp_path / "phonons" / "phonon_bands.csv").is_file()


def test_root_mean_square_displacement_and_short_md():
    reference = np.zeros((2, 3))
    positions = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    assert root_mean_square_displacement(positions, reference) == 1.0
    with raises(ValueError, match="shape"):
        root_mean_square_displacement([1, 2], [1, 2])

    result = rmsd_trajectories(
        _periodic_model(),
        1000,
        repetitions=(1, 1, 1),
        steps=1,
        runs=1,
    )
    assert result.time_fs.tolist() == [0.0, 2.0]
    assert result.individual_angstrom.shape == (1, 2)
    assert result.mean_angstrom[0] == 0
    assert result.standard_error_angstrom.tolist() == [0.0, 0.0]


def test_ase_md_uses_provided_state_and_reference():
    velocity = np.array([[0.01, 0, 0], [0, 0.02, 0]])
    atoms = Atoms("Si2", positions=np.zeros((2, 3)), cell=[10, 10, 10], pbc=True)
    atoms.set_velocities(velocity)
    trajectory = run_md_trajectory(
        _ZeroForceModel(), MDInitialState(atoms, np.zeros((2, 3))), 12631, steps=2
    )
    expected = root_mean_square_displacement(
        velocity * 2 * 0.09822694788464063, np.zeros((2, 3))
    )
    assert trajectory == approx([0.0, expected, 2 * expected])


def test_prepared_md_ensemble_loads_and_runs_with_ase(tmp_path):
    ensemble_dir = tmp_path / "ensemble"
    positions = np.arange(6, dtype=float).reshape(2, 3) / 10
    write_property_table(
        ensemble_dir / "reference_positions.csv",
        {
            f"position_angstrom_{axis}": positions[:, i] - 0.1
            for i, axis in enumerate("xyz")
        },
    )
    write_property_table(
        ensemble_dir / "run_01.csv",
        {
            **{
                f"position_angstrom_{axis}": positions[:, i]
                for i, axis in enumerate("xyz")
            },
            **{
                f"velocity_angstrom_per_ase_time_{axis}": np.full(2, 0.01)
                for axis in "xyz"
            },
        },
    )
    (ensemble_dir / "metadata.json").write_text(
        json.dumps(
            {
                "electronic_temperature_kelvin": 12631,
                "timestep_fs": 2.0,
                "cell_angstrom": np.diag([10.0, 10.0, 10.0]).tolist(),
                "pbc": [True] * 3,
                "reference_positions_file": "reference_positions.csv",
                "run_files": ["run_01.csv"],
            }
        )
    )
    ensemble = load_md_initial_state_ensemble(ensemble_dir)
    assert ensemble.electronic_temperature_kelvin == 12631
    result = run_md_ensemble(_ZeroForceModel(), ensemble.states, 12631, steps=1)
    assert result.individual_angstrom.shape == (1, 2)
    assert result.mean_angstrom[0] == approx(np.sqrt(0.03))

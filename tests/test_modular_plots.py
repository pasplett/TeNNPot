import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from ase import Atoms

from tennpot.cli import plot_properties
from tennpot.figure_plots import (
    plot_cohesive,
    plot_heat_capacity,
    plot_internal_energy,
    plot_phonons,
    plot_rmsd,
    resolve_reference,
)
from tennpot.physics import millihartree_to_kelvin
from tennpot.properties import PhononBands, RMSDTrajectories, rmsd_trajectories

matplotlib.use("Agg")


class ZeroForceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, positions, temperature, **kwargs):
        return positions.sum(dim=(1, 2))[:, None] * 0, positions * 0


def test_custom_structure_md_uses_input_velocities_and_keeps_input_unchanged():
    atoms = Atoms("Si2", positions=[[0, 0, 0], [2, 0, 0]], cell=[10, 10, 10], pbc=True)
    atoms.set_velocities([[0.01, 0, 0], [-0.01, 0, 0]])
    original = atoms.positions.copy()
    result = rmsd_trajectories(ZeroForceModel(), 12631, atoms=atoms, steps=2, runs=2)
    assert result.individual_angstrom.shape == (2, 3)
    assert result.mean_angstrom[0] == 0
    assert result.mean_angstrom[-1] > 0
    assert np.allclose(result.individual_angstrom[0], result.individual_angstrom[1])
    assert np.array_equal(atoms.positions, original)


def test_auto_reference_rejects_unavailable_temperature(tmp_path):
    (tmp_path / "cohesive").mkdir()
    with pytest.raises(FileNotFoundError, match="no unique cohesive"):
        resolve_reference(
            "cohesive",
            True,
            reference_dir=tmp_path,
            electronic_temperature_kelvin=20000,
        )
    with pytest.raises(FileNotFoundError, match="no unique rmsd"):
        resolve_reference(
            "rmsd",
            True,
            reference_dir=tmp_path,
            electronic_temperature_kelvin=20000,
        )


def test_auto_reference_requires_a_prepared_database(tmp_path):
    with pytest.raises(FileNotFoundError, match="Supply a prepared"):
        resolve_reference(
            "cohesive",
            True,
            reference_dir=tmp_path / "absent",
            electronic_temperature_kelvin=12631,
        )


def test_auto_reference_reads_prepared_database_without_writing(tmp_path):
    from tennpot.properties import write_property_table

    reference = tmp_path / "cohesive" / "40mHa.csv"
    write_property_table(
        reference,
        {
            "diamond_lattice_constant_angstrom": [4.0, 5.0],
            "diamond_energy_ev_per_atom": [-1.0, -2.0],
        },
    )
    reference = resolve_reference(
        "cohesive",
        True,
        reference_dir=tmp_path,
        electronic_temperature_kelvin=12631,
    )
    assert reference.is_file()
    assert list(tmp_path.rglob("*.csv")) == [reference]
    table = np.genfromtxt(reference, delimiter=",", names=True)
    assert np.allclose(table["diamond_energy_ev_per_atom"], [-1.0, -2.0])


def test_reference_lookup_distinguishes_temperature_and_md_cell(tmp_path):
    from tennpot.properties import write_property_table

    for temperature in (40, 50):
        write_property_table(
            tmp_path / "phonon_bands" / f"{temperature}mHa.csv",
            {"path_coordinate": [0.0, 1.0], "mode_1_thz": [0.0, 1.0]},
        )
    for atom_count in (288, 640):
        write_property_table(
            tmp_path / "rmsd" / f"N{atom_count}_50mHa.csv",
            {
                "time_fs": [0.0, 2.0],
                "mean_angstrom": [0.0, 0.1],
                "standard_error_angstrom": [0.0, 0.01],
            },
        )
    temperature = millihartree_to_kelvin(50)
    selected = resolve_reference(
        "phonon_bands",
        True,
        reference_dir=tmp_path,
        electronic_temperature_kelvin=temperature,
    )
    assert selected.name == "50mHa.csv"
    selected = resolve_reference(
        "rmsd",
        True,
        reference_dir=tmp_path,
        electronic_temperature_kelvin=temperature,
        atom_count=640,
    )
    assert selected.name == "N640_50mHa.csv"
    with pytest.raises(FileNotFoundError, match="no unique rmsd"):
        resolve_reference(
            "rmsd",
            True,
            reference_dir=tmp_path,
            electronic_temperature_kelvin=temperature,
        )


def test_standalone_cohesive_plot_uses_selected_structure_and_reference(
    tmp_path, monkeypatch
):
    from tennpot.properties import write_property_table

    reference = tmp_path / "reference.csv"
    write_property_table(
        reference,
        {
            "diamond_lattice_constant_angstrom": [4.0, 5.0],
            "diamond_energy_ev_per_atom": [-1.0, -1.5],
        },
    )
    observed = {}

    def fake_curves(model, temperature, lattice, *, structures, device):
        observed.update(temperature=temperature, structures=structures)
        return {"diamond": np.array([-0.5, -0.6])}

    monkeypatch.setattr(
        plot_properties, "_load_selected_model", lambda *args: (object(), "model")
    )
    monkeypatch.setattr(plot_properties, "cohesive_energy_curves", fake_curves)
    output = tmp_path / "output"
    plot_properties.cohesive_main(
        [
            "--output",
            str(output),
            "--reference",
            str(reference),
            "--temperature",
            "12631",
            "--structures",
            "diamond",
            "--lattice-min",
            "4",
            "--lattice-max",
            "5",
            "--points",
            "2",
        ]
    )
    assert observed == {"temperature": 12631.0, "structures": ["diamond"]}
    assert (output / "cohesive.pdf").is_file()
    assert (output / "cohesive.png").is_file()
    assert (output / "cohesive.csv").is_file()


def test_md_uses_prepared_ensemble_and_matching_reference(tmp_path, monkeypatch):
    from tennpot.properties import write_property_table

    database = tmp_path / "ab_initio_reference"
    write_property_table(
        database / "rmsd/N640_40mHa.csv",
        {
            "time_fs": [0, 2],
            "mean_angstrom": [0.1, 0.2],
            "standard_error_angstrom": [0.01, 0.01],
        },
    )
    observed = {}

    def fake_trajectories(model, states, temperature, **kwargs):
        observed.update(temperature=temperature, states=states, kwargs=kwargs)
        return RMSDTrajectories(
            np.array([0.0, 2.0]),
            np.array([0.1, 0.2]),
            np.array([0.01, 0.01]),
            np.array([[0.1, 0.2]]),
        )

    monkeypatch.setattr(
        plot_properties, "_load_selected_model", lambda *args: (object(), "model")
    )
    from tennpot.properties import MDInitialState, MDInitialStateEnsemble

    ensemble = MDInitialStateEnsemble(
        (
            MDInitialState(
                Atoms("Si640", positions=np.zeros((640, 3)), cell=[5, 5, 5], pbc=True)
            ),
        ),
        12631,
        2.0,
    )
    monkeypatch.setattr(
        plot_properties,
        "load_md_initial_state_ensemble",
        lambda *args, **kwargs: ensemble,
    )
    monkeypatch.setattr(plot_properties, "run_md_ensemble", fake_trajectories)
    output = tmp_path / "output"
    plot_properties.tennpot_md_main(
        [
            "--output",
            str(output),
            "--reference-db-dir",
            str(database),
            "--ab-initio-reference",
            "--initial-state-ensemble",
            str(database / "rmsd/N640_40mHa"),
            "--runs",
            "1",
            "--steps",
            "1",
        ]
    )
    assert observed["temperature"] == 12631
    assert observed["states"] is ensemble.states
    assert (output / "md.pdf").is_file()
    assert (output / "md.csv").is_file()


def test_all_panel_plotters_accept_explicit_reference_csvs(tmp_path):
    from tennpot.properties import write_property_table

    reference = tmp_path / "reference"
    write_property_table(
        reference / "cohesive.csv",
        {
            "diamond_lattice_constant_angstrom": [4.0, 5.0],
            "diamond_energy_ev_per_atom": [-1.0, -1.5],
        },
    )
    write_property_table(
        reference / "phonon_bands.csv",
        {"path_coordinate": [0.0, 1.0], "mode_1_thz": [0.0, 2.0]},
    )
    write_property_table(
        reference / "thermodynamics.csv",
        {
            "temperature_kelvin": [1000.0, 2000.0],
            "internal_energy_ev": [0.0, 0.1],
            "heat_capacity_ev_per_kelvin": [0.0, 1e-4],
        },
    )
    write_property_table(
        reference / "rmsd_12631.csv",
        {
            "time_fs": [0.0, 2.0],
            "mean_angstrom": [0.1, 0.2],
            "standard_error_angstrom": [0.01, 0.02],
        },
    )
    figure, axes = plt.subplots(5, 1)
    plot_cohesive(
        axes[0],
        {
            "lattice_constant_angstrom": np.array([4.0, 5.0]),
            "diamond": np.array([-0.5, -0.6]),
        },
        ("diamond",),
        reference / "cohesive.csv",
    )
    bands = PhononBands(
        np.array([0.0, 1.0]),
        np.array([[0.0], [1.0]]),
        np.array([0.0, 1.0]),
        ["G", "X"],
    )
    plot_phonons(axes[1], bands, reference / "phonon_bands.csv")
    thermo = {
        "temperature_kelvin": np.array([1000.0, 2000.0]),
        "internal_energy_ev": np.array([1.0, 1.2]),
        "heat_capacity_ev_per_kelvin": np.array([0.0, 1e-4]),
    }
    plot_internal_energy(axes[2], thermo, reference / "thermodynamics.csv")
    plot_heat_capacity(axes[3], thermo, reference / "thermodynamics.csv")
    trajectory = RMSDTrajectories(
        np.array([0.0, 2.0]),
        np.array([0.0, 0.1]),
        np.array([0.0, 0.01]),
        np.array([[0.0, 0.1]]),
    )
    plot_rmsd(axes[4], trajectory, reference / "rmsd_12631.csv")
    assert [len(axis.lines) for axis in axes] == [3, 2, 2, 2, 2]
    plt.close(figure)


@pytest.mark.parametrize(
    "property_name", ["phonons", "internal-energy", "specific-heat", "tennpot-md"]
)
def test_standalone_commands_write_results(tmp_path, monkeypatch, property_name):
    from ase.io import write

    monkeypatch.setattr(
        plot_properties, "_load_selected_model", lambda *args: (object(), "model")
    )
    bands = PhononBands(
        np.array([0.0, 1.0]),
        np.array([[0.0], [1.0]]),
        np.array([0.0, 1.0]),
        ["G", "X"],
    )
    monkeypatch.setattr(
        plot_properties, "phonon_band_structure", lambda *args, **kwargs: bands
    )

    class Thermo:
        temperature_kelvin = torch.tensor([1000.0, 2000.0])
        free_energy_ev = torch.tensor([0.0, 0.1])
        internal_energy_ev = torch.tensor([0.0, 0.2])
        heat_capacity_ev_per_kelvin = torch.tensor([0.0, 1e-4])

    monkeypatch.setattr(
        plot_properties, "diamond_thermodynamics", lambda *args, **kwargs: Thermo()
    )
    trajectory = RMSDTrajectories(
        np.array([0.0, 2.0]),
        np.array([0.0, 0.1]),
        np.array([0.0, 0.01]),
        np.array([[0.0, 0.1]]),
    )
    monkeypatch.setattr(
        plot_properties, "rmsd_trajectories", lambda *args, **kwargs: trajectory
    )
    structure = tmp_path / "structure.xyz"
    write(structure, Atoms("Si", positions=[[0, 0, 0]]))
    output = tmp_path / property_name
    options = ["--temperature", "12631"]
    if property_name == "tennpot-md":
        options = [str(structure), *options]
    elif property_name in ("internal-energy", "specific-heat"):
        options = ["--points", "2"]
    command = getattr(plot_properties, property_name.replace("-", "_") + "_main")
    command(["--output", str(output), *options])
    stem = "md" if property_name == "tennpot-md" else property_name.replace("-", "_")
    assert (output / f"{stem}.csv").is_file()
    assert (output / f"{stem}.pdf").is_file()

from pathlib import Path

import numpy as np

from tennpot.cli import reproduce_fig3
from tennpot.cli.reproduce_fig3 import (
    PAPER_COHESIVE_XLIM_NM,
    PAPER_COHESIVE_YLIM_EV,
    PAPER_MD_XLIM_PS,
    PAPER_MD_YLIM_PM,
    _load_selected_model,
    _plot_figure,
)
from tennpot.checkpoint import save_checkpoint
from tennpot.config import ModelConfig, build_model
from tennpot.properties import CRYSTAL_STRUCTURES, write_property_table


def test_paper_cohesive_curves_use_negative_reference_domain(monkeypatch):
    dtype = []
    for structure in CRYSTAL_STRUCTURES:
        dtype.extend(
            [
                (f"{structure}_lattice_constant_angstrom", float),
                (f"{structure}_energy_ev_per_atom", float),
            ]
        )
    reference = np.zeros(3, dtype=dtype)
    for structure in CRYSTAL_STRUCTURES:
        reference[f"{structure}_lattice_constant_angstrom"] = [2.0, 3.0, 14.0]
        reference[f"{structure}_energy_ev_per_atom"] = [1.0, -1.0, -0.1]

    evaluated = {}

    def fake_curves(model, temperature, lattice_constants, *, structures, device):
        evaluated[structures[0]] = lattice_constants
        return {structures[0]: np.zeros_like(lattice_constants)}

    monkeypatch.setattr(reproduce_fig3, "cohesive_energy_curves", fake_curves)
    result = reproduce_fig3._paper_cohesive_curves(
        object(), reference, quick=False, device="cpu"
    )

    assert all(values.tolist() == [3.0] for values in evaluated.values())
    assert result["diamond_lattice_constant_angstrom"][0] == 3.0


def test_plot_figure_writes_pdf_and_png_with_paper_panel_a_ranges(
    tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt

    closed_figures = []
    monkeypatch.setattr(plt, "close", closed_figures.append)
    lattice_constants = np.array([4.0, 5.0])
    cohesive = {"lattice_constant_angstrom": lattice_constants}
    cohesive.update(
        {structure: np.array([-0.1, -0.2]) for structure in CRYSTAL_STRUCTURES}
    )
    thermodynamics = {
        "temperature_kelvin": np.array([1000.0, 2000.0]),
        "internal_energy_ev": np.array([0.0, 0.1]),
        "heat_capacity_ev_per_kelvin": np.array([0.0, 1.0e-4]),
    }
    reference_dir = tmp_path / "reference"
    write_property_table(
        reference_dir / "cohesive.csv",
        {
            name: (
                np.array([4.0, 5.0])
                if "lattice_constant" in name
                else np.array([-0.1, -0.2])
            )
            for structure in CRYSTAL_STRUCTURES
            for name in (
                f"{structure}_lattice_constant_angstrom",
                f"{structure}_energy_ev_per_atom",
            )
        },
    )
    write_property_table(
        reference_dir / "phonon_bands.csv",
        {
            "path_coordinate": np.array([0.0, 1.0]),
            "mode_1_thz": np.array([0.0, 1.0]),
        },
    )
    write_property_table(
        reference_dir / "thermodynamics.csv",
        {
            "temperature_kelvin": np.array([1000.0, 2000.0]),
            "internal_energy_ev": np.array([0.0, 0.08]),
            "heat_capacity_ev_per_kelvin": np.array([0.0, 0.8e-4]),
        },
    )
    for temperature in (12631, 22104):
        write_property_table(
            reference_dir / f"rmsd_{temperature}.csv",
            {
                "time_fs": np.array([0.0, 2.0]),
                "mean_angstrom": np.array([0.1, 0.2]),
                "standard_error_angstrom": np.array([0.01, 0.02]),
            },
        )

    _plot_figure(
        tmp_path,
        cohesive,
        phonons=None,
        thermodynamics=thermodynamics,
        trajectories={},
        reference_paths={
            "cohesive": reference_dir / "cohesive.csv",
            "phonon_bands": reference_dir / "phonon_bands.csv",
            "thermodynamics": reference_dir / "thermodynamics.csv",
            ("rmsd", 12631): reference_dir / "rmsd_12631.csv",
            ("rmsd", 22104): reference_dir / "rmsd_22104.csv",
        },
    )

    for filename in ("figure3.pdf", "figure3.png"):
        output = Path(tmp_path, filename)
        assert output.is_file()
        assert output.stat().st_size > 0

    axis = closed_figures[0].axes[0]
    assert axis.get_xlim() == PAPER_COHESIVE_XLIM_NM
    assert axis.get_ylim() == PAPER_COHESIVE_YLIM_EV
    assert [line.get_label() for line in axis.lines[:4]] == [
        "sc",
        "bcc",
        "fcc",
        "dia",
    ]
    assert [line.get_color() for line in axis.lines[:4]] == [
        "blue",
        "magenta",
        "limegreen",
        "orange",
    ]
    for axis, temperature in zip(closed_figures[0].axes[4:], (12631, 22104)):
        assert axis.get_xlim() == PAPER_MD_XLIM_PS
        assert axis.get_ylim() == PAPER_MD_YLIM_PM[temperature]
        assert axis.get_xlabel() == "time (ps)"
        assert axis.get_ylabel() == "RMSD (pm)"


def test_figure3_loader_uses_the_selected_checkpoint(tmp_path):
    config = ModelConfig(
        schema_version=1,
        name="Figure 3 test model",
        radial_cutoff_radii=(2.5,),
        angular_cutoff_radii=(2.5,),
        angular_lambda=(1.0,),
        angular_zeta=(1.0,),
        scale_factors=(1.0, 1.0),
        periodic=(False, False, False),
        cell=(10.0, 10.0, 10.0),
        hidden_nodes=(2,),
        activations=("tanh",),
    )
    model = build_model(config)
    checkpoint = tmp_path / "custom.pt"
    save_checkpoint(checkpoint, model, config)

    selected, cache_key = _load_selected_model(checkpoint, "cpu")

    assert selected.config.name == "Figure 3 test model"
    assert len(cache_key) == 12
    assert cache_key != "937494642d2f"

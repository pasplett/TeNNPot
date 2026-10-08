# Research workflows with external data

This guide is for working with the original CHIVES training archive and prepared
ab-initio reference calculations. These datasets and MD initial states are not
distributed with TeNNPot. The commands below require you to supply them separately.
Run commands from the repository root after installing the package.

## Local data layout

The examples use this layout:

```text
data/
  si_thin_film_chives.npz
  ab_initio_reference/
    cohesive/
    phonon_bands/
    thermodynamics/
    rmsd/
      N640_40mHa/
      N288_70mHa/
```

Dataset files under `data/` are ignored by Git. Keep them outside publication
copies. The [dataset reference](#dataset-reference) below describes the archive
schema, units, and prepared reference database layout.

## Training on the original CHIVES archive

The NPZ archive is loaded directly, retaining its original train/test partitions
and fixed cell metadata. It does not need conversion to Extended XYZ.

```bash
tennpot-train data/si_thin_film_chives.npz --output runs/si_thin_film
```

For a small workflow check:

```bash
tennpot-train data/si_thin_film_chives.npz --model-config silicon_tiny --output results/silicon_tiny --device cpu --dtype float32 --epochs 3 --batch-size 1 --max-train-samples 32 --max-validation-samples 8
```

The sample caps select deterministic subsets of the original partitions.
`silicon_tiny` has three radial descriptors with cutoffs of 5.0, 5.5, and
6.0 bohr, no angular descriptors, and eight hidden nodes. This checks the
training workflow; it does not produce a useful scientific fit. Its best
checkpoint is written to `results/silicon_tiny/best.pt`.

## Overlaying ab-initio reference curves

Install the plotting dependencies:

```bash
python -m pip install ".[reproduction]"
```

Add `--ab-initio-reference` to a property command to read matching curves from
the prepared database at `data/ab_initio_reference/`. For example:

```bash
cohesive --output results/cohesive --temperature 12631 --structures diamond fcc --ab-initio-reference
```

Use `--reference-db-dir path/to/database` for another database location, or
`--reference path/to/reference.csv` to select a particular CSV directly.
Automatic phonon and thermodynamic references use the paper's diamond lattice
constant. When changing that lattice constant, supply a matching reference CSV.

## Reproducing the paper's MD panels

The prepared initial-state ensembles provide the cell, electronic temperature,
initial velocities, RMSD reference coordinates, and the 10 or 40 member runs
for the respective panels. The default trajectory length is 500 steps.

```bash
tennpot-md --ab-initio-reference --output results/silicon_2023/figure3e --initial-state-ensemble data/ab_initio_reference/rmsd/N640_40mHa
tennpot-md --ab-initio-reference --output results/silicon_2023/figure3f --initial-state-ensemble data/ab_initio_reference/rmsd/N288_70mHa
```

For a quick check, add `--runs 1 --steps 2`. Omit `--ab-initio-reference`
to run the ensemble without a reference overlay. The initial-state ensemble
is still required.

MD checkpoints are saved separately for each ensemble member under
`<output>/md_cache/`. Repeat the same command to continue from the last saved
timestep and reuse completed members. `--steps` sets the total trajectory length;
increasing it extends saved trajectories. Increasing `--runs` adds members.
Checkpoints are saved every 10 completed steps and at completion; adjust this
with `--checkpoint-interval`. An interruption can lose progress since the last
checkpoint, which is recomputed on restart.

Matching requires the same model, initial conditions, RMSD reference, temperature,
timestep, software versions and compute settings. Changed settings use separate
checkpoints. Use a new output directory for a fresh run, or `--no-resume` to run
without reading or writing MD checkpoints. Checkpointing supports unconstrained
atoms and ASE `FixAtoms`/`FixCartesian` constraints. Other constraints require
`--no-resume`.

## Combined Figure 3 layout

```bash
tennpot-figure3 --checkpoint runs/my-model/best.pt --ab-initio-reference --output results/figure3
```

Omit `--checkpoint` to use the bundled model. Omit `--ab-initio-reference`
to draw model curves without reference overlays. Without `--md-structure`,
the script still requires the prepared MD initial states under
`data/ab_initio_reference/rmsd/`.

Use `--reference-db-dir` for another prepared database location.
`--cohesive-temperature` and `--phonon-temperature` change panels a/b;
with reference overlays, these temperatures must match available reference CSVs.
To use your own MD structure and temperatures, supply both `--md-structure`
(an ASE-readable file) and `--md-temperatures`.

Phonon force calculations and MD trajectories are reused on repeated invocations.
The combined command uses the same MD restart mechanism as `tennpot-md`, with
`--md-steps` setting the total target and `--md-checkpoint-interval` controlling
how often state is saved. `--no-resume` disables MD checkpointing. `--quick`,
`--skip-phonons`, and `--skip-md` are intended for development checks.

## Dataset reference

### Prepared reference database

Automatic reference overlays use the prepared CSV database, rather than the
original calculation source files. The expected files are:

- `cohesive/<temperature>mHa.csv` for diamond, fcc, bcc, and sc at
  1, 30, 40, 50, 58, 60, and 70 mHa;
- `phonon_bands/<temperature>mHa.csv` at the same temperatures;
- `thermodynamics/diamond.csv`;
- `rmsd/N<atom-count>_<temperature>mHa.csv` for N640 at 40/50 mHa
  and N288 at 60/70 mHa.

The MD initial-state ensembles contain `rmsd/N640_40mHa/run_01.csv` through
`run_10.csv`, and `rmsd/N288_70mHa/run_01.csv` through `run_40.csv`. Each
ensemble has a `metadata.json` recording the cell, PBC, electronic temperature,
timestep, shared RMSD reference, and member runs. Run files contain positions
and ASE-compatible velocities; their CSV headers specify the units.

Other archived structure calculations are not included in automatic lookup
because their coordinate conversions have not been established for these workflows.

### CHIVES physical system

The archive contains Te-dependent DFT trajectories of laser-excited silicon thin
films and artificial compression and expansion configurations.

- Material: silicon, diamond-structure thin film.
- Number of atoms per sample: 320 Si atoms.
- Thin-film construction: a `2 x 2 x 10` conventional-cell supercell with vacuum
  inserted along `z`; film thickness: 5.3 nm.
- Boundary conditions: Periodic.
- Reference method: CHIVES, a Te-dependent DFT code for laser-excited solids.

### Units

The dataset is stored in atomic units.

| Quantity | Array(s) | Stored unit | Notes |
| --- | --- | --- | --- |
| Atomic positions | `positions_train`, `positions_test` | Bohr | Shape is `(n_samples, 320, 3)` with Cartesian `x, y, z` coordinates. |
| Electronic temperature | `el_temp_train`, `el_temp_test` | `0.1 Ha = 100 mHa` | Stored value is `Te / 100 mHa`. Thus `0.01 = 1 mHa`, `0.5 = 50 mHa`, `1.0 = 100 mHa`. |
| Energy | `energy_train`, `energy_test` | Ha | Total Helmholtz free cohesive energy of the 320-atom structure. |
| Atomic forces | `forces_train`, `forces_test` | Ha / Bohr | Forces are the negative gradient of the energy with respect to positions. |
| Timestep/index | `timestep_train`, `timestep_test` | integer index | For MD and expansion data, timestep index `t` corresponds to time `2 fs * t`. For compression data, it is the compression-configuration index `0..24`. |
| Simulation type | `sim_type_train`, `sim_type_test` | categorical code | Stored as float64 values `0.0`, `1.0`, `2.0`; see below. |

Useful conversions:

- `Te_mHa = 100 * el_temp`
- `Te_Ha = 0.1 * el_temp`
- `1 mHa` corresponds to about `316 K` in the thesis examples.

### Array schema

All main arrays are little-endian NumPy arrays. Positions, energies, forces,
electronic temperatures, and simulation types are `float64`; timesteps are
`int32`. Each quantity has a `_train` and a `_test` array. In the table below,
`n_samples` is the size of the corresponding partition.

| Keys | Shape | Meaning |
| --- | --- | --- |
| `positions_train`, `positions_test` | `(n_samples, 320, 3)` | Atomic coordinates. |
| `energy_train`, `energy_test` | `(n_samples, 1)` | Cohesive Helmholtz free energy of each complete structure. |
| `forces_train`, `forces_test` | `(n_samples, 320, 3)` | Force vector for each atom. |
| `el_temp_train`, `el_temp_test` | `(n_samples, 1)` | Encoded electronic temperature. |
| `sim_type_train`, `sim_type_test` | `(n_samples, 1)` | Sample origin code. |
| `timestep_train`, `timestep_test` | `(n_samples, 1)` | Timestep or configuration index. |

### Split and temperature grid

The split is by electronic temperature:

- Training temperatures: `1, 10, 20, ..., 100 mHa`
- Test temperatures: `5, 15, 25, ..., 95 mHa`

The stored `el_temp` values are therefore:

- Training: `0.01, 0.10, 0.20, ..., 1.00`
- Test: `0.05, 0.15, 0.25, ..., 0.95`

### Simulation-type codes

The `sim_type_*` arrays encode the three dataset subsets described in the thesis.

| Code | Meaning | `timestep` interpretation |
| ---: | --- | --- |
| `0` | MD simulation of the laser-excited Si thin film after suddenly raising `Te` to a constant value. | Time index. Most trajectories have `0..500`, corresponding to up to `1000 fs = 1 ps` at `2 fs` per step. |
| `1` | Stepwise compression of the thin film initialized at `1 mHa`. | Compression index `0..24`; 25 configurations per electronic temperature. |
| `2` | Simulated mechanical expansion of the thin film. | Time/configuration index `0..500`; generated only for lower `Te` because the film expands naturally at higher `Te`. |

### Loading the archive

```python
import numpy as np

data = np.load("data/si_thin_film_chives.npz")

positions = data["positions_train"]      # Bohr, shape (n_train, 320, 3)
te = data["el_temp_train"]               # Te / 100 mHa, shape (n_train, 1)
energy = data["energy_train"]            # Ha, shape (n_train, 1)
forces = data["forces_train"]            # Ha/Bohr, shape (n_train, 320, 3)
sim_type = data["sim_type_train"].astype(int)
timestep = data["timestep_train"]

te_mha = 100.0 * te
time_fs = 2.0 * timestep                 # valid for sim_type 0 and 2
```

# TeNNPot silicon 2023 model card

## Model

This repository distributes the trained TeNNPot model accompanying
Plettenberg, Bernd Bauerhenne, and Garcia, *Neural network interatomic potential for
laser-excited materials*, Communications Materials **4**, 63 (2023),
https://doi.org/10.1038/s43246-023-00389-w.

The version-controlled checkpoint is
`src/tennpot/saved_models/silicon_2023.pt`. It contains a PyTorch state
dictionary and reviewed model configuration and is loaded with PyTorch's
restricted `weights_only` loader. Its identity and conversion provenance are
recorded in the adjacent JSON manifest.

## Intended use

The checkpoint predicts the electronic-temperature-dependent cohesive
Helmholtz free energy and atomic forces of elemental silicon. It can be used
through the low-level PyTorch API or the ASE calculator included in TeNNPot.

The model is intended for silicon structures reasonably represented by the
training environments and electronic temperatures between 1 and 100 mHa
(approximately 316 to 31,578 K). Results outside that domain are extrapolations
and do not have a validated accuracy guarantee.

## Limitations

- The checkpoint supports silicon only.
- It is a single-element model and cannot distinguish chemical species.
- It is not an uncertainty model and does not detect all extrapolative
  structures.
- The training data consist of finite-temperature DFT configurations of the
  silicon thin-film system described in the paper.

## Units and energy reference

The neural network uses positions in bohr, cohesive energies in hartree,
forces in hartree per bohr, and electronic temperature in units of 0.1 hartree.
The ASE calculator converts to angstrom, electronvolt, electronvolt per
angstrom, and kelvin. Total free energy is obtained by adding the documented
isolated-atom free-energy reference to the cohesive network output.

The ASE calculator returns total free energy by default. Set
`energy_reference="cohesive"` to return the cohesive free energy instead.

## Precision

The original TensorFlow training used float32. The converted parameters are
stored losslessly in the PyTorch checkpoint and the public loader currently
uses float64 by default for stable force and property evaluation. Explicit
float32 inference remains supported.

## License and citation

The model checkpoint is distributed as part of this repository under the MIT
license. Cite the paper above when using the model in scientific work.

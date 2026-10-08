# Model regression fixture

`test_tennpot.npz` uses `positions_train[42]` and `el_temp_train[42]` from the
local silicon thin-film archive. This single example sample is approved for
redistribution with the code under the repository MIT license. It has 320 Si
atoms; the full dataset is not distributed. The cell
and periodic boundaries come from the bundled `silicon_2023` configuration.

Descriptors, descriptor gradients, cohesive energy, and forces in this file
were computed with the bundled `silicon_2023.pt` checkpoint in float64. The
archive's DFT energy and force arrays were not read or copied. The fixture
records the checkpoint SHA-256 hash to identify its numerical source.

Maintainers with the private archive can regenerate it with
`python tests/generate_model_fixture.py`. The regression test protects the
published model's current numerical behavior; it is not independent DFT
validation.

============
Contributing
============

Issues and pull requests are welcome at
https://github.com/pasplett/TeNNPot. Bug reports should include the operating
system, Python and PyTorch versions, a minimal reproducer, and the full error
message. Scientific changes should also state the units and expected physical
invariant or reference result.

Development setup
=================

Create an isolated environment and install all development dependencies::

    python -m pip install -e ".[testing,mlflow,reproduction]"
    python -m pip install pre-commit black
    pre-commit install

Before opening a pull request, run::

    black --check src tests
    flake8 src tests
    pytest

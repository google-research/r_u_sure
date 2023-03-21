# R-U-SURE? Uncertainty-Aware Code Suggestions By Maximizing Utility Across Random User Intents

[![Unittests](https://github.com/google-research/r_u_sure/actions/workflows/unittests.yml/badge.svg)](https://github.com/google-research/r_u_sure/actions/workflows/unittests.yml)

This is the repository accompanying the paper
["R-U-SURE? Uncertainty-Aware Code Suggestions By Maximizing Utility Across Random User Intents"][rusure].

[rusure]: https://arxiv.org/abs/2303.00732

If you use the code released through this repository, please cite the following paper:

```
@article{johnson2023rusure,
title     = {{R-U-SURE?} Uncertainty-Aware Code Suggestions By Maximizing
             Utility Across Random User Intents},
author    = {Daniel D. Johnson and
             Daniel Tarlow and
             Christian Walder},
journal   = {arXiv preprint arXiv:2303.00732},
year      = {2023},
}
```


---

## Demo

If you would like to try out the R-U-SURE system, you can open our demo notebook
in Google Colab:

[![Open R-U-SURE Demo In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook_demo]

[notebook_demo]: https://colab.research.google.com/github/google-research/r_u_sure/blob/main/r_u_sure/notebooks/R_U_SURE_Demo.ipynb

You might also be interested in the
[intro][notebook-udd-intro] and [details][notebook-udd-details]
notebooks for our utility function decision diagram representation.

[notebook-udd-intro]: https://colab.research.google.com/github/google-research/r_u_sure/blob/main/r_u_sure/notebooks/Utility_decision_diagrams_intro.ipynb
[notebook-udd-details]: https://colab.research.google.com/github/google-research/r_u_sure/blob/main/r_u_sure/notebooks/Utility_decision_diagrams_details.ipynb

## Installation

If you would like to install the R-U-SURE library on your own system, you can
follow the instructions below.

### Prerequisite: Setting up a virtual environment

It is highly recommended to install this package into a virtual environment,
as it currently depends on a patched version of `numba` that may be incompatible
with a global installation.

To create and activate a virtual environment, you can run the Bash commands

```
# you can use any path here
venv_path="$HOME/venvs/rusure"
python3 -m venv $venv_path
source $venv_path/bin/activate
echo "Active virtual environment is: $VIRTUAL_ENV"
```

(On Linux, you may need to run `sudo apt-get install python3-venv` first.)

### Installing the package directly from GitHub

If you want to use the `r_u_sure` package from Python without modifying it, you
can directly install it from GitHub using `pip`:

```
# Optional: disable some unused numba features to prevent build errors
export NUMBA_DISABLE_TBB=1
export NUMBA_DISABLE_OPENMP=1

pip install "r_u_sure @ git+https://github.com/google-research/r_u_sure"
```

`pip` will then automatically install the most recent version of the package
and make it available from Python via `import r_u_sure`.

Note that you can also add `r_u_sure @ git+https://github.com/google-research/r_u_sure`
to your `requirements.txt` or `pyproject.toml` files if you are developing a
package that depends on R-U-SURE.

### Installing from source

If you prefer to download the R-U-SURE source files manually, or if you would
like to contribute to the R-U-SURE library, you can perform a local installation.
Start by cloning this GitHub repository:

```
git clone https://github.com/google-research/r_u_sure
cd r_u_sure
```

Next, install it:

```
# Optional: disable some unused numba features to prevent build errors
export NUMBA_DISABLE_TBB=1
export NUMBA_DISABLE_OPENMP=1

pip install -e .
```

Local edits to the source files will now be reflected properly in the python
interpreter.

(If you'd prefer, you can also omit the `export NUMBA_DISABLE_{X}=1` lines to
compile those features into numba. Those features have additional dependencies;
see the [Numba documentation][numba-opt-deps].)

[numba-opt-deps]: https://numba.readthedocs.io/en/stable/user/installing.html#build-time-environment-variables-and-configuration-of-optional-components

### Running tests

To run the R-U-SURE tests, you can use the command

```
python -m r_u_sure.testing.run_tests
```

Note that some tests require jit-compiling large programs, which can take a few
minutes. To run a faster subset of the tests, you can instead run

```
python -m r_u_sure.testing.run_tests --skip_jit_tests
```


---

*This is not an officially supported Google product.*

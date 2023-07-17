![Masskit logo](src/masskit/docs/_static/img/masskit_logo.png)

--------------------------------------------------------------------------------

Masskit is a Python package for easy-to-use, efficient and flexible computing on mass spectra.
Masskit does this by taking advantage of modern software and hardware architectures.  It features:

- Functions to filter, normalize, annotate, plot and compute on mass spectra.
- Searching and indexing of spectral libraries.
- Reading and writing of multiple file formats.
- Computational algorithms for small molecules, including reactions.
- Algorithms and encodings for peptides.
- Use of columnar memory for optimal efficiency in distributed, multicore computation.

<!-- toc -->
## Table of Contents

- [Installation](#installation)
- [API Documentation](https://pages.nist.gov/masskit)
- [The Team](https://chemdata.nist.gov/)
- [License](LICENSE.md)

<!-- tocstop -->

## Installation

### Requirements

- Masskit is installed using the package managers [`conda`](https://conda.io/) and [`mamba`](https://mamba.readthedocs.io/).
If you do not have either installed we recommend installing [mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

### Windows

- If you intend to use Masskit AI, please use the [Masskit AI instructions](https://github.com/usnistgov/masskit_ai#installation) instructions instead.
- Download the latest Masskit Windows zip file from the
[Releases](https://github.com/usnistgov/masskit/releases) page.
- Extract the contents of the file to a directory on your computer by navigating explorer to the
Downloads folder, right clicking on the zip file, and selecting `Extract all...`.
- Run `Miniforge Prompt` or `Anaconda Prompt` from the Start menu depending on whether you
installed `mambaforge` or a `conda` distribution.
- In the prompt window, use `cd` to change to the directory starting with `masskit` in
the directory you extracted the downloads to.
- Run `call init_masskit.bat` to create the `masskit` package environment.
- Run `pip install masskit-1.1.0-py3-none-any.whl`.

Whenever using the programs in Masskit, please make sure you initialize the appropriate package environment:

- If you installed `mambaforge` run `Miniforge Prompt` from the Start menu otherwise run `Anaconda Prompt` from the Start menu.
- Run `conda activate masskit`.

### Linux or macOS

- Our code is downloaded via `git`. If `git` is not installed, installers can be found at the [Git website](https://git-scm.com/).
- Change to a directory that will hold the masskit directory.
- Run `git clone https://github.com/usnistgov/masskit.git`
- Run `cd masskit`.
- If you are going to use [Masskit AI](https://github.com/usnistgov/masskit_ai.git):
  - Run `source environments/init_masskit.sh -m` if you will be using GPUs.
  - Run `source environments/init_masskit.sh -c` if you will be using CPUs.
- If you are not using Masskit AI, run `source environments/init_masskit.sh`
- Run `pip install .`

Whenever using the programs in Masskit, please make sure you initialize the appropriate package environment:

- If you use `mamba` run `mamba activate masskit`
- If you use `conda` run `conda activate masskit`.

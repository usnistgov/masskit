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

Masskit is currently only supported in a Python conda environment. If you do not have one installed we highly recommend installing [mambaforge](https://github.com/conda-forge/miniforge#mambaforge)

On a Linux or macOS computer that has Conda and [Git](https://git-scm.com/) installed, run on the command line:

- change to a directory that will hold the masskit directory
- `git clone https://github.com/usnistgov/masskit.git`
- `cd masskit`
- if you are going to use [Masskit_ai](https://github.com/usnistgov/masskit_ai.git), run `source environments/init_masskit.sh -m`, otherwise run
`source environments/init_masskit.sh`
  - if you are going to use [Masskit_ai](https://github.com/usnistgov/masskit_ai.git) in cpu-only mode, run `source environments/init_masskit.sh -c` instead.
- run `pip install .`


On a Windows computer that has Conda installed, download the masskit windows zip file from the
[Releases](https://github.com/usnistgov/masskit/releases) page and extract the contents. Then run the
following from the Conda prompt:

- change to the directory that contains the extracted files.
- run `call init_masskit.bat`
- run `pip install masskit-1.1.0-py3-none-any.whl`

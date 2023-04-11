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

On a Linux, Windows or macOS computer that has [Anaconda](https://www.anaconda.com/) and [Git](https://git-scm.com/) installed, run on the command line:

- `git clone https://github.com/usnistgov/masskit.git`
- `cd masskit`
- if using Windows, run `bash`
- in the masskit directory, run `source environments/init_masskit.sh -m`.
  - if you wish to use a cpu-only version of the library, run `source environments/init_masskit.sh -c` instead.
- if using Windows, `control-D` to exit bash
- run `conda activate masskit_ai`
- run `pip install -e .`

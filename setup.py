from glob import glob
import os
import setuptools
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension
# import distutils.command.bdist_conda

# To setup and locally for development:
#    python setup.py build_ext --inplace
#    python setup.py develop

# To create and install conda bdist package:
#    python setup.py bdist_conda
#    conda install $CONDA_PREFIX/conda-bld/linux-64/massspec-1.0-py39_0.tar.bz2

import numpy as np
import pyarrow as pa
import pybind11

# Remove the annoying "-Wstrict-prototypes" compiler option, which isn't valid for C++.
# This is mostly cosmetic to remove a wrning from the build process
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")


ext_modules = [
    Pybind11Extension(
        "massspec_ext",
        sorted(glob("src/massspec_ext/*.cpp")),  # Sort source files for reproducibility
    ),
]


for ext in ext_modules:
    # The Numpy C headers are currently required
    ext.include_dirs.append(np.get_include())
    ext.include_dirs.append(pa.get_include())
    ext.include_dirs.append(pybind11.get_include())
    ext.include_dirs.append(os.getenv('CONDA_PREFIX') + "/include")
    # ext.include_dirs.append(os.getenv('CONDA_PREFIX') + "/include/rdkit")
    
    #if os.name == 'nt':  # windows
        # only for windows we link
    ext.libraries.extend(pa.get_libraries())
    # ext.libraries.append("RDKitDataStructs")
    ext.library_dirs.extend(pa.get_library_dirs())
    ext.library_dirs.append(os.getenv('CONDA_PREFIX') + "/lib")

    if os.name == 'posix':
        ext.extra_compile_args.append('-std=c++11')

    # Try uncommenting the following line on Linux
    # if you get weird linker errors or runtime crashes
    # ext.define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))


setup(
    name="massspec",
    version="1.0",
    author="Lewis Geer, Douglas Slotta",
    author_email="lewis.geer@nist.gov",
    description="Utilities for handling mass spec data",
    # tell setuptools to look for any packages under 'src'
    packages=find_packages('src'),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    package_dir={'':'src'},
    include_package_data=True,
    # package_data={'': ['*.pkl','*.sdf']},
    # add the extension modules
    ext_modules=ext_modules,
    # distclass=distutils.command.bdist_conda.CondaDistribution,
)

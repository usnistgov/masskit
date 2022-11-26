(return 0 2>/dev/null) && sourced=1 || sourced=0

if [ $sourced -ne 1 ]; then
    echo "Sorry, to work correctly, this file must be sourced"
    echo "and not executed in a subshell. As such:"
    echo
    echo "        source $0"
    echo
    exit 1
fi

ENVNAME=arrow
REMOVE=0
if [ "$1" == "-f" ]; then
    REMOVE=1
fi

umask 0007
for arg in "$@"; do shift; done  # Don't pass args to activate
source /opt/conda/bin/activate


if [ $REMOVE -eq 1 ]; then
    # forcibly remove any existing environment
    echo "Trying to remove existing $ENVNAME environment.";
    conda env remove --name $ENVNAME || true
    # conda remove --name $ENVNAME --all
fi

echo "Initializing the conda $ENVNAME environment."
if ! conda activate $ENVNAME; then
    echo "Creating conda $ENVNAME environment."
    conda create -y -n $ENVNAME
    conda activate $ENVNAME
    mamba install -y -c conda-forge \
          arrow-cpp=9.0.* \
          conda-build \
          cython \
          hydra-core \
          imageio \
          jsonpickle \
          jupyter \
          matplotlib \
          molvs \
          numba \
          numpy \
          pandas \
          pyarrow=9.0.* \
          pybind11 \
          pynndescent \
          pytest \
          python=3 \
          quaternion \
          rdkit=2021.09.4 \
          sqlalchemy \
          sqlparse

    echo "Dumping conda environment..."
    conda env export --name arrow > arrow_conda.yml
fi

# To install last known good env
# mamba env create -f arrow_conda.yml

# To check existing environments:
# conda info --envs

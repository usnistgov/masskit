(return 0 2>/dev/null) && sourced=1 || sourced=0

if [ $sourced -ne 1 ]; then
    echo "Sorry, to work correctly, this file must be sourced"
    echo "and not executed in a subshell. As such:"
    echo
    echo "        source $0"
    echo
    exit 1
fi

# name of environment to set up
ENVNAME=msml
# environment used to set up ENVNAME
# SETUP_ENVNAME=msml_setup
# remove previous versions of environments if set to 1
REMOVE=0
# name of the pytorch package
PYTORCH=pytorch
# name of the borchvision package
TORCHVISION=torchvision
# name of the cuda toolkit
CUDATOOLKIT=cudatoolkit\=11.3

while [[ $# -gt 0 ]]; do
  case $1 in
    -f)
      REMOVE=1
      shift # past argument
      ;;
    -c)
      # cpu only environment
      ENVNAME=msml_cpu
      PYTORCH=pytorch
      TORCHVISION=torchvision
      CUDATOOLKIT=cpuonly
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      return 1
      ;;
  esac
done

umask 0007
for arg in "$@"; do shift; done  # Don't pass args to activate
local_conda="${CONDA_PREFIX:-/opt/conda}"
source $local_conda/bin/activate

if [[ $CONDA_SHLVL != 1 ]]
then
    echo "must be in base environment or otherwise mamba will not work properly"
fi

if [ $REMOVE -eq 1 ]; then
    # forcibly remove any existing environment
    echo "Trying to remove existing $ENVNAME environment.";
    conda env remove --name $ENVNAME || true
    # conda env remove --name $SETUP_ENVNAME || true
    # conda remove --name $ENVNAME --all
fi

if ! command -v mamba &> /dev/null
then
    if ! conda install mamba -n base -c conda-forge; then
        echo  "if you don't have write access to the base enviroment, install your own copy of anaconda"
        return 2
    fi
fi

# echo "Initializing the conda $SETUP_ENVNAME environment"
# if ! conda activate $SETUP_ENVNAME; then
#     echo "Creating conda setup environment"
#     conda create -y -n $SETUP_ENVNAME
#     # mamba is installed into $ENVNAME since on some computers mamba cannot be installed into the base environment
#     conda install -y -n $SETUP_ENVNAME -c conda-forge mamba
# fi

echo "Initializing the conda $ENVNAME environment"
if ! conda activate $ENVNAME; then
    # conda activate $SETUP_ENVNAME
    echo "Creating conda environment"
    # Notes:
    # Please keep list alphabetical
    # sqlparse added as it is necessary for some functionality in mlflow-skinny
    conda create -y -n $ENVNAME
    mamba install -y -n $ENVNAME -c pytorch -c nvidia -c conda-forge \
          boto3 \
          conda-build \
          $CUDATOOLKIT \
          cython \
          hydra-core \
          imageio \
          imbalanced-learn \
          jsonpickle \
          jupyter \
          levenshtein \
          matplotlib \
          mlflow-skinny \
          molvs \
          numba \
          numpy \
          pandas \
          pyarrow=7 \
          pybind11 \
          pynndescent \
          pytest \
          python=3 \
          $PYTORCH \
          pytorch-lightning \
          quaternion \
          rdkit=2021.09.4 \
          scikit-learn \
          sqlalchemy \
          sqlparse \
          torchmetrics \
          $TORCHVISION 


    conda activate $ENVNAME
    # # Check for bayesian_torch
    # (python -c "import bayesian_torch" 2> /dev/null) && has_lib=1 || has_lib=0

    # if [ $has_lib -ne 1 ]; then
    #     pushd /tmp
    #     git clone https://github.com/IntelLabs/bayesian-torch.git
    #     cd bayesian-torch
    #     pip install .
    #     cd ..
    #     rm -rf bayesian-torch
    #     popd
    # fi
fi

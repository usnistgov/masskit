(return 0 2>/dev/null) && sourced=1 || sourced=0

if [ $sourced -ne 1 ]; then
    echo "Sorry, to work correctly, this file must be sourced"
    echo "and not executed in a subshell. As such:"
    echo
    echo "        source $0"
    echo
    exit 1
fi

# Get the script location
SCRIPT_FILE=`realpath ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}`
SCRIPT_DIR="$(dirname "${SCRIPT_FILE}")"

# default name of environment to set up
ENVNAME=masskit
# Allow customizable suffixes, e.g. branch specific CI/CD testing
SUFFIX=
# remove previous versions of environments if set to 1
REMOVE=0
# Enable machine learning packages if set to 1
USE_ML=0
# Only use the CPU for machine learning packages if set to 1
CPUONLY=0

while [[ $# -gt 0 ]]; do
  case $1 in
    -f | --force)
      REMOVE=1
      shift # remove argument
      ;;
    -m | --ml)
      # machine learning environment
      ENVNAME=masskit_ai
      USE_ML=1
      shift # remove argument
      ;;
    -s | --suffix)
      # Add suffix to environment
      shift # remove argument
      SUFFIX=$1
      shift # remove argument
      ;;
    -c | --cpu)
      # machine_learning, cpu only environment
      ENVNAME=masskit_ai_cpu
      USE_ML=1
      CPUONLY=1
      shift # remove argument
      ;;
    -* | --*)
      echo "Unknown option $1"
      return 1
      ;;
  esac
done

for arg in "$@"; do shift; done  # Don't pass args to activate
if [ -d "$HOME/mambaforge" ]; then
    local_dir=$HOME/mambaforge
elif [ -d "$HOME/miniconda3" ]; then
    local_dir=$HOME/miniconda3
else
    local_dir=$HOME/Anaconda3
fi
local_conda="${CONDA_PREFIX:-$local_dir}"
source $local_conda/bin/activate

if [[ $CONDA_SHLVL != 1 ]]
then
    echo "must be in base environment or otherwise mamba will not work properly"
    return 2
fi

if [[ ! -z "$SUFFIX" ]] ; then
  ENVNAME=${ENVNAME}_${SUFFIX}
fi

if [ $REMOVE -eq 1 ]; then
    # forcibly remove any existing environment
    echo "Trying to remove existing $ENVNAME environment.";
    conda env remove --name $ENVNAME || true
fi

if ! command -v mamba &> /dev/null
then
    if ! conda install -y mamba -n base -c conda-forge; then
        echo  "if you don't have write access to the base enviroment, please install your own copy of anaconda"
        return 2
    fi
fi

export MAMBA_NO_BANNER=1

# Note: Please keep lists alphabetical

# The gxx package is installed because the arrow-cpp package is built with 
# a newer version than both AL2022 and the rest of conda-forge 
# (11.3 at the time of this comment) Why? I dunno.
# gxx is not available on windows

BASE_PACKAGES="--file=${SCRIPT_DIR}/base_packages.txt"

ML_CHANNELS=
ML_PACKAGES=
if [ $USE_ML -eq 1 ]; then
  ML_CHANNELS="-c pytorch -c nvidia"
  CUDATOOLKIT=""
  if [ $CPUONLY -eq 1 ]; then
    CUDATOOLKIT="${SCRIPT_DIR}/nocuda_packages.txt"
  else
    CUDATOOLKIT="${SCRIPT_DIR}/cuda_packages.txt"
  fi

  ML_PACKAGES="--file=$CUDATOOLKIT --file=${SCRIPT_DIR}/ml_packages.txt"
fi

echo "Initializing the conda $ENVNAME environment"
if ! conda activate $ENVNAME; then
    # conda activate $SETUP_ENVNAME
    echo "Creating conda environment"
    mamba create -y -n $ENVNAME
    mamba install -y -n $ENVNAME -c nodefaults ${ML_CHANNELS} -c conda-forge ${BASE_PACKAGES} ${ML_PACKAGES}
    conda activate $ENVNAME
fi

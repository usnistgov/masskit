(return 0 2>/dev/null) && sourced=1 || sourced=0

if [ $sourced -ne 1 ]; then
    echo "Sorry, to work correctly, this file must be sourced"
    echo "and not executed in a subshell. As such:"
    echo
    echo "        source $0"
    echo
    exit 1
fi
echo "Initializing the sphinx conda environment"

umask 0007
source /opt/conda/bin/activate
if ! conda activate sphinx; then
    echo "Creating conda environment"
    conda create -y -n sphinx
    conda activate sphinx
fi
conda install -y -c pytorch -c conda-forge --override-channels sphinx myst-parser nbsphinx awscli boto3 hydra-core rdkit scikit-learn mlflow molvs numba quaternion pytorch-lightning torchvision imageio jsonpickle torchmetrics

# Check for bayesian_torch

if [ $has_lib -ne 1 ]; then
    pushd /tmp
    git clone https://github.com/IntelLabs/bayesian-torch.git
    cd bayesian-torch
    pip install .
    cd ..
    rm -rf bayesian-torch
    popd
fi

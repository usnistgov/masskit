#!/bin/bash
set -eux

if [ -d "anaconda3" ] ; then
    echo "Anaconda directory exists, presuming it is already installed"

else
    wget -nv -O Anaconda3-2020.02-Linux-x86_64.sh https://repo.continuum.io/archive/Anaconda3-2020.02-Linux-x86_64.sh
    bash Anaconda3-2020.02-Linux-x86_64.sh -b -p anaconda3
    rm Anaconda3-2020.02-Linux-x86_64.sh    
    eval "$(./anaconda3/bin/conda shell.bash hook)"
    conda init
    conda create -y -n prop_predictor python=3.6.6
    conda activate prop_predictor
    conda install -y rdkit -c rdkit
    conda install -y pytorch-cpu torchvision-cpu -c pytorch
    conda install -y matplotlib cmake cairo eigen pkg-config boost-cpp py-boost scikit-learn flask tqdm gitpython scikit-image libiconv pyinstaller
    conda install -y -c conda-forge uwsgi quaternion molvs
fi

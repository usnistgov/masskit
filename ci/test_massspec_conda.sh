# To be executed in the libraries directory

mamba remove -y --force masskit || true
rm -rf $CONDA_PREFIX/pkgs/masskit-*
#mamba install -y -c $CONDA_PREFIX/conda-bld/linux-64/ --override-channels --no-deps masskit
#mamba install --use-local masskit
mamba install -y ./masskit-*.tar.bz2
pytest

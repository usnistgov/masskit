# To be executed in the libraries directory

mamba remove -y --force massspec || true
rm -rf $CONDA_PREFIX/pkgs/massspec-*
#mamba install -y -c $CONDA_PREFIX/conda-bld/linux-64/ --override-channels --no-deps massspec
#mamba install --use-local massspec
mamba install -y ./massspec-*.tar.bz2
pytest

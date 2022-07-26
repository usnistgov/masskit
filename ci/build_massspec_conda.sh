# To be executed in the libraries directory

rm -rf $CONDA_PREFIX/conda-bld/*
python setup.py bdist_conda
cp $CONDA_PREFIX/conda-bld/linux-64/massspec*.tar.bz2 .

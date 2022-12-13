# nist_ms: spectra and small molecule processing library

I put a full list of dependencies in requirements.txt, most likely not all are needed.  
Some packages that do need to be installed if not installed already: 
- numba
- numpy
- rdkit
- scipy
- matplotlib
- py3Dmol
- quaternion
- skimage
- pspearman (this is a sibling of this repository on github)

PYTHONPATH should be set to include the modules, e.g. 
PYTHONPATH=/home/lyg/source/deep/spectra/ei/nistms2:/home/lyg/source/deep/spectra/ei/pspearman

The demo jupyter notebook is in the docs/ directory.  If you have jupyter installed, just run `jupyter notebook` from the command line.
Data files can be found in `Share4All\Users\Lewis\data\deep\spectra\ei`

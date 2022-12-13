@echo OFF

rem This script is intended to be executed by CI/CD in the msdc_services\libraries directory

rmdir /s/q %CONDA_PREFIX%\conda-bld\*
python setup.py bdist_conda
copy %CONDA_PREFIX%\conda-bld\win-64\masskit*.tar.bz2 .
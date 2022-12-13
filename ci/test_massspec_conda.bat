rem To be executed in the libraries directory
rem @echo OFF

cmd /c mamba remove -y --force masskit
del /f %CONDA_PREFIX%\pkgs\masskit-*
cmd /c mamba install -y masskit-1.0-py39_0.tar.bz2
pytest

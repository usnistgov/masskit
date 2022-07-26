rem To be executed in the libraries directory
rem @echo OFF

cmd /c mamba remove -y --force massspec
del /f %CONDA_PREFIX%\pkgs\massspec-*
cmd /c mamba install -y massspec-1.0-py39_0.tar.bz2
pytest

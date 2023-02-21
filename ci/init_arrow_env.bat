@echo OFF

rem Define here the path to your conda installation
set CONDAPATH=C:\Tools\Miniconda3
set CONDABIN=%CONDAPATH%\condabin\conda.bat
set MAMBABIN=%CONDAPATH%\condabin\mamba.bat
set ENVNAME=arrow
set ENVPATH=%CONDAPATH%\envs\%ENVNAME%

if [%1]==[] ( set REMOVE=0 & goto noarg )
if %~1==/f set REMOVE=1
:noarg

if %REMOVE%==1 if exist %ENVPATH% (
    echo Removing existing environment
    rem call %CONDABIN% env remove --name %ENVNAME%
    call %CONDABIN% remove --name %ENVNAME% --all
    rem Why is this needed now? It wasn't before. Sigh.
    rmdir /s /q \tools\miniconda3\envs\arrow  
)

if not exist %ENVPATH% (
    echo Installing a new environment
    call %MAMBABIN% create --name %ENVNAME% -y -c conda-forge^
    "arrow-cpp=8.0.*"^
    conda-build^
    cython^
    hydra-core^
    imageio^
    jsonpickle^
    jupyter^
    matplotlib^
    numba^
    numpy^
    pandas^
    parquet-tools^
    "pyarrow=8.0.*"^
    pybind11^
    pynndescent^
    pytest^
    python=3^
    rdkit=2023.03.2^
    sqlalchemy^
    sqlparse
)   

rem Activate the conda environment
call %CONDABIN% activate %ENVNAME%

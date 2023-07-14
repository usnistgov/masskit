@echo off
SETLOCAL 
REM EnableDelayedExpansion
SET ENVNAME=masskit

:processargs
SET ARG=%1
IF DEFINED ARG (
    IF "%ARG%"=="/f" (
        SET REMOVE=1
        SHIFT
        GOTO processargs
    )
    IF "%ARG%"=="/force" (
        SET REMOVE=1
        SHIFT
        GOTO processargs
    )
    IF "%ARG%"=="/m" (
        SET ENVNAME=masskit_ai
        SET USE_ML=1
        SHIFT
        GOTO processargs
    )
    IF "%ARG%"=="/ml" (
        SET ENVNAME=masskit_ai
        SET USE_ML=1
        SHIFT
        GOTO processargs
    )
    IF "%ARG%"=="/c" (
        SET ENVNAME=masskit_ai_cpu
        SET USE_ML=1
        SET CPUONLY=1
        SHIFT
        GOTO processargs
    )
    IF "%ARG%"=="/cpu" (
        SET ENVNAME=masskit_ai_cpu
        SET USE_ML=1
        SET CPUONLY=1
        SHIFT
        GOTO processargs
    )
    IF "%ARG%"=="/s" (
        SHIFT
        SET SUFFIX=%2
        SHIFT
        GOTO processargs
    )
    IF "%ARG%"=="/suffix" (
        SHIFT
        SET SUFFIX=%2
        SHIFT
        GOTO processargs
    )
    ECHO Unknown argument: %ARG%
    SHIFT
    GOTO processargs
)

IF DEFINED SUFFIX (
    SET ENVNAME=%ENVNAME%_%SUFFIX%
)

IF DEFINED REMOVE (
    echo Trying to remove existing %ENVNAME% environment
    call conda env remove --name %ENVNAME%
)

:: Install mamba if needed
where /q mamba
IF ERRORLEVEL 1 (
    ECHO Mamba not found, attemping to install mamba.
    conda install -y mamba -n base -c conda-forge
)

SET BASE_PACKAGES=^
arrow-cpp=12.*^
 conda-build^
 cmake^
 cython^
 hydra-core^
 imageio^
 jsonpickle^
 jupyter^
 matplotlib^
 matplotlib-venn^
 numba^
 numpy^
 pandas^
 pyarrow=12.*^
 pybind11^
 pynndescent^
 pyside6^
 pytest^
 python=3^
 rdkit=2021.09.4^
 rich^
 ruff^
 scikit-build-core^
 sqlalchemy^
 sqlparse

SET ML_CHANNELS=
SET ML_PACKAGES=

IF DEFINED USE_ML (
  SET ML_CHANNELS=-c pytorch -c nvidia
  SET CUDATOOLKIT=cudatoolkit=11.*
  IF DEFINED CPUONLY (
    SET CUDATOOLKIT=cpuonly
  )
)

IF DEFINED USE_ML (
  SET ML_PACKAGES=%CUDATOOLKIT%^
 boto3^
 imbalanced-learn^
 jupyter^
 levenshtein^
 mlflow-skinny^
 pytorch^
 pytorch-lightning^
 scikit-learn^
 torchmetrics^
 torchvision
)

ECHO Initializing the conda %ENVNAME% environment
call conda activate %ENVNAME%
IF ERRORLEVEL 1 (
    ECHO Creating conda environment
    mamba create -y -n %ENVNAME%
    mamba install -y -n %ENVNAME% -c nodefaults %ML_CHANNELS% -c conda-forge %BASE_PACKAGES% %ML_PACKAGES%
    call conda activate %ENVNAME%
)

ENDLOCAL
::EXIT /B

@echo off
SETLOCAL 
REM EnableDelayedExpansion
SET ENVNAME=masskit

SET SCRIPT_DIR=%~dp0

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
    conda env remove --name %ENVNAME%
)

:: Install mamba if needed
where /q mamba
IF ERRORLEVEL 1 (
    ECHO Mamba not found, attemping to install mamba.
    conda install -y mamba -n base -c conda-forge
)

SET BASE_PACKAGES=--file=%SCRIPT_DIR%base_packages.txt

SET ML_CHANNELS=
SET ML_PACKAGES=

IF DEFINED USE_ML (
  SET ML_CHANNELS=-c pytorch -c nvidia
  SET CUDATOOLKIT=%SCRIPT_DIR%cuda_packages.txt
  IF DEFINED CPUONLY (
    SET CUDATOOLKIT=%SCRIPT_DIR%nocuda_packages.txt
  )
)

IF DEFINED USE_ML (
  SET ML_PACKAGES=--file=%CUDATOOLKIT% --file=%SCRIPT_DIR%ml_packages.txt
)

ECHO Initializing the conda %ENVNAME% environment
call conda activate %ENVNAME%
IF ERRORLEVEL 1 (
    ECHO Creating conda environment
    mamba create --no-banner -y -n %ENVNAME%
    mamba install --no-banner -y -n %ENVNAME% -c nodefaults %ML_CHANNELS% -c conda-forge %BASE_PACKAGES% %ML_PACKAGES%
    call conda activate %ENVNAME%
)

ENDLOCAL
::EXIT /B %ERRORLEVEL%

# - Find PyArrow
# This module finds the libraries corresponding to the pyarrow library
# 
# This code sets the following variables:
#
#  PYARROW_FOUND           - has the PyArrow installation been found
#  PYARROW_LIBRARY         - path to the pyarrow library
#  PYTHON_INCLUDE_DIR      - path to where pyarrow.h is found

if(PYARROW_FOUND)
  return()
endif()

if(PyArrow_FIND_QUIETLY)
  set(_pyarrow_quiet QUIET)
else()
  set(_pyarrow_quiet "")
endif()

if(PyArrow_FIND_REQUIRED)
  set(_pyarrow_required REQUIRED)
endif()

include(CMakeFindDependencyMacro)
find_dependency(Python3 COMPONENTS Interpreter)

if(NOT Python3_Interpreter_FOUND)
  set(PYARROW_FOUND FALSE)
  return()
endif()

execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" "-c" "
import pyarrow as pa;
print(pa.cpp_version);
print(pa.get_include());
print('arrow_python' if 'arrow_python' in set(pa.get_libraries()) else '');
print('|'.join(pa.get_library_dirs()));
"
  RESULT_VARIABLE _PYARROW_SUCCESS
  OUTPUT_VARIABLE _PYARROW_VALUES
  ERROR_VARIABLE _PYARROW_ERROR_VALUE)

if(NOT _PYARROW_SUCCESS MATCHES 0)
  if(PyArrow_FIND_REQUIRED)
    message(FATAL_ERROR "Python config failure:\n${_PYARROW_ERROR_VALUE}")
  endif()
  set(PYARROW_FOUND FALSE)
  return()
endif()

option(
  PYBIND11_PYTHONLIBS_OVERWRITE
  "Overwrite cached values read from Python library (classic search). Turn off if cross-compiling and manually setting these values."
  ON)
# Can manually set values when cross-compiling
macro(_PYBIND11_GET_IF_UNDEF lst index name)
  if(PYBIND11_PYTHONLIBS_OVERWRITE OR NOT DEFINED "${name}")
    list(GET "${lst}" "${index}" "${name}")
  endif()
endmacro()

# Convert the process output into a list
if(WIN32)
  # Make sure all directory separators are '/'
  string(REGEX REPLACE "\\\\" "/" _PYARROW_VALUES ${_PYARROW_VALUES})
endif()
#string(REGEX REPLACE ";" "\\\\;" _PYARROW_VALUES ${_PYARROW_VALUES})
#string(REGEX REPLACE ";" "|" _PYARROW_VALUES ${_PYARROW_VALUES})
string(REGEX REPLACE "\n" ";" _PYARROW_VALUES ${_PYARROW_VALUES})
_pybind11_get_if_undef(_PYARROW_VALUES 0 PYARROW_VERSION_STRING)
_pybind11_get_if_undef(_PYARROW_VALUES 1 PYARROW_INCLUDE_DIR)
_pybind11_get_if_undef(_PYARROW_VALUES 2 PYARROW_LIB)
_pybind11_get_if_undef(_PYARROW_VALUES 3 PYARROW_LIBDIRS)

# Make sure all directory separators are '/'
#string(REGEX REPLACE "\\\\" "/" PYARROW_INCLUDE_DIR "${PYARROW_INCLUDE_DIR}")
#string(REGEX REPLACE "\\\\" "/" PYARROW_LIBDIRS "${PYARROW_LIBDIRS}")
string(REGEX REPLACE "\\|" " " PYARROW_LIBDIRS ${PYARROW_LIBDIRS})

find_library(
  PYARROW_LIBRARY
  NAMES "${PYARROW_LIB}"
  HINTS ${PYARROW_LIBDIRS}
  NO_DEFAULT_PATH)

set(PYARROW_INCLUDE_DIR "${PYARROW_INCLUDE_DIR}")
set(PYARROW_LIBRARY "${PYARROW_LIBRARY}")

message(STATUS "Found PyArrowLib: ${PYARROW_LIBRARY} (found version \"${PYARROW_VERSION_STRING}\")")
message(STATUS "pyarrow.h include dir: ${PYARROW_INCLUDE_DIR}")

set(PYARROW_FOUND TRUE)



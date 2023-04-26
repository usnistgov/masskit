# CMakeList.txt : CMake project for masskit_ext, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.15)
message(STATUS "Building using CMake version: ${CMAKE_VERSION}")
message(STATUS "Building on: ${CMAKE_HOST_SYSTEM_NAME}")

if(NOT DEFINED ENV{CONDA_PREFIX})
    if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
        set(VCPKG_LOCATION $ENV{HOME}/code/vcpkg/scripts/buildsystems/vcpkg.cmake)
    elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
        set(VCPKG_LOCATION $ENV{HOMEPATH}\\code\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake)
    endif()
    message(STATUS "Not building in a conda env, looking for ${VCPKG_LOCATION}")
    if(EXISTS ${VCPKG_LOCATION})
        # We should check to see if CMAKE_TOOLCHAIN_FILE is set as an environmental variable
        set(CMAKE_TOOLCHAIN_FILE ${VCPKG_LOCATION}
            CACHE STRING "Vcpkg toolchain file")
        if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
            set(VCPKG_TARGET_TRIPLET "x64-linux" CACHE STRING "VCPKG Target Triplet to use")
        elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
            set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "VCPKG Target Triplet to use")
        endif()
        message(STATUS "Using Triplet: ${VCPKG_TARGET_TRIPLET}")
    endif()
endif()

project ("masskit-ext")

find_package(Arrow CONFIG REQUIRED)
find_package(ArrowDataset CONFIG REQUIRED)
find_package(pybind11 CONFIG)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(PyArrow)
if (${PYARROWLIBS_FOUND})
  set(CREATE_PYTHON_LIBS ON)
else()
  set(CREATE_PYTHON_LIBS OFF)
endif()

# set(CREATE_PYTHON_LIBS ON)
# if(DEFINED PYTHON_EXECUTABLE)
#     execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pyarrow as pa;print(pa.get_include(),end='')"
#                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                     RESULT_VARIABLE PYARROW_INCLUDE_RESULT
#                     OUTPUT_VARIABLE PYARROW_INCLUDE_DIR)
#     if(NOT PYARROW_INCLUDE_RESULT EQUAL "0")
#         message(STATUS "Unable to locate the include directory for pyarrow.h")
#         message(STATUS "    Python support will not be included")
#         set(CREATE_PYTHON_LIBS OFF)
#     else()
#         message(STATUS "pyarrow.h include dir: ${PYARROW_INCLUDE_DIR}")
#     endif()

#     execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pyarrow as pa;print(pa.get_libraries(),end='')"
#                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                     RESULT_VARIABLE PYARROW_LIBRARIES_RESULT
#                     OUTPUT_VARIABLE PYARROW_LIBRARIES)
#     if(NOT PYARROW_LIBRARIES_RESULT EQUAL "0")
#         message(STATUS "Unable to locate the libraries for pyarrow")
#         message(STATUS "    Python support will not be included")
#         set(CREATE_PYTHON_LIBS OFF)
#     else()
#         message(STATUS "pyarrow libraries: ${PYARROW_LIBRARIES}")
#     endif()

#     execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pyarrow as pa;print(pa.get_library_dirs(),end='')"
#                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                     RESULT_VARIABLE PYARROW_LIBRARY_DIRS_RESULT
#                     OUTPUT_VARIABLE PYARROW_LIBRARY_DIRS)
#     if(NOT PYARROW_LIBRARY_DIRS_RESULT EQUAL "0")
#         message(STATUS "Unable to locate the libraries directory for pyarrow")
#         message(STATUS "    Python support will not be included")
#         set(CREATE_PYTHON_LIBS OFF)
#     else()
#         message(STATUS "pyarrow library dirs: ${PYARROW_LIBRARY_DIRS}")
#     endif()
# endif()

# set(CREATE_PYTHON_LIBS ON)
# if(NOT DEFINED PYARROW_INCLUDE_DIR)
#     #set(PYARROW_INCLUDE_DIR "/home/slottad/miniconda3/envs/masskit/lib/python3.9/site-packages/pyarrow/include")
#     execute_process(COMMAND python -c "import pyarrow as pa;print(pa.get_include(),end='')"
#                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                     RESULT_VARIABLE PYARROW_INCLUDE_RESULT
#                     OUTPUT_VARIABLE PYARROW_INCLUDE_DIR)
#     if(NOT PYARROW_INCLUDE_RESULT EQUAL "0")
#         message(STATUS "Unable to locate the include directory for pyarrow.h")
#         message(STATUS "    Python support will not be included")
#         set(CREATE_PYTHON_LIBS OFF)
#     endif()
# endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# We require a C++17 compliant compiler
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#option(ARROW_LINK_SHARED "Link to the Arrow shared library" ON)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    # add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# message(STATUS "Arrow version: ${ARROW_VERSION}")
# message(STATUS "Arrow SO version: ${ARROW_FULL_SO_VERSION}")

if(${CREATE_PYTHON_LIBS})
    message(STATUS "pyarrow.h include dir: ${PYARROW_INCLUDE_DIR}")
endif()

if(${ARROW_BUILD_STATIC})
    message(STATUS "Linking Arrow libs statically")
else()
    message(STATUS "Linking Arrow libs dynamically")
endif()

if ((${CMAKE_BUILD_TYPE} MATCHES "Release") OR (${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo"))
    # To show which lines have been automatically vectorized:
    # SET(GCC_VECTORIZATION_COMPILE_FLAGS "-O3 -ffast-math -mavx -fopt-info-vec-optimized")
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        SET(GCC_VECTORIZATION_COMPILE_FLAGS "-O3 -ffast-math -mavx")
        SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_VECTORIZATION_COMPILE_FLAGS}")
    endif()
endif()

add_subdirectory(src)
add_subdirectory(examples/cpp)

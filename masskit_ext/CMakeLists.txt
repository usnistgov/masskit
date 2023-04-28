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
if(${ARROW_BUILD_STATIC})
    message(STATUS "Linking Arrow libs statically")
else()
    message(STATUS "Linking Arrow libs dynamically")
endif()

find_package(pybind11 CONFIG)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(PyArrow)
if (${PYARROW_FOUND})
  set(CREATE_PYTHON_LIBS ON)
else()
  set(CREATE_PYTHON_LIBS OFF)
endif()

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

if (${PYARROW_FOUND})
    install(TARGETS masskit_ext LIBRARY DESTINATION .)
endif()
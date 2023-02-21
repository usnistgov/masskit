# CMakeList.txt : CMake project for masskit_ext, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.15)
message(STATUS "Building using CMake version: ${CMAKE_VERSION}")
message(STATUS "Building on: ${CMAKE_HOST_SYSTEM_NAME}")

if(NOT DEFINED ENV{CONDA_PREFIX})
	if(EXISTS $ENV{HOME}/code/vcpkg/scripts/buildsystems/vcpkg.cmake)
		set(CMAKE_TOOLCHAIN_FILE $ENV{HOME}/code/vcpkg/scripts/buildsystems/vcpkg.cmake
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
find_package(pybind11 CONFIG REQUIRED)

set(CREATE_PYTHON_LIBS ON)
if(NOT DEFINED PYARROW_INCLUDE_DIR)
	#set(PYARROW_INCLUDE_DIR "/home/slottad/miniconda3/envs/masskit/lib/python3.9/site-packages/pyarrow/include")
	execute_process(COMMAND python -c "import pyarrow as pa;print(pa.get_include(),end='')"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE PYARROW_INCLUDE_RESULT
					OUTPUT_VARIABLE PYARROW_INCLUDE_DIR)
	if(NOT PYARROW_INCLUDE_RESULT EQUAL "0")
		message(STATUS "Unable to locate the include directory for pyarrow.h")
		message(STATUS "    Python support will not be included")
		set(CREATE_PYTHON_LIBS OFF)
	endif()
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
	SET(GCC_VECTORIZATION_COMPILE_FLAGS "-O3 -ffast-math -mavx")
	SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_VECTORIZATION_COMPILE_FLAGS}")
endif()

add_subdirectory(src)
add_subdirectory(examples/cpp)

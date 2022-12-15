# CMakeList.txt : CMake project for masskit_ext, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.15)
message(STATUS "Building using CMake version: ${CMAKE_VERSION}")
message(STATUS "Building on: ${CMAKE_HOST_SYSTEM_NAME}")

# if(NOT DEFINED ENV{CONDA_PREFIX})
# 	if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake)
# 		set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
# 			CACHE STRING "Vcpkg toolchain file")
# 		if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
# 			set(VCPKG_TARGET_TRIPLET "x64-linux-dynamic" CACHE STRING "VCPKG Target Triplet to use")
# 		elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
# 			set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "VCPKG Target Triplet to use")
# 		endif()
# 		message(STATUS "Using Triplet: ${VCPKG_TARGET_TRIPLET}")
# 	endif()
# endif()

project ("masskit-ext")

find_package(Arrow CONFIG REQUIRED)
find_package(ArrowDataset CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
endif()

# We require a C++11 compliant compiler
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

message(STATUS "pyarrow.h include dir: ${PYARROW_INCLUDE_DIR}")

add_subdirectory(src)
add_subdirectory(examples/cpp)
add_subdirectory(search)

# Add source to this project's executable.
add_executable (masskit_example "masskit_ext.cpp" "masskit_ext.h")
target_link_libraries(masskit_example PRIVATE Arrow::arrow_shared)

# python extension
pybind11_add_module(masskit_ext ext.cpp fingerprints.cpp index.cpp pybind_module.cpp)
target_include_directories(masskit_ext PUBLIC ${PYARROW_INCLUDE_DIR})
target_link_libraries(masskit_ext PRIVATE search Arrow::arrow_shared)

add_subdirectory(search)

# Add source to this project's executable.
add_executable (masskit_example "masskit_ext.cpp" "masskit_ext.h")

# python extension
pybind11_add_module(masskit_ext ext.cpp fingerprints.cpp index.cpp pybind_module.cpp)
target_link_libraries(masskit_ext PRIVATE search)
target_link_libraries(masskit_ext PRIVATE arrow_shared arrow_python_shared parquet_shared)

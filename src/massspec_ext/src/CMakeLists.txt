add_subdirectory(search)

# Add source to this project's executable.
add_executable (massspec_example "massspec_ext.cpp" "massspec_ext.h")

# python extension
pybind11_add_module(massspec_ext ext.cpp fingerprints.cpp index.cpp pybind_module.cpp)
target_link_libraries(massspec_ext PRIVATE search)
target_link_libraries(massspec_ext PRIVATE arrow_shared arrow_python_shared parquet_shared)

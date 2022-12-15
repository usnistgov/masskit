add_subdirectory(search)

# Add source to this project's executable.
add_executable (massspec_example "massspec_ext.cpp" "massspec_ext.h")
target_link_libraries(massspec_example PRIVATE Arrow::arrow_shared)

# python extension
pybind11_add_module(massspec_ext ext.cpp fingerprints.cpp index.cpp pybind_module.cpp)
target_include_directories(massspec_ext PUBLIC ${PYARROW_INCLUDE_DIR})
target_link_libraries(massspec_ext PRIVATE search Arrow::arrow_shared)

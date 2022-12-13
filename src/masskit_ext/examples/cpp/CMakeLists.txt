project(examples)

add_executable (search_example "search_example.cpp")
target_link_libraries(search_example PRIVATE search)

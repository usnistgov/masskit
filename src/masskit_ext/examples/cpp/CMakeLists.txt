project(examples)

add_executable (search_example "search_example.cpp")
target_link_libraries(search_example PRIVATE search "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>")

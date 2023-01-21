project(search)

################################################################################
# Source groups
################################################################################
set(Header_Files
    search.hpp
)
source_group("Header Files" FILES ${Header_Files})

add_library(search STATIC ${Header_Files} search.cpp)
set_property(TARGET search PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(search PUBLIC .)

target_link_libraries(search PUBLIC
    "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>"
    "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,ArrowDataset::arrow_dataset_static,ArrowDataset::arrow_dataset_shared>")

add_executable (test_search ${Header_Files} "test_search.cpp")
target_link_libraries(test_search PRIVATE search "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>")

add_executable (udf_example "udf_example.cpp")
target_link_libraries(udf_example PRIVATE "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>")

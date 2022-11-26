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
target_link_libraries(search PUBLIC arrow_shared arrow_dataset_shared parquet_shared)

add_executable (test_search ${Header_Files} "test_search.cpp")
target_link_libraries(test_search PRIVATE search arrow_shared parquet_shared)

add_executable (udf_example "udf_example.cpp")
target_link_libraries(udf_example PRIVATE arrow_shared parquet_shared)

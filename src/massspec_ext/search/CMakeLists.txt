project(search)

################################################################################
# Source groups
################################################################################
set(Header_Files
    search.hpp
)
source_group("Header Files" FILES ${Header_Files})

add_library(search STATIC ${Header_Files} search.cpp)
target_include_directories(search PUBLIC .)
target_link_libraries(search PUBLIC ${ARROW_LIBRARIES})
target_link_libraries(search PUBLIC ${PARQUET_LIBRARIES})

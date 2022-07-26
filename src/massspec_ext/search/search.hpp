#include <arrow/api.h>
//#include <arrow/array/builder_primitive.h>
//#include <arrow/chunked_array.h>
//#include <arrow/table.h>


std::shared_ptr<arrow::Table> tanimoto_search(
    std::shared_ptr<arrow::Scalar> query,
    std::shared_ptr<arrow::Scalar> query_count,
    std::shared_ptr<arrow::Table> table
);

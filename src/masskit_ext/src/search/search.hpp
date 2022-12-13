#include <arrow/api.h>
#include <arrow/compute/api.h>

namespace cp = ::arrow::compute;

// std::shared_ptr<arrow::Table> tanimoto_search(
//     std::shared_ptr<arrow::Scalar> query,
//     std::shared_ptr<arrow::Scalar> query_count,
//     std::shared_ptr<arrow::Table> table
// );

arrow::Status RegisterSearchFunctions(cp::FunctionRegistry* registry);
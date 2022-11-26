#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Python.h>

#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/chunked_array.h>
#include <arrow/table.h>

#include <string>

#include "caster.hpp"

namespace py = pybind11;


// py::bytes mz_fingerprint(std::shared_ptr<arrow::DoubleArray> mz);
int64_t arrow_chunk(std::shared_ptr<arrow::ChunkedArray>& source);
int64_t table_info(std::shared_ptr<arrow::Table>& source);

std::shared_ptr<arrow::DoubleArray> arrow_add(std::shared_ptr<arrow::DoubleArray>& a,
                                        std::shared_ptr<arrow::DoubleArray>& b,
                                        std::shared_ptr<arrow::DoubleArray>& c);

py::array_t<double> numpy_add(py::array_t<double> input1, py::array_t<double> input2);

std::shared_ptr<arrow::Table> calc_start_stops(std::shared_ptr<arrow::Table> table);

std::shared_ptr<arrow::Table> tanimoto_search(
    std::shared_ptr<arrow::Scalar> query,
    std::shared_ptr<arrow::Scalar> query_count,
    std::shared_ptr<arrow::Table> table
);

class BruteForceIndex {
public:
    BruteForceIndex(std::string name) : name(name) {}
    int64_t create(std::shared_ptr<arrow::Table>& query);
    int64_t search(std::shared_ptr<arrow::Table>& query);

private:
    std::string name;
    std::shared_ptr<arrow::Table> source = nullptr;
};

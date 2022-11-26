#include "ext.hpp"

PYBIND11_MODULE(massspec_ext, m) {
    arrow::py::import_pyarrow();
    m.doc() = "massspec c++ extension functions";
    m.def("arrow_chunk", &arrow_chunk, "Examine ChunkedArray usage");
    m.def("table_info", &table_info, "Examine Table usage");
    m.def("arrow_add", &arrow_add, "Add two PyArrow arrays");
    m.def("numpy_add", &numpy_add, "Add two NumPy arrays");
    // m.def("mz_fingerprint", &mz_fingerprint);
    m.def("calc_start_stops", &calc_start_stops, "Add start and stop columns to arrow table");
    m.def("tanimoto_search", &tanimoto_search, "Perform a Tanimoto search on a given table");
    py::class_<BruteForceIndex>(m, "BruteForceIndex")
        .def(py::init<const std::string &>())
        .def("create", &BruteForceIndex::create)
        .def("search", &BruteForceIndex::search);
}

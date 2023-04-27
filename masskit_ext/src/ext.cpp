
//#include <rdkit/DataStructs/BitVects.h>
//#include <rdkit/DataStructs/BitOps.h>

#include <iostream>
#include <memory>
#include <math.h>

#include "ext.hpp"

// namespace py = pybind11;
//using DoubleArray = arrow::DoubleArray;
//using Array = arrow::Array;

int calc_bit(double a, double b) {
    return static_cast<int>(floor(abs(b-a)*10));
}

//ExplicitBitVect mz_fingerprint_impl(std::shared_ptr<arrow::DoubleArray> mz) {
//    ExplicitBitVect fp(20000);
//    fp.setBit(10);
//
//    //std::cout << "Hello, world!" << std::endl;
//    for(int i = 0; i < mz->length(); i++) {
//        for(int j = i+1; j < mz->length(); j++) {
//            fp.setBit(calc_bit(mz->Value(i), mz->Value(j)));
//        }
//    }
//    return fp;
//}
//
//
//py::bytes mz_fingerprint(std::shared_ptr<arrow::DoubleArray> mz) {
//    return BitVectToBinaryText( mz_fingerprint_impl(mz) );
//}

int64_t arrow_chunk(std::shared_ptr<arrow::ChunkedArray>& source) {
    return source->length();
}

int64_t table_info(std::shared_ptr<arrow::Table>& source) {
    return source->num_columns();
}

std::shared_ptr<arrow::DoubleArray> arrow_add(std::shared_ptr<arrow::DoubleArray>& a,
                                        std::shared_ptr<arrow::DoubleArray>& b,
                                        std::shared_ptr<arrow::DoubleArray>& c) {
    if ((a->length() != b->length()) || (a->length() != c->length())) {
        throw std::length_error("Arrays are not of equal length");
    }
    arrow::DoubleBuilder builder;
    arrow::Status status = builder.Resize(a->length());
    if (!status.ok()) {
        throw std::bad_alloc();
    }
    for(int i = 0; i < a->length(); i++) {
        builder.UnsafeAppend(a->Value(i) + b->Value(i) * c->Value(i));
    }
    std::shared_ptr<arrow::DoubleArray> array;
    arrow::Status st = builder.Finish(&array);
    return array;
}

py::array_t<double> numpy_add(py::array_t<double> input1, py::array_t<double> input2) {
    py::buffer_info buf1 = input1.request(), buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shapes must match");

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<double>(buf1.size);

    py::buffer_info buf3 = result.request();

    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *ptr2 = static_cast<double *>(buf2.ptr);
    double *ptr3 = static_cast<double *>(buf3.ptr);

    for (int64_t idx = 0; idx < buf1.shape[0]; idx++)
        ptr3[idx] = ptr1[idx] + ptr2[idx];

    return result;
}

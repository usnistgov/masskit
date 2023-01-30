#include <arrow/api.h>
#include <arrow/compute/api.h>

#include <iostream>
#include <memory>
#include <cstdint>

#include "search.hpp"

#ifdef _MSC_VER
  #include <intrin.h>
  #define my_popcount(m) __popcnt64(m)
#else
   #define my_popcount(m) __builtin_popcountll(m)
#endif

// All compute functions must be documented. Ironically, how to set
// this up is not well documented in the online docs, one must use the
// source.
const cp::FunctionDoc tanimoto_func_doc{
    // Summary
    "User-defined-function to perform a tanimoto match",
    // Description
    "returns the tanimoto score for every element for the given query input",
    // Arguments
    { "query_fingerprint", "query_fingerprint_count",
      "fingerprint_array", "fingerprint_array_counts"},
    // Options (optional) 
    "UDFOptions",
    // Are options required?
    false};


// The actual compute function. This is a scalar function, so the
// output will generally have the same number of rows as the
// input. The function will be called multiple times, each on a
// different subset of rows (or span).  Possible inputs are either
// scalars or vectors.
arrow::Status TanimotoFunction(cp::KernelContext* ctx,
                               const cp::ExecSpan& batch,
                               cp::ExecResult* out) {

    // Useful for figuring out what exactly is being passed to functions like this
    // for (int64_t n = 0; n < batch.num_values(); ++n) {
    // 	std::cout << "Batch: " << n
    //            << "\tType: " << (batch[n].is_scalar() ? "scalar" : "array") << ", "
    //            << batch.GetTypes()[n] << std::endl;
    // }

    // As one can see, the scalar inputs are striaghtforward(!?) to
    // work with. I'm sure there is a better way, and one day I'll
    // find it, right before they change it.
    // auto query = std::static_pointer_cast<arrow::UInt8Array>
    //     ( static_cast<const arrow::LargeListScalar*>(batch[0].scalar)->value );
    // const uint64_t* query64 = reinterpret_cast<const uint64_t*>(query->raw_values());

    // A better way?
    auto query = batch[0].scalar_as<arrow::LargeListScalar>().value;
    auto query64 = query->data()->GetValues<uint64_t>(1);

    if (query->length() % 8 != 0) {
        return arrow::Status::TypeError("Query array is non-conformant");
    }
    int64_t query_length = query->length() / 8;

    // // Output for debugging
    // for (int64_t i = 0; i < query_length; i++) {
    // 	std::cout << query64[i] << " ";
    // }
    // std::cout << std::endl;

    const int32_t query_count = batch[1].scalar_as<arrow::Int32Scalar>().value;

    // A list of lists is a little more convoluted to work with. They
    // use a poorly documented class called ArraySpan to pass the data
    // in and out. The offsets are in the top-level ArraySpan and the
    // data is in the child ArraySpan.
    // See arrow/cpp/src/arrow/compute/kernels/scalar_nested.cc for
    // more examples of using ArraySpan.
    const arrow::ArraySpan& list = batch[2].array;
    const arrow::ArraySpan& list_values = list.child_data[0];
    const int64_t* offsets = list.GetValues<int64_t>(1);

    const int32_t* arr_count = batch[3].array.GetValues<int32_t>(1);

    // Get the output array so we may place our results there.
    //arrow::ArraySpan* out_arr = out->array_span();
    //auto out_values = out_arr->GetValues<float>(1);
    auto out_values = out->array_span()->GetValues<float>(1);

    // Iterate over our subset of rows.
    for (int64_t i = 0; i < list.length; ++i) {
        // List of lists are handled with offsets into one giant list.
        // The offsets stored the top level ArraySpan, and the actual
        // data in the child_data member.
        int32_t data_length = (offsets[i + 1] - offsets[i]) / 8;

        // The following does not work because the offset is interpreted as 
        // applying uint64 array, not a uint8 array, as it is stored 
        //   const uint64_t* data64 = list_values.GetValues<uint64_t>(1, offsets[i]);
        // So we need to advance the pointer, and then reinterpret the array.
        const uint8_t* data = list_values.GetValues<uint8_t>(1, offsets[i]);
        const uint64_t* data64 = reinterpret_cast<const uint64_t*>(data);

        // Compute the tanimoto score.
        int32_t	sum_and(0);
        int32_t sum_or(0);
        for (int64_t j = 0; j < data_length; ++j) {
            sum_and += my_popcount(query64[j] & data64[j]);
            sum_or += my_popcount(query64[j] | data64[j]);
        }
        *out_values++ = static_cast<float_t>(sum_and) / static_cast<float_t>(sum_or);
    }
  return arrow::Status::OK();
}

const cp::FunctionDoc cosine_score_doc{
    // Summary
    "User-defined-function to calculate the cosine score",
    // Description
    "returns the cosine score between the query and elements in the reference set",
    // Arguments
    { "query_mz", "query_intensity", "query_massinfo", "reference_mz", "reference_intensity", "reference_massinfo"},
    // Options (optional) 
    "CosineScoreOptions",
    // Are options required?
    false};

enum Tolerance_Type { NONE, PPM, DALTONS};

struct CSpectrum {
    std::vector<double> _mz;
    std::vector<double> _intensity;
    std::vector<double> _starts;
    std::vector<double> _stops;
    double tolerance;
    Tolerance_Type tol_type;

    CSpectrum(std::shared_ptr<arrow::DoubleArray> mzArr,
              std::shared_ptr<arrow::DoubleArray> intensityArr,
              double tolerance, Tolerance_Type type) 
              : tolerance(tolerance), tol_type(type) {
        const double *mzbuf = mzArr->raw_values();
        const double *intbuf = intensityArr->raw_values();
        // Strip out ions with zero intensities
        for (auto i = 0; i < mzArr->length(); i++) {
            if ( intbuf[i] > 0) {
                _mz.push_back(mzbuf[i]);
                _intensity.push_back(intbuf[i]);
            }
        }
        this->set_starts_stops();
    }

    CSpectrum(const double mzArr[], const double intensityArr[], int32_t length,
              double tolerance, Tolerance_Type type) 
              : tolerance(tolerance), tol_type(type) {
        // Strip out ions with zero intensities
        for (auto i = 0; i < length; i++) {
            if ( intensityArr[i] > 0) {
                _mz.push_back(mzArr[i]);
                _intensity.push_back(intensityArr[i]);
            }
        }
        this->set_starts_stops();
    }

    double half_tolerance(double mz) {
        switch (this->tol_type) {
            case NONE: 
                return 0;
            case PPM:
                return mz * this->tolerance / 1000000.0;
            case DALTONS:
                return this->tolerance;
        }
        return 0;
    }

    void set_starts_stops() {
        for (auto mz : _mz) {
            double hTol = this->half_tolerance(mz);
            _starts.push_back(mz - hTol);
            _stops.push_back(mz + hTol);
        }
    }

};

arrow::Status CosineScore(cp::KernelContext* ctx,
                               const cp::ExecSpan& batch,
                               cp::ExecResult* out) {

    // Query data
    auto query_mz = std::static_pointer_cast<arrow::DoubleArray>(
        batch[0].scalar_as<arrow::LargeListScalar>().value);
    auto query_intensity = std::static_pointer_cast<arrow::DoubleArray>(
        batch[1].scalar_as<arrow::LargeListScalar>().value);
    auto query_massinfo = batch[2].scalar_as<arrow::StructScalar>();
    auto query_tol = std::static_pointer_cast<arrow::DoubleScalar>(query_massinfo.field("tolerance").ValueOrDie());
    auto query_toltype = std::static_pointer_cast<arrow::DictionaryScalar>(
        query_massinfo.field("tolerance_type").ValueOrDie()
        )->GetEncodedValue().ValueOrDie()->ToString();

    CSpectrum query(query_mz, query_intensity, query_tol->value, PPM);

    // Reference Data
    auto ref_mz = batch[3].array;
    auto ref_mz_offsets = ref_mz.GetValues<int64_t>(1);
    auto ref_mz_data = ref_mz.child_data[0];

    auto ref_intensity = batch[4].array;
    auto ref_intensity_offsets = ref_intensity.GetValues<int64_t>(1);
    auto ref_intensity_data = ref_intensity.child_data[0];

    auto ref_massinfo = std::static_pointer_cast<arrow::StructArray>(batch[5].array.ToArray());
    auto ref_tol = std::static_pointer_cast<arrow::DoubleArray>(ref_massinfo->GetFieldByName("tolerance"));
    auto tol_type = std::static_pointer_cast<arrow::DictionaryArray>(ref_massinfo->GetFieldByName("tolerance_type"));
    auto tol_dict = tol_type->dictionary();
    auto tol_idx = std::static_pointer_cast<arrow::Int32Array>(tol_type->indices())->raw_values();

    // Output array for results
    auto out_values = out->array_span()->GetValues<float>(1);

    for (int64_t i = 0; i < batch.length; i++) {
        int32_t mzlength = (ref_mz_offsets[i + 1] - ref_mz_offsets[i]);
        int32_t intlength = (ref_intensity_offsets[i + 1] - ref_intensity_offsets[i]);
        if (mzlength != intlength) return arrow::Status::Invalid("Spectrum mz and intensity array lengths do not match");
        const double* mzdata = ref_mz_data.GetValues<double>(1, ref_mz_offsets[i]);
        const double* intensitydata = ref_intensity_data.GetValues<double>(1, ref_intensity_offsets[i]);

        CSpectrum reference(mzdata, intensitydata, mzlength, tol_idx[i], PPM);
        auto val = tol_idx[i];
        *out_values++ = val;
    }

    return arrow::Status::OK();
}


// All search UDFs should be placed here for registration
arrow::Status RegisterSearchFunctions(cp::FunctionRegistry* registry) {
    // Prepare the functions for registration
    auto tanimoto_func = std::make_shared<cp::ScalarFunction>("tanimoto", cp::Arity(4, false), tanimoto_func_doc);
    auto cosine_score = std::make_shared<cp::ScalarFunction>("cosine_score", cp::Arity(6, false), cosine_score_doc);

    // The kernel links the inputs to a given C++ function. It might be
    // nice to use templates to simplify writing the handling functions
    cp::ScalarKernel tanimoto_kernel({
            cp::InputType(arrow::Type::LARGE_LIST), arrow::int32(),
            cp::InputType(arrow::Type::LARGE_LIST), arrow::int32()},
        arrow::float32(), TanimotoFunction);
    tanimoto_kernel.mem_allocation = cp::MemAllocation::PREALLOCATE;
    tanimoto_kernel.null_handling = cp::NullHandling::INTERSECTION;

    cp::ScalarKernel cosine_score_kernel({
            cp::InputType(arrow::Type::LARGE_LIST), 
            cp::InputType(arrow::Type::LARGE_LIST),
            cp::InputType(arrow::Type::STRUCT),
            cp::InputType(arrow::Type::LARGE_LIST),
            cp::InputType(arrow::Type::LARGE_LIST),
            cp::InputType(arrow::Type::STRUCT)},
            arrow::float32(),
            CosineScore);
    cosine_score_kernel.mem_allocation = cp::MemAllocation::PREALLOCATE;
    cosine_score_kernel.null_handling = cp::NullHandling::INTERSECTION;

    // Add the kernal to the functions. Multiple kernels may be added to
    // a function to make it polymorphic.
    ARROW_RETURN_NOT_OK(tanimoto_func->AddKernel(std::move(tanimoto_kernel)));
    ARROW_RETURN_NOT_OK(cosine_score->AddKernel(std::move(cosine_score_kernel)));

    // Finally, register the functions
    ARROW_RETURN_NOT_OK(registry->AddFunction(std::move(tanimoto_func)));
    ARROW_RETURN_NOT_OK(registry->AddFunction(std::move(cosine_score)));

    return arrow::Status::OK();
}

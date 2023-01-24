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
    "User-defined-function to usage to perform a tanimoto match",
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
    auto query = std::static_pointer_cast<arrow::UInt8Array>
        ( static_cast<const arrow::LargeListScalar*>(batch[0].scalar)->value );

    if (query->length() % 8 != 0) {
        return arrow::Status::TypeError("Query array is non-conformant");
    }
    int64_t query_length = query->length() / 8;
    const uint64_t* query64 = reinterpret_cast<const uint64_t*>(query->raw_values());

    // // Output for debugging
    // for (int64_t i = 0; i < query_length; i++) {
    // 	std::cout << query64[i] << " ";
    // }
    // std::cout << std::endl;

    const int32_t query_count = static_cast<const arrow::Int32Scalar*>(batch[1].scalar)->value;

    //std::cout << "query\t" << query_offsets[0] << "\t" << query_count << std::endl;
    //std::cout << "query_count: " << query_count << std::endl;

    // A list of lists is a little more convoluted to work with. They
    // use a poorly documented class called ArraySpan to pass the data
    // in and out. The offsets are in the top-level ArraySpan and the
    // data is in the child ArraySpan. Why? I dunno. The ArraySpan
    // seems to have 3 buffers, but only buffer(1) seems to contain
    // any useful data. Why? I dunno.
    //
    // See arrow/cpp/src/arrow/compute/kernels/scalar_nested.cc for
    // more examples of using ArraySpan.
    const arrow::ArraySpan& arr = batch[2].array;
    const arrow::ArraySpan& arr_values = arr.child_data[0];
    const int64_t* offsets = arr.GetValues<int64_t>(1);
    const int32_t* arr_count = batch[3].array.GetValues<int32_t>(1);

    // Get the output array so we may place our results there.
    arrow::ArraySpan* out_arr = out->array_span();
    auto out_values = out_arr->GetValues<float>(1);

    // Iterate over our subset of rows.
    for (int64_t i = 0; i < arr.length; ++i) {
        // List of lists are handled with offsets into one giant list. Why
        // are the offsets stored the top level ArraySpan, and the actual
        // data in the child_data member? I still have no idea!
        int32_t data_length = (offsets[i + 1] - offsets[i]) / 8;
        //std::cout << offsets[i] << std::endl;
        const uint8_t* data = arr_values.GetValues<uint8_t>(1, offsets[i]);
        const uint64_t* data64 = reinterpret_cast<const uint64_t*>(data);

        // Compute the tanimoto score.
        int32_t	sum_and(0);
        int32_t sum_or(0);
        for (int64_t j = 0; j < data_length; ++j) {
            //std::cout << data64[j] << " ";
            sum_and += my_popcount(query64[j] & data64[j]);
            sum_or += my_popcount(query64[j] | data64[j]);
        }
        //std::cout << std::endl;
        *out_values++ = static_cast<float_t>(sum_and) / static_cast<float_t>(sum_or);
        // *out_values++ = static_cast<float_t>(offsets[i + 1] - offsets[i]);
        // std::cout << i << "\t" << static_cast<float_t>(offsets[i + 1] - offsets[i]) << "\t" << arr_count[i] << std::endl;
        //std::cout << i << "\t" << arr_count[i] << std::endl;
    }
  return arrow::Status::OK();
}

const cp::FunctionDoc cosine_score_doc{
    // Summary
    "User-defined-function to usage to calculate the cosine score",
    // Description
    "returns the cosine score for every element for the given query input",
    // Arguments
    { "query_mz", "query_intensity", "mz", "intensity"},
    // Options (optional) 
    "UDFOptions",
    // Are options required?
    false};

arrow::Status CosineScore(cp::KernelContext* ctx,
                               const cp::ExecSpan& batch,
                               cp::ExecResult* out) {

    // Useful for figuring out what exactly is being passed to functions like this
    // for (int64_t n = 0; n < batch.num_values(); ++n) {
    // 	std::cout << "Batch: " << n
    //            << "\tType: " << (batch[n].is_scalar() ? "scalar" : "array") << ", "
    //            << batch.GetTypes()[n] << std::endl;
    // }


  return arrow::Status::OK();
}


// All search UDFs should be placed here for registration
arrow::Status RegisterSearchFunctions(cp::FunctionRegistry* registry) {
    // Prepare the functions for registration
    auto tanimoto_func = std::make_shared<cp::ScalarFunction>("tanimoto", cp::Arity(4, false), tanimoto_func_doc);
    auto cosine_score = std::make_shared<cp::ScalarFunction>("cosine_score", cp::Arity(4, false), cosine_score_doc);

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
            cp::InputType(arrow::Type::LARGE_LIST),
            cp::InputType(arrow::Type::LARGE_LIST)},
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

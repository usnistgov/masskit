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

class CosineScoreOptionsType : public cp::FunctionOptionsType {
    const char* type_name() const override { return "CosineScoreOptionsType"; }
    std::string Stringify(const cp::FunctionOptions&) const override {
        return "CosineScoreOptionsType";
    }
    bool Compare(const cp::FunctionOptions&, const cp::FunctionOptions&) const override {
        return true;
    }
    std::unique_ptr<cp::FunctionOptions> Copy(const cp::FunctionOptions&) const override;
    // optional: support for serialization
    // Result<std::shared_ptr<Buffer>> Serialize(const FunctionOptions&) const override;
    // Result<std::unique_ptr<FunctionOptions>> Deserialize(const Buffer&) const override;
};

cp::FunctionOptionsType* GetCosineScoreOptionsType() {
    static CosineScoreOptionsType options_type;
    return &options_type;
}

std::unique_ptr<cp::FunctionOptions> CosineScoreOptionsType::Copy(
    const cp::FunctionOptions&) const {
        return std::make_unique<CosineScoreOptions>();
}


CosineScoreOptions::CosineScoreOptions( float mz_power,
                                 float intensity_power,
                                 int scale,
                                 bool skip_denom,
                                 TieBreaker tiebreaker )
        : cp::FunctionOptions(GetCosineScoreOptionsType()),
          mz_power(mz_power),
          intensity_power(intensity_power),
          scale(scale),
          skip_denom(skip_denom),
          tiebreaker(tiebreaker) {}

const cp::FunctionDoc cosine_score_doc{
    // Summary
    "User-defined-function to calculate the cosine score",
    // Description
    "returns the cosine score between the query and elements in the reference set",
    // Arguments
    { "query_mz", "query_intensity", "query_massinfo", "reference_mz", "reference_intensity", "reference_massinfo", "mask"},
    // Options (optional) 
    "CosineScoreOptions",
    // Are options required?
    false};

enum Tolerance_Type { NONE, PPM, DALTONS};
typedef std::pair<double, double> tInterval;
typedef std::pair<int32_t, int32_t> tMatch;
bool overlap(tInterval &a, tInterval &b) {
    // overlap if max(L1, L2) <= min(R1, R2)}
    // Non-overlapping if max(L1, L2) > min(R1, R2)}
    return (std::max(a.first, b.first) <= std::min(a.second, b.second));
};

void compute_weight(const double *mz,
                    const double *intensity,
                    int32_t length,
                    double mz_power,
                    double intensity_power, 
                    std::vector<double> &weight) {
    for (int32_t i=0; i<length; i++) {
        double val = std::pow(intensity[i],intensity_power);
        if (mz_power > 0) val *= std::pow(mz[i],mz_power);
        weight.push_back(val);
    }
}

struct CSpectrum {
    // std::vector<double> m_mz;
    // std::vector<double> m_intensity;
    const double *m_mz;
    const double *m_intensity;
    int64_t m_length;
    std::vector<tInterval> m_intervals; 
    double m_tolerance;
    Tolerance_Type m_toltype;

    // CSpectrum(std::shared_ptr<arrow::DoubleArray> mzArr,
    //           std::shared_ptr<arrow::DoubleArray> intensityArr,
    //           double m_tolerance, Tolerance_Type type) 
    //           : m_tolerance(m_tolerance), m_toltype(type) {
    //     const double *mzbuf = mzArr->raw_values();
    //     const double *intbuf = intensityArr->raw_values();
    //     // Strip out ions with zero intensities
    //     for (auto i = 0; i < mzArr->length(); i++) {
    //         if ( intbuf[i] > 0) {
    //             m_mz.push_back(mzbuf[i]);
    //             m_intensity.push_back(intbuf[i]);
    //         }
    //     }
    //     this->create_intervals();
    // }

    CSpectrum(const double *mzArr, const double *intensityArr, int64_t length,
              double m_tolerance, Tolerance_Type type) 
              : m_mz(mzArr), m_intensity(intensityArr), m_length(length), 
                m_tolerance(m_tolerance), m_toltype(type) {
        // Strip out ions with zero intensities
        // for (auto i = 0; i < length; i++) {
        //     if ( intensityArr[i] > 0) {
        //         m_mz.push_back(mzArr[i]);
        //         m_intensity.push_back(intensityArr[i]);
        //     }
        // }
        this->create_intervals();
    }

    float cosine_score(CSpectrum &other) {
        std::vector<tMatch> intersections;
        this->intersect(other, intersections);

        //if ((m_length > 2) && (other.m_length > 2) && (intersections.size() < 2))
        if ((std::min(m_length, other.m_length) > 2) && (intersections.size() < 2))
            return -42.0;

        double mz_power = 0.0;
        double intensity_power = 0.5;
        double scale = 999;
        bool skip_denom = false;

        std::vector<double> weight_spec1;
        std::vector<double> weight_spec2;

        compute_weight(m_mz, m_intensity, m_length, mz_power, intensity_power, weight_spec1);
        compute_weight(other.m_mz, other.m_intensity, other.m_length, mz_power, intensity_power, weight_spec2);

        double sum = 0;
        for (tMatch match : intersections) {
            sum += (weight_spec1[match.first] * weight_spec2[match.second]);
        }
        double score = std::pow(sum,2);

        if (!skip_denom) {
            double d1 = 0.0;
            for (int32_t i = 0; i < weight_spec1.size(); i++) {
                d1 += std::pow(weight_spec1[i],2);
            }
            double d2 = 0.0;
            for (int32_t j = 0; j < weight_spec2.size(); j++) {
                d2 += std::pow(weight_spec2[j],2);
            }
            score /= (d1 * d2);
        }

        return static_cast<float>(score * scale);
    }

    void intersect(CSpectrum &other, std::vector<tMatch> &intersections) {

        int32_t curIdx = 0;
        int32_t othMinIdx = 0;
        int32_t othCurIdx = 0;
        intersections.clear();

        while (curIdx < m_intervals.size()) {
            othCurIdx = othMinIdx;
            while ((curIdx < m_intervals.size()) &&
                   (othCurIdx < other.m_intervals.size()) &&
                   (m_intervals[curIdx].first <= other.m_intervals[othCurIdx].second)) {
                if (overlap(m_intervals[curIdx], other.m_intervals[othCurIdx])) {
                    intersections.push_back(std::make_pair(curIdx, othCurIdx));
                }
                ++othCurIdx;
            }
            ++curIdx;
            if (curIdx < m_intervals.size()) {
                while ((othMinIdx < other.m_intervals.size()) &&
                       (m_intervals[curIdx].first > other.m_intervals[othMinIdx].second)) {
                    ++othMinIdx;
                }
            }
        }
    }

    double half_tolerance(double mz) {
        switch (this->m_toltype) {
            case NONE: 
                return 0;
            case PPM:
                return mz * this->m_tolerance / 1000000.0;
            case DALTONS:
                return this->m_tolerance;
        }
        return 0;
    }

    void create_intervals() {
        m_intervals.clear();
        // for (auto mz : m_mz) {
        for (int32_t i=0; i<m_length; i++) {
            double hTol = this->half_tolerance(m_mz[i]);
            m_intervals.push_back(std::make_pair(m_mz[i] - hTol, m_mz[i] + hTol));            
        }
    }

};

arrow::Status CosineScore(cp::KernelContext* ctx,
                          const cp::ExecSpan& batch,
                          cp::ExecResult* out) {
    // Options
    //auto opt = std::static_pointer_cast<cp::ScalarKernel>(ctx->state())->options;
    
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

    CSpectrum query(query_mz->raw_values(), query_intensity->raw_values(),
                    query_mz->length(), query_tol->value, PPM);

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

    auto mask_arr = batch[6].array;
    auto mask = mask_arr.GetValues<int8_t>(1);
    //auto mask = std::static_pointer_cast<arrow::BooleanArray>(batch[6].array.ToArray());

    // Output array for results
    auto out_values = out->array_span()->GetValues<float>(1);

    for (int64_t i = 0; i < batch.length; i++) {
        //if (mask->GetScalar(i).ValueOrDie()) {
        if (mask[i]) {
            int64_t mzlength = (ref_mz_offsets[i + 1] - ref_mz_offsets[i]);
            int64_t intlength = (ref_intensity_offsets[i + 1] - ref_intensity_offsets[i]);
            if (mzlength != intlength) return arrow::Status::Invalid("Spectrum mz and intensity array lengths do not match");
            const double* mzdata = ref_mz_data.GetValues<double>(1, ref_mz_offsets[i]);
            const double* intensitydata = ref_intensity_data.GetValues<double>(1, ref_intensity_offsets[i]);

            CSpectrum reference(mzdata, intensitydata, mzlength, tol_idx[i], PPM);

            *out_values++ = query.cosine_score(reference);
        } else {
            *out_values++ = -1.0;
        }
    }

    return arrow::Status::OK();
}


// All search UDFs should be placed here for registration
arrow::Status RegisterSearchFunctions(cp::FunctionRegistry* registry) {
    // Prepare the functions for registration
    auto tanimoto_func = std::make_shared<cp::ScalarFunction>("tanimoto", cp::Arity(4, false), tanimoto_func_doc);
    auto cosine_score = std::make_shared<cp::ScalarFunction>("cosine_score", cp::Arity(7, false), cosine_score_doc);

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
            cp::InputType(arrow::Type::STRUCT),
            cp::InputType(arrow::Type::INT8)},
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

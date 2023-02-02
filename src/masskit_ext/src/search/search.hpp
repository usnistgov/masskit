#include <arrow/api.h>
#include <arrow/compute/api.h>

namespace cp = ::arrow::compute;

enum class TieBreaker {
    MZ,
    INTENSITY,
    NONE
};

// class CosineScoreOptionsType : public cp::FunctionOptionsType {
//   const char* type_name() const override { return "CosineScoreOptionsType"; }
//   std::string Stringify(const cp::FunctionOptions&) const override {
//     return "CosineScoreOptionsType";
//   }
//   bool Compare(const cp::FunctionOptions&, const cp::FunctionOptions&) const override {
//     return true;
//   }
//   std::unique_ptr<cp::FunctionOptions> Copy(const cp::FunctionOptions&) const override;
//   // optional: support for serialization
//   // Result<std::shared_ptr<Buffer>> Serialize(const FunctionOptions&) const override;
//   // Result<std::unique_ptr<FunctionOptions>> Deserialize(const Buffer&) const override;
// };

// cp::FunctionOptionsType* GetCosineScoreOptionsType();

// cp::FunctionOptionsType* GetCosineScoreOptionsType() {
//   static CosineScoreOptionsType options_type;
//   return &options_type;
// }


class  CosineScoreOptions : public cp::FunctionOptions {
    public:
    explicit CosineScoreOptions( float mz_power=0.0,
                                 float intensity_power=0.5,
                                 int scale=999,
                                 bool skip_denom=false,
                                 TieBreaker tiebreaker=TieBreaker::NONE );
        // : cp::FunctionOptions(GetCosineScoreOptionsType()),
        //   mz_power(mz_power),
        //   intensity_power(intensity_power),
        //   scale(scale),
        //   skip_denom(skip_denom),
        //   tiebreaker(tiebreaker) {};

    static constexpr char const kTypeName[] = "CosineScoreOptions";
    static CosineScoreOptions Defaults() { return CosineScoreOptions(); }

    float mz_power;
    float intensity_power;
    int scale;
    bool skip_denom=false;
    TieBreaker tiebreaker;
};


arrow::Status RegisterSearchFunctions(cp::FunctionRegistry* registry);
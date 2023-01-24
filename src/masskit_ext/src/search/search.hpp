#include <arrow/api.h>
#include <arrow/compute/api.h>

namespace cp = ::arrow::compute;

enum class TieBreaker {
    MZ,
    INTENSITY,
    NONE
};

class  CosineScoreOptions : public cp::FunctionOptions {
 public:
  explicit CosineScoreOptions(  float mz_power=0.0,
                                float intensity_power=0.5,
                                int scale=999,
                                bool skip_denom=false,
                                TieBreaker tiebreaker=TieBreaker::NONE );
  static constexpr char const kTypeName[] = "CosineScoreOptions";
  static CosineScoreOptions Defaults() { return CosineScoreOptions(); }

  /// Sorting order
  //SortOrder order;
  /// Whether nulls and NaNs are placed at the start or at the end
  //NullPlacement null_placement;
};


arrow::Status RegisterSearchFunctions(cp::FunctionRegistry* registry);
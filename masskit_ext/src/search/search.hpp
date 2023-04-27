#include <arrow/api.h>
#include <arrow/compute/api.h>

namespace cp = ::arrow::compute;

template <typename C, typename T>
struct DataMemberProperty {
  using Class = C;
  using Type = T;

  constexpr const Type& get(const Class& obj) const { return obj.*ptr_; }

  void set(Class* obj, Type value) const { (*obj).*ptr_ = std::move(value); }

  constexpr std::string_view name() const { return name_; }

  std::string_view name_;
  Type Class::*ptr_;
};

template <typename Class, typename Type>
constexpr DataMemberProperty<Class, Type> DataMember(std::string_view name,
                                                     Type Class::*ptr) {
  return {name, ptr};
}


/// KernelState adapter for the common case of kernels whose only
/// state is an instance of a subclass of FunctionOptions.
/// Default FunctionOptions are *not* handled here.
template <typename OptionsType>
struct MyOptionsWrapper : public cp::KernelState {
    explicit MyOptionsWrapper(OptionsType options) : options(std::move(options)) {}

    static arrow::Result<std::unique_ptr<cp::KernelState>> Init(cp::KernelContext* ctx,
                                                         const cp::KernelInitArgs& args) {
        if (auto options = static_cast<const OptionsType*>(args.options)) {
            return std::make_unique<MyOptionsWrapper>(*options);
        }

        return arrow::Status::Invalid(
            "Attempted to initialize KernelState from null FunctionOptions");
    }

    static const OptionsType& Get(const cp::KernelState& state) {
        //return ::arrow::internal::checked_cast<const MyOptionsWrapper&>(state).options;
        return static_cast<const MyOptionsWrapper&>(state).options;
    }

    static const OptionsType& Get(cp::KernelContext* ctx) { return Get(*ctx->state()); }

    OptionsType options;
};

enum class TieBreaker {
    MZ,
    INTENSITY,
    NONE
};

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

cp::FunctionOptionsType* GetCosineScoreOptionsType();

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
                                 TieBreaker tiebreaker=TieBreaker::NONE )
        : cp::FunctionOptions(GetCosineScoreOptionsType()),
          mz_power(mz_power),
          intensity_power(intensity_power),
          scale(scale),
          skip_denom(skip_denom),
          tiebreaker(tiebreaker) {};

    static constexpr char const kTypeName[] = "CosineScoreOptions";
    static CosineScoreOptions Defaults() { return CosineScoreOptions(); }

    float mz_power;
    float intensity_power;
    int scale;
    bool skip_denom=false;
    TieBreaker tiebreaker;
};


arrow::Status RegisterSearchFunctions(cp::FunctionRegistry* registry);
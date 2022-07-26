#pragma once

// #include <arrow/array/array_primitive.h>

namespace pybind11 {
    namespace detail {
        template <> struct type_caster< std::shared_ptr<arrow::Table> > {
        public:
            // this doesn't work: PYBIND11_TYPE_CASTER(std::shared_ptr<ArrayType>, _(ArrayType::TypeClass::type_name()));
            PYBIND11_TYPE_CASTER(std::shared_ptr<arrow::Table>, _("pyarrow::lib::Table"));
            // Python -> C++
            bool load(handle src, bool) {
                PyObject *source = src.ptr();
                arrow::Result< std::shared_ptr<arrow::Table> > result = arrow::py::unwrap_table(source);
                if(!result.ok())
                    return false;
                //value = std::static_pointer_cast< std::shared_ptr<arrow::ChunkedArray> >(result.ValueOrDie());
                value = result.ValueOrDie();
                return true;
            }
            // C++ -> Python
            static handle cast(std::shared_ptr<arrow::Table> src, return_value_policy /* policy */, handle /* parent */) {
                return arrow::py::wrap_table(src);
            }
        };

        template <> struct type_caster< std::shared_ptr<arrow::ChunkedArray> > {
        public:
            // this doesn't work: PYBIND11_TYPE_CASTER(std::shared_ptr<ArrayType>, _(ArrayType::TypeClass::type_name()));
            PYBIND11_TYPE_CASTER(std::shared_ptr<arrow::ChunkedArray>, _("pyarrow::lib::ChunkedArray"));
            // Python -> C++
            bool load(handle src, bool) {
                PyObject *source = src.ptr();
                arrow::Result< std::shared_ptr<arrow::ChunkedArray> > result = arrow::py::unwrap_chunked_array(source);
                if(!result.ok())
                    return false;
                //value = std::static_pointer_cast< std::shared_ptr<arrow::ChunkedArray> >(result.ValueOrDie());
                value = result.ValueOrDie();
                return true;
            }
            // C++ -> Python
            static handle cast(std::shared_ptr<arrow::ChunkedArray> src, return_value_policy /* policy */, handle /* parent */) {
                return arrow::py::wrap_chunked_array(src);
            }
        };

        template <> struct type_caster< std::shared_ptr<arrow::Scalar> > {
        public:
            // this doesn't work: PYBIND11_TYPE_CASTER(std::shared_ptr<ArrayType>, _(ArrayType::TypeClass::type_name()));
            PYBIND11_TYPE_CASTER(std::shared_ptr<arrow::Scalar>, _("pyarrow::lib::Scalar"));
            // Python -> C++
            bool load(handle src, bool) {
                PyObject* source = src.ptr();
                arrow::Result< std::shared_ptr<arrow::Scalar> > result = arrow::py::unwrap_scalar(source);
                if (!result.ok())
                    return false;
                //value = std::static_pointer_cast< std::shared_ptr<arrow::ChunkedArray> >(result.ValueOrDie());
                value = result.ValueOrDie();
                return true;
            }
            // C++ -> Python
            static handle cast(std::shared_ptr<arrow::Scalar> src, return_value_policy /* policy */, handle /* parent */) {
                return arrow::py::wrap_scalar(src);
            }
        };

        template <typename ArrayType> struct gen_type_caster {
        public:
            // this doesn't work: PYBIND11_TYPE_CASTER(std::shared_ptr<ArrayType>, _(ArrayType::TypeClass::type_name()));
            PYBIND11_TYPE_CASTER(std::shared_ptr<ArrayType>, _("pyarrow::Array"));
            // Python -> C++
            bool load(handle src, bool) {
                PyObject *source = src.ptr();
                if (!arrow::py::is_array(source))
                    return false;
                arrow::Result<std::shared_ptr<arrow::Array>> result = arrow::py::unwrap_array(source);
                if(!result.ok())
                    return false;
                value = std::static_pointer_cast<ArrayType>(result.ValueOrDie());
                return true;
            }
            // C++ -> Python
            static handle cast(std::shared_ptr<ArrayType> src, return_value_policy /* policy */, handle /* parent */) {
                return arrow::py::wrap_array(src);
            }
        };

        template <>
        struct type_caster<std::shared_ptr<arrow::DoubleArray>> : public gen_type_caster<arrow::DoubleArray> {};

    }

} // namespace pybind11::detail

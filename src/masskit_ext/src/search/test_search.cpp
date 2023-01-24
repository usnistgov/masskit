#include <arrow/io/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/file_parquet.h>
#include <arrow/filesystem/filesystem.h>
#include <iostream>

#include "search.hpp"

namespace ds = arrow::dataset;
namespace fs = arrow::fs;

// Not using <parquet/arrow/reader.h> because of the following error message
//
// terminate called after throwing an instance of 'parquet::ParquetStatusException'
//   what():  NotImplemented: Nested data conversions not implemented for chunked array outputs
// Aborted (core dumped)
//
// Apparently Python currently uses the arrow::dataset method as well,
// otherwise, how else would it work?  See:
// https://issues.apache.org/jira/browse/ARROW-10958
//
// This is a problem, because: 
//
//     "Dataset is currently unstable. APIs subject to change without notice."
//
arrow::Result<std::shared_ptr<arrow::Table>> read_data(std::string filename) {
    std::string root_path;
    ARROW_ASSIGN_OR_RAISE(auto fs, fs::FileSystemFromUriOrPath(filename, &root_path));
    //std::cout << root_path << std::endl;
    std::shared_ptr<ds::FileFormat> format = std::make_shared<ds::ParquetFileFormat>();
    ARROW_ASSIGN_OR_RAISE(
                          auto factory, ds::FileSystemDatasetFactory::Make(fs, 
                                                                           { filename }, 
                                                                           format,
                                                                           ds::FileSystemFactoryOptions()));
    ARROW_ASSIGN_OR_RAISE(auto dataset, factory->Finish());
    // Read specified columns with a row filter
    ARROW_ASSIGN_OR_RAISE(auto scan_builder, dataset->NewScan());
    ARROW_RETURN_NOT_OK(scan_builder->Project({ "spectrum_fp", "spectrum_fp_count"}));
    ARROW_ASSIGN_OR_RAISE(auto scanner, scan_builder->Finish());
    return scanner->ToTable();
    //return scanner->Head(50);
}

arrow::Status Execute(const std::string FILENAME) {
    // Adding compute functions to the central registry is a runtime
    // operation, even arrow does this for itself. At some point we'll
    // have a single function that calls all of the sub-registry
    // functions to ensure that the masskit compute functions are
    // available to all.
    auto registry = cp::GetFunctionRegistry();
    ARROW_RETURN_NOT_OK(RegisterSearchFunctions(registry));
  
    // Load the two columns we need from the given parquet file.
    //const std::string FILENAME = "/home/slottad/nist/data/hr_msms_nist.parquet";
    std::shared_ptr<arrow::Table> table;
    ARROW_ASSIGN_OR_RAISE(table, read_data(FILENAME));
    std::cout << "Loaded " << table->num_rows() << " rows in " << table->num_columns() << " columns." << std::endl;

    // The fingerprints to be searched
    auto spec = table->GetColumnByName("spectrum_fp");
    auto count = table->GetColumnByName("spectrum_fp_count");

    // The query fingerprint conveniently location in the first position
    // of our array to be searched. I wonder if it will match anything?
    ARROW_ASSIGN_OR_RAISE(auto query, spec->GetScalar(0));
    ARROW_ASSIGN_OR_RAISE(auto query_count, count->GetScalar(0));

    // The parameters for the compute function are given in the first
    // array. Since there are a mixture of scalars and arrays, we provide
    // the size of the return array. 
    cp::ExecBatch batch( {
            arrow::Datum(query),
            arrow::Datum(query_count),
            arrow::Datum(spec),
            arrow::Datum(count)
        },
        table->num_rows() );

    // Use the convenience function we apply the tanimoto UDF to the
    // previously batched data
    ARROW_ASSIGN_OR_RAISE(auto tanimoto_results, cp::CallFunction("tanimoto", batch));

    // The type of array returned should match the input arrays
    // auto tanimoto_results_array = tanimoto_results.make_array(); // segfaults
    auto tanimoto_results_array = tanimoto_results.chunked_array();
    std::cout << "Tanimoto Result:" << std::endl << tanimoto_results_array->ToString() << std::endl;

    // In the future, we might use the streaming execution engine,
    // once it is less experimental and more stable. However, our data
    // is small enough that memory is mostly not a problem.
    //
    // For now, we will just run one computation after another.

    cp::ArraySortOptions sort_options(cp::SortOrder::Descending);
    ARROW_ASSIGN_OR_RAISE(auto sort_results, cp::CallFunction("array_sort_indices", {tanimoto_results}, &sort_options));
    auto sort_results_array = sort_results.chunked_array();
    std::cout << "Sort Result:" << std::endl << sort_results_array->ToString() << std::endl;
    
    std::shared_ptr<arrow::Scalar> filter_value = arrow::MakeScalar(0.1);
    ARROW_ASSIGN_OR_RAISE(auto gte_results, cp::CallFunction("greater_equal", {tanimoto_results, filter_value}));
    auto gte_results_array = gte_results.chunked_array();
    std::cout << "GTE Result:" << std::endl << gte_results_array->ToString() << std::endl;
    
    return arrow::Status::OK();
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << argv[0] << ": missing parquet input file!\n";
        std::cerr << "\tusage: " << argv[0] << " <input_file> ...\n";
        return EXIT_FAILURE;
    }

    auto status = Execute(std::string(argv[1]));
    if (!status.ok()) {
        std::cerr << "Error occurred : " << status.message() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

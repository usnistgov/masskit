#include <arrow/io/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/file_parquet.h>
#include <arrow/filesystem/filesystem.h>
#include <iostream>
#include <chrono>

#include "search.hpp"

namespace ds = arrow::dataset;
namespace fs = arrow::fs;

class Timer
{
public:
    void start() {
        m_StartTime = std::chrono::system_clock::now();
        m_bRunning = true;
    }
    
    void stop() {
        m_EndTime = std::chrono::system_clock::now();
        m_bRunning = false;
    }
    
    double elapsedMilliseconds() {
        std::chrono::time_point<std::chrono::system_clock> endTime;
        
        if(m_bRunning) {
            endTime = std::chrono::system_clock::now();
        }
        else {
            endTime = m_EndTime;
        }
        
        return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count();
    }
    
    double elapsedSeconds() {
        return elapsedMilliseconds() / 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_StartTime;
    std::chrono::time_point<std::chrono::system_clock> m_EndTime;
    bool m_bRunning = false;
};

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
    ARROW_RETURN_NOT_OK(scan_builder->Project({ 
        "id",
        "precursor_mz",
        "product_massinfo",
        "mz",
        "intensity",
        "spectrum_fp",
        "spectrum_fp_count"
        }));
    ARROW_ASSIGN_OR_RAISE(auto scanner, scan_builder->Finish());
    return scanner->ToTable();
    //return scanner->Head(50);
}

arrow::Status initialize(const std::string FILENAME, std::shared_ptr<arrow::Table> &table) {
    // Adding compute functions to the central registry is a runtime
    // operation, even arrow does this for itself. At some point we'll
    // have a single function that calls all of the sub-registry
    // functions to ensure that the masskit compute functions are
    // available to all.
    auto registry = cp::GetFunctionRegistry();
    ARROW_RETURN_NOT_OK(RegisterSearchFunctions(registry));
  
    // Load the columns we need from the given parquet file.
    //const std::string FILENAME = "/home/slottad/nist/data/hr_msms_nist.parquet";
    auto result = read_data(FILENAME);
    //std::shared_ptr<arrow::Table> table;
    ARROW_ASSIGN_OR_RAISE(table, result);
    std::cout << "Loaded " << table->num_rows() << " rows in " << table->num_columns() << " columns." << std::endl;

    return arrow::Status::OK();
}

arrow::Status run_cosine_score(std::shared_ptr<arrow::Table> table) {

    double ppm = 20;

    auto precursor_mz = table->GetColumnByName("precursor_mz");
    ARROW_ASSIGN_OR_RAISE(auto query_precursor_mz, precursor_mz->GetScalar(0));
    double qPrecursorMZ = (std::static_pointer_cast<arrow::DoubleScalar>(query_precursor_mz))->value;
    double tol = qPrecursorMZ * ppm / 1000000.0;
    arrow::Datum maxMZ = arrow::DoubleScalar(qPrecursorMZ + tol);
    arrow::Datum minMZ = arrow::DoubleScalar(qPrecursorMZ - tol);
    //auto minMZ = arrow::DoubleScalar(qPrecursorMZ - tol);
    
    ARROW_ASSIGN_OR_RAISE(auto minDatum, arrow::compute::CallFunction("greater_equal",{precursor_mz, minMZ}));
    ARROW_ASSIGN_OR_RAISE(auto maxDatum, arrow::compute::CallFunction("less_equal",{precursor_mz, maxMZ}));
    ARROW_ASSIGN_OR_RAISE(auto precursorWindow, arrow::compute::And(minMZ, maxMZ));
    
    // The fingerprints to be searched
    auto mz = table->GetColumnByName("mz");
    auto intensity = table->GetColumnByName("intensity");
    auto massinfo = table->GetColumnByName("product_massinfo");

    // The query fingerprint conveniently location in the first position
    // of our array to be searched. I wonder if it will match anything?
    ARROW_ASSIGN_OR_RAISE(auto query_mz, mz->GetScalar(0));
    ARROW_ASSIGN_OR_RAISE(auto query_intensity, intensity->GetScalar(0));
    ARROW_ASSIGN_OR_RAISE(auto query_massinfo, massinfo->GetScalar(0));
    
    CosineScoreOptions cso(0,0.5,999,true,TieBreaker::MZ);

    // The parameters for the compute function are given in the first
    // array. Since there are a mixture of scalars and arrays, we provide
    // the size of the return array. 
    cp::ExecBatch batch( {
            arrow::Datum(query_mz),
            arrow::Datum(query_intensity),
            arrow::Datum(query_massinfo),
            arrow::Datum(mz),
            arrow::Datum(intensity),
            arrow::Datum(massinfo)
        },
        table->num_rows() );

    // Use the convenience function we apply the cosine score UDF to the
    // previously batched data
    ARROW_ASSIGN_OR_RAISE(auto cosine_score_results, cp::CallFunction("cosine_score", batch, &cso));

    // The type of array returned should match the input arrays
    auto cosine_score_results_array = cosine_score_results.chunked_array();
    std::cout << "Cosine Score Result:" << std::endl << cosine_score_results_array->ToString() << std::endl;

    // cp::ArraySortOptions sort_options(cp::SortOrder::Descending);
    // ARROW_ASSIGN_OR_RAISE(auto sort_results, cp::CallFunction("array_sort_indices", {cosine_score_results}, &sort_options));
    // auto sort_results_array = sort_results.chunked_array();
    // std::cout << "Sort Result:" << std::endl << sort_results_array->ToString() << std::endl;

    // cp::CountOptions count_options(cp::CountOptions::CountMode::ALL);
    // ARROW_ASSIGN_OR_RAISE(auto count_results, cp::CallFunction("hash_distinct", {cosine_score_results}, &count_options));
    // auto count_results_array = count_results.array();
    // std::cout << "Count Results:" << std::endl << count_results.ToString() << std::endl;

    return arrow::Status::OK();
}

arrow::Status run_tanimoto(std::shared_ptr<arrow::Table> table) {

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
    std::string filename(argv[1]);
    Timer timer;

    timer.start();
    std::shared_ptr<arrow::Table> table;
    auto status = initialize(filename, table);
    if (!status.ok()) {
        std::cerr << "Error occurred : " << status.message() << std::endl;
        return EXIT_FAILURE;
    }
    timer.stop();
    std::cout << "Time to load data: " << timer.elapsedSeconds() << " seconds.\n";

    timer.start();
    status = run_cosine_score(table);
    if (!status.ok()) {
        std::cerr << "Error occurred : " << status.message() << std::endl;
        return EXIT_FAILURE;
    }
    timer.stop();
    std::cout << "Time to calculate cosine score: " << timer.elapsedSeconds() << " seconds.\n";
    std::cout << "Rate of cosine score: \n";
    std::cout << "\t" << timer.elapsedSeconds()/table->num_rows() << " spectra matches/second.\n";
    std::cout << "\t" << timer.elapsedSeconds()/table->num_rows()*1000.0 << " spectra matches/millisecond.\n";
    std::cout << "\t" << timer.elapsedSeconds()/table->num_rows()*1000000.0 << " spectra matches/microsecond.\n";
    std::cout << "\t" << timer.elapsedSeconds()/table->num_rows()*1000000000.0 << " spectra matches/nanosecond.\n";

    // timer.start();
    // status = run_tanimoto(table);
    // if (!status.ok()) {
    //     std::cerr << "Error occurred : " << status.message() << std::endl;
    //     return EXIT_FAILURE;
    // }
    // timer.stop();
    // std::cout << "Time to run tanimoto fingerprint search: " << timer.elapsedSeconds() << " seconds.\n";

    return EXIT_SUCCESS;
}

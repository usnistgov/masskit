#include <arrow/io/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/file_parquet.h>
#include <arrow/filesystem/filesystem.h>
#include <iostream>
#include <chrono>
#include <vector>

#include "search.hpp"

namespace ds = arrow::dataset;
namespace fs = arrow::fs;

const int64_t TEST_SIZE = 10000;
const int64_t TOPN_HITS = 10;

// Global variables as a last minute hack, don't tell anyone that you saw this!
int64_t first_matches = 0;
int64_t topn_matches = 0;

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
        } else {
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
        //"spectrum_fp",
        //"spectrum_fp_count",
        "peptide",
        "mod_names",
        "mod_positions"
        }));
    ARROW_ASSIGN_OR_RAISE(auto scanner, scan_builder->Finish());
    return scanner->ToTable();
    //return scanner->Head(50);
}

arrow::Status initialize() {
    // Adding compute functions to the central registry is a runtime
    // operation, even arrow does this for itself. At some point we'll
    // have a single function that calls all of the sub-registry
    // functions to ensure that the masskit compute functions are
    // available to all.
    auto registry = cp::GetFunctionRegistry();
    ARROW_RETURN_NOT_OK(RegisterSearchFunctions(registry));
    return arrow::Status::OK();
}

arrow::Status load_db(const std::string FILENAME, std::shared_ptr<arrow::Table> &table) {
    // Load the columns we need from the given parquet file.
    auto result = read_data(FILENAME);
    ARROW_ASSIGN_OR_RAISE(table, result);
    std::cout << "Loaded " << table->num_rows() << " rows in " << table->num_columns() << " columns." << std::endl;

     return arrow::Status::OK();
}

arrow::Status load_db(std::vector<std::string> const& filenames, std::shared_ptr<arrow::Table> &table) {
    std::vector<std::shared_ptr<arrow::Table>> tables;
    for (auto file : filenames) {
        std::cout << file << std::endl;
        std::shared_ptr<arrow::Table> subtable;
        ARROW_RETURN_NOT_OK(load_db(file, subtable));
        tables.push_back(subtable);
    }
    ARROW_ASSIGN_OR_RAISE(table, arrow::ConcatenateTables(tables));
    std::cout << "Complete library loaded " << table->num_rows() << " rows in " << table->num_columns() << " columns." << std::endl;
    return arrow::Status::OK();
}

arrow::Status check_cosine_score(int64_t query_row, 
                               std::shared_ptr<arrow::Table> const& query_table, 
                               std::shared_ptr<arrow::Table> const& library_table,
                               arrow::Datum results) {

    //ARROW_ASSIGN_OR_RAISE(auto query_peptide, query_table->GetColumnByName("peptide")->GetScalar(0));
    auto query_peptide = query_table->GetColumnByName("peptide")->GetScalar(query_row).ValueOrDie()->ToString();

    ARROW_ASSIGN_OR_RAISE(auto selectk_results, cp::SelectKUnstable(results, cp::SelectKOptions::TopKDefault(TOPN_HITS)));
    // std::cout << "Select K Result:" << std::endl << selectk_results->ToString() << std::endl;

    ARROW_ASSIGN_OR_RAISE(auto cosine_scores, cp::Take(results,selectk_results));
    // std::cout << "Top N cosine scores:" << std::endl << cosine_scores.chunked_array()->ToString() << std::endl;

    ARROW_ASSIGN_OR_RAISE(auto matches_datum, cp::Take(arrow::Datum(library_table->GetColumnByName("peptide")), selectk_results));
    //std::shared_ptr<arrow::Array> matches = std::move(matches_datum).make_array();

    // ARROW_ASSIGN_OR_RAISE(auto all_contains, cp::CallFunction("equal", {matches, query_peptide}));
    // ARROW_ASSIGN_OR_RAISE(auto contains, cp::Any(all_contains));
    // //std::cout << "Contains:" << std::endl << contains.scalar_as<arrow::BooleanType>() << std::endl;
    // //contains.scalar_as<arrow::BooleanType>()
    // ARROW_ASSIGN_OR_RAISE(auto contains_bool, contains.scalar());
    // if (contains.scalar() {
    for (int64_t i=0; i < matches_datum.chunked_array()->length(); i++) {
        auto match_peptide = matches_datum.chunked_array()->GetScalar(i).ValueOrDie()->ToString();
        auto cosine_score = cosine_scores.chunked_array()->GetScalar(i).ValueOrDie()->ToString();
        // std::cout << query_peptide << "\t" << match_peptide << "\n";
        if (query_peptide == match_peptide) {
            //std::cout << i << "\tScore: " << cosine_score << "\tQuery: " << query_peptide << "\tMatch: " << match_peptide << std::endl;
            if (i == 0) {
                ++first_matches;
            }
            ++topn_matches;
            break;
        }
    }
    return arrow::Status::OK();
}

arrow::Status run_cosine_score(int64_t query_row, 
                               std::shared_ptr<arrow::Table> const& query_table, 
                               std::shared_ptr<arrow::Table> const& library_table) {

    // Extract the query elements
    ARROW_ASSIGN_OR_RAISE(auto query_precursor_mz, query_table->GetColumnByName("precursor_mz")->GetScalar(query_row));
    ARROW_ASSIGN_OR_RAISE(auto query_mz, query_table->GetColumnByName("mz")->GetScalar(query_row));
    ARROW_ASSIGN_OR_RAISE(auto query_intensity, query_table->GetColumnByName("intensity")->GetScalar(query_row));
    ARROW_ASSIGN_OR_RAISE(auto query_massinfo, query_table->GetColumnByName("product_massinfo")->GetScalar(query_row));

    // Create a mask, essentially a column of boolean values to denote which rows will be scored
    // and which will be skipped. Right now, it is based on the precursor fitting within a given 
    // window. However, more complex criteria may be used in the future.
    double ppm = 20; // Should be a parameter
    auto precursor_mz = library_table->GetColumnByName("precursor_mz");
    double qPrecursorMZ = (std::static_pointer_cast<arrow::DoubleScalar>(query_precursor_mz))->value;
    double tol = qPrecursorMZ * ppm / 1000000.0;
    arrow::Datum maxMZ = arrow::DoubleScalar(qPrecursorMZ + tol);
    arrow::Datum minMZ = arrow::DoubleScalar(qPrecursorMZ - tol);
    ARROW_ASSIGN_OR_RAISE(auto minDatum, arrow::compute::CallFunction("greater_equal",{precursor_mz, minMZ}));
    ARROW_ASSIGN_OR_RAISE(auto maxDatum, arrow::compute::CallFunction("less_equal",{precursor_mz, maxMZ}));
    ARROW_ASSIGN_OR_RAISE(auto precursorMask, arrow::compute::And(minDatum, maxDatum));

    // The library to be searched
    auto mz = library_table->GetColumnByName("mz");
    auto intensity = library_table->GetColumnByName("intensity");
    auto massinfo = library_table->GetColumnByName("product_massinfo");
    
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
            arrow::Datum(massinfo),
            precursorMask
        },
        library_table->num_rows() );

    // To simplify debugging
    // ARROW_RETURN_NOT_OK(arrow::SetCpuThreadPoolCapacity(1));
    // auto cpus_result = arrow::GetCpuThreadPoolCapacity();
    // std::cout << "Arrow CPU count: " << cpus_result << "\n";
    // return arrow::Status::OK();

    // Use the convenience function we apply the cosine score UDF to the
    // previously batched data
    ARROW_ASSIGN_OR_RAISE(auto cosine_score_results, cp::CallFunction("cosine_score", batch, &cso));

    // The type of array returned should match the input arrays
    auto cosine_score_results_array = cosine_score_results.chunked_array();
    //std::cout << "Cosine Score Result:" << std::endl << cosine_score_results_array->ToString() << std::endl;

    ARROW_RETURN_NOT_OK(check_cosine_score(query_row, query_table, library_table, cosine_score_results));
    // ARROW_ASSIGN_OR_RAISE(auto selectk_results, cp::SelectKUnstable(cosine_score_results, cp::SelectKOptions::TopKDefault(5)));
    // std::cout << "Select K Result:" << std::endl << selectk_results->ToString() << std::endl;

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
    // std::cout << "CPU Thread Pool Capacity: " << arrow::GetCpuThreadPoolCapacity() << "\n"; 
    // std::cout << "IO Thread Pool Capacity:  " << arrow::io::GetIOThreadPoolCapacity() << "\n"; 
    // return EXIT_SUCCESS;

    // if (argc < 2) {
    //     std::cerr << argv[0] << ": missing parquet input file!\n";
    //     std::cerr << "\tusage: " << argv[0] << " <input_file> ...\n";
    //     return EXIT_FAILURE;
    // }
    // std::string filename(argv[1]);
    std::string query_file("/home/djs10/asms2023/test_filtered.parquet");

    // std::vector<std::string> library_files{
    //     "/home/djs10/asms2023/library/predicted_cho_uniprot_tryptic_2_0.parquet",
    //     "/home/djs10/asms2023/library/predicted_cho_uniprot_tryptic_2_1.parquet",
    //     "/home/djs10/asms2023/library/predicted_cho_uniprot_tryptic_2_2.parquet",
    //     "/home/djs10/asms2023/library/predicted_cho_uniprot_tryptic_2_3.parquet",
    //     "/home/djs10/asms2023/library/predicted_cho_uniprot_tryptic_2_4.parquet",
    //     "/home/djs10/asms2023/library/predicted_cho_uniprot_tryptic_2_5.parquet"
    // };
    // std::vector<std::string> library_files{
    //     "/home/djs10/asms2023/library/predicted_cho_uniprot_tryptic_2_5.parquet"
    // };
    std::vector<std::string> library_files{
        "/home/djs10/asms2023/library/predicted_cho_uniprot_tryptic_2.parquet"
    };
    // std::vector<std::string> library_files{
    //     "/home/djs10/asms2023/test_filtered.parquet"
    // };

    auto status = initialize();
    if (!status.ok()) {
        std::cerr << "Error occurred : " << status.message() << std::endl;
        return EXIT_FAILURE;
    }

    Timer timer;
    timer.start();
    std::shared_ptr<arrow::Table> query_table;
    status = load_db(query_file, query_table);
    if (!status.ok()) {
        std::cerr << "Error occurred : " << status.message() << std::endl;
        return EXIT_FAILURE;
    }


    timer.stop();
    std::cout << "Time to load query data: " << timer.elapsedSeconds() << " seconds.\n";

    timer.start();
    std::shared_ptr<arrow::Table> search_table;
    status = load_db(library_files, search_table);
    if (!status.ok()) {
        std::cerr << "Error occurred : " << status.message() << std::endl;
        return EXIT_FAILURE;
    }
    timer.stop();
    std::cout << "Time to load library data: " << timer.elapsedSeconds() << " seconds.\n";

    int64_t num_tests;
    if (TEST_SIZE > query_table->num_rows()) {
        num_tests = query_table->num_rows();
    } else {
        num_tests = TEST_SIZE;
    }

    double tps;
    Timer timer2;
    timer2.start();
    for (int64_t i=0; i<num_tests; ++i) {
        timer.start();
        status = run_cosine_score(i, query_table, search_table);
        if (!status.ok()) {
            std::cerr << "Error occurred : " << status.message() << std::endl;
            return EXIT_FAILURE;
        }
        timer.stop();
        std::cout << "Query: " << i << "\t search time: " << timer.elapsedSeconds() << " seconds.\tRate: ";
        tps = search_table->num_rows()/timer.elapsedSeconds();
        // std::cout << tps << " spectra matches/second.\n";
        // std::cout << tps / 1000.0 << " spectra matches/millisecond.\n";
        std::cout << tps / 1000000.0 << " spectra matches/microsecond.\n";
        // std::cout << tps / 1000000000.0 << " spectra matches/nanosecond.\n";
    }
    timer2.stop();

    std::cout << "\n";
    std::cout << "Num queries: " << num_tests << "\n";
    std::cout << "Library Spectra: " << search_table->num_rows() << "\n";
    std::cout << "Total search time: " << timer2.elapsedSeconds() << "\n";
    tps = (num_tests * search_table->num_rows())/timer2.elapsedSeconds();
    std::cout << "Rate: " << tps / 1000000.0 << " spectra matches/microsecond.\n";
    std::cout << "\n";
    std::cout << "First place matches: \t" << first_matches << "\n";
    std::cout << "Top N matches: \t" << topn_matches << "\n";

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

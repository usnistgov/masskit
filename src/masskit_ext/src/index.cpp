#include <iostream>
#include "ext.hpp"

using arrow::DoubleBuilder;
using arrow::ListBuilder;

struct dict_values {
	int8_t ppm;
	int8_t daltons;
	std::shared_ptr<arrow::Int8Array> pIdx;
};

struct dict_values dict_decode(std::shared_ptr<arrow::DictionaryArray> pDictArr) {
	int8_t ppm = -1;
	int8_t daltons = -1;
	auto pIdx = std::static_pointer_cast<arrow::Int8Array>(pDictArr->indices());
	auto pDict = std::static_pointer_cast<arrow::StringArray>(pDictArr->dictionary());
	for (int64_t i=0; i<pDict->length(); i++) {
        if (pDict->Value(i) == "ppm") ppm = i;
        if (pDict->Value(i) == "daltons") daltons = i;
	}
	return {ppm, daltons, pIdx};
}

double half_tolerance(double mz, double tol, bool isPPM) {
	if (isPPM) {
		return (mz * tol / 1000000.0);
	}
	return tol;
}

std::shared_ptr<arrow::Table> calc_start_stops(std::shared_ptr<arrow::Table> table) {
    //std::cout << table->ToString()  << std::endl;
    int64_t num_chunks = table->column(0)->num_chunks();
    auto mz_chunks = table->GetColumnByName("mz");
    auto pmi_chunks = table->GetColumnByName("product_massinfo");
    //std::cout << mz_chunks->length() << pmi_chunks->length() << std::endl;

    arrow::MemoryPool* pool = arrow::default_memory_pool();

    ListBuilder starts_list_builder(pool, std::make_shared<DoubleBuilder>(pool));
	DoubleBuilder* starts_value_builder = (static_cast<DoubleBuilder*>(starts_list_builder.value_builder()));

    ListBuilder stops_list_builder(pool, std::make_shared<DoubleBuilder>(pool));
	DoubleBuilder* stops_value_builder = (static_cast<DoubleBuilder*>(stops_list_builder.value_builder()));

    for (int64_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        auto mz_list = std::static_pointer_cast<arrow::LargeListArray>(mz_chunks->chunk(chunk_id));
        auto mz = std::static_pointer_cast<arrow::DoubleArray>(mz_list->values());
        auto pmi_struct = std::static_pointer_cast<arrow::StructArray>(pmi_chunks->chunk(chunk_id));
        auto tol = std::static_pointer_cast<arrow::DoubleArray>(pmi_struct->GetFieldByName("tolerance"));
        struct dict_values tol_type = dict_decode(
        	std::static_pointer_cast<arrow::DictionaryArray>(pmi_struct->GetFieldByName("tolerance_type")));
        for (int64_t i = 0; i < mz_list->length(); i++) {
            // std::cout << tol->Value(i) << " : ";
            // std::cout << (int)tt_idx->Value(i) << " : ";
            int8_t ttidx = tol_type.pIdx->Value(i);
            bool isPPM = false;
            if ( ttidx == tol_type.ppm) {
            	isPPM = true;
            	// std::cout << "ppm" << " : ";
            } else if (ttidx == tol_type.daltons) {
            	// std::cout << "daltons" << " : ";
            } else {
			    // std::cout << "unknown" << " : ";
            }
            // std::cout.precision(12);	
            double tolval = tol->Value(i);
            starts_list_builder.Append();
            stops_list_builder.Append();
            for (int64_t j = mz_list->value_offset(i); j < mz_list->value_offset(i+1); j++) {
            	double mzval = mz->Value(j);
            	double mod = half_tolerance(mzval, tolval, isPPM);
                // std::cout << "[" << mzval - mod
                // 		  << "," << mzval 
                // 		  << "," << mzval + mod  << "], ";
         		starts_value_builder->Append(mzval - mod);
         		stops_value_builder->Append(mzval + mod);
            }
            // std::cout << std::endl;
        }
    }
    std::shared_ptr<arrow::Array> starts_array;
    starts_list_builder.Finish(&starts_array);
    auto starts_field = arrow::field("starts", arrow::list(arrow::float64()));
    table = table->AddColumn(table->num_columns(), 
    						 starts_field, 
    						 std::make_shared<arrow::ChunkedArray>(arrow::ChunkedArray(starts_array))).ValueOrDie();

    std::shared_ptr<arrow::Array> stops_array;
    stops_list_builder.Finish(&stops_array);
    auto stops_field = arrow::field("stops", arrow::list(arrow::float64()));
    table = table->AddColumn(table->num_columns(), 
    						 stops_field, 
    						 std::make_shared<arrow::ChunkedArray>(arrow::ChunkedArray(stops_array))).ValueOrDie();

	return table;
}

int64_t BruteForceIndex::create(std::shared_ptr<arrow::Table>& source) {
	return source->num_rows();
}

int64_t BruteForceIndex::search(std::shared_ptr<arrow::Table>& query) {
	return query->num_columns();
}

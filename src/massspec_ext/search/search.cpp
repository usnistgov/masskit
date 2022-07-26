#include <iostream>
#include <memory>
#include <cstdint>

#include <search.hpp>

std::shared_ptr<arrow::Table> tanimoto_search(
	std::shared_ptr<arrow::Scalar> query,
	std::shared_ptr<arrow::Scalar> query_count,
	std::shared_ptr<arrow::Table> table) {

	// Transform input types

	// Convert query to c-style int64_t array
	auto pQueryArr = std::static_pointer_cast<arrow::UInt8Array>(
		std::static_pointer_cast<arrow::ListScalar>(query)->value );

	if (pQueryArr->length() % 8 != 0) {
		// todo: raise some error here
		std::cout << "Query array is non-conformant" << std::endl;
		return table;
	}
	int64_t query_length = pQueryArr->length() / 8;
	const uint64_t* raw_qv = reinterpret_cast<const uint64_t*>(pQueryArr->raw_values());

	// Output for debugging
	for (int64_t i = 0; i < query_length; i++) {
		std::cout << raw_qv[i] << " ";
	}
	std::cout << std::endl;

	// Query_count to int32_t
	auto pqc = std::static_pointer_cast<arrow::Int32Scalar>(query_count);
	int32_t qc = pqc->value;

	// Output for debugging
	std::cout << qc << std::endl;



	return table;
}
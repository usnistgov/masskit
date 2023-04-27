#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>

#include <string>
#include <iostream>

#include "search.hpp"

#ifdef linux
  // const std::string FILENAME = "/data/aiomics/massspec_cache/hr_msms_nist2020_v42_0.parquet";
  const std::string FILENAME = "~/nist/data/hr_msms_nist.parquet";
#else
  const std::string FILENAME = "\\nist\\data\\aiomics\\massspec_cache\\hr_msms_nist2020_v42_0.parquet";
#endif

// #0 Build dummy data to pass around
// To have some input data, we first create an Arrow Table that holds
// some data.
std::shared_ptr<arrow::Table> generate_table() {
  arrow::Int64Builder i64builder;
  PARQUET_THROW_NOT_OK(i64builder.AppendValues({1, 2, 3, 4, 5}));
  std::shared_ptr<arrow::Array> i64array;
  PARQUET_THROW_NOT_OK(i64builder.Finish(&i64array));

  arrow::StringBuilder strbuilder;
  PARQUET_THROW_NOT_OK(strbuilder.Append("some"));
  PARQUET_THROW_NOT_OK(strbuilder.Append("string"));
  PARQUET_THROW_NOT_OK(strbuilder.Append("content"));
  PARQUET_THROW_NOT_OK(strbuilder.Append("in"));
  PARQUET_THROW_NOT_OK(strbuilder.Append("rows"));
  std::shared_ptr<arrow::Array> strarray;
  PARQUET_THROW_NOT_OK(strbuilder.Finish(&strarray));

  std::shared_ptr<arrow::Schema> schema = arrow::schema(
      {arrow::field("int", arrow::int64()), arrow::field("str", arrow::utf8())});

  return arrow::Table::Make(schema, {i64array, strarray});
}

// #1 Write out the data as a Parquet file
void write_parquet_file(const arrow::Table& table) {
  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  PARQUET_ASSIGN_OR_THROW(
      outfile, arrow::io::FileOutputStream::Open("parquet-arrow-example.parquet"));
  // The last argument to the function call is the size of the RowGroup in
  // the parquet file. Normally you would choose this to be rather large but
  // for the example, we use a small value to have multiple RowGroups.
  PARQUET_THROW_NOT_OK(
      parquet::arrow::WriteTable(table, arrow::default_memory_pool(), outfile, 3));
}

// #2: Fully read in the file
std::shared_ptr<arrow::Table> read_whole_file(std::string filename) {
  std::cout << "Reading parquet-arrow-example.parquet at once" << std::endl;
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(infile,
                          arrow::io::ReadableFile::Open(filename,
                                                        arrow::default_memory_pool()));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

  std::shared_ptr<::arrow::Schema> schema;
  PARQUET_THROW_NOT_OK(reader->GetSchema(&schema));
  std::vector<int> column_indices;
  column_indices.push_back(schema->GetFieldIndex("spectrum_fp"));
  column_indices.push_back(schema->GetFieldIndex("spectrum_fp_count"));

  std::shared_ptr<arrow::Table> table;
  PARQUET_THROW_NOT_OK(reader->ReadTable(column_indices, &table));
  std::cout << "Loaded " << table->num_rows() << " rows in " << table->num_columns()
            << " columns." << std::endl;
  PARQUET_THROW_NOT_OK(arrow::PrettyPrint(*table, 4, &std::cout));
  return table;
}

// #4: Read only a single column of the whole parquet file
void read_single_column() {
  std::cout << "Reading first column of parquet-arrow-example.parquet" << std::endl;
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(infile,
                          arrow::io::ReadableFile::Open("parquet-arrow-example.parquet",
                                                        arrow::default_memory_pool()));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
  std::shared_ptr<arrow::ChunkedArray> array;
  PARQUET_THROW_NOT_OK(reader->ReadColumn(0, &array));
  PARQUET_THROW_NOT_OK(arrow::PrettyPrint(*array, 4, &std::cout));
  std::cout << std::endl;
  PARQUET_THROW_NOT_OK(reader->ReadColumn(1, &array));
  PARQUET_THROW_NOT_OK(arrow::PrettyPrint(*array, 4, &std::cout));
  std::cout << std::endl;
}

int main(int argc, char** argv) {
    std::shared_ptr<arrow::Table> table = generate_table();
    write_parquet_file(*table);
    std::shared_ptr<arrow::Table> table2 = read_whole_file(FILENAME);
    read_single_column();
}

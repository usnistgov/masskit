#!/usr/bin/env python
import sys
import pyarrow.parquet as pq

if len(sys.argv) < 2:
    print("Need filename")
    sys.exit(-1)


#parquet_file = pq.ParquetFile('hr_msms_nist.parquet')
parquet_file = pq.ParquetFile(sys.argv[1])
metadata = parquet_file.metadata
print(metadata)

#schema = parquet_file.schema_arrow
schema = parquet_file.schema

item_counts = {}
for i in range(metadata.num_columns):
    print(i)
    item_counts[i] = 0
print(item_counts)
print()

for rg in range(metadata.num_row_groups):
    for col in range(metadata.num_columns):
        item_counts[col] += metadata.row_group(rg).column(col).statistics.num_values

print(len(item_counts))
print(len(schema.names))
print(schema.names)
for col in sorted(item_counts, key=item_counts.get, reverse=True):
    print(col, schema.names[col], item_counts[col])
#    ic = item_counts[col]
#    if ic > 0:
#       print(col, schema.field(col).name, ic)

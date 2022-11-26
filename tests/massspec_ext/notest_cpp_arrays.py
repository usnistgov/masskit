#!/bin/env python
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
# import pyarrow.compute as pc
import os
import sys

# import massspec
from massspec.utils.tables import create_dataset
import massspec.spectrum.spectrum as spec
import massspec_ext

FILENAME = 'test_data.parquet'

# Create table
table = create_dataset(rows=5, cols=[list, float, float])
pq.write_table(table, FILENAME)
table = pq.read_table(FILENAME)

print(table.column('a')[2])
# print(help(table.column('a')))
print(massspec_ext.arrow_chunk(table.column('a')))
print(massspec_ext.table_info(table))

s = massspec_ext.BruteForceIndex("george")
print(s.create(table))
print(s.search(table))

sys.exit()

a = table.column('a')
b = table.column('b')
c = table.column('c')
result = []
print(a.num_chunks)
for i in range(a.num_chunks):
    result.append(massspec_ext.arrow_add(a.chunk(i), b.chunk(i), c.chunk(i)))
ca = pa.chunked_array(result)

print(table.append_column("result", ca).to_pandas())

sys.exit()

a = pa.array(table.column('a').to_pandas())
b = pa.array(table.column('b').to_pandas())
c = pa.array(table.column('c').to_pandas())

result = massspec_ext.numpy_add(table.column('a').to_numpy(), table.column('b').to_numpy())
table = table.append_column("result", pa.array(result))
print(table.to_pandas())


df = pd.DataFrame({'one': [-1, np.nan, 2.5, 35, 243, 12],
                   'two': ['foo', 'bar', 'baz', 'fub', 'zab', 'rab'],
                   'three': [True, False, True, False, False, True]})
table = pa.Table.from_pandas(df)
print(table.to_pydict())
pq.write_table(table, FILENAME)

# Read table
only_cols = ["one", "two"]
filter_list = [('three', '=', False)]
new_table = pq.read_table(FILENAME, columns=only_cols, filters=filter_list).combine_chunks()

print(new_table.to_pydict())

p = massspec_ext.Pet('George')
print(p.getName())

# Cleanup
os.remove(FILENAME)

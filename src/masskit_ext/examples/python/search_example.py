#!/bin/env python
import os
from masskit_ext import tanimoto_search
import pyarrow.parquet as pq

data_dir = r'C:\nist\data\aiomics\massspec_cache'
table_name = os.path.join(data_dir,'hr_msms_nist2020_v42_0.parquet')

table = pq.read_table(table_name, columns=["spectrum_fp", "spectrum_fp_count"])

query = table.column("spectrum_fp")[0]
query_count = table.column("spectrum_fp_count")[0]

print(query_count)

tanimoto_search(query, query_count, table)

# tanimoto_search(query, query_count, table, 0.1, predicate=predicate)


"""
# sample search
import pyarrow.parquet as pq
from search import tanimoto_search
import numpy as np

table = pq.read_table('hr_msms_nist2020_v42_0.parquet')
fingerprints_list = table['spectrum_fp'].combine_chunks()
fingerprints = fingerprints_list.values.to_numpy()
fingerprints = np.reshape(fingerprints, (-1, fingerprints_list.offsets[1].as_py()))
predicate[:fingerprints.shape[0]//2] = False
tanimoto_search(fingerprints[0], fingerprints, table['spectrum_fp_count'][0].as_py(), table['spectrum_fp_count'].to_numpy().astype(np.uint64), 0.1, predicate=predicate)
"""


#!/bin/env python
import numpy as np

import masskit.spectra
import masskit.spectra.ions as mkions
import masskit_ext
# import masskit
from masskit.utils.files import spectra_to_array

# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
# import pyarrow.compute as pc
# import os
# import sys


FILENAME = "test.msp"

hi_res1 = masskit.spectra.Spectrum()
hi_res1.from_arrays(
    [100.0001, 200.0002, 300.0003],
    [999, 1, 50],
    row={
        "id": 1,
        "retention_time": 4.5,
        "name": "hi_res1",
        "precursor_mz": 500.5,
    },
    precursor_mz=555.0,
    product_mass_info=mkions.MassInfo(10.0, "ppm", "monoisotopic", "", 1),
)

hi_res2 = masskit.spectra.Spectrum()
hi_res2.from_arrays(
    [100.0002, 200.0062, 500.0, 300.0009],
    [999, 1, 50, 120],
    row={
        "id": 2,
        "retention_time": 4.5,
        "name": "hi_res2",
        "precursor_mz": 500.5,
    },
    product_mass_info=mkions.MassInfo(10.0, "ppm", "monoisotopic", "", 1),
)

dup1 = masskit.spectra.Spectrum().from_arrays(
    np.array([100.0001, 200.0002, 300.0003, 200.0000]),
    np.array([1, 3, 4, 2]),
    row={
        "id": 3,
        "retention_time": 4.5,
        "name": "dup1",
        "precursor_mz": 500.5,
    },
    product_mass_info=mkions.MassInfo(10.0, "ppm", "monoisotopic", "", 1),
    copy_arrays=False)

dup2 = masskit.spectra.Spectrum().from_arrays(
    np.array([100.0001, 100.0001, 200.0, 200.0002, 400.0003]),
    np.array([1, 1, 2, 3, 4]),
    row={
        "id": 4,
        "retention_time": 4.5,
        "name": "dup2",
        "precursor_mz": 500.5,
    },
    product_mass_info=mkions.MassInfo(10.0, "ppm", "monoisotopic", "", 1),
    copy_arrays=False)

spectra = [hi_res1, hi_res2, dup1, dup2]
table = spectra_to_array(spectra)

table = masskit_ext.calc_start_stops(table)
# print(table.column_names)

print(table.to_pandas())

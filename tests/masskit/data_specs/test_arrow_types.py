import pytest
from pytest import approx
import numpy as np
import masskit.spectrum.spectrum as mss
import masskit.data_specs.arrow_types as mda
import pyarrow as pa

def test_arrow_types(SRM1950_lumos_structarray):
    spectrum_type = mda.MolSpectrumArrowType()
    array = pa.ExtensionArray.from_storage(spectrum_type, SRM1950_lumos_structarray)
    spectrum = array[0].as_py()
    assert spectrum.name == 'N-Acetyl-L-alanine'
    
# pd.Series(array, dtype=pd.ArrowDtype(mda.MolSpectrumArrowType()))
# mapit = lambda x: mda.MolSpectrumArrowType() if x ==  mda.MolSpectrumArrowType else x
# tt =pa.Table.from_arrays([xx], names=['spectrum'])
# tt.to_pandas(types_mapper=mapit)
# error due to if not hasattr(pandas_dtype, '__from_arrow__')


import pytest
from pytest import approx
import numpy as np
from masskit.data_specs.schemas import table2structarray, table_add_structarray
import masskit.spectrum.spectrum as mss
import masskit.data_specs.arrow_types as mda
import pyarrow as pa

def test_arrow_types(SRM1950_lumos_table):
    spectrum_type = mda.SpectrumArrowType()
    array = table2structarray(SRM1950_lumos_table, spectrum_type)
    spectrum = array[0].as_py()
    assert spectrum.name == 'N-Acetyl-L-alanine'
    spectrum_list = array.to_pylist()
    assert spectrum_list[0].name == 'N-Acetyl-L-alanine'
    spectrum_array = array.to_numpy()
    assert spectrum_array[0].name == 'N-Acetyl-L-alanine'
    
def test_table2structarray(SRM1950_lumos_table):
    structarray = table2structarray(SRM1950_lumos_table, mda.SpectrumArrowType())
    table = table_add_structarray(SRM1950_lumos_table, structarray)
    mol_array = pa.ExtensionArray.from_storage(mda.MolArrowType(), table['mol'].chunk(0))
    table = table.set_column(table.schema.get_field_index('mol'), 'mol', mol_array)
    assert table['spectrum'][0].as_py().name == 'N-Acetyl-L-alanine'
    df = table.to_pandas()
    assert df['spectrum'].iloc[0].name == 'N-Acetyl-L-alanine'
    assert df['mol'].iat[0].GetProp('NAME')  == 'N-Acetyl-L-alanine'

# pd.Series(array, dtype=pd.ArrowDtype(mda.MolSpectrumArrowType()))
# mapit = lambda x: mda.MolSpectrumArrowType() if x ==  mda.MolSpectrumArrowType else x
# tt =pa.Table.from_arrays([xx], names=['spectrum'])
# tt.to_pandas(types_mapper=mapit)
# error due to if not hasattr(pandas_dtype, '__from_arrow__')


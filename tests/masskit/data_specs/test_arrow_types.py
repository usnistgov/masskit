
def test_arrow_types(SRM1950_lumos_table):
    array = SRM1950_lumos_table['spectrum']
    spectrum = array[0].as_py()
    assert spectrum.name == 'N-Acetyl-L-alanine'
    spectrum_list = array.to_pylist()
    assert spectrum_list[0].name == 'N-Acetyl-L-alanine'
    # to_numpy() only works on Arrays, not ChunkedArray
    spectrum_array = array.chunk(0).to_numpy()
    assert spectrum_array[0].name == 'N-Acetyl-L-alanine'
    
def test_table_to_structarray(SRM1950_lumos_table):
    assert SRM1950_lumos_table['spectrum'][0].as_py().name == 'N-Acetyl-L-alanine'
    df = SRM1950_lumos_table.to_pandas()
    assert df['spectrum'].iloc[0].name == 'N-Acetyl-L-alanine'
    assert df['mol'].iat[0].GetProp('NAME')  == 'N-Acetyl-L-alanine'

# pd.Series(array, dtype=pd.ArrowDtype(mda.MolSpectrumArrowType()))
# mapit = lambda x: mda.MolSpectrumArrowType() if x ==  mda.MolSpectrumArrowType else x
# tt =pa.Table.from_arrays([xx], names=['spectrum'])
# tt.to_pandas(types_mapper=mapit)


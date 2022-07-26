from matplotlib.pyplot import annotate
import pytest
from pytest import approx
import numpy as np
import massspec.spectrum.spectrum as mss
import pyarrow.parquet as pq
from massspec.utils.index import ArrowLibraryMap
from massspec.data_specs.spectral_library import LibraryAccessor

@pytest.fixture
def library_df():
    table = ArrowLibraryMap.from_parquet('libraries/tests/data/cho_uniq_short.parquet')
    return table.to_pandas()

def test_df_to_msp(library_df, tmpdir):
    msp_file = (tmpdir / 'test_spectral_library.msp')
    library_df.lib.to_msp(msp_file.open("w+"), annotate=True)
    assert msp_file.read().startswith("Name: AAAACALTPGPLADLAAR/2_1(4,C,Carbamidomethyl)")

def test_df_to_mgf(library_df, tmpdir):
    msp_file = (tmpdir / 'test_spectral_library.mgf')
    library_df.lib.to_mgf(msp_file.open("w+"))

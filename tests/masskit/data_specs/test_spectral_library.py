from matplotlib.pyplot import annotate
import pytest
from pytest import approx
import numpy as np
import masskit.spectrum.spectrum as mss
import pyarrow.parquet as pq
from masskit.utils.tablemap import ArrowLibraryMap
from masskit.data_specs.spectral_library import LibraryAccessor

@pytest.fixture
def library_df(cho_uniq_short_parquet):
    table = ArrowLibraryMap.from_parquet(cho_uniq_short_parquet)
    return table.to_pandas()

def test_df_to_msp(library_df, tmpdir):
    msp_file = (tmpdir / 'test_spectral_library.msp')
    library_df.lib.to_msp(msp_file.open("w+"), annotate=True)
    assert msp_file.read().startswith("Name: AAAACALTPGPLADLAAR/2_1(4,C,CAM)")# Name: AAAACALTPGPLADLAAR/2_1(4,C,Carbamidomethyl)

def test_df_to_mgf(library_df, tmpdir):
    msp_file = (tmpdir / 'test_spectral_library.mgf')
    library_df.lib.to_mgf(msp_file.open("w+"))

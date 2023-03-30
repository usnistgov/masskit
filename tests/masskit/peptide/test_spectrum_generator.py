import pytest
from pytest import approx
import masskit.peptide.spectrum_generator as msps
import numpy as np

def test_generate_peptide_library():
    df = msps.generate_peptide_library(num=2, min_length=5, max_length=30, min_charge=1, max_charge=8, min_ev=10,
                                    max_ev=60, mod_list='Acetyl,Phospho{S/T0}',
                                    mod_probability=1.0)
    assert 0 in df['mod_positions'].iloc[0] and 1 in df['mod_names'].iloc[0]
    
def test_create_peptide_name():
    output = msps.create_peptide_name("AAS", 3, ['Acetyl', 'Phospho'], np.array([1,2]), 50)
    assert output == 'AAS/3_2(2,A,Acetyl)(3,S,Phospho)_eV50'

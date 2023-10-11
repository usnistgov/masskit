from typing import Any

import numpy as np
import pytest
from pytest import approx

import masskit.spectra as mkspectra
import masskit.spectra.theoretical_spectrum as msts
from masskit.utils.tablemap import ArrowLibraryMap


@pytest.fixture
def peptide_spectrum():

    mz_values = [
        65.5468,
        74.0600,
        78.0626,
        84.0808,
        86.5759,
        88.0393,
        92.0600,
        100.5733,
        101.1073,
        120.0808,
        126.0913,
        129.1022,
        130.0863,
        139.0810,
        147.1128,
        147.5942,
        155.1179,
        161.5618,
        170.0750,
        172.1444,
        175.5592,
        183.1128,
        184.0725,
        196.1024,
        196.5944,
        200.1394,
        205.1077,
        211.0478,
        219.0752,
        227.5885,
        232.5807,
        233.0727,
        235.1077,
        239.0427,
        241.5860,
        245.0921,
        250.5912,
        255.0377,
        263.1026,
        265.0220,
        277.1547,
        279.6016,
        280.0936,
        283.0326,
        288.6069,
        292.6094,
        294.1812,
        235.1077,
        239.0427,
        241.5860,
        245.0921,
        250.5912,
        255.0377,
        263.1026,
        265.0220,
        277.1547,
        279.6016,
        280.0936,
        283.0326,
        288.6069,
        292.6094,
        294.1812,
        301.1227,
        306.1149,
        306.6069,
        315.1202,
        315.1202,
        315.6122,
        322.1162,
        324.1255,
        324.1255,
        326.0748,
        336.0591,
        339.1428,
        350.1112,
        354.0697,
        367.1377,
        379.1677,
        379.6597,
        301.1227,
        306.1149,
        306.6069,
        315.1202,
        315.1202,
        315.6122,
        322.1162,
        324.1255,
        324.1255,
        326.0748,
        336.0591,
        339.1428,
        350.1112,
        354.0697,
        367.1377,
        379.1677,
        379.6597,
        388.1729,
        391.1976,
        392.1816,
        402.1061,
        409.2082,
        412.0904,
        430.1010,
        437.1432,
        454.1697,
        464.1541,
        465.1381,
        473.1432,
        482.1647,
        483.1275,
        500.1752,
        501.1381,
        558.1960,
        559.1800,
        576.2065,
        584.2116,
        601.2382,
        611.2225,
        612.2065,
        629.2331,
        629.2331,
        630.2171,
        647.2436,
        647.2436,
        757.3280,
        758.3120,
        775.3386,
    ]

    spectrum = mkspectra.Spectrum()
    spectrum.from_arrays(
        mz_values,
        [1.0] * len(mz_values),
        row={
            "id": 1234,
            "name": "KAsDFK",
            "peptide": "KASDFK",
            "mod_names": ["Phospho"],
            "mod_positions": [2],
            "precursor_mz": 388.1729,
            "charge": 2,
        },
    )
    return spectrum

@pytest.fixture
def peptide_parent_spectrum():
    """
    parent minus NH3 loss
    """
    mz_values = [
        736.7285,
        ]

    spectrum = mkspectra.Spectrum()
    spectrum.from_arrays(
        mz_values,
        [1.0] * len(mz_values),
        row={
            "id": 1235,
            "name": "QALIQEQEAQIKEQEAQIK",
            "peptide": "QALIQEQEAQIKEQEAQIK",
            "mod_names": [],
            "mod_positions": [],
            "precursor_mz": 388.1729,
            "charge": 3,
        },
    )
    return spectrum


@pytest.fixture
def peptide_exp_spectrum(data_dir):
    arrow_map = ArrowLibraryMap.from_msp(data_dir / "Exp_AENNCLYIEYGINEK_2_1(4,C,Carbamidomethyl)_57eV_NCE40_msp.txt",
                                         spectrum_type='msp_peptide')
    return arrow_map[0]['spectrum']

@pytest.fixture
def peptide_pred_spectrum(data_dir):
    arrow_map = ArrowLibraryMap.from_msp(data_dir / "Predicted_AENNCLYIEYGINEK_2_1(4,C,Carbamidomethyl)_msp.txt",
                                         spectrum_type='msp_peptide')
    return arrow_map[0]['spectrum']

def test_annotate_peptide_exp_spectrum(peptide_exp_spectrum: Any):
    msts.annotate_peptide_spectrum(peptide_exp_spectrum)
    output = peptide_exp_spectrum.to_msp()
    assert(output.find('IEA/1.0ppm') != -1)
    assert(output.find('y1/2.0ppm') != -1)
    assert(output.find('y2-H2O/3.7ppm') != -1)
    assert(output.find('y3+i/-1.7ppm') != -1)
    assert(output.find('Int/YIEYGIN+i/7.2ppm') != -1)
    assert(output.find('p-H2O/0.0ppm') != -1)

def test_annotate_peptide_parent_spectrum(peptide_parent_spectrum: mkspectra.Spectrum):
    msts.annotate_peptide_spectrum(peptide_parent_spectrum)
    output = peptide_parent_spectrum.to_msp()
    # note that there is a collision here between p-H2O+i and p-NH3
    assert(output.find('p-H2O+i/0.4ppm') != -1)

def test_annotate_peptide_spectrum(peptide_spectrum: mkspectra.Spectrum):
    msts.annotate_peptide_spectrum(peptide_spectrum)
    output = peptide_spectrum.to_msp()
    assert(output.find('294.1812\t1\t"y2/-0.1ppm"') != -1)

def test_theoretical_spectrum():
    spectrum = msts.TheoreticalPeptideSpectrum('AAA', ion_types=[('b', 1), ('b', 2), ('y', 1), ('y', 2)], charge=2)
    np.testing.assert_array_almost_equal(spectrum.products.mz, [45.531115,  46.032793,  72.04439 ,  72.546068,  81.049672,
                                                                81.55135 ,  90.054955,  91.05831 , 143.081504, 144.084859,
                                                                161.092069, 162.095424])
    np.testing.assert_array_almost_equal(spectrum.products.intensity, [999.0] * 12)

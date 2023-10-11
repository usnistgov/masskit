import pytest
from pytest import approx
import masskit.spectra

import masskit.spectra.join as mssj
import masskit.spectra.ions as mkions
import masskit.spectra.theoretical_spectrum as msts
import masskit.utils.tablemap as msut


@pytest.fixture
def theo_spectrum():
    return msts.TheoreticalPeptideSpectrum('ACDE', ion_types=[('b', 1), ('b', 2), ('y', 1), ('y', 2)], charge=2)
    # [175.0536, 290.0805, 88.0304, 145.5439, 366.0966, 263.0874, 148.0604, 183.5519, 132.0473, 74.5339]

@pytest.fixture
def exp_spectrum():
    return masskit.spectra.Spectrum().from_arrays(
        [175.0536, 290.0805, 88.0304, 1000.051, 366.0966, 263.0874, 148.0604, 183.5519, 132.0473, 74.533, 74.544],
        [60, 90, 20, 999, 100, 80, 50, 70, 30, 10, 11],
        row={
            "id": 1234,
            "retention_time": 4.5,
            "name": "hello",
            "charge": 2,
            "precursor_mz": 219.0705,
        },
        product_mass_info=mkions.MassInfo(10.0, "ppm", "monoisotopic", "", 1),
        copy_arrays=False
    )

@pytest.fixture
def pred_spectrum():
    return masskit.spectra.Spectrum().from_arrays(
        [175.1, 290.1, 88.0, 1000.0, 366.1, 263.1, 148.1, 1100.0, 1000.1],
        [60, 90, 20, 999, 100, 80, 50, 888, 777],
        product_mass_info=mkions.MassInfo(
            tolerance=0.05,
            tolerance_type="daltons",
            mass_type="monoisotopic",
            evenly_spaced=True
        )
    )

@pytest.fixture
def pred_spectrum_2():
    return masskit.spectra.Spectrum().from_arrays(
        [175.1, 290.1, 88.0, 1000.0, 366.1, 263.1, 148.1, 1100.0, 74.5],
        [60, 90, 20, 999, 100, 80, 50, 888, 777],
        product_mass_info=mkions.MassInfo(
            tolerance=0.05,
            tolerance_type="daltons",
            mass_type="monoisotopic",
            evenly_spaced=True
        )
    )


def test_pairwise_join_none(exp_spectrum: masskit.spectra.Spectrum, pred_spectrum_2: masskit.spectra.Spectrum):
        result = mssj.Join.join_2_spectra(exp_spectrum, pred_spectrum_2, tiebreaker=None)
        assert result == ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                            [0, 0, 1, None, 2, 3, None, 4, 5, 6, 7])
        j2 = mssj.PairwiseJoin(msut.ListLibraryMap([exp_spectrum, exp_spectrum]), msut.ListLibraryMap([pred_spectrum_2, pred_spectrum_2])).do_join(tiebreaker="intensity")
        pass

def test_pairwise_join_delete(exp_spectrum: masskit.spectra.Spectrum, pred_spectrum_2: masskit.spectra.Spectrum):
        result = mssj.Join.join_2_spectra(pred_spectrum_2, exp_spectrum, tiebreaker="delete")
        assert result == ([1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 5, 7, 8, 9, 10, None])
                                            
        j2 = mssj.PairwiseJoin(msut.ListLibraryMap([exp_spectrum, exp_spectrum]), msut.ListLibraryMap([pred_spectrum_2, pred_spectrum_2])).do_join(tiebreaker="intensity")
        pass

def test_pairwise_join(theo_spectrum: msts.TheoreticalPeptideSpectrum, exp_spectrum: masskit.spectra.Spectrum, pred_spectrum: masskit.spectra.Spectrum):
        result = mssj.Join.join_2_spectra(exp_spectrum, pred_spectrum, tiebreaker="intensity")
        assert result == ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                            [None, None, 0, None, 1, 2, None, 3, 4, 5, 6])
        j2 = mssj.PairwiseJoin(msut.ListLibraryMap([exp_spectrum, exp_spectrum]), msut.ListLibraryMap([pred_spectrum, pred_spectrum])).do_join(tiebreaker="intensity")
        pass

def test_threeway_join(theo_spectrum: msts.TheoreticalPeptideSpectrum, exp_spectrum: masskit.spectra.Spectrum, pred_spectrum: masskit.spectra.Spectrum):
        result = mssj.Join.join_3_spectra(exp_spectrum, pred_spectrum, theo_spectrum)
        assert result == ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, None],
                                            [None, None, 0, None, 1, 2, None, 3, 4, 5, 6, 8, 7],
                                            [None, None, 2, 4, 8, 10, 12, 14, 16, 18, None, None, None])
        j3 = mssj.ThreewayJoin(msut.ListLibraryMap([exp_spectrum, exp_spectrum]), msut.ListLibraryMap([pred_spectrum, pred_spectrum]),
                            msut.ListLibraryMap([theo_spectrum, theo_spectrum])).do_join()

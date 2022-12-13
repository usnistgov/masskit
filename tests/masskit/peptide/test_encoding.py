import pytest
from pytest import approx
import masskit.peptide.encoding as mspe
import numpy as np


def test_encoding():
    ions = mspe.calc_ions_mz('AAA', ('a', 1), num_isotopes=1)
    np.testing.assert_array_almost_equal(ions[0], [115.086589])
    # ions[2] are the annotations and field 4 is the position
    np.testing.assert_array_almost_equal(ions[2].field(4).to_numpy(), [2])

    ions = mspe.calc_ions_mz('AAA', ('b', 1), num_isotopes=1)
    np.testing.assert_array_almost_equal(ions[0], [143.081504])
    np.testing.assert_array_almost_equal(ions[2].field(4).to_numpy(), [2])

    ions = mspe.calc_ions_mz('AAA', ('c', 1), num_isotopes=1)
    np.testing.assert_array_almost_equal(ions[0], [89.070939, 160.108053])
    np.testing.assert_array_almost_equal(ions[2].field(4).to_numpy(), [1, 2])

    ions = mspe.calc_ions_mz('AAA', ('x', 1), num_isotopes=1)
    np.testing.assert_array_almost_equal(ions[0], [116.03422, 187.071334])
    np.testing.assert_array_almost_equal(ions[2].field(4).to_numpy(), [1, 2])

    ions = mspe.calc_ions_mz('AAA', ('y', 1), num_isotopes=1)
    np.testing.assert_array_almost_equal(ions[0], [90.054955, 161.092069])
    np.testing.assert_array_almost_equal(ions[2].field(4).to_numpy(), [1, 2])

    ions = mspe.calc_ions_mz('AAA', ('z', 1), num_isotopes=1)
    np.testing.assert_array_almost_equal(ions[0], [74.036231, 145.073345])
    np.testing.assert_array_almost_equal(ions[2].field(4).to_numpy(), [1, 2])

def test_encoding_ptm():
    ions = mspe.calc_ions_mz('ASA', ('y', 1), np.array([21]), np.array([1]), num_isotopes=1)  # Phospho
    np.testing.assert_array_almost_equal(ions[0], [90.054955, 257.053314])
    np.testing.assert_array_almost_equal(ions[2].field(4).to_numpy(), [1, 2])

def test_encoding_charge():
    ions = mspe.calc_ions_mz('AAA', (('y', 1), ('y', 2)), num_isotopes=1)
    np.testing.assert_array_almost_equal(ions[0], [90.054955, 161.092069, 45.531115, 81.049672])
    np.testing.assert_array_almost_equal(ions[2].field(4).to_numpy(), [1, 2, 1, 2])

def test_encoding_ion_type():
    ions = mspe.calc_ions_mz('AAA', (('b', 1), ('y', 1)), num_isotopes=1)
    np.testing.assert_array_almost_equal(ions[0], [143.081504, 90.054955, 161.092069])

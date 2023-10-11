import numpy as np
import pytest
from pytest import approx
from rdkit import Chem

import masskit.spectra.spectrum as mss


def test_load_spectrum():
    spectrum = mss.Spectrum()
    spectrum.from_arrays(
        [100.1, 200.2],
        [999, 1],
        row={
            "id": 1234,
            "retention_time": 4.5,
            "name": "hello",
            "precursor_mz": 500.5,
        },
    )
    assert spectrum.precursor.mz == 500.5
    assert spectrum.products.mz[1] == 200.2
    assert spectrum.products.intensity[1] == 1
    assert spectrum.products.rank[1] == 2
    assert spectrum.id == 1234
    return

def test_rdkit(data_dir):
    suppl = Chem.SDMolSupplier(str(data_dir / "test.new.sdf"), sanitize=False)
    for mol in suppl:
        spectrum = mss.Spectrum()
        spectrum.from_mol(mol)
        assert spectrum.precursor.mz == 180
        assert spectrum.products.mz.size == 36
        assert spectrum.formula == "C9H8O4"
        break

@pytest.fixture
def hi_res1():
    hr = mss.Spectrum()
    hr.from_arrays(
        [100.0001, 200.0002, 300.0003],
        [999, 1, 50],
        row={
            "id": 1234,
            "retention_time": 4.5,
            "name": "hello",
            "precursor_mz": 500.5,
        },
        precursor_mz=555.0,
        product_mass_info=mss.MassInfo(10.0, "ppm", "monoisotopic", "", 1),
    )
    return hr

@pytest.fixture
def hi_res2():
    hr = mss.Spectrum()
    hr.from_arrays(
        [100.0002, 200.0062, 500.0, 300.0009],
        [999, 1, 50, 120],
        row={
            "id": 1234,
            "retention_time": 4.5,
            "name": "hello",
            "precursor_mz": 500.5,
        },
        product_mass_info=mss.MassInfo(10.0, "ppm", "monoisotopic", "", 1),
    )
    return hr

@pytest.fixture
def predicted_spectrum1():
    ps = mss.Spectrum()
    ps.from_arrays(
        [173.1, 201.1, 527.3, 640.4, 769.5, 856.5, 955.6],
        [
            266.66589355,
            900.99719238,
            895.36578369,
            343.2482605,
            296.28485107,
            999.0,
            427.2822876,
        ],
        product_mass_info=mss.MassInfo(
            tolerance=0.05,
            tolerance_type="daltons",
            mass_type="monoisotopic",
            evenly_spaced=True
        )
    )
    return ps

@pytest.fixture
def spectrum1():
    s = mss.Spectrum()
    s.from_arrays(
        [173.0928, 201.088, 527.3327, 640.4177, 769.4603, 856.4924, 955.5608],
        [
            11619800.0,
            21305800.0,
            25972200.0,
            9451650.0,
            8369560.0,
            33015900.0,
            14537000.0,
        ],
        product_mass_info=mss.MassInfo(
            tolerance=10.0,
            tolerance_type="ppm",
            mass_type="monoisotopic",
            evenly_spaced=False
        )
    )
    return s

@pytest.fixture
def spectrum2():
    s = mss.Spectrum()
    s.from_arrays(
        [173.0928, 201.088, 527.3327, 640.4177, 769.4603, 856.4924, 955.5608, 955.6],
        [
            11619800.0,
            21305800.0,
            25972200.0,
            9451650.0,
            8369560.0,
            33015900.0,
            0.1,
            14537000.0,
        ],
        product_mass_info=mss.MassInfo(
            tolerance=10.0,
            tolerance_type="ppm",
            mass_type="monoisotopic",
            evenly_spaced=False
        )
    )
    return s

@pytest.fixture
def hi_res3():
    hr = mss.Spectrum()
    hr.from_arrays(
        np.array([100.0001, 200.0002, 300.0003]),
        np.array([999, 1, 50]),
        stddev=np.array([1, 2, 3]),
        row={
            "id": 1234,
            "retention_time": 4.5,
            "name": "hello",
            "precursor_mz": 500.5,
        },
        precursor_mz=300.0,
        product_mass_info=mss.MassInfo(10.0, "ppm", "monoisotopic", "", 1),
        copy_arrays=False
    )
    return hr

def test_intersect_hires_spectrum(hi_res1: mss.Spectrum, hi_res2: mss.Spectrum):
    index1, index2 = hi_res1.products.intersect(hi_res2.products)
    assert index1.tolist() == [0, 2]
    assert index2.tolist() == [0, 2]
    return

def test_intersect_hires_spectrum_duplicates():
    dup1 = mss.Spectrum().from_arrays(
        np.array([100.0001, 200.0002, 300.0003, 200.0000]),
        np.array([1, 3, 4, 2]),
        product_mass_info=mss.MassInfo(10.0, "ppm", "monoisotopic", "", 1),
        copy_arrays=False)

    dup2 = mss.Spectrum().from_arrays(
        np.array([100.0001, 100.0001, 200.0, 200.0002, 400.0003]),
        np.array([1, 1, 2, 3, 4]),
        product_mass_info=mss.MassInfo(10.0, "ppm", "monoisotopic", "", 1),
        copy_arrays=False)

    index1, index2 = dup1.products.intersect(dup2.products)
    assert index1.tolist() == [0, 0, 1, 2, 1, 2]
    assert index2.tolist() == [0, 1, 2, 2, 3, 3]

def test_copy_ions(hi_res1: mss.Spectrum):
    filtered = hi_res1.products.copy(min_mz=150, max_mz=250)
    assert filtered.mz.tolist() == [200.0002]
    filtered = hi_res1.products.copy(min_intensity=15, max_intensity=100)
    assert filtered.mz.tolist() == [300.0003]
    return

def test_filter_ions(hi_res1: mss.Spectrum):
    filtered = hi_res1.products.copy()
    filtered.filter(min_mz=150, max_mz=250, inplace=True)
    assert filtered.mz.tolist() == [200.0002]
    filtered = hi_res1.products.copy()
    filtered.filter(min_intensity=15, max_intensity=100, inplace=True)
    assert filtered.mz.tolist() == [300.0003]
    return

def test_normalize(hi_res1: mss.Spectrum):
    filtered = hi_res1.products.norm(10000)
    assert filtered.intensity.tolist() == [10000, 10, 500]
    filtered = hi_res1.products.norm(1.0, ord=2, keep_type=False)
    assert filtered.intensity.tolist() == approx([0.99874935, 0.00099975, 0.04998745])
    return

def test_mask(hi_res1: mss.Spectrum):
    filtered = hi_res1.products.mask([1, 2])
    assert filtered.intensity.tolist() == [999]
    assert filtered.mz.tolist() == [100.0001]
    return

def test_mask_bool(hi_res1: mss.Spectrum):
    filtered = hi_res1.products.mask(hi_res1.products.mz < 200.0)
    assert filtered.intensity.tolist() == [1, 50]
    assert filtered.mz.tolist() == [200.0002, 300.0003]
    return

def test_merge(hi_res1: mss.Spectrum, hi_res2: mss.Spectrum):
    merge1 = hi_res1.products.copy()  # make a copy
    merge2 = hi_res2.products.copy()  # make a copy
    merged = merge1.merge(merge2)
    assert merged.intensity.tolist() == [999, 999, 1, 1, 50, 120, 50]
    return

def test_evenly_space(hi_res1: mss.Spectrum):
    initial_spectrum = hi_res1.copy()  # make a copy
    initial_spectrum.evenly_space(
        tolerance=0.05,
        inplace=True,
    )
    assert initial_spectrum.products.starts.tolist() == approx([99.95, 199.95, 299.95])
    assert initial_spectrum.products.stops.tolist() == approx([100.05, 200.05, 300.05])

def test_evenly_space_stddev(hi_res3: mss.Spectrum):
    initial_spectrum = hi_res3.copy()  # make a copy
    initial_spectrum.evenly_space(
        tolerance=0.05,
        inplace=True,
    )
    assert initial_spectrum.products.starts.tolist() == approx([99.95, 199.95, 299.95])
    assert initial_spectrum.products.stops.tolist() == approx([100.05, 200.05, 300.05])
    assert initial_spectrum.products.stddev.tolist() == approx([1, 2, 3])

def test_cosine_score(spectrum1: mss.Spectrum, predicted_spectrum1: mss.Spectrum):
    # according to mspepsearch should be 549.  spectra 484, first one in Qian's plot
    score = spectrum1.cosine_score(predicted_spectrum1)
    assert score == approx(992.2658666598618)

def test_cosine_score_tiebreak(spectrum2: mss.Spectrum, predicted_spectrum1: mss.Spectrum):
    # according to mspepsearch should be 549.  spectra 484, first one in Qian's plot
    score = predicted_spectrum1.cosine_score(spectrum2, tiebreaker='mz')
    assert score == approx(992.2658666598618)

def test_to_msp(hi_res1: mss.Spectrum):
    ret_value = hi_res1.to_msp()
    len(ret_value) > 10

def test_accumulator(hi_res1: mss.Spectrum, hi_res2: mss.Spectrum):
    acc = mss.AccumulatorSpectrum(mz=np.linspace(0.1, 2000, 20000), tolerance=0.05)
    acc1 = hi_res1.copy()  # make a copy
    acc2 = hi_res2.copy()  # make a copy
    acc.add(acc1)
    acc.add(acc2)
    assert acc.products.intensity[999] == 999.0
    assert acc.products.intensity[4999] == 25.0

def test_shift_mz(hi_res1: mss.Spectrum):
    test_spectrum = hi_res1.copy()  # make a copy
    test_spectrum = test_spectrum.shift_mz(-200.0)
    assert test_spectrum.products.mz.tolist() == approx([0.0002, 100.0003])

def test_windowed_filter(predicted_spectrum1: mss.Spectrum):
    test_spectrum = predicted_spectrum1.copy()
    test_spectrum = test_spectrum.windowed_filter(mz_window=100, num_ions=2)
    assert test_spectrum.products.mz.tolist() == approx([173.1, 201.1, 527.3, 640.4, 856.5, 955.6])

def test_parent_filter(hi_res3: mss.Spectrum):
    test_spectrum = hi_res3.copy()
    test_spectrum = test_spectrum.parent_filter(h2o=True, inplace=False)
    assert test_spectrum.products.mz.tolist() == approx([100.0001, 200.0002])

def test_evenly_space_cosine_score(hi_res3: mss.Spectrum):
    initial_spectrum = hi_res3.copy()  # make a copy
    even_nozero = initial_spectrum.evenly_space(
        tolerance=0.05,
        include_zeros=False,
        )
    even_zero = initial_spectrum.evenly_space(
        tolerance=0.05,
        include_zeros=True,
        )
    assert(even_nozero.cosine_score(even_zero) == 999.0)
    
def test_single_match(hi_res3: mss.Spectrum):
    assert(hi_res3.single_match(hi_res3)[2] == 1.0)

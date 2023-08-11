import masskit.spectrum.spectrum as mss
import pytest

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

def test_cosine_score(spectrum1, predicted_spectrum1):
    # according to mspepsearch should be 549.  spectra 484, first one in Qian's plot
    score = spectrum1.cosine_score(predicted_spectrum1)
    return score
    #assert score == approx(992.2658666598618)

# ps1 = predicted_spectrum1()
# s1 = spectrum1()
# res = test_cosine_score(s1, ps1)
# print(res)

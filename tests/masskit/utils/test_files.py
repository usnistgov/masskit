import pytest
from masskit.utils.files import load_mzTab

@pytest.fixture
def maxQuant_hitlist():
    hitlist = load_mzTab("data/example.mzTab")
    return hitlist

@pytest.fixture
def mascot_hitlist():
    hitlist = load_mzTab("data/F981123.mztab")
    return hitlist

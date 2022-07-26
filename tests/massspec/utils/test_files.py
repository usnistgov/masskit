import pytest
from massspec.utils.files import load_mzTab

@pytest.fixture
def maxQuant_hitlist():
    hitlist = load_mzTab("libraries/tests/data/example.mzTab")
    return hitlist

@pytest.fixture
def mascot_hitlist():
    hitlist = load_mzTab("libraries/tests/data/F981123.mztab")
    return hitlist

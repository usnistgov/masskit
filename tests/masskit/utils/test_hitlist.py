from masskit.utils.hitlist import Hitlist, IdentityRecall
import pytest
import pandas as pd

@pytest.fixture
def maxQuant_hitlist():
    df = pd.read_pickle("data/2022-07-24_predict_hitlist_trunc.pkl")
    hitlist = Hitlist(df) 
    return hitlist

@pytest.mark.skip(reason="fixture used is old pickle file with old module names")
def test_identity(maxQuant_hitlist):
    df_results = IdentityRecall(comparison_score='raw_score').compare(maxQuant_hitlist)
    pass
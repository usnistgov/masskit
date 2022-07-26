import pytest

import pyarrow as pa
# import numpy as np

# from massspec_ext import massspec_ext
import massspec_ext as ext
from rdkit import DataStructs

def test_madd():

    a = pa.array([1, 2, 3], type=pa.float64())
    b = pa.array([0, 2, 4], type=pa.float64())
    c = pa.array([1, 1, 5], type=pa.float64())
    assert ext.madd(a, b, c).to_pylist() == [1, 4, 23]

    with pytest.raises(ValueError):
        assert ext.madd(a[1:], b, c).to_pylist() == [1, 4, 23]


def test_mz_fingerprint():
    x = pa.array([1, 2, 3.1, 57.9, 1642.78], type=pa.float64())
    fp = ext.mz_fingerprint(x)
    print(type(fp))
    print(fp)
    print(DataStructs.cDataStructs.CreateFromBinaryText(fp))
    # print(DataStructs.cDataStructs.BitVectToText(fp))
    # assert ext.mz_fingerprint(x) == np.sum(np.array(x))
    # assert ext.sum(x[1:]) == np.sum(np.array(x[1:]))

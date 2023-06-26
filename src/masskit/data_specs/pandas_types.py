import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype
from pandas.api.extensions import register_extension_dtype, take
import pandas as pd
from masskit.spectrum.spectrum import Spectrum
from masskit.utils.tables import row_view
import numbers

@register_extension_dtype
class SpectrumPandasDtype(ExtensionDtype):
    type = Spectrum
    name = "Spectrum"
    na_value = np.nan

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.
        Returns
        -------
        type
        """
        return SpectrumPandasArray

    def __from_arrow__(self, arr, name=None):
        """
        input can be table, struct array, or chunked struct array
        """
        if name is None:
            name = 'spectrum'

        if type(arr) == pa.Table:
            table = arr
        elif type(arr) == pa.ChunkedArray:
            table = pa.Table.from_arrays(arr, names=[name])
        elif type(arr) == pa.StructArray:
            table = pa.Table.from_batches([pa.RecordBatch.from_struct_array(arr)])

        row = row_view(table)

        output_array = np.empty((len(arr),), dtype=object)

        for i in range(len(arr)):
            row.idx = i
            output_array[i] = Spectrum(row=row)

        return SpectrumPandasArray(arr)

"""
getting values out of struct array
sa[0]['isomeric_smiles'].as_py()
sa[0]['precursor_massinfo']['tolerance'].as_py()
sa[0]['mz'].values.to_numpy()
to initialize Spectrum with arrow struct, would have to:
- make separate argument for struct in Spectrum constuctor
- make separate argument for massinfo struct in MassInfo constructor
- create function like 
"""
"""

class SpectrumPandasArray(ExtensionArray):

    dtype = SpectrumPandasDtype()

    def __init__(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("Need to pass a numpy array as values")
        for val in values:
            if not isinstance(val, self.dtype.type) and not None:
                raise TypeError("All values must be of type " + str(self.dtype.type))
        self.data = values

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        data = np.empty(len(scalars), dtype=object)
        data[:] = scalars
        return cls(data)
    
    @classmethod
    def _from_factorized(cls, values):
        raise NotImplementedError
    
    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return self.data[item]
        else:
            # slice, list-like, mask
            return type(self)(self.data[item])

    def __len__(self) -> int:
        return len(self.data)
    
    def __eq__(self, other) -> int:
        return np.all(self.data == other.data)

    def isna(self):
        return np.array(
            [x is None for x in self.data], dtype=bool
        )
    
    def nbytes(self) -> int:
        # this is not correct -- have to ask Spectrum objects for their size
        return self.data.nbytes

    def take(self, indexes):
        indexes = np.asarray(indexes)
        output = np.take(self.data, indexes)
        return type(self)(output)

    def copy(self):
        return type(self)(self.data[:])

    @classmethod
    def _concat_same_type(cls, to_concat):
        data = np.concatenate([x.data for x in to_concat])
        return cls(data)

"""
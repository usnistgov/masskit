import numpy as np
import pyarrow as pa
import pandas as pd
from masskit.data_specs.schemas import molecules_struct, peptide_struct
from masskit.spectrum.spectrum import Spectrum
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype
from pandas.api.extensions import register_extension_dtype, take
import numbers
from rdkit import Chem


class MasskitArrowArray(pa.ExtensionArray):
    """
    Extension array for arrow arrays.
    """
    def to_pylist(self):
        """
        Convert to list of Spectrum objects
        """
        array = pa.ExtensionArray.from_storage(self.type, self.storage)
        output = []
        for i in range(len(array)):
            output.append(array[i].as_py())
        return output

    def to_numpy(self):
        """
        Convert to numpy array of Spectrum objects
        """
        array = pa.ExtensionArray.from_storage(self.type, self.storage)
        output = np.empty((len(array),), dtype=object)
        for i in range(len(array)):
            output[i] = array[i].as_py()
        return output
        

class MasskitPandasArray(ExtensionArray):
    """
    Base class for pandas extension arrays that contain a numpy array of objects
    """

    dtype = None

    def __init__(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("Need to pass a numpy array as values")
        for val in values:
            if not isinstance(val, self.dtype.type) and not None:
                raise TypeError("All values must be of type " + str(self.dtype.type))
        self.data = values

    @classmethod
    def _from_sequence(cls, scalars):
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


class MasskitPandasDtype(ExtensionDtype):
    type = None
    name = None
    na_value = None

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.
        Returns
        -------
        type
        """
        return None

    def __from_arrow__(self, arr):
        """
        input can be table, struct array, or chunked struct array
        """
        output = []
        for chunk in arr.iterchunks():
            output.append(chunk.to_numpy())
        numpy_arr = np.concatenate(output)
        return self.construct_array_type()(numpy_arr)


class SpectrumArrowScalarType(pa.ExtensionScalar):
    """
    arrow scalar extension class for spectra
    """

    def as_py(self):
        if self.value is None:
            return None
        else:
            return Spectrum(struct=self.value)

    
class SpectrumArrowType(pa.PyExtensionType):
    """
    arrow type extension class for spectra
    parameterized by storage_type, which can be molecules_struct or peptide_struct
    """

    def __init__(self, storage_type=molecules_struct):
        pa.PyExtensionType.__init__(self, storage_type)

    def __reduce__(self):
        """
        used by pickle to understand how to serialize this class
        :return: a callable object and a tuple of arguments
        """
        #todo: should save size as metadata
        return SpectrumArrowType, (self.storage_type,)

    def __arrow_ext_scalar_class__(self):
        return SpectrumArrowScalarType
    
    def __arrow_ext_class__(self):
        return SpectrumArrowArray

    def to_pandas_dtype(self):
        """
        returns pandas extension dtype
        """
        return SpectrumPandasDtype()


class SpectrumArrowArray(MasskitArrowArray):
    """
    arrow array that holds spectra
    """


@register_extension_dtype
class SpectrumPandasDtype(MasskitPandasDtype):
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


class SpectrumPandasArray(MasskitPandasArray):
    dtype = SpectrumPandasDtype()
    
    def __init__(self, values):
        super().__init__(values)


class MolArrowScalarType(pa.ExtensionScalar):
    """
    arrow scalar extension class for spectra
    """

    def as_py(self):
        if self.value is None:
            return None
        else:
            mol = Chem.rdMolInterchange.JSONToMols(self.value.as_py())[0]
            return mol

    
class MolArrowType(pa.PyExtensionType):
    """
    arrow type extension class for Mols
    """

    def __init__(self):
        pa.PyExtensionType.__init__(self, pa.string())

    def __reduce__(self):
        """
        used by pickle to understand how to serialize this class
        :return: a callable object and a tuple of arguments
        """
        #todo: should save size as metadata
        return MolArrowType, (self.storage_type,)

    def __arrow_ext_scalar_class__(self):
        return MolArrowScalarType
    
    def __arrow_ext_class__(self):
        return MolArrowArray

    def to_pandas_dtype(self):
        """
        returns pandas extension dtype
        """
        return MolPandasDtype()


class MolArrowArray(MasskitArrowArray):
    """
    Extension array for MolArrowType
    """
        

@register_extension_dtype
class MolPandasDtype(MasskitPandasDtype):
    type = Chem.rdchem.Mol
    name = "Mol"
    na_value = np.nan

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.
        Returns
        -------
        type
        """
        return MolPandasArray


class MolPandasArray(MasskitPandasArray):
    dtype = MolPandasDtype()
    
    def __init__(self, values):
        super().__init__(values)

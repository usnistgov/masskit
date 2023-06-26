import pyarrow as pa
import pandas as pd
from masskit.data_specs.schemas import molecules_struct, peptide_struct
from masskit.spectrum.spectrum import Spectrum
from masskit.data_specs.arrow_types import MolSpectrumArrowType

class SpectrumArrowScalarType(pa.ExtensionScalar):
    """
    arrow scalar extension class for spectra
    """

    def as_py(self):
        if self.value is None:
            return None
        else:
            return Spectrum(struct=self.value)

    
class MolSpectrumArrowType(pa.PyExtensionType):
    """
    arrow type extension class for small molecule spectra
    """

    storage_type = molecules_struct

    def __init__(self):
        pa.PyExtensionType.__init__(self, self.storage_type)

    def __reduce__(self):
        """
        used by pickle to understand how to serialize this class
        See https://docs.python.org/3/library/pickle.html#object.__reduce__
        https://stackoverflow.com/questions/19855156/whats-the-exact-usage-of-reduce-in-pickler

        :return: a callable object and a tuple of arguments
        """
        #todo: should save size as metadata
        return MolSpectrumArrowType, ()

    def __arrow_ext_scalar_class__(self):
        return SpectrumArrowScalarType
    
    # def __arrow_ext_class__(self):

    #     return SpectrumArrowArray

    def to_pandas_dtype(self):
        """
        returns pandas extension dtype
        """
        return MolSpectrumArrowType()


class PeptideSpectrumArrowType(pa.PyExtensionType):
    """
    arrow type extension class for peptide spectra
    """

    storage_type = peptide_struct

    def __init__(self):
        pa.PyExtensionType.__init__(self, self.storage_type)

    def __reduce__(self):
        """
        used by pickle to understand how to serialize this class
        See https://docs.python.org/3/library/pickle.html#object.__reduce__
        https://stackoverflow.com/questions/19855156/whats-the-exact-usage-of-reduce-in-pickler

        :return: a callable object and a tuple of arguments
        """
        #todo: should save size as metadata
        return PeptideSpectrumArrowType, ()

    def __arrow_ext_scalar_class__(self):
        return SpectrumArrowScalarType
    
    # def __arrow_ext_class__(self):

    #     return SpectrumArrowArray

    def to_pandas_dtype(self):
            return pd.PeriodDtype(freq=self.freq)


# class SpectrumArrowArray(pa.ExtensionArray):
#     """
#     See Arrow docs for customizing extension arrays:
#     https://arrow.apache.org/docs/python/extending_types.html#custom-extension-array-class
#     """
#     def to_pylist(self):
#         """
#         Convert to list of Spectrum
#         """

#     def to_numpy(self):
#         """
#         Convert to numpy array of Spectrum
#         """


#     OFFSET_DTYPE = np.int32

#     @classmethod
#     def from_numpy(
#         cls, arr: Union[np.ndarray, Iterable[np.ndarray]]
#     ) -> Union["ArrowTensorArray", "ArrowVariableShapedTensorArray"]:
#         """
#         Convert an ndarray or an iterable of ndarrays to an array of homogeneous-typed
#         tensors. If given fixed-shape tensor elements, this will return an
#         ``ArrowTensorArray``; if given variable-shape tensor elements, this will return
#         an ``ArrowVariableShapedTensorArray``.

#         Args:
#             arr: An ndarray or an iterable of ndarrays.

#         Returns:
#             - If fixed-shape tensor elements, an ``ArrowTensorArray`` containing
#               ``len(arr)`` tensors of fixed shape.
#             - If variable-shaped tensor elements, an ``ArrowVariableShapedTensorArray``
#               containing ``len(arr)`` tensors of variable shape.
#             - If scalar elements, a ``pyarrow.Array``.
#         """
#         if isinstance(arr, (list, tuple)) and arr and isinstance(arr[0], np.ndarray):
#             # Stack ndarrays and pass through to ndarray handling logic below.
#             try:
#                 arr = np.stack(arr, axis=0)
#             except ValueError:
#                 # ndarray stacking may fail if the arrays are heterogeneously-shaped.
#                 arr = np.array(arr, dtype=object)
#         if isinstance(arr, np.ndarray):
#             if len(arr) > 0 and np.isscalar(arr[0]):
#                 # Elements are scalar so a plain Arrow Array will suffice.
#                 return pa.array(arr)
#             if _is_ndarray_variable_shaped_tensor(arr):
#                 # Tensor elements have variable shape, so we delegate to
#                 # ArrowVariableShapedTensorArray.
#                 return ArrowVariableShapedTensorArray.from_numpy(arr)
#             if not arr.flags.c_contiguous:
#                 # We only natively support C-contiguous ndarrays.
#                 arr = np.ascontiguousarray(arr)
#             pa_dtype = pa.from_numpy_dtype(arr.dtype)
#             if pa.types.is_string(pa_dtype):
#                 if arr.dtype.byteorder == ">" or (
#                     arr.dtype.byteorder == "=" and sys.byteorder == "big"
#                 ):
#                     raise ValueError(
#                         "Only little-endian string tensors are supported, "
#                         f"but got: {arr.dtype}",
#                     )
#                 pa_dtype = pa.binary(arr.dtype.itemsize)
#             outer_len = arr.shape[0]
#             element_shape = arr.shape[1:]
#             total_num_items = arr.size
#             num_items_per_element = np.prod(element_shape) if element_shape else 1

#             # Data buffer.
#             if pa.types.is_boolean(pa_dtype):
#                 # NumPy doesn't represent boolean arrays as bit-packed, so we manually
#                 # bit-pack the booleans before handing the buffer off to Arrow.
#                 # NOTE: Arrow expects LSB bit-packed ordering.
#                 # NOTE: This creates a copy.
#                 arr = np.packbits(arr, bitorder="little")
#             data_buffer = pa.py_buffer(arr)
#             data_array = pa.Array.from_buffers(
#                 pa_dtype, total_num_items, [None, data_buffer]
#             )

#             # Offset buffer.
#             offset_buffer = pa.py_buffer(
#                 cls.OFFSET_DTYPE(
#                     [i * num_items_per_element for i in range(outer_len + 1)]
#                 )
#             )

#             storage = pa.Array.from_buffers(
#                 pa.list_(pa_dtype),
#                 outer_len,
#                 [None, offset_buffer],
#                 children=[data_array],
#             )
#             type_ = ArrowTensorType(element_shape, pa_dtype)
#             return pa.ExtensionArray.from_storage(type_, storage)
#         elif isinstance(arr, Iterable):
#             return cls.from_numpy(list(arr))
#         else:
#             raise ValueError("Must give ndarray or iterable of ndarrays.")

#     def _to_numpy(self, index: Optional[int] = None, zero_copy_only: bool = False):
#         """
#         Helper for getting either an element of the array of tensors as an
#         ndarray, or the entire array of tensors as a single ndarray.

#         Args:
#             index: The index of the tensor element that we wish to return as
#                 an ndarray. If not given, the entire array of tensors is
#                 returned as an ndarray.
#             zero_copy_only: If True, an exception will be raised if the
#                 conversion to a NumPy array would require copying the
#                 underlying data (e.g. in presence of nulls, or for
#                 non-primitive types). This argument is currently ignored, so
#                 zero-copy isn't enforced even if this argument is true.

#         Returns:
#             The corresponding tensor element as an ndarray if an index was
#             given, or the entire array of tensors as an ndarray otherwise.
#         """
#         # TODO(Clark): Enforce zero_copy_only.
#         # TODO(Clark): Support strides?
#         # Buffers schema:
#         # [None, offset_buffer, None, data_buffer]
#         buffers = self.buffers()
#         data_buffer = buffers[3]
#         storage_list_type = self.storage.type
#         value_type = storage_list_type.value_type
#         ext_dtype = value_type.to_pandas_dtype()
#         shape = self.type.shape
#         if pa.types.is_boolean(value_type):
#             # Arrow boolean array buffers are bit-packed, with 8 entries per byte,
#             # and are accessed via bit offsets.
#             buffer_item_width = value_type.bit_width
#         else:
#             # We assume all other array types are accessed via byte array
#             # offsets.
#             buffer_item_width = value_type.bit_width // 8
#         # Number of items per inner ndarray.
#         num_items_per_element = np.prod(shape) if shape else 1
#         # Base offset into data buffer, e.g. due to zero-copy slice.
#         buffer_offset = self.offset * num_items_per_element
#         # Offset of array data in buffer.
#         offset = buffer_item_width * buffer_offset
#         if index is not None:
#             # Getting a single tensor element of the array.
#             offset_buffer = buffers[1]
#             offset_array = np.ndarray(
#                 (len(self),), buffer=offset_buffer, dtype=self.OFFSET_DTYPE
#             )
#             # Offset into array to reach logical index.
#             index_offset = offset_array[index]
#             # Add the index offset to the base offset.
#             offset += buffer_item_width * index_offset
#         else:
#             # Getting the entire array of tensors.
#             shape = (len(self),) + shape
#         if pa.types.is_boolean(value_type):
#             # Special handling for boolean arrays, since Arrow bit-packs boolean arrays
#             # while NumPy does not.
#             # Cast as uint8 array and let NumPy unpack into a boolean view.
#             # Offset into uint8 array, where each element is a bucket for 8 booleans.
#             byte_bucket_offset = offset // 8
#             # Offset for a specific boolean, within a uint8 array element.
#             bool_offset = offset % 8
#             # The number of uint8 array elements (buckets) that our slice spans.
#             # Note that, due to the offset for a specific boolean, the slice can span
#             # byte boundaries even if it contains less than 8 booleans.
#             num_boolean_byte_buckets = 1 + ((bool_offset + np.prod(shape) - 1) // 8)
#             # Construct the uint8 array view on the buffer.
#             arr = np.ndarray(
#                 (num_boolean_byte_buckets,),
#                 dtype=np.uint8,
#                 buffer=data_buffer,
#                 offset=byte_bucket_offset,
#             )
#             # Unpack into a byte per boolean, using LSB bit-packed ordering.
#             arr = np.unpackbits(arr, bitorder="little")
#             # Interpret buffer as boolean array.
#             return np.ndarray(shape, dtype=np.bool_, buffer=arr, offset=bool_offset)
#         # Special handling of binary/string types. Assumes unicode string tensor columns
#         if pa.types.is_fixed_size_binary(value_type):
#             ext_dtype = np.dtype(
#                 f"<U{value_type.byte_width // NUM_BYTES_PER_UNICODE_CHAR}"
#             )
#         return np.ndarray(shape, dtype=ext_dtype, buffer=data_buffer, offset=offset)

#     def to_numpy(self, zero_copy_only: bool = True):
#         """
#         Convert the entire array of tensors into a single ndarray.

#         Args:
#             zero_copy_only: If True, an exception will be raised if the
#                 conversion to a NumPy array would require copying the
#                 underlying data (e.g. in presence of nulls, or for
#                 non-primitive types). This argument is currently ignored, so
#                 zero-copy isn't enforced even if this argument is true.

#         Returns:
#             A single ndarray representing the entire array of tensors.
#         """
#         return self._to_numpy(zero_copy_only=zero_copy_only)

import logging
import math
from abc import ABC, abstractmethod

import numpy as np

try:
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:
    pass

def calc_bit(a, b, max_mz_in, tolerance_in):
    logging.warning('this function is deprecated')
    return min(int(abs(b - a) / tolerance_in), max_mz_in / tolerance_in - 1)


def calc_interval_fingerprint(spectrum, max_mz=2000, tolerance=0.1, min_intensity=50, mz_window=14,
                              min_mz=1, peaks=None):
    """
    create a pairwise interval fingerprint from the filtered peaks of a spectrum

    :param spectrum: input spectrum
    :param max_mz: maximum mz used for the fingerprint
    :param tolerance: mass tolerance to use
    :param min_intensity: minimum intensity to allow for creating the fingerprint
    :param mz_window: noise filter mz window
    :param min_mz: minimum mz to use
    :param peaks: a list of additional mz values to use for creating the intervals
    :return: the fingerprint
    """
    logging.warning('this function is deprecated')

    filtered_spectrum = spectrum.norm().filter(
        min_intensity=min_intensity, inplace=True).windowed_filter(
        inplace=True, mz_window=mz_window)

    if peaks is None:
        if filtered_spectrum.precursor is None:
            peaks = [0]
        else:
            peaks = [0, filtered_spectrum.precursor.mz]
        peaks.extend(filtered_spectrum.products.mz)

    fp = DataStructs.ExplicitBitVect(int(max_mz/tolerance))
    for i in peaks:
        for j in filtered_spectrum.products.mz:
            if abs(j-i) > min_mz:
                fp.SetBit(calc_bit(i, j, max_mz, tolerance))
    return fp


def calc_dynamic_fingerprint(spectrum, max_mz=2000, tolerance=0.1, min_intensity=50, mz_window=14,
                             min_mz=2.5, max_rank=20, hybrid_mask=None):
    """
    create a fingerprint created from the top max_rank peaks and the OR of the fingerprint of the top
    peaks, where each fingerprint is shifted by the mz of one of the top peaks

    :param spectrum: input spectrum
    :param max_mz: maximum mz used for the fingerprint
    :param tolerance: mass tolerance to use
    :param min_intensity: minimum intensity to allow for creating the fingerprint
    :param mz_window: noise filter mz window
    :param min_mz: minimum mz to use
    :param max_rank: the maximum rank used for the fingerprint
    :return: the fingerprint
    """
    logging.warning('this function is deprecated')

    filtered_spectrum = spectrum.norm().filter(
        min_intensity=min_intensity, inplace=True).windowed_filter(
        inplace=True, mz_window=mz_window)

    filtered_spectrum = filtered_spectrum.mask(filtered_spectrum.products.rank <= max_rank)

    if filtered_spectrum.precursor is None:
        peaks = [0]
    else:
        peaks = [0, filtered_spectrum.precursor.mz]
    peaks.extend(filtered_spectrum.products.mz)

    offset = int(max_mz/tolerance)
    fp = DataStructs.ExplicitBitVect(offset)
    # note that since calc_bit returns absolute value, most bits are set twice
    for i in peaks:
        for j in filtered_spectrum.products.mz:
            if abs(j-i) > min_mz:
                fp.SetBit(calc_bit(i, j, max_mz, tolerance))
    if hybrid_mask:
        fp = fp & hybrid_mask

    return fp


def fingerprint_search_numpy(query_fingerprint, fingerprint_array, tanimoto_cutoff,
                             query_fingerprint_count=None, fingerprint_array_count=None):
    """
    return a list of fingerprint hits to a query fingerprint

    :param query_fingerprint: query fingerprint
    :param fingerprint_array: array-like list of fingerprints
    :param tanimoto_cutoff: Tanimoto cutoff
    :param query_fingerprint_count: number of bits set in query
    :param fingerprint_array_count: number of bits set in array fingerprints
    :return: tanimoto scores (if below threshold, set to 0.0)

    Notes: numba doesn't work with arrays of objects.  pyarrow creates a numpy array of objects, where each object
    is a numpy array.  even arrow fixed size lists do this.  fixedbinary uses byte objects.
    """
    logging.warning('this function is deprecated')
    def divide_0(a, b, fill=0.0):
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
        if np.isscalar(c):
            return c if np.isfinite(c) \
                else fill
        else:
            c[~ np.isfinite(c)] = fill
            return c

    if query_fingerprint_count is None or fingerprint_array_count is None:
        return_value = divide_0(np.unpackbits(np.bitwise_and(query_fingerprint, fingerprint_array), axis=-1).
                                sum(axis=-1), np.unpackbits(np.bitwise_or(query_fingerprint, fingerprint_array),
                                                            axis=-1).sum(axis=-1))
    else:
        condition = (query_fingerprint_count >= fingerprint_array_count*tanimoto_cutoff) &\
                    (fingerprint_array_count >= query_fingerprint_count*tanimoto_cutoff)
        return_value = \
            np.where(condition,
                     divide_0(np.unpackbits(np.bitwise_and(query_fingerprint, fingerprint_array), axis=-1).sum(axis=-1),
                              np.unpackbits(np.bitwise_or(query_fingerprint, fingerprint_array),
                                            axis=-1).sum(axis=-1)), 0.0)
    return_value[return_value < tanimoto_cutoff] = 0.0
    return return_value


class Fingerprint(ABC):
    """
    class for encapsulating and calculating a fingerprint

    :param dimension: size of the fingerprint (includes other features like counts
    :type dimension: int
    """
    def __init__(self, dimension=2000, stride=256, column_name=None, *args, **kwargs):
        """
        :param dimension: size of the fingerprint
        :param stride: the fingerprint will be fit into contiguous memory that is a multiple of stride
        :param column_name: the column name used to store the fingerprint
        """
        super().__init__(*args, **kwargs)
        self.dimension = dimension
        # the vector instruction stride of the fingerprint in bits
        self.stride = stride
        self.fingerprint = None
        self.column_name = column_name
        self.id = None

    def size(self) -> int:
        """
        size of the fingerprint in bytes, where the size is a multiple of the stride

        :return: size of fingerprint
        """
        return ((self.dimension + self.stride - 1)//self.stride) * 32

    @abstractmethod
    def to_bitvec(self):
        """
        convert fingerprint to rdkit ExplicitBitVect

        :return: the fingerprint
        :rtype: DataStructs.ExplicitBitVect
        """
        raise NotImplementedError()

    @abstractmethod
    def to_numpy(self):
        """
        convert fingerprint to numpy array

        :return: the fingerprint
        :rtype: np.ndarray
        """
        raise NotImplementedError()

    def bitvec2numpy(self):
        """
        version of to_numpy used when fingerprint is a bitvec
        """
        return_value = np.zeros((0,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(self.fingerprint, return_value)
        return_value = np.packbits(return_value)
        if return_value.shape[0] % self.size():
            return_value = np.pad(return_value, (0, self.size() - return_value.shape[0]))
        return return_value

    @abstractmethod
    def from_bitvec(self, array):
        """
        create fingerprint from rdkit ExplicitBitVect

        :param array: input ExplicitBitVect
        :type array: DataStructs.ExplicitBitVect
        """
        raise NotImplementedError()

    @abstractmethod
    def from_numpy(self, array):
        """
        create fingerprint from numpy array

        :param array: input numpy array
        :type array: np.ndarray
        """
        raise NotImplementedError()

    def numpy2bitvec(self, array):
        """
        version of from_numpy used when fingerprint is a bitvec

        :param array: the fingerprint as a numpy array
        """

        bitstring = "".join(array.astype(int).astype(str))
        self.fingerprint = DataStructs.cDataStructs.CreateFromBitString(bitstring)

    @abstractmethod
    def object2fingerprint(
            self,
            obj, dtype=np.float32):
        """
        convert an object into a fingerprint

        :param obj: the object to convert to a fingerprint
        :type obj: object
        :param dtype: data type of output array
        :type dtype: np.dtype
        """
        raise NotImplementedError()

    def get_num_on_bits(self):
        """
        retrieve the number of nonzero features
        """
        return self.to_bitvec().GetNumOnBits()


class MolFingerprint(ABC):
    """
    class for encapsulating and calculating a molecule fingerprint

    :param dimension: size of the fingerprint (includes other features like counts
    :type dimension: int
    """
    def __init__(self, dimension=4096, *args, **kwargs):
        super().__init__(dimension=dimension, *args, **kwargs)

    @abstractmethod
    def to_bitvec(self):
        """
        convert fingerprint to rdkit ExplicitBitVect

        :return: the fingerprint
        :rtype: DataStructs.ExplicitBitVect
        """
        raise NotImplementedError()

    @abstractmethod
    def to_numpy(self):
        """
        convert fingerprint to numpy array

        :return: the fingerprint
        :rtype: np.ndarray
        """
        raise NotImplementedError()

    @abstractmethod
    def from_bitvec(self, array):
        """
        create fingerprint from rdkit ExplicitBitVect

        :param array: input ExplicitBitVect
        :type array: DataStructs.ExplicitBitVect
        """
        raise NotImplementedError()

    @abstractmethod
    def from_numpy(self, array):
        """
        create fingerprint from numpy array

        :param array: input numpy array
        :type array: np.ndarray
        """
        raise NotImplementedError()

    @abstractmethod
    def object2fingerprint(
            self,
            obj, dtype=np.float32):
        """
        convert an object into a fingerprint

        :param obj: the object to convert to a fingerprint
        :type obj: object
        :param dtype: data type of output array
        :type dtype: np.dtype
        """
        raise NotImplementedError()


class ECFPFingerprint(Fingerprint):
    """
    rdkit version of ECFP fingerprint for a small molecule structure

    """
    def __init__(self, dimension=4096, radius=2, *args, **kwargs):
        super().__init__(dimension=dimension, *args, **kwargs)
        self.ecfp = rdFingerprintGenerator.GetMorganGenerator(fpSize=dimension, radius=radius,
                                                              countSimulation=True, includeChirality=False)

    def to_bitvec(self):
        """
        convert fingerprint to rdkit ExplicitBitVect

        :return: the fingerprint
        :rtype: DataStructs.ExplicitBitVect
        """
        return self.fingerprint

    def to_numpy(self):
        """
        convert fingerprint to numpy array

        :return: the fingerprint
        :rtype: np.ndarray
        """
        return self.bitvec2numpy()

    def from_bitvec(self, array):
        """
        create fingerprint from rdkit ExplicitBitVect

        :param array: input ExplicitBitVect
        :type array: DataStructs.ExplicitBitVect
        """
        self.fingerprint = array

    def from_numpy(self, array):
        """
        create fingerprint from numpy array

        :param array: input numpy array
        :type array: np.ndarray
        """
        raise NotImplementedError()

    def object2fingerprint(
            self,
            obj, dtype=np.float32):
        """
        convert an object into a fingerprint

        :param obj: the object to convert to a fingerprint
        :type obj: object
        :param dtype: data type of output array
        :type dtype: np.dtype
        """
        self.fingerprint = self.ecfp.GetFingerprint(obj)


class SpectrumFingerprint(Fingerprint):
    """
    base class for spectrum fingerprint

    """

    def __init__(self, dimension=2000, bin_size=1.0, first_bin_left=0.5, *args, **kwargs):
        """
        :param dimension: overall dimension of fingerprint
        :type dimension: int
        :param bin_size: size of each bin
        :type bin_size: float
        :param first_bin_left: the position of the first bin left side
        :type first_bin_left: float
        """
        super().__init__(dimension=dimension, *args, **kwargs)
        # maximum number of counting bits, ignoring 2^0 bit
        self.bin_size = bin_size
        self.first_bin_left = first_bin_left

    @abstractmethod
    def to_bitvec(self):
        raise NotImplementedError()

    @abstractmethod
    def to_numpy(self):
        raise NotImplementedError()

    @abstractmethod
    def from_bitvec(self, array):
        raise NotImplementedError()

    @abstractmethod
    def from_numpy(self, array):
        raise NotImplementedError()

    @abstractmethod
    def object2fingerprint(
            self,
            obj, dtype=np.float32):
        raise NotImplementedError()


class SpectrumFloatFingerprint(SpectrumFingerprint):
    """
    create a spectral fingerprint that also contains a count by powers of two, skipping 1.  In other words,
    the count bits are set if the number of peaks exceeds >=2, >=4, ...

    """

    def __init__(self, dimension=2000, count_max=2000, mz_ratio: bool = False, column_name=None, *args, **kwargs):
        """
        :param dimension: overall dimension of fingerprint
        :type dimension: int
        :param bin_size: size of each bin
        :type bin_size: float
        :param first_bin_left: the position of the first bin left side
        :type first_bin_left: float
        :param count_max: the maximum count to allow.
        :type count_max: int
        :param mz_ratio: allow peak to be in two bins with scaled intensity
        """
        if column_name is None:
            self.column_name = 'spectral_fp'
        super().__init__(dimension=dimension, *args, **kwargs)
        # maximum number of counting bits, ignoring 2^0 bit
        self.count_max = int(math.log(count_max, 2))
        self.mz_ratio = mz_ratio

    def to_bitvec(self):
        raise NotImplementedError()

    def to_numpy(self):
        return self.fingerprint

    def from_bitvec(self, array):
        raise NotImplementedError()

    def from_numpy(self, array):
        self.fingerprint = array

    def object2fingerprint(
            self,
            obj, dtype=np.float32):
        """
        fill out an array of fixed size with the ions.  note that this func assumes spectra sorted by mz
        """

        array = np.zeros((self.dimension,), dtype=np.float32)
        max_bin = self.dimension - self.count_max

        for i in range(len(obj.products.mz)):
            mz = obj.products.mz[i] - self.first_bin_left

            which_bin = int(mz / self.bin_size)
            intensity = math.sqrt(obj.products.intensity[i])
            if self.mz_ratio:
                ratio = (mz/self.bin_size - which_bin) / self.bin_size
                if ratio > 0.5:
                    which_bin2 = which_bin + 1
                    ratio1 = 1.5 - ratio
                else:
                    which_bin2 = which_bin - 1
                    ratio1 = ratio + 0.5
                ratio2 = 1.0 - ratio1
                if 0 <= which_bin < max_bin and 0 <= which_bin2 < max_bin:
                    # compare to the previous peak and skip if the last peak was in the same mz bin and was more intense
                    array[which_bin] += intensity * ratio1
                    array[which_bin2] += intensity * ratio2
            else:
                if 0 <= which_bin < max_bin:
                    array[which_bin] += intensity

        # add on counts as power of two, skipping the first bit
        if len(obj.products.mz) > 1:
            array_max = array.max()
            for i in range(1, self.count_max):
                if len(obj.products.mz) >= 2 ** i:
                    # set bit if count is greater than current power of 2
                    array[max_bin + i - 1] = array_max/2.0
                else:
                    # scale last count by remainder of last power of 2
                    array[max_bin + i - 1] = array_max/2.0 * (math.log(len(obj.products.mz), 2) + 1 - i)
                    break

        # l2 norm
        array = array/np.linalg.norm(array, ord=2)
        if dtype == np.uint8:
            array = array * 255
            array[(array < 1.0) & (array > 0.0)] = 1.0
            array = array.astype(np.uint8)

        self.fingerprint = array
        return array


class SpectrumTanimotoFingerPrint(SpectrumFingerprint):
    """
    create a spectral tanimoto fingerprint

    :param dimension: overall dimension of fingerprint
    :param bin_size: size of each bin
    :param first_bin_left: the position of the first bin left side
    """

    def __init__(self, dimension=2000, column_name=None, *args, **kwargs):
        if column_name is None:
            self.column_name = 'tanimoto_spectral_fp'
        super().__init__(dimension, *args, **kwargs)

    def to_bitvec(self):
        return self.fingerprint

    def to_numpy(self):
        return self.bitvec2numpy()

    def from_bitvec(self, array):
        self.fingerprint = array

    def from_numpy(self, array):
        self.numpy2bitvec(array)

    def object2fingerprint(self, obj, dtype=np.float32):
        # create a rdkit ExplicitBitVect that corresponds to the spectra.  Each bit position
        # corresponds to an integer mz value and is set if the intensity is above min_intensity
        self.fingerprint = DataStructs.ExplicitBitVect(self.dimension)
        for i in range(len(obj.products.mz)):
            mz = int(obj.products.mz[i] - self.first_bin_left)
            which_bin = int(mz / self.bin_size)
            if 0 < which_bin < self.dimension:
                self.fingerprint.SetBit(mz)


if __name__ == "__main__":
    import unittest

    from ..spectra.spectrum import MassInfo, Spectrum

    class TestFingerprintMethods(unittest.TestCase):
        """
        unit tests for fingerprints
        """

        predicted_spectrum1 = Spectrum()
        predicted_spectrum1.from_arrays(
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
            product_mass_info=MassInfo(
                tolerance=0.05,
                tolerance_type="daltons",
                mass_type="monoisotopic",
                evenly_spaced=True
            )
        )

        spectrum1 = Spectrum()
        spectrum1.from_arrays(
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
            product_mass_info=MassInfo(
                tolerance=10.0,
                tolerance_type="ppm",
                mass_type="monoisotopic",
                evenly_spaced=False
            )
        )

        spectrum2 = Spectrum()
        spectrum2.from_arrays(
            [173.0928, 527.3327, 955.5608],
            [
                11619800.0,
                25972200.0,
                9451650.0,
            ],
            product_mass_info=MassInfo(
                tolerance=10.0,
                tolerance_type="ppm",
                mass_type="monoisotopic",
                evenly_spaced=False
            )
        )

        def test_dynamic(self):
            fp = SpectrumTanimotoFingerPrint(dimension=1000)
            fp.object2fingerprint(self.spectrum2)
            self.assertSequenceEqual(list(fp.to_bitvec().GetOnBits()), [172, 526, 955])
            return

    unittest.main()


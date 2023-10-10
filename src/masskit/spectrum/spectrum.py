import copy
import logging
import math
import random
import re
from abc import ABC, abstractmethod
from base64 import b64encode
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.stats as sts

from .. import accumulator as _mkaccumulator
from ..data_specs import schemas as _mkschemas
from ..peptide import encoding as _mkencoding
from ..utils import fingerprints as _mkfingerprints
from ..utils import textalloc as _mktextalloc
from . import ipython as _mkipython
from . import spectrum_plotting as _mkspectrum_plotting

# try:
#     from numba import jit
# except ImportError:
#     # when numba is not available, define jit decorator
#     def jit(nopython=True):
#         def decorator(func):
#             def newfn(*args, **kwargs):
#                 return func(*args, **kwargs)

#             return newfn

#         return decorator


def nce2ev(nce, precursor_mz, charge):
    """
    convert nce to ev.  Equation for QE and taken from 
    http://proteomicsnews.blogspot.com/2014/06/normalized-collision-energy-calculation.html

    :param nce: normalized collision energy
    :param precursor_mz: precursor m/z
    :param charge: charge
    :return: ev
    """
    charge = abs(charge)
    if charge <= 1:
        factor = 1.0
    elif charge == 2:
        factor = 0.9
    elif charge == 3:
        factor = 0.85
    elif charge == 4:
        factor = 0.8
    elif charge >= 5:
        factor = 0.75
    return factor * nce * precursor_mz/500.0


class MassInfo:
    """
    information about mass measurements of an ion peak
    """

    def __init__(
            self,
            tolerance: float = None,
            tolerance_type: str = None,
            mass_type: str = None,
            neutral_loss: str = None,
            neutral_loss_charge: int = None,
            evenly_spaced = False,
            arrow_struct_accessor = None,
            arrow_struct_scalar = None,
    ):
        """
        initialize

        :param tolerance: mass tolerance.  If 0.5 daltons, this is unit mass
        :param tolerance_type: type of tolerance: "ppm", "daltons"
        :param mass_type: "monoisotopic" or "average"
        :param neutral_loss: # neutral loss chemical formula
        :param neutral_loss_charge: sign of neutral loss
        :param evenly_spaced: spectra m/z values are evenly spaced
        :param arrow_struct_accessor: the same as above, but taken from an arrow row_view() object
        :param arrow_struct_scalar: the same as above, but taken from an arrow StructScalar
        """
        if arrow_struct_accessor:
            self.tolerance = arrow_struct_accessor.tolerance()
            self.tolerance_type = arrow_struct_accessor.tolerance_type()
            self.mass_type = arrow_struct_accessor.mass_type()
            self.neutral_loss = arrow_struct_accessor.neutral_loss()
            self.neutral_loss_charge = arrow_struct_accessor.neutral_loss_charge()
            self.evenly_spaced = arrow_struct_accessor.evenly_spaced()
        elif arrow_struct_scalar:
            self.tolerance = arrow_struct_scalar['tolerance'].as_py()
            self.tolerance_type = arrow_struct_scalar['tolerance_type'].as_py()
            self.mass_type = arrow_struct_scalar['mass_type'].as_py()
            self.neutral_loss = arrow_struct_scalar['neutral_loss'].as_py()
            self.neutral_loss_charge = arrow_struct_scalar['neutral_loss_charge'].as_py()
            self.evenly_spaced = arrow_struct_scalar['evenly_spaced'].as_py()
        else:
            self.tolerance = tolerance
            self.tolerance_type = tolerance_type
            self.mass_type = mass_type
            self.neutral_loss = neutral_loss
            self.neutral_loss_charge = neutral_loss_charge
            self.evenly_spaced = evenly_spaced


class IonsIterator:
    """
    iterator over ion mz and intensity
    """

    def __init__(self, ions):
        self._ions = ions
        self._index = 0

    def __next__(self):
        if self._index < len(self._ions.mz):
            return_val = self._ions.mz[self._index], self._ions.intensity[self._index]
            self._index += 1
            return return_val
        raise StopIteration


class Ions(ABC):
    """
    base class for a series of ions
    """

    def __init__(
            self,
            mz=None,
            intensity=None,
            stddev=None,
            annotations=None,
            mass_info: MassInfo = None,
            jitter=0,
            copy_arrays=True,
            tolerance=None,
                ):
        """
        :param mz: mz values in array-like or scalar
        :param intensity: corresponding intensities in array-like or scalar
        :param stddev: the standard deviation of the intensity
        :param annotations: peak annotations
        :param mass_info:  dict containing mass type, tolerance, tolerance type
        :param jitter: used to add random jitter value to the mz values.  useful for creating fake data
        :param copy_arrays: if the inputs are numpy arrays, make copies
        :param tolerance: mass tolerance array
        """

        def init_arrays(array_in, copy_arrays_in):
            """
            return numpy arrays of input, avoiding copying if requested

            :param array_in: input array
            :param copy_arrays_in: should the array be copied?
            """
            if array_in is not None:
                # numpy array
                if isinstance(array_in, np.ndarray):
                    if not copy_arrays_in:
                        return array_in
                    else:
                        return np.array(array_in)

                # pyarrow Array
                if hasattr(array_in, "type") and (
                        pa.types.is_large_list(array_in.type) or pa.types.is_list(array_in.type) or\
                        pa.types.is_struct(array_in.type)):
                    if not copy_arrays_in:
                        return array_in
                    else:
                        return copy.deepcopy(array_in)

                # a py list
                return np.array(array_in)

            else:
                return None

        if hasattr(mz, "__len__") and hasattr(
                intensity, "__len__"
        ):  # check to see if both are array-like
            if len(mz) != len(intensity):
                raise ValueError("mz and intensity arrays are of different length")

            self.mz = init_arrays(mz, copy_arrays)
            self.intensity = init_arrays(intensity, copy_arrays)
            self.stddev = init_arrays(stddev, copy_arrays)
            self.annotations = init_arrays(annotations, copy_arrays)

            # sort by mz if not sorted already.  This is required for spectrum matching
            # note that this does create new arrays
            # investigate using in place sort: https://stackoverflow.com/a/52916896
            if not np.all(np.diff(self.mz) >= 0):
                sorted_indexes = self.mz.argsort()
                self.mz = self.mz[sorted_indexes]
                self.intensity = self.intensity[sorted_indexes]
                if stddev is not None:
                    self.stddev = self.stddev[sorted_indexes]
                if annotations is not None:
                    # TODO: the relationship of annotations is one to many, so this needs to be fixed.
                    self.annotations = self.annotations.take(pa.array(sorted_indexes))

            # peak rank by intensity
            self.rank = None
        else:
            # precursor
            self.mz = mz
            self.intensity = intensity
            self.annotations = annotations
            self.stddev = stddev
            self.rank = 1

        self.mass_info = mass_info
        self.jitter = jitter
        self.tolerance = tolerance  # used for high resolution spectra, mass tolerance
        self.join = None
        return

    def __iter__(self):
        """
        return an iterator

        :return: iterator
        """
        return IonsIterator(self)

    def __len__(self):
        """
        return the length of the ions

        :return: int length
        """
        return len(self.mz)

    def __getitem__(self, item):
        """
        return an ion by index

        :param item: index
        :return: mz, intensity
        """
        if item >= len(self.mz) or item < 0:
            return IndexError
        return self.mz[item], self.intensity[item]

    # def __getstate__(self):
    #     """
    #     omit members from pickle and jsonpickle
    #
    #     :return:
    #     """
    #     state = self.__dict__.copy()
    #     for k in ['_rank', '_starts', '_stops']:
    #         state.pop(k, None)
    #     return state
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    def clear_computed_properties(self):
        """
        clear properties that are lazy computed

        :return: returns self
        """
        self.tolerance = None
        self.rank = None
        return self

    @staticmethod
    def cast_mz(mz):
        """
        cast a single mz value
        """
        return mz

    @staticmethod
    def cast_intensity(intensity):
        """
        cast a single mz value
        """
        return intensity

    @property
    def tolerance(self):
        if self.mass_info is None:
            return None
        return self.mass_info.tolerance

    @property
    def tolerance_type(self):
        if self.mass_info is None:
            return None
        return self.mass_info.tolerance_type

    @property
    def mass_type(self):
        if self.mass_info is None:
            return None
        return self.mass_info.mass_type

    @property
    def neutral_loss(self):
        if self.mass_info is None:
            return None
        return self.mass_info.neutral_loss

    @property
    def neutral_loss_charge(self):
        if self.mass_info is None:
            return None
        return self.mass_info.neutral_loss_charge

    @property
    def mz(self):
        """
        :return: the mz of the precursor.  could be a numpy array.  optionally adds a jitter to the m/z values
        """
        # check to see if jitter exists as old versions of data don't have it
        if hasattr(self, "jitter") and self.jitter != 0:
            return self._mz + self.jitter
        else:
            return self._mz

    @mz.setter
    def mz(self, value):
        self._mz = value
        return

    @property
    def intensity(self):
        """
        :return: the intensity of the precursor.  could be a numpy array
        """
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = value
        return

    @property
    def rank(self):
        """
        return ranks of peaks by intensity
        1=most intense, rank is integer over the size of the intensity matrix

        :return: the rank of the ions by intensity.  could be a numpy array. 
        """
        if not hasattr(self, "_rank") or self._rank is None:
            self.rank_ions()
        return self._rank

    @rank.setter
    def rank(self, value):
        self._rank = value
        return

    @property
    def stddev(self):
        """
        :return: the std dev of the intensity per peak
        """
        return self._stddev

    @stddev.setter
    def stddev(self, value):
        self._stddev = value
        return

    @property
    def annotations(self):
        """
        per peak annotations
        """
        return self._annotations

    @annotations.setter
    def annotations(self, value):
        self._annotations = value
        return

    @property
    def join(self):
        """
        returns a dictionary with information on how to join this ion set to another ion set, such as for annotation
        """
        return self._join

    @join.setter
    def join(self, value):
        self._join = value
        return

    def rank_ions(self):
        """
        rank the ions. intensity rank, 1=most intense, rank is integer over the size of the intensity matrix
        """
        # question:  why is this subtracting from the size of the array?  Obviously to change sign, but otherwise why?
        self.rank = sts.rankdata(
            self.intensity.shape[0] - self.intensity + 1, method="ordinal"
        )
        self.rank = self.rank.astype(int)
        return

    def copy(self, min_mz=-1, max_mz=0, min_intensity=-1, max_intensity=0):
        """
        create filtered version of self.  This is essentially a copy constructor

        :param min_mz: minimum mz value
        :param max_mz: maximum mz value.  0 = ignore
        :param min_intensity: minimum intensity value
        :param max_intensity: maximum intensity value.  0 = ignore
        :return: copy
        """
        return self.filter(
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            inplace=False,
        )

    def filter(
            self, min_mz=-1, max_mz=0, min_intensity=-1, max_intensity=0, inplace=False
    ):
        """
        filter ions by mz and/or intensity.

        :param min_mz: minimum mz value, exclusive
        :param max_mz: maximum mz value, inclusive.  0 = ignore
        :param min_intensity: minimum intensity value, exclusive
        :param max_intensity: maximum intensity value, inclusive.  0 = ignore
        :param inplace: do operation on current ions, otherwise create copy
        :return: filtered copy if not inplace, otherwise current ions
        """
        if not inplace:
            return_ions = copy.deepcopy(self)
        else:
            return_ions = self
        mask = (return_ions.mz > min_mz) & (return_ions.intensity > min_intensity)
        if max_mz:
            mask &= return_ions.mz <= max_mz
        if max_intensity:
            mask &= return_ions.intensity <= max_intensity
        return Ions.mask_ions(mask, return_ions)

    @staticmethod
    def mask_ions(mask, return_ions):
        """
        mask out a set of ions

        :param mask: boolean mask
        :param return_ions: ions to be masked
        :return: masked ions
        """
        return_ions.mz = return_ions.mz[mask]
        return_ions.intensity = return_ions.intensity[mask]
        if return_ions.annotations is not None:
            # TODO: the relationship of annotations is one to many, so this needs to be fixed. 
            return_ions.annotations = return_ions.annotations.filter(pa.array(mask))
        if return_ions.stddev is not None:
            return_ions.stddev = return_ions.stddev[mask]
        # join information is deleted when masking is performed
        return_ions.join = None
        return_ions.clear_computed_properties()
        return return_ions

    def parent_filter(
            self, h2o=True, inplace=False, precursor_mz=0.0, charge=None
    ):
        """
        filter parent ions, including water losses.

        :param h2o: filter out water losses
        :param inplace: do operation on current ions, otherwise create copy
        :param precursor_mz: precursor m/z
        :param charge: charge of precursor
        :return: filtered copy if not inplace, otherwise current ions
        """
        if not inplace:
            return_ions = copy.deepcopy(self)
        else:
            return_ions = self
        if precursor_mz == 0:
            return return_ions
        tolerance = self.half_tolerance(precursor_mz)
        mask = (return_ions.mz < precursor_mz - tolerance) | (return_ions.mz > precursor_mz + tolerance)
        if h2o and charge is not None and charge != 0:
            h2o_neutral_loss = (precursor_mz * charge - _mkencoding.h2o_mass)/charge
            mask &= (return_ions.mz < h2o_neutral_loss - tolerance) | (return_ions.mz > h2o_neutral_loss + tolerance)
            h2o_neutral_loss = (precursor_mz * charge - 2 * _mkencoding.h2o_mass)/charge
            mask &= (return_ions.mz < h2o_neutral_loss - tolerance) | (return_ions.mz > h2o_neutral_loss + tolerance)
        return Ions.mask_ions(mask, return_ions)

    def windowed_filter(
            self, mz_window=7, num_ions=5, inplace=False
    ):
        """
        filter ions by examining peaks in order of intensity and filtering peaks within a window

        :param mz_window: half size of mz_window for filtering.  0 = no filtering
        :param num_ions: number of ions allowed in full mz_window
        :param inplace: do operation on current ions, otherwise create copy
        :return: filtered copy if not inplace, otherwise current ions
        """
        if not inplace:
            return_ions = copy.deepcopy(self)
        else:
            return_ions = self

        if mz_window == 0:
            return return_ions

        mask = np.full(self.mz.shape, True)
        rank_indices = np.argsort(self.rank)
        for pos in rank_indices:
            # if already masked, ignore
            if not mask[pos]:
                continue
            # create mask of ions around ps
            windowed_ions = (self.mz < self.mz[pos] + mz_window) & (self.mz > self.mz[pos] - mz_window)
            # get the ranks of the windowed ions
            windowed_ranks = self.rank[windowed_ions]
            # if there are more ions than allowed in the window
            if len(windowed_ranks) > num_ions:
                # sort the rank of the windowed ions
                sorted_ranks = np.sort(windowed_ranks)
                # find the rank value of the first peak to be deleted
                partition_rank = sorted_ranks[num_ions]
                # create mask of all peaks to be deleted
                windowed_ions = windowed_ions & (self.rank >= partition_rank)
                # add the result to the mask
                mask = mask & ~windowed_ions

        # mask out the spectrum
        return Ions.mask_ions(mask, return_ions)

    def norm(self, max_intensity_in=999, keep_type=True, inplace=False, ord=None):
        """
        norm the intensities

        :param max_intensity_in: the intensity of the most intense peak
        :param keep_type: keep the type of the intensity array
        :param inplace: do operation on current ions, otherwise create copy
        :param ord: if set, normalize using norm order as in np.linalg.norm. 2 = l2
        :returns: normed copy if not inplace, otherwise current ions
        """
        if not inplace:
            return_ions = self.copy()
        else:
            return_ions = self
        if len(return_ions.intensity) != 0:
            if ord is None:
                max_intensity = np.max(return_ions.intensity)
            else:
                max_intensity = np.linalg.norm(return_ions.intensity, ord=ord)
            d_type = return_ions.intensity.dtype
            return_ions.intensity = (
                    return_ions.intensity / float(max_intensity) * max_intensity_in
            )
            if return_ions.stddev is not None:
                return_ions.stddev = (
                        return_ions.stddev / float(max_intensity) * max_intensity_in
                )
            if keep_type:
                return_ions.intensity = return_ions.intensity.astype(
                    d_type
                )  # cast back to original type
                if return_ions.stddev is not None:
                    return_ions.stddev = return_ions.stddev.astype(d_type)
        return return_ions

    def mask(self, indices, inplace=False):
        """
        mask out ions that are pointed to by the indices

        :param indices: indices of ions to screen out or numpy boolean mask
        :param inplace: do operation on current ions
        :returns: masked copy if not inplace, otherwise current ions
        """
        if not inplace:
            return_ions = self.copy()
        else:
            return_ions = self
        if isinstance(indices, np.ndarray) and indices.dtype == np.bool_:
            # flip the index mask so that we select the non selected elements
            indices = ~indices
            return_ions.mz = return_ions.mz[indices]
            return_ions.intensity = return_ions.intensity[indices]
            if return_ions.annotations is not None:
                # TODO: the relationship of annotations is one to many, so this needs to be fixed.
                return_ions.annotations = return_ions.annotations.take(pa.array(indices))
            if return_ions.stddev is not None:
                return_ions.stddev = return_ions.stddev[indices]
        else:
            return_ions.mz = np.delete(return_ions.mz, indices)
            return_ions.intensity = np.delete(return_ions.intensity, indices)
            if return_ions.annotations is not None:
                # TODO: the relationship of annotations is one to many, so this needs to be fixed. 
                mask = np.ones(len(return_ions.annotations), np.bool)
                mask[indices] = False
                return_ions.annotations = return_ions.annotations.filter(pa.array(mask))
            if return_ions.stddev is not None:
                return_ions.stddev = np.delete(return_ions.stddev, indices)
        # join information is deleted when masked
        return_ions.join = None
        return_ions.clear_computed_properties()
        return return_ions

    def shift_mz(self, shift, inplace=False):
        """
        shift the mz values of all ions by the value of shift.  Negative ions are masked out

        :param shift: value to shift mz
        :param inplace: do operation on current ions
        :returns: masked copy if not inplace, otherwise current ions
        """
        if not inplace:
            return_ions = self.copy()
        else:
            return_ions = self
        return_ions.mz += shift
        mask = return_ions.mz > 0
        return_ions.mz = return_ions.mz[mask]
        return_ions.intensity = return_ions.intensity[mask]
        if return_ions.annotations is not None:
            # TODO: the relationship of annotations is one to many, so this needs to be fixed.
            return_ions.annotations = return_ions.filter(pa.array(mask))
        if return_ions.stddev is not None:
            return_ions.stddev = return_ions.stddev[mask]
        return_ions.join = None
        return_ions.clear_computed_properties()
        return return_ions

    def merge(self, merge_ions, inplace=False):
        """
        merge another set of ions into this one.

        :param merge_ions: the ions to add in
        :param inplace: do operation on current ions
        :returns: merged copy if not inplace, otherwise current ions
        """
        if not hasattr(self.mz, "__len__") or not hasattr(merge_ions.mz, "__len__"):
            raise NotImplementedError(
                "merging ions without mz arrays is not implemented"
            )
        if not inplace:
            return_ions = self.copy()
        else:
            return_ions = self

        return_ions.mz = np.concatenate((return_ions.mz, merge_ions.mz))
        # get indices to sort the mz array
        sorted_indexes = return_ions.mz.argsort()
        return_ions.mz = return_ions.mz[sorted_indexes]

        return_ions.intensity = np.concatenate(
            (return_ions.intensity, merge_ions.intensity)
        )
        return_ions.intensity = return_ions.intensity[sorted_indexes]
        if return_ions.stddev is not None and merge_ions.stddev is not None:
            return_ions.stddev = np.concatenate((return_ions.stddev, merge_ions.stddev))
            return_ions.stddev = return_ions.stddev[sorted_indexes]
        else:
            return_ions.stddev = None
        if return_ions.annotations is not None or merge_ions.annotations is not None:
            # TODO: the relationship of annotations is one to many, so this needs to be fixed. 
            return_ions.annotations = pa.concat_arrays([return_ions.annotations, merge_ions.annotations])
            return_ions.annotations = return_ions.annotations.take(pa.array(sorted_indexes))
        else:
            return_ions.annotations = None
        # join information is deleted when merged
        return_ions.join = None
        return_ions.clear_computed_properties()

        return return_ions

    def half_tolerance(self, mz):
        """
        calculate 1/2 of the tolerance interval

        :param mz: mz value
        :return: 1/2 tolerance interval
        """
        is_mz_array = isinstance(mz,(list,pd.core.series.Series,np.ndarray,pa.Array))
        if self.mass_info is None:
            if is_mz_array:
                return np.full_like(mz, 0.0, dtype=np.float64)
            else:
                return 0.0
        if self.mass_info.tolerance_type == "ppm":
            return mz * self.mass_info.tolerance / 1000000.0
        elif self.mass_info.tolerance_type == "daltons":
            if is_mz_array:
                return np.full_like(mz, self.mass_info.tolerance, dtype=np.float64)
            else:
                return self.mass_info.tolerance
        else:
            raise ValueError(
                f"mass tolerance type {self.mass_info.tolerance_type} not supported"
            )

    @abstractmethod
    def intersect(self, comparison_ions, tiebreaker=None):
        pass

    def copy_annot(self, ion2, index1, index2):
        """
        copy annotations from ion2 to this set of ions using the matched ion indices

        :param ion2: the ions to compare against
        :param index1: matched ions in this set of ions
        :param index2: matched ions in ions2
        """

        if ion2.annotations is None:
            return

        indices = []
        index1 = np.array(index1)
        for i in range(len(self.mz)):
            pos = np.argwhere(index1 == i)
            if len(pos) > 0 and len(pos[0]) > 0:
                indices.append(index2[pos[0][0]])
            else:
                indices.append(None)
        # TODO: the relationship of annotations is one to many, so this needs to be fixed.  Also,
        # sometimes the annotations is a LargeListScalar, and sometimes an Array.  why?
        self.annotations = ion2.annotations.take(indices)

    def clear_and_intersect(self, ion2, index1, index2, tiebreaker=None):
        """_
        if indices are not provided, clear the ions of both self and ion2 of zero intensity peaks
        then intersect

        :param ion2: comparison ions
        :param index1: intersection indices for self (can be None)
        :param index2: intersection indices for ion2 (can be None)
        :param tiebreaker: how to deal with one to multiple matches to peaks in spectra1. mz is closest mz value, intensity is closest intensity, None is report multiple matches
        :return: _description_
        """
        if index1 is None or index2 is None:
            if not np.all(self.intensity):
                ion1 = self.filter(min_intensity=0.0)
            else:
                ion1 = self
            if not np.all(ion2.intensity):
                ion2 = ion2.filter(min_intensity=0.0)
            
            index1, index2 = ion1.intersect(ion2, tiebreaker=tiebreaker)
        else:
            ion1 = self
        return ion1, ion2, index1, index2
            
    def cosine_score(
            self,
            ion2,
            index1=None,
            index2=None,
            mz_power=0.0,
            intensity_power=0.5,
            scale=999,
            skip_denom=False,
            tiebreaker=None
    ):
        """
        calculate the cosine score between this set of ions and ions2

        :param ion2: the ions to compare against
        :param index1: matched ions in this set of ions
        :param index2: matched ions in ions2
        :param mz_power: what power to raise the mz value for each peak
        :param intensity_power: what power to raise the intensity for each peak
        :param scale: what value to scale the score by
        :param skip_denom: skip computing the denominator
        :param tiebreaker: how to deal with one to multiple matches to peaks in self. mz is closest mz value, intensity is closest intensity, None is no tiebreaking
        :return: cosine score
        """
        if self.mz is None or ion2.mz is None:
            return 0.0
        
        # if necessary, screen out any zero intensity peaks as zero intensity peaks mess up the cosine
        # score calculation when there are many to many matches of peaks
        ion1, ion2, index1, index2 = self.clear_and_intersect(ion2, index1, index2, tiebreaker=tiebreaker)            
            
        return cosine_score_calc(
            ion1.mz,
            ion1.intensity,
            ion2.mz,
            ion2.intensity,
            index1,
            index2,
            mz_power=mz_power,
            intensity_power=intensity_power,
            scale=scale,
            skip_denom=skip_denom
        )

    def total_intensity(self):
        """
        total intensity of ions
        :return: total intensity
        """
        return np.sum(self.intensity)

    def num_ions(self):
        """
        number of ions in spectrum
        :return: number of ions in spectrum
        """
        if self.mz is None:
            return 0
        else:
            return len(self.mz)

    def ions2array(
            self,
            array,
            channel,
            bin_size=1.0,
            precursor=0,
            intensity_norm=1.0,
            insert_mz=False,
            mz_norm=2000.0,
            rand_intensity=0.0,
            down_shift=0.0,
            channel_first=True,
            take_max=True,
            stddev_channel=None,
            take_sqrt=False,
    ):
        """
        fill out an array of fixed size with the ions.  note that this func assumes spectra sorted by mz

        :param array: the array to fill out
        :param channel: which channel to fill out in the array
        :param bin_size: the size of each bin in the array
        :param precursor: if nonzero, use this value to invert the spectra by subtracting mz from this value
        :param intensity_norm: value to norm the intensity
        :param insert_mz: instead of putting the normalized intensity in the array, put in the normalized mz
        :param mz_norm: the value to use to norm the mz values inserted
        :param rand_intensity: if not 0, multiply each intensity by random value 1 +/- rand_intensity
        :param down_shift: shift mz down by this value in Da
        :param channel_first: channel before spectrum in input array (pytorch style).  tensorflow is channel last.
        :param take_max: take the maximum intensity in a bin rather the sum of peaks in a bin
        :param stddev_channel: which channel contains the std dev.  None means no std dev
        :param take_sqrt: take the square root of the intensity
        """
        if self.mz is None:
            return
        last_which_bin = -1
        last_intensity = -1
        last_stddev = -1
        augment_intensity = rand_intensity != 0.0
        lo_fraction = 1.0 - rand_intensity  # lower bound of augmentation
        hi_fraction = 1.0 + rand_intensity  # higher bound of augmentation
        if channel_first:
            max_bin = array.shape[1]
        else:
            max_bin = array.shape[0]
        if stddev_channel is not None and self.stddev is not None:
            do_stddev = True
        else:
            do_stddev = False
        if take_sqrt:
            intensities = np.sqrt(self.intensity)
        else:
            intensities = self.intensity

        for i in range(self.mz.size):
            mz = self.mz[i] - down_shift
            if precursor != 0:
                mz = precursor - mz
            which_bin = int(mz / bin_size)
            if 0 <= which_bin < max_bin:
                # compare to the previous peak and skip if the last peak was in the same mz bin and was more intense
                intensity = intensities[i]
                if do_stddev:
                    stddev = self.stddev[i]
                else:
                    stddev = 0
                if which_bin == last_which_bin:
                    if take_max:
                        if intensity <= last_intensity:
                            continue
                    else:
                        intensity += last_intensity
                        if do_stddev:
                            stddev = math.sqrt(stddev ** 2 + last_stddev ** 2)

                last_which_bin = which_bin
                last_intensity = intensity
                if do_stddev:
                    last_stddev = stddev

                if insert_mz:
                    if channel_first:
                        array[channel, which_bin] = self.mz[i] / mz_norm
                    else:
                        array[which_bin, channel] = self.mz[i] / mz_norm
                else:
                    if channel_first:
                        array[channel, which_bin] = intensity / intensity_norm
                        if do_stddev:
                            array[stddev_channel, which_bin] = stddev / intensity_norm
                        if augment_intensity:
                            array[channel, which_bin] *= random.uniform(
                                lo_fraction, hi_fraction
                            )
                    else:
                        array[which_bin, channel] = intensity / intensity_norm
                        if do_stddev:
                            array[which_bin, stddev_channel] = stddev / intensity_norm
                        if augment_intensity:
                            array[which_bin, channel] *= random.uniform(
                                lo_fraction, hi_fraction
                            )

    def evenly_space(self, tolerance=None, take_max=True, max_mz=None, include_zeros=False, take_sqrt=False):
        """
        convert ions to  product ions with evenly spaced m/z bins.  The m/z bins are centered on
        multiples of tolerance * 2.  Multiple ions that map to the same bin are either summed or the max taken of the
        ion intensities.

        :param tolerance: the mass tolerance of the evenly spaced m/z bins (bin width is twice this value) in daltons
        :param take_max: for each bin take the maximum intensity ion, otherwise sum all ions mapping to the bin
        :param max_mz: maximum mz value, 2000 by default
        :param include_zeros: fill out array including bins with zero intensity
        :param take_sqrt: take the sqrt of the intensities
        """
        if tolerance is None and self.mass_info is not None:
            if self.mass_info.tolerance_type == "daltons":
                tolerance = self.mass_info.tolerance
            else:
                raise ValueError('please specify fixed mass tolerance for evenly_space')
        if tolerance is None:
            raise ValueError("unable to evenly space ions")

        if max_mz is None:
            max_mz = 2000.0
        num_bins = int(max_mz / (2 * tolerance))

        if self.stddev is not None:
            # we don't subtract one due to down_shift
            intensity_array = np.zeros((2, num_bins))
            stddev_channel = 1
        else:
            # we don't subtract one due to down_shift
            intensity_array = np.zeros((1, num_bins))
            stddev_channel = None

        self.ions2array(
            intensity_array,
            0,
            bin_size=tolerance * 2,
            precursor=0,
            insert_mz=False,
            down_shift=tolerance,
            channel_first=True,
            take_max=take_max,
            stddev_channel=stddev_channel,
            take_sqrt=take_sqrt,
        )

        if include_zeros:
            self.mz = np.linspace(tolerance * 2, max_mz, num_bins)
            self.intensity = intensity_array[0]
            if stddev_channel is not None:
                self.stddev = intensity_array[stddev_channel]
        else:
            mzs = []
            intensities = []
            stddevs = []

            for i in np.nonzero(intensity_array[0])[0]:
                mzs.append((i + 1) * 2 * tolerance)  # i + 1 to account for the down_shift
                intensities.append(intensity_array[0, i])
                if stddev_channel is not None:
                    stddevs.append(intensity_array[stddev_channel, i])

            self.mz = np.array(mzs)
            self.intensity = np.array(intensities)
            if stddev_channel is not None:
                self.stddev = np.array(stddevs)

        if stddev_channel is None:
            self.stddev = None
        self.annotations = None
        self.join = None
        self.clear_computed_properties()

        if self.mass_info is not None:
            self.mass_info.tolerance_type = "daltons"
            self.mass_info.tolerance = tolerance
            self.mass_info.evenly_spaced = True
        else:
            self.mass_info = MassInfo(
                tolerance_type="daltons", tolerance=tolerance, evenly_spaced=True
            )



class HiResIons(Ions):
    """
    for containing high mass resolution ions
    """

    def __init__(self, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        if (
                "mass_info" not in kwargs or kwargs["mass_info"] is None
        ):  # set up some reasonable defaults
            self.mass_info = MassInfo(20.0, "ppm", "monoisotopic", "", 1)

    @property
    def tolerance(self):
        """
        :return: the mass tolerance for each peak bin
        """
        if not hasattr(self, "_tolerance") or self._tolerance is None:
            self.create_tolerance()
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value
        return
    
    @property
    def starts(self):
        """
        :return: the start positions for each peak bin
        """
        return self.mz - self.tolerance

    @property
    def stops(self):
        """
        :return: the stop positions for each peak bin
        """
        return self.mz + self.tolerance

    def create_tolerance(self):
        """
        create start and stop arrays
        """
        self.tolerance = self.half_tolerance(self.mz)

    def intersect(self, comparison_ions, tiebreaker=None):
        """
        find the intersections between two high resolution ion series.  calls standalone function to allow use of
        numba

        :param comparison_ions: the ion series to compare to
        :param tiebreaker: how to deal with one to multiple matches to peaks in spectra1. mz is closest mz value, intensity is closest intensity, None is report multiple matches
        :return: matched peak indexes in self, matched peak indexes in comparison_ions
        """
        index1, index2 = intersect_hires(
            self.starts,
            self.stops,
            comparison_ions.starts,
            comparison_ions.stops,
        )

        if tiebreaker is not None:
            index1, index2 = dedup_matches(self, comparison_ions, index1, index2, tiebreaker=tiebreaker, skip_nomatch=True)
            # cast to numpy, cast to deal with empty arrays, which don't cast safe to int
            index1 = np.array(index1).astype(np.int64)
            index2 = np.array(index2).astype(np.int64)

        return index1, index2

    @staticmethod
    def cast_mz(mz):
        """
        cast a single mz value
        """
        return float(mz)

    @staticmethod
    def cast_intensity(intensity):
        """
        cast a single mz value
        """
        return float(intensity)

    def change_mass_info(self, mass_info, inplace=False, take_max=True):
        """
        given a new mass info, recalculate tolerance bins

        :param mass_info: the MassInfo structure to change to
        :param inplace: if true, change in place, otherwise return copy
        :param take_max: for each bin take the maximum intensity ion, otherwise sum all ions mapping to the bin
        """
        if not inplace:
            return_ions = copy.deepcopy(self)
        else:
            return_ions = self

        return_ions.mass_info = copy.deepcopy(mass_info)
        if hasattr(mass_info, "evenly_spaced") and mass_info.evenly_spaced:
            self.evenly_space(take_max=take_max)
        return_ions.create_tolerance()
        return return_ions


#@jit(nopython=True, cache=True)
def my_intersect1d(ar1, ar2):
    """
    simplified version of numpy intersect1d.  Pull outside of class so it can be jit compiled by numba (numba has only
    experimental class support).
    Note: this function does not work if there are ions in each spectra with identical mz!

    :param ar1: mz values for one spectra
    :param ar2: mz values for another spectra
    :return: index of matches into array 1, index of matches into array 2
    """
    aux = np.concatenate((ar1, ar2))
    aux_sort_indices = np.argsort(aux, kind="mergesort")
    aux = aux[aux_sort_indices]

    mask = aux[1:] == aux[:-1]

    ar1_indices = aux_sort_indices[:-1][mask]
    ar2_indices = aux_sort_indices[1:][mask] - ar1.size
    return ar1_indices, ar2_indices


#@jit(nopython=True, parallel=True, cache=True)
def cosine_score_calc(
        spectrum1_mz,
        spectrum1_intensity,
        spectrum2_mz,
        spectrum2_intensity,
        index1,
        index2,
        mz_power=0.0,
        intensity_power=0.5,
        scale=999,
        skip_denom=False
):
    """
    the Stein and Scott 94 cosine score.  By convention, sqrt of score is taken and
    multiplied by 999.  separated out from class and placed here so that can be jit compiled by numba.

    :param spectrum1_mz: query spectrum mz
    :param spectrum1_intensity: query spectrum intensity
    :param spectrum2_mz: the comparison spectrum2 mz
    :param spectrum2_intensity: the comparison spectrum2 intensity
    :param index1: matched ions in spectrum1.  may include duplicate matches
    :param index2: matched ions in spectrum2.  may include duplicate matches
    :param mz_power: what power to raise the mz value for each peak
    :param intensity_power: what power to raise the intensity for each peak
    :param scale: what value to scale the score by
    :param skip_denom: skip computing the denominator
    :return: the cosine score
    """
    weighted_spectrum1_intensity = spectrum1_intensity ** intensity_power
    weighted_spectrum2_intensity = spectrum2_intensity ** intensity_power
    if mz_power != 0.0:
        weighted_spectrum1_intensity *= spectrum1_mz ** mz_power
        weighted_spectrum2_intensity *= spectrum2_mz ** mz_power

    a = np.take(weighted_spectrum1_intensity, index1)
    b = np.take(weighted_spectrum2_intensity, index2)

    score = np.sum(np.multiply(a, b)) ** 2

    if not skip_denom:
        mask = np.ones_like(weighted_spectrum1_intensity, dtype=np.bool_)
        mask[index1] = False
        denominator = np.sum(weighted_spectrum1_intensity[index1]**2) + np.sum(weighted_spectrum1_intensity[mask]**2)

        mask = np.ones_like(weighted_spectrum2_intensity, dtype=np.bool_)
        mask[index2] = False
        denominator *= np.sum(weighted_spectrum2_intensity[index2]**2) + np.sum(weighted_spectrum2_intensity[mask]**2)

        if denominator != 0:
            score /= denominator

    return score * scale


#@jit(nopython=True, parallel=True, cache=True)
def intersect_hires(ions1_starts, ions1_stops, ions2_starts, ions2_stops):
    """
    find the intersections between two high resolution ion series

    :param ions1_starts: start positions of the first ion series to compare
    :param ions1_stops: stop positions of the first ion series to compare
    :param ions2_starts: start positions of the second ion series to compare
    :param ions2_stops: stop positions of the second ion series to compare
    :return: matched peak indexes in ions1, matched peak indexes in ion2
    """
    # for each index i in ions2, the start index in ion1 that overlaps
    start_indx = np.searchsorted(ions1_stops, ions2_starts, "left")
    # for each index i in ions2, the end index in ion1 that overlaps
    end_indx = np.searchsorted(ions1_starts, ions2_stops, "right")
    # list out the corresponding index into ions2
    ions2_index = np.arange(0, len(start_indx))
    # mask out no intersections
    mask = end_indx > start_indx
    start_indx = start_indx[mask]
    end_indx = end_indx[mask]
    ions2_index = ions2_index[mask]
    # now create numpy arrays with the intersection indices in ions1
    num_intersections = end_indx - start_indx
    size = np.sum(num_intersections)
    index1 = np.empty((size,), dtype=np.int64)
    index2 = np.empty((size,), dtype=np.int64)
    # current writing position in output arrays
    position = 0
    for i in range(len(start_indx)):
        # number of intersections for current peak in ions2
        num_slots = num_intersections[i]
        index1[position: position + num_slots] = np.arange(start_indx[i], end_indx[i])
        index2[position: position + num_slots] = ions2_index[i]
        position += num_slots

    return index1, index2


def dedup_matches(products1, products2, index1, index2, tiebreaker='mz', skip_nomatch=True):
    """
    given a series of indices to matched peaks in two product ion sets, get rid of duplicate matches to
    peaks in the first product ion set, using a tiebreaker.

    :param products1: first set of product ions
    :param products2: second set of product ions
    :param index1: indices into first set of product ions
    :param index2: indices into the second set of product ions
    :param tiebreaker: tiebreak by 'intensity' or 'mz' of duplicate matches. 'delete' means don't match either.  defaults to 'mz'
    :param skip_nomatch: in the return values, skip missing matches to the first set of produc tions, defaults to True
    :return: matches of the first product ion set, matches of the second product ion set
    """
    join_1_2 = []
    join_2_1 = []

    def dedup_matches_mz(products1, products2, index2, join_1_2, join_2_1, exp_peak_index, pos):
        first = True
        for i in range(pos.shape[0]):
            dmz = abs(products1.mz[exp_peak_index] - products2.mz[index2[pos[i, 0]]])
            if first or dmz < min_mz:
                min_i = index2[pos[i, 0]]
                min_mz = dmz
                first = False
        join_2_1.append(min_i)
        join_1_2.append(exp_peak_index)

    for exp_peak_index in range(len(products1.mz)):
        pos = np.argwhere(index1 == exp_peak_index)
        if pos.shape[0] != 0:
            if tiebreaker == "mz":
                dedup_matches_mz(products1, products2, index2, join_1_2, join_2_1, exp_peak_index, pos)
            elif tiebreaker == "intensity":
                first = True
                for i in range(pos.shape[0]):
                    dintensity = abs(products1.intensity[exp_peak_index] - products2.intensity[index2[pos[i, 0]]])
                    if first or dintensity < min_intensity:
                        min_i = index2[pos[i, 0]]
                        min_intensity = dintensity
                        first = False
                join_2_1.append(min_i)
                join_1_2.append(exp_peak_index)
            elif tiebreaker == "delete":
                if pos.shape[0] <= 1:
                    dedup_matches_mz(products1, products2, index2, join_1_2, join_2_1, exp_peak_index, pos)
            else:
                # don't use a tiebreaker
                for i in range(pos.shape[0]):
                    join_2_1.append(index2[pos[i, 0]])
                    join_1_2.append(exp_peak_index)
        elif not skip_nomatch:
            join_2_1.append(None)
            join_1_2.append(exp_peak_index)
    
    return join_1_2, join_2_1 

class Spectrum:
    """
    Base class for spectrum with called ions.
    The props attribute is a dict that contains any structured data
    """

    def __init__(self, precursor_mass_info=None, product_mass_info=None, name=None, id=None, 
                 ev=None, nce=None, charge=None, ion_class=HiResIons, mz=None, intensity=None,
                 row=None, struct=None, precursor_mz=None, precursor_intensity=None, stddev=None,
                 annotations=None, tolerance=None, copy_arrays=False):
        """
        construct a spectrum.  Can initialize with arrays (mz, intensity) or arrow row object

        :param mz: mz array
        :param intensity: intensity array
        :param stddev: standard deviation of intensity
        :param row: dict containing parameters and precursor info or arrow row object
        :param struct: arrow struct containing parameters and precursor info
        :param precursor_mz: precursor_mz value, used preferentially to row
        :param precursor_intensity: precursor intensity, optional
        :param annotations: annotations on the ions
        :param precursor_mass_info: MassInfo mass measurement information for the precursor
        :param product_mass_info: MassInfo mass measurement information for the product
        :param tolerance: mass tolerance array
        :param copy_arrays: if the inputs are numpy arrays, make copies
        """
        
        self.joins = []  # join data structures
        self.joined_spectra = []  # corresponding spectra to joins
        self.props = {}
        self.prop_names = None
        self.precursor_class = ion_class
        self.product_class = ion_class
        if precursor_mass_info is None:
            self.precursor_mass_info = MassInfo(20.0, "ppm", "monoisotopic", "", 1)
        else:
            self.precursor_mass_info = precursor_mass_info
        if product_mass_info is None:
            self.product_mass_info = MassInfo(20.0, "ppm", "monoisotopic", "", 1)
        else:
            self.product_mass_info = product_mass_info
        self.name = name
        self.id = id
        self.charge = charge
        self.ev= ev
        self.nce=nce
        self.precursor = None
        self.products = None
        self.filtered = None  # filtered version of self.products

        if mz is not None and intensity is not None:
            self.from_arrays(
                mz,
                intensity,
                row=row,
                precursor_mz=precursor_mz,
                precursor_intensity=precursor_intensity,
                stddev=stddev,
                annotations=annotations,
                precursor_mass_info=precursor_mass_info,
                product_mass_info=product_mass_info,
                copy_arrays=copy_arrays,
                tolerance=tolerance,
                )
        elif row is not None:
            self.from_arrow(row, copy_arrays=copy_arrays)
        elif struct is not None:
            self.from_struct(struct, copy_arrays=copy_arrays)
        

    # def __getstate__(self):
    #     """
    #     omit members from pickle, jsonpickle, and deep copy
    #
    #     :return:
    #     """
    #     state = self.__dict__.copy()
    #     state.pop('_filtered', None)
    #     return state
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    @property
    def filtered(self):
        """
        filtered version of product ions

        :return:
        """
        return self._filtered

    @filtered.setter
    def filtered(self, value):
        self._filtered = value
        return

    @property
    def props(self):
        """
        get property list

        :return: property list
        """
        return self._props

    @props.setter
    def props(self, value):
        self._props = value
        return

    def get_prop(self, name):
        """
        return a property from property list with None as default

        :param name: property name
        :return: value
        """
        return self.props.get(name)
    
    def get_props(self):
        """
        returns back all properties for this object

        :return: list of properties
        """
        return [p for p in dir(self.__class__) if isinstance(getattr(self.__class__, p), property)]

    def copy_props_from_dict(self, dict_in):
        """
        given a dictionary, e.g. a row, copy allowed properties in the spectrum props
        
        :param dict_in: the input dictionary.  Allowed properties are in self.prop_names
        """
        for prop in dict_in:
            if prop in self.get_props():
                self.props[prop] = copy.deepcopy(dict_in[prop])        

    def add_join(self, join, joined_spectrum):
        """
        add a join and a corresponding joined spectrum

        :param join: the result of the join expressed as a arrow struct
        :param joined_spectrum: the spectrum object that is joined to this spectrum via the join
        """
        self.joins.append(join)
        self.joined_spectra.append(joined_spectrum)

    def from_arrow(self,
                   row,
                   copy_arrays=False):
        """
        Update or initialize from an arrow row object

        :param row: row object from which to extract the spectrum information
        :param copy_arrays: if the inputs are numpy arrays, make copies
        """

        # loop through the experimental fields and if there is data, save it to the spectrum
        for field in _mkschemas.property_fields:
            attribute = row.get(field.name)
            if attribute is not None:
                setattr(self, field.name, attribute())
        
        self.charge = row.charge() if row.get('charge') is not None else None

        stddev = row.stddev() if row.get('stddev') is not None else None
        annotations = row.annotations() if row.get('annotations') is not None else None
        precursor_intensity = row.precursor_intensity() if row.get('precursor_intensity') is not None else None        

        self.precursor = self.precursor_class(
            mz=row.precursor_mz(),
            intensity=precursor_intensity,
            mass_info=MassInfo(arrow_struct_accessor=row.precursor_massinfo),
        )

        tolerance = row.tolerance() if row.get('tolerance') is not None else None

        self.products = self.product_class(
            row.mz(),
            row.intensity(),
            stddev=stddev,
            mass_info=MassInfo(arrow_struct_accessor=row.product_massinfo),
            annotations=annotations,
            copy_arrays=copy_arrays,
            tolerance=tolerance
        )
        return self

    def from_struct(self,
                   struct,
                   copy_arrays=False):
        """
        Update or initialize from an arrow struct object

        :param row: row object from which to extract the spectrum information
        :param copy_arrays: if the inputs are numpy arrays, make copies
        """

        # loop through the experimental fields and if there is data, save it to the spectrum
        for field in _mkschemas.property_fields:
            attribute = struct.get(field.name)
            if attribute is not None:
                if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
                    if attribute.values is not None:
                        # zero copy only is False to handle string lists
                        setattr(self, field.name, attribute.values.to_numpy(zero_copy_only=False))
                elif pa.types.is_struct(field.type):
                    if attribute.values is not None:
                        setattr(self, field.name, attribute.values)
                else:
                    setattr(self, field.name, attribute.as_py())
        
        self.charge = struct['charge'].as_py() if struct.get('charge') is not None else None

        def struct2numpy(struct, attribute):
            if struct.get(attribute) is not None:
                if struct[attribute].values is not None:
                    return struct[attribute].values.to_numpy()
            return None
        
        stddev = struct2numpy(struct, 'stddev')
        precursor_intensity = struct['precursor_intensity'].as_py() if struct.get('precursor_intensity') is not None else None

        self.precursor = self.precursor_class(
            mz=struct.get('precursor_mz'),
            intensity=precursor_intensity,
            mass_info=MassInfo(arrow_struct_scalar=struct['precursor_massinfo']),
        )

        tolerance = struct2numpy(struct, 'tolerance')

        annotations = struct.get("annotations", None)
        if annotations is not None:
            # unwrap the struct array
            annotations = annotations.values
        
        self.products = self.product_class(
            struct2numpy(struct, 'mz'),
            struct2numpy(struct, 'intensity'),
            stddev=stddev,
            mass_info=MassInfo(arrow_struct_scalar=struct['product_massinfo']),
            annotations=annotations,
            copy_arrays=copy_arrays,
            tolerance=tolerance
        )
        return self
    
    def from_arrays(
            self,
            mz,
            intensity,
            row=None,
            precursor_mz=None,
            precursor_intensity=None,
            stddev=None,
            annotations=None,
            precursor_mass_info=None,
            product_mass_info=None,
            copy_arrays=True,
            tolerance=None,
    ):
        """
        Update or initialize from a series of arrays and the information in rows.  precursor information
        is pulled from rows unless precursor_mz and/or procursor_intensity are provided.

        :param mz: mz array
        :param intensity: intensity array
        :param stddev: standard deviation of intensity
        :param row: dict containing parameters and precursor info
        :param precursor_mz: precursor_mz value, used preferentially to row
        :param precursor_intensity: precursor intensity, optional
        :param annotations: annotations on the ions
        :param precursor_mass_info: MassInfo mass measurement information for the precursor
        :param product_mass_info: MassInfo mass measurement information for the product
        :param tolerance: mass tolerance array
        :param copy_arrays: if the inputs are numpy arrays, make copies
        """

        if precursor_mass_info is None:
            precursor_mass_info = self.precursor_mass_info
        if product_mass_info is None:
            product_mass_info = self.product_mass_info

        # todo: we should turn these into properties for the spectrum that use what is in the product/precursor Ions
        self.precursor_mass_info = precursor_mass_info
        self.product_mass_info = product_mass_info

        if row:
            self.precursor = self.precursor_class(
                row.get("precursor_mz", None), mass_info=precursor_mass_info
            )
            self.name = row.get("name", None)
            self.id = row.get("id", None)
            self.retention_time = row.get("retention_time", None)
            # make a copy of the props
            self.props.update(row)
        if precursor_mz:
            self.precursor = self.precursor_class(
                precursor_mz, mass_info=precursor_mass_info
            )
        if precursor_intensity and self.precursor is not None:
            self.precursor.intensity = precursor_intensity
        # numpy array of peak intensity
        self.products = self.product_class(
            mz,
            intensity,
            stddev=stddev,
            mass_info=product_mass_info,
            annotations=annotations,
            copy_arrays=copy_arrays,
            tolerance=tolerance,
        )
        return self

    def get_string_prop(self, mol, prop_name):
        """
        read a molecular property, dealing with unicode decoding error (rdkit uses UTF-8)

        :param mol: the rdkit molecule
        :param prop_name: the name of the property
        :return: property value
        """
        prop = None
        if mol.HasProp(prop_name):
            try:
                prop = mol.GetProp(prop_name)
            except UnicodeDecodeError as err:
                logging.debug(
                    f"Invalid unicode character in property {prop_name} for spectrum {self.id} with error {err}"
                )
                prop = None
        return prop

    def get_float_prop(self, mol, prop_name):
        """
        read a molecular property, and parse it as a float, ignoring non number characters
        doesn't currently deal with exponentials

        :param mol: the rdkit molecule
        :param prop_name: the name of the property
        :return: property value
        """
        prop = None
        if mol.HasProp(prop_name):
            try:
                prop = mol.GetProp(prop_name)
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", prop)
                if matches:
                    prop = float(matches[0])
                else:
                    logging.debug(
                        f"No float in property {prop_name} for spectrum {self.id}"
                    )
                    prop = None
            except UnicodeDecodeError as err:
                logging.debug(
                    f"Invalid unicode character in property {prop_name} for spectrum {self.id} with error {err}"
                )
                prop = None
        return prop

    def get_int_prop(self, mol, prop_name):
        """
        read a molecular property, and parse it as a int, ignoring non number characters

        :param mol: the rdkit molecule
        :param prop_name: the name of the property
        :return: property value
        """
        prop = None
        if mol.HasProp(prop_name):
            try:
                prop = mol.GetProp(prop_name)
                matches = re.findall(r"[-+]?\d+", prop)
                if matches:
                    prop = int(matches[0])
                else:
                    logging.debug(
                        f"No int in property {prop_name} for spectrum {self.id}"
                    )
                    prop = None
            except UnicodeDecodeError as err:
                logging.debug(
                    f"Invalid unicode character in property {prop_name} for spectrum {self.id} with error {err}"
                )
                prop = None
        return prop

    def to_msp(self, annotate_peptide=False, ion_types=None):
        """
        convert a spectrum to an msp file entry, encoded as a string

        :param annotate_peptide: annotate as a peptide
        :param ion_types: ion types for annotation
        :return: string containing spectrum in msp format
        """
        #todo: check to see if annotation should be turned on
        if annotate_peptide:
            from masskit.spectrum.theoretical_spectrum import \
                annotate_peptide_spectrum
            annotate_peptide_spectrum(self, ion_types=ion_types)

        ret_value = ""
        if hasattr(self, "name") and self.name is not None:
            ret_value += f"Name: {self.name}\n"
        else:
            ret_value += f"Name: {self.id}\n"
        if hasattr(self, "precursor") and self.precursor is not None and hasattr(self.precursor, "mz") and self.precursor.mz is not None:
            ret_value += f"PRECURSORMZ: {self.precursor.mz}\n"
        if hasattr(self, "formula") and self.formula is not None:
            ret_value += f"Formula: {self.formula}\n"
        if hasattr(self, "ev") and self.ev is not None:
            ret_value += f"eV: {self.ev}\n"
        if hasattr(self, "nce") and self.nce is not None:
            ret_value += f"NCE: {self.nce}\n"
        if hasattr(self, "protein_id") and self.protein_id is not None:
            ret_value += f"ProteinId: {','.join(self.protein_id)}\n"
        ret_value += f"DB#: {self.id}\n"
        # spectrum = self.copy(min_mz=1.0, min_intensity=min_intensity)
        num_peaks = len(self.products.mz)
        ret_value += f"Num Peaks: {num_peaks}\n"
        for i in range(num_peaks):
            ret_value += f"{self.products.mz[i]:.4f}\t{self.products.intensity[i]:.8g}"
            if self.joined_spectra and self.joins:
                ret_value += '\t"'
                # find join to annotation (for now, just one annotation to scan.  make join property handlers)
                # may be more than one annotation spectrum -- ignore this for now

                #   scan exp_peaks in join table, get matching theo_peaks
                annotation_strings = self.get_ion_annotation(i)
                if len(annotation_strings) > 0:
                    for k, annotation_string in enumerate(annotation_strings):
                        if k != 0:
                            ret_value += ","
                        ret_value += annotation_string
                else:
                    ret_value += '?'
                ret_value += '"\n'
            else:
                # if no annotation, print "?"
                ret_value += '\t"?"\n'
        ret_value += "\n"
        return ret_value

    def get_ion_annotation(self, i, show_ppm=True, tex_style=False, show_mz=True):
        """
        get the text annotations for a given ion

        :param i: index of the ion
        :param show_ppm: show a calcuated ppm
        :param tex_style: text annotations use tex for formatting
        :param show_mz: show mz values in the text
        """
        exp_peaks = self.joins[0].column('exp_peaks')
        theo_peaks = self.joins[0].column('theo_peaks')
        # from theo spectrum, get annotation struct.  see ion_annot_fields for fields
        annotations = self.joined_spectra[0].products.annotations
        ion_type = annotations.field(0)
        isotope = annotations.field(2)
        ion_subtype = annotations.field(3)
        position = annotations.field(4)
        end_position = annotations.field(5)
        product_charge = annotations.field(1)
        annotation_strings = []
        for j in range(len(exp_peaks)):
            if exp_peaks[j].is_valid and theo_peaks[j].is_valid and exp_peaks[j].as_py() == i:
                annotation_string = ""
                theo_peak = theo_peaks[j].as_py()
                current_ion_type = ion_type[theo_peak].as_py()
                current_ion_subtype = ion_subtype[theo_peak].as_py()
                        
                if current_ion_type == 'internalb':
                    fragment = self.peptide[position[theo_peak].as_py(): end_position[theo_peak].as_py()]
                    annotation_string += f'Int/{fragment}'
                elif current_ion_type == 'immonium':
                    annotation_string += f'{current_ion_subtype}'
                elif current_ion_type == 'parent':
                    annotation_string += f'p'
                else:
                    if tex_style:
                        annotation_string += f'$\mathregular{{{current_ion_type}_{{{position[theo_peak].as_py()}}}}}$'
                    else:
                        annotation_string += f'{current_ion_type}{position[theo_peak].as_py()}'

                if current_ion_subtype is not None and current_ion_type != 'immonium':
                    if tex_style:
                        # subscript atom counts
                        neutral_loss = re.sub(r"(\d+)", r"$\\mathregular{_{\1}}$", current_ion_subtype)
                        annotation_string += f'-{neutral_loss}'
                    else:
                        annotation_string += f'-{current_ion_subtype}'

                if isotope[theo_peak].as_py() == 1:
                    annotation_string += '+i'
                elif isotope[theo_peak].as_py() > 1:
                    annotation_string += f'+{isotope[theo_peak].as_py()}i' 
                if product_charge[theo_peak].as_py() > 1 and current_ion_type != 'parent':
                    if tex_style:
                        annotation_string += f'$\mathregular{{^{{{product_charge[theo_peak].as_py()}+}}}}$'
                    else:
                        annotation_string += f'^{product_charge[theo_peak].as_py()}'
                
                # calculate ppm
                if show_ppm:
                    ppm = (self.products.mz[i] - self.joined_spectra[0].products.mz[theo_peaks[j].as_py()]) / self.products.mz[i] * 1000000
                    annotation_string += f"/{ppm:.1f}ppm"

                if tex_style and show_mz:
                    annotation_string += f'\n{self.products.mz[i]:.2f}'
                annotation_strings.append(annotation_string)
        return annotation_strings

    def from_mol(
            self, mol, skip_expensive=False, id_field="NISTNO", id_field_type="int"
    ):
        """
        Initialize from rdkit mol

        :param mol: rdkit mol
        :param skip_expensive: skip the expensive calculations
        :param id_field: field to use for the mol id, such as NISTNO, ID or _NAME (the sdf title field)
        :param id_field_type: the id field type, such as int or str
        """

        if mol.HasProp("MW") or mol.HasProp("PRECURSOR M/Z"):
            if mol.HasProp("PRECURSOR M/Z"):
                precursor_mz = self.get_float_prop(mol, "PRECURSOR M/Z")
            else:
                precursor_mz = self.get_float_prop(mol, "MW")
            self.precursor = self.precursor_class(
                self.precursor_class.cast_mz(precursor_mz),
                mass_info=self.precursor_mass_info,
            )
        self.name = self.get_string_prop(mol, "NAME")
        self.casno = self.get_string_prop(mol, "CASNO")
        self.synonyms = self.get_string_prop(mol, "SYNONYMS")
        if self.synonyms is not None:
            self.synonyms = self.synonyms.splitlines()
        if type(id_field) is not int:
            if id_field_type == "int":
                self.id = self.get_int_prop(mol, id_field)
            else:
                self.id = self.get_string_prop(mol, id_field)

        if mol.HasProp("EXPERIMENTAL RI MEDIAN/DEVIATION/#DATA"):
            ri_string = mol.GetProp("EXPERIMENTAL RI MEDIAN/DEVIATION/#DATA")
            ris = ri_string.split()
            for ri_string in ris:
                ri = re.split("[=/]", ri_string)
                if len(ri) == 4:
                    if ri[0] == "SemiStdNP":
                        self.column = ri[0]
                        self.experimental_ri = float(ri[1])
                        self.experimental_ri_error = float(ri[2])
                        self.experimental_ri_data = int(ri[3])
                    elif ri[0] == "StdNP":
                        self.stdnp = float(ri[1])
                        self.stdnp_error = float(ri[2])
                        self.stdnp_data = int(ri[3])
                    elif ri[0] == "StdPolar":
                        self.stdpolar = float(ri[1])
                        self.stdpolar_error = float(ri[2])
                        self.stdpolar_data = int(ri[3])
        elif mol.HasProp("RIDATA_01"):
            self.column = self.get_string_prop(mol, "RIDATA_15")
            self.experimental_ri = self.get_float_prop(mol, "RIDATA_01")
            self.experimental_ri_error = 0.0
            self.experimental_ri_data = 1
        self.estimated_ri = self.get_float_prop(mol, "ESTIMATED KOVATS RI")
        self.inchi_key = self.get_string_prop(mol, "INCHIKEY")
        self.estimated_ri_error = self.get_float_prop(mol, "RI ESTIMATION ERROR")
        self.formula = self.get_string_prop(mol, "FORMULA")
        self.exact_mass = self.get_float_prop(mol, "EXACT MASS")
        self.ion_mode = self.get_string_prop(mol, "ION MODE")
        self.charge = self.get_int_prop(mol, "CHARGE")
        self.instrument = self.get_string_prop(mol, "INSTRUMENT")
        self.instrument_type = self.get_string_prop(mol, "INSTRUMENT TYPE")
        self.ionization = self.get_string_prop(mol, "IONIZATION")
        self.collision_gas = self.get_string_prop(mol, "COLLISION GAS")
        self.sample_inlet = self.get_string_prop(mol, "SAMPLE INLET")
        self.spectrum_type = self.get_string_prop(mol, "SPECTRUM TYPE")
        self.precursor_type = self.get_string_prop(mol, "PRECURSOR TYPE")
        notes = self.get_string_prop(mol, "NOTES")
        if notes is not None:
            match = re.search(r"Vial_ID=(\d+)", notes)
            if match:
                self.vial_id = int(match.group(1))
        ce = self.get_string_prop(mol, "COLLISION ENERGY")
        if ce is not None:
            match = re.search(r"NCE=(\d+)% (\d+)eV|NCE=(\d+)%|(\d+)|", ce)
            if match:
                if match.group(4) is not None:
                    self.ev = float(match.group(4))
                    self.collision_energy = float(self.ev)
                elif match.group(3) is not None:
                    self.nce = float(match.group(3))
                    if precursor_mz is not None and self.charge is not None:
                        self.collision_energy = nce2ev(self.nce, precursor_mz, self.charge)
                elif match.group(1) is not None and match.group(2) is not None:
                    self.nce = float(match.group(1))
                    self.ev = float(match.group(2))
                    self.collision_energy = self.ev
        self.insource_voltage = self.get_int_prop(mol, "IN-SOURCE VOLTAGE")

        skip_props = set([
            "MASS SPECTRAL PEAKS",
            "NISTNO",
            "NAME",
            "MW",
            "EXPERIMENTAL RI MEDIAN/DEVIATION/#DATA",
            "ESTIMATED KOVATS RI",
            "INCHIKEY",
            "RI ESTIMATION ERROR",
            "FORMULA",
            "SYNONYMS",
            "RIDATA_01",
            "RIDATA_15",
            "PRECURSOR M/Z",
            "EXACT MASS",
            "ION MODE",
            "CHARGE",
            "INSTRUMENT",
            "INSTRUMENT TYPE",
            "IONIZATION",
            "COLLISION ENERGY",
            "COLLISION GAS",
            "SAMPLE INLET",
            "SPECTRUM TYPE",
            "PRECURSOR TYPE",
            "NOTES",
            "IN-SOURCE VOLTAGE",
            "ID",
        ])
        # populate spectrum props with mol props
        for k in mol.GetPropNames():
            # skip over props handled elsewhere
            if k in skip_props:
                continue
            self.props[k] = self.get_string_prop(mol, k)

        # get spectrum string and break into list of lines
        if mol.HasProp("MASS SPECTRAL PEAKS"):
            spectrum = mol.GetProp("MASS SPECTRAL PEAKS").splitlines()
            mz_in = []
            intensity_in = []
            annotations_in = []
            has_annotations = False
            for peak in spectrum:
                values = peak.replace("\n", "").split(
                    maxsplit=2
                )  # get rid of any newlines then strip
                intensity = self.product_class.cast_intensity(values[1])
                # there are intensity 0 ions in some spectra
                if intensity != 0:
                    mz_in.append(self.product_class.cast_mz(values[0]))
                    intensity_in.append(intensity)
                    annotations = []
                    if len(values) > 2 and not skip_expensive:
                        has_annotations = True
                        end = "".join(values[2:])
                        for match in re.finditer(
                                r"(\?|((\w+\d*)+((\+?-?)((\w|\d)+))?(=((\w+)(\+?-?))?(\w+\d*)+((\+?-?)((\w|\d)+))?)?("
                                r"\/(-?\d+.?\d+))(\w*)))*(;| ?(\d+)\/(\d+))",
                                end,
                        ):
                            annotation = {
                                "begin": match.group(10),
                                "change": match.group(11),
                                "loss": match.group(12),
                                "mass_diff": match.group(18),
                                "diff_units": match.group(19),
                                "fragment": match.group(3),
                            }
                            annotations.append(annotation)
                            # group 10 is the beginning molecule, e.g. "p" means precursor
                            # group 11 indicates addition or subtraction from the beginning molecule
                            # group 12 is the chemical formula of the rest of the beginning molecule
                            # group 18 is the mass difference, e.g. "-1.3"
                            # group 19 is the mass difference unit, e.g. "ppm"
                            # group 3 is the chemical formula of the peak
                    annotations_in.append(annotations)
            if has_annotations and not skip_expensive:
                logging.warning('spectrum annotations from molfile currently unimplemented')
                self.products = self.product_class(
                    mz_in,
                    intensity_in,
                    mass_info=self.product_mass_info,
                    # annotations=annotations_in,
                )
            else:
                self.products = self.product_class(
                    mz_in, intensity_in, mass_info=self.product_mass_info
                )
        return

    @staticmethod
    def weighted_intensity(intensity, mz):
        """
        Stein & Scott 94 intensity weight

        :param intensity: peak intensity
        :param mz: peak mz in daltons
        :return: weighted intensity
        """
        return intensity ** 0.6 * mz ** 3

    def single_match(self, spectrum, minimum_match=1, cosine_threshold=0.3, cosine_score_scale=1.0):
        """
        try to match two spectra and calculate the probability of a match

        :param spectrum: the spectrum to match against
        :param cosine_threshold: minimum score to return results
        :param minimum_match: minimum number of matching peaks
        :param cosine_score_scale: max value of cosine score
        :return: query id, hit id, cosine score, number of matching peaks
        """
        ion1, ion2, index1, index2 = self.products.clear_and_intersect(spectrum.products, None, None)

        # intersect the spectra
        matches = len(index1)
        if matches >= minimum_match:
            cosine_score = cosine_score_calc(
                ion1.mz,
                ion1.intensity,
                ion2.mz,
                ion2.intensity,
                index1,
                index2,
                scale=cosine_score_scale,
            )
            if cosine_score >= cosine_threshold:
                return self.id, spectrum.id, cosine_score, matches
        return None, None, None, None

    def identity(self, spectrum, identity_name=False, identity_energy=False):
        """
        check to see if this spectrum and the passed in spectrum have the same chemical structure

        :param spectrum: comparison spectrum
        :param identity_name: require the name to match in addition to the inchi_key
        :param identity_energy: require the collision_energy to match in addition to the inchi_key
        :return: are they identical?
        """
        return_val = self.inchi_key == spectrum.inchi_key
        if self.inchi_key is None or spectrum.inchi_key is None:
            return_val = False
        if identity_name:
            return_val = return_val and (self.name == spectrum.name)
        if identity_energy:
            return_val = return_val and (
                    self.collision_energy == spectrum.collision_energy
            )
        return return_val

    def copy_annot(self, spectrum2):
        """
        copy annotations from spectrum2 to this spectrum using the matched ion indices

        :param spectrum2: the ions to compare against
        """
        index1, index2 = self.products.intersect(spectrum2.products)
        return self.products.copy_annot(spectrum2.products, index1, index2)

    def cosine_score(
            self,
            spectrum2,
            use_same_tolerance=False,
            mz_power=0.0,
            intensity_power=0.5,
            scale=999,
            skip_denom=False,
            tiebreaker=None
    ):
        """
        cosine score on product ions

        :param spectrum2: spectrum to compare against
        :param use_same_tolerance: evaluate cosine score by using the mass tolerance for this spectrum for spectrum2
        :param mz_power: what power to raise the mz value for each peak
        :param intensity_power: what power to raise the intensity for each peak
        :param scale: what value to scale the score by
        :param skip_denom: skip computing the denominator
        :param tiebreaker: how to deal with one to multiple matches to peaks in spectra1. mz is closest mz value, intensity is closest intensity, None is report multiple matches
        :return: cosine score
        """
        if use_same_tolerance:
            spectrum2 = spectrum2.change_mass_info(self.product_mass_info)
        return self.products.cosine_score(
            spectrum2.products,
            mz_power=mz_power,
            intensity_power=intensity_power,
            scale=scale,
            skip_denom=skip_denom,
            tiebreaker=tiebreaker
        )

    def intersect(self, spectrum2, tiebreaker=None):
        """
        intersect product ions with another spectrum's product ions

        :param spectrum2: comparison spectrum
        :param tiebreaker: how to deal with one to multiple matches to peaks in spectra1. mz is closest mz value, intensity is closest intensity, None is report multiple matches
        :return: matched ions indices in this spectrum, matched ion indices in comparison spectrum
        """
        return self.products.intersect(spectrum2.products, tiebreaker=tiebreaker)

    def copy(self, min_mz=-1, max_mz=0, min_intensity=-1, max_intensity=0):
        """
        create filtered version of self.  This is essentially a copy constructor

        :param min_mz: minimum mz value
        :param max_mz: maximum mz value.  0 = ignore
        :param min_intensity: minimum intensity value
        :param max_intensity: maximum intensity value.  0 = ignore
        :return: copy
        """
        return self.filter(
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            inplace=False,
        )

    def filter(
            self, min_mz=-1, max_mz=0, min_intensity=-1, max_intensity=0, inplace=False
    ):
        """
        filter a spectrum by mz and/or intensity.

        :param min_mz: minimum mz value
        :param max_mz: maximum mz value.  0 = ignore
        :param min_intensity: minimum intensity value
        :param max_intensity: maximum intensity value.  0 = ignore
        :param inplace: do operation on current spectrum, otherwise create copy
        :return: filtered copy if not inplace, otherwise current ions
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.filter(
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            inplace=True,
        )
        # zero out annotations as they are no longer valid
        return_spectrum.joins = []
        return_spectrum.joined_spectra = []
        return return_spectrum

    def parent_filter(self, h2o=True, inplace=False):
        """
        filter parent ions, including water losses.

        :param h2o: filter out water losses
        :param inplace: do operation on current ions, otherwise create copy
        :return: filtered copy if not inplace, otherwise current ions
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        if self.precursor is not None:
            return_spectrum.products.parent_filter(h2o=h2o, precursor_mz=self.precursor.mz, charge=self.charge,
                                                   inplace=True)
        # zero out annotations as they are no longer valid
        return_spectrum.joins = []
        return_spectrum.joined_spectra = []
        return return_spectrum

    def windowed_filter(self, mz_window=14, num_ions=5, inplace=False):
        """
        filter ions by examining peaks in order of intensity and filtering peaks within a window

        :param mz_window: half size of mz_window for filtering
        :param num_ions: number of ions allowed in full mz_window
        :param inplace: do operation on current ions, otherwise create copy
        :return: filtered copy if not inplace, otherwise current ions
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.windowed_filter(mz_window=mz_window, num_ions=num_ions, inplace=True)

        # zero out annotations as they are no longer valid
        return_spectrum.joins = []
        return_spectrum.joined_spectra = []
        return return_spectrum

    def norm(self, max_intensity_in=999, keep_type=True, inplace=False, ord=None):
        """
        norm the product intensities

        :param max_intensity_in: the intensity of the most intense peak
        :param keep_type: keep the type of the intensity array
        :param inplace: do operation on current spectrum, otherwise create copy
        :param ord: if set, normalize using norm order as in np.linalg.norm. 2 = l2
        :returns: normed copy if not inplace, otherwise current ions
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.norm(
            max_intensity_in=max_intensity_in, keep_type=keep_type, inplace=True, ord=ord
        )
        return return_spectrum

    def merge(self, merge_spectrum, inplace=False):
        """
        merge the product ions of another spectrum into this one.

        :param merge_spectrum: the spectrum whose product ions will be merged in
        :param inplace: do operation on current spectrum
        :returns: merged copy if not inplace, otherwise current spectrum
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.merge(merge_spectrum.products, inplace=True)

        # zero out annotations as they are no longer valid
        return_spectrum.joins = []
        return_spectrum.joined_spectra = []
        return return_spectrum

    def shift_mz(self, shift, inplace=False):
        """
        shift the mz values of all product ions by the value of shift.  Negative ions are masked out

        :param shift: value to shift mz
        :param inplace: do operation on current ions
        :returns: masked copy if not inplace, otherwise current ions
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.shift_mz(shift=shift, inplace=True)
        return return_spectrum

    def mask(self, indices, inplace=False):
        """
        mask out product ions that are pointed to by the indices

        :param indices: indices of ions to screen out or numpy boolean mask
        :param inplace: do operation on current ions
        :returns: masked copy if not inplace, otherwise current spectrum
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.mask(indices=indices, inplace=True)

        # zero out annotations as they are no longer valid
        return_spectrum.joins = []
        return_spectrum.joined_spectra = []
        return return_spectrum

    def change_mass_info(self, mass_info, inplace=False, take_max=True):
        """
        given a new mass info for product ions, recalculate tolerance bins

        :param mass_info: the MassInfo structure to change to
        :param inplace: if true, change in place, otherwise return copy
        :param take_max: for each bin take the maximum intensity ion, otherwise sum all ions mapping to the bin
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.change_mass_info(
            mass_info, inplace=True, take_max=take_max
        )
        return return_spectrum

    def total_intensity(self):
        """
        total intensity of product ions

        :return: total intensity
        """
        return self.products.total_intensity()

    def num_ions(self):
        """
        number of product ions

        :return: number of ions
        """
        return self.products.num_ions()

    def plot(
            self,
            axes=None,
            mirror_spectrum=None,
            mirror=True,
            title=None,
            xlabel="m/z",
            ylabel="Intensity",
            title_size=None,
            label_size=None,
            max_mz=None,
            min_mz=0,
            color=(0, 0, 1, 1),
            mirror_color=(1, 0, 0, 1),
            stddev_color=(0.3, 0.3, 0.3, 0.5),
            left_label_color=(1, 0, 0, 1),
            normalize=None,
            plot_stddev=False,
            vertical_cutoff=0.0,
            vertical_multiplier=1.1,
            right_label=None,
            left_label=None,
            right_label_size=None,
            left_label_size=None,
            no_xticks=False,
            no_yticks=False,
            linewidth=None,
            annotate=False,
        ):
        """
        make a spectrum plot using matplotlib.  if mirror_spectrum is specified, will do a mirror plot

        :param axes: matplotlib axis
        :param mirror_spectrum: spectrum to mirror plot (optional)
        :param mirror: if true, mirror the plot if there are two spectra.  Otherwise plot the two spectra together
        :param title: title of plot
        :param xlabel: xlabel of plot
        :param ylabel: ylabel of plot
        :param title_size: size of title font
        :param label_size: size of x and y axis label fonts
        :param max_mz: maximum mz to plot
        :param min_mz: minimum mz to plot
        :param color: color of spectrum specified as RBGA tuple
        :param mirror_color: color of mirrored spectrum specified as RGBA tuple
        :param stddev_color: color of error bars
        :param left_label_color: color of the left top label
        :param normalize: if specified, norm the spectra to this value.
        :param plot_stddev: if true, plot the standard deviation
        :param vertical_cutoff: if the intensity/max_intensity is below this value, don't plot the vertical line
        :param vertical_multiplier: multiply times y max values to create white space
        :param right_label: label for the top right of the fiture
        :param left_label: label for the top left of the figure
        :param right_label_size: size of label for the top right of the fiture
        :param left_label_size: size of label for the top left of the figure
        :param no_xticks: turn off x ticks and labels
        :param no_yticks: turn off y ticks and lables
        :param linewidth: width of plotted lines
        :param annotate: if peptide spectra, annotate ions
        :return: peak_collection, mirror_peak_collection sets of peaks for picking
        """
        #TODO: move this code into plotting.py so that it can use matplotlib, e.g use the default axes.
        if mirror_spectrum:
            mirror_ions = mirror_spectrum.products
            mirror_intensity = mirror_ions.intensity
            mirror_mz = mirror_ions.mz
            if plot_stddev:
                mirror_stddev = mirror_ions.stddev
            else:
                mirror_stddev = None
        else:
            mirror_ions = None
            mirror_intensity = None
            mirror_mz = None
            mirror_stddev = None

        if plot_stddev:
            stddev = self.products.stddev
        else:
            stddev = None

        line_collections = _mkspectrum_plotting.spectrum_plot(
            axes,
            self.products.mz,
            self.products.intensity,
            stddev,
            mirror_mz=mirror_mz,
            mirror_intensity=mirror_intensity,
            mirror_stddev=mirror_stddev,
            mirror=mirror,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            title_size=title_size,
            label_size=label_size,
            max_mz=max_mz,
            min_mz=min_mz,
            color=color,
            mirror_color=mirror_color,
            stddev_color=stddev_color,
            left_label_color=left_label_color,
            normalize=normalize,
            vertical_cutoff=vertical_cutoff,
            vertical_multiplier=vertical_multiplier,
            right_label=right_label,
            left_label=left_label,
            right_label_size=right_label_size,
            left_label_size=left_label_size,
            no_xticks=no_xticks,
            no_yticks=no_yticks,
            linewidth=linewidth
        )

        if annotate:
            annots = []
            xx = []
            yy = []
            for j in range(len(self.products)):
                annot = self.get_ion_annotation(j, show_ppm=False, tex_style=True, show_mz=False)
                if len(annot) > 0:
                    annots.append(annot[0])
                    xx.append(self.products.mz[j])
                    yy.append(self.products.intensity[j])

            _mktextalloc.allocate_text(axes,xx,yy,
                            annots,
                            x_lines=[np.array([self.products.mz[i],self.products.mz[i]]) for i in range(len(self.products))],
                            y_lines=[np.array([0,self.products.intensity[i]]) for i in range(len(self.products))], 
                            textsize=7,
                            margin=0.0,
                            min_distance=0.005,
                            max_distance=0.05,
                            linewidth=0.7,
                            nbr_candidates=200,
                            textcolor="black",
                            draw_all=False,
                            ylims=(axes.get_ylim()[1]/20, axes.get_ylim()[1]))
            
        return line_collections


    def __repr__(self):
        """
        text representation of spectrum

        :return: text
        """
        if self.precursor is not None:
            return f"<spectrum {self.id}; {self.precursor.mz}Da precursor; {self.num_ions()} ions>"
        elif self.products is not None:
            return f"<spectrum {self.id}; {self.num_ions()} ions>"
        else:
            return f"<spectrum {self.id}>"

    def draw_spectrum(self, fig_format, output):
        return _mkspectrum_plotting.draw_spectrum(self, fig_format, output)

    def _repr_png_(self):
        """
        png representation of spectrum

        :return: png
        """
        return self.draw_spectrum("png", BytesIO())

    def _repr_svg_(self):
        """
        png representation of spectrum

        :return: svg
        """
        return self.draw_spectrum("svg", StringIO())

    def __str__(self):
        if _mkipython.is_notebook():
            val = b64encode(self._repr_png_()).decode("ascii")
            return \
                f'<img data-content="masskit/spectrum" src="data:image/png;base64,{val}" alt="spectrum {self.name}"/>'
        else:
            return self.__repr__()

    def create_fingerprint(self, max_mz=2000):
        """
        create a fingerprint that corresponds to the spectra.  Each bit position
        corresponds to an integer mz value and is set if the intensity is above min_intensity

        :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
        :return: SpectrumTanimotoFingerPrint
        """
        fp = _mkfingerprints.SpectrumTanimotoFingerPrint(dimension=max_mz)
        fp.object2fingerprint(self)
        return fp
    
    def finalize(self):
        """
        function used to clean up spectrum after creation
        """
        pass

    def evenly_space(self, tolerance=None, take_max=True, max_mz=None, include_zeros=False, inplace=False,
                    take_sqrt=False):
        """
        convert product ions to product ions with evenly spaced m/z bins.  The m/z bins are centered on
        multiples of tolerance * 2.  Multiple ions that map to the same bin are either summed or the max taken of the
        ion intensities.

        :param tolerance: the mass tolerance of the evenly spaced m/z bins (bin width is twice this value) in daltons
        :param take_max: for each bin take the maximum intensity ion, otherwise sum all ions mapping to the bin
        :param max_mz: maximum mz value, 2000 by default
        :param include_zeros: fill out array including bins with zero intensity
        :param inplace: do operation on current spectrum, otherwise create copy
        :param take_sqrt: take the sqrt of the intensities
        :returns: normed copy if not inplace, otherwise current ions
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)

        return_spectrum.products.evenly_space(tolerance=tolerance, take_max=take_max, max_mz=max_mz,
                                                include_zeros=include_zeros, take_sqrt=take_sqrt)
        return return_spectrum


# Add properties from the schema to Spectrum
_mkschemas.populate_properties(Spectrum)

class AccumulatorSpectrum(Spectrum, _mkaccumulator.Accumulator):
    """
    used to contain a spectrum that accumulates the sum of many spectra
    includes calculation of standard deviation
    """
    prop_names = None

    def __init__(self, mz=None, tolerance=None, precursor_mz=None, count_spectra=False, take_max=True, *args, **kwargs):
        """
        initialize predicted spectrum

        :param mz: array of mz values
        :param tolerance: mass tolerance in daltons
        :param precursor_mz: m/z of precursor
        :param count_spectra: count peaks instead of summing intensity
        :param take_max: when converting new_spectrum to evenly spaced bins, take max value per bin, otherwise sum
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        # keep count of the number of spectra being averaged into each bin
        self.count = np.zeros_like(mz)
        self.from_arrays(mz, np.zeros_like(mz), stddev=np.zeros_like(mz),
                         product_mass_info=MassInfo(tolerance, "daltons", "monoisotopic", evenly_spaced=True), 
                         precursor_mz=precursor_mz, precursor_intensity=999.0,
                         precursor_mass_info=MassInfo(0.0, "ppm", "monoisotopic"))
        self.count_spectra = count_spectra
        self.take_max = take_max

    def add(self, new_item):
        """
        add a spectrum to the average.  Keeps running total of average and std deviation using
        Welford's algorithm.  This API assumes that the spectrum being added is also evenly spaced
        with the same mz values.  However, the new spectrum doesn't have to have the same max_mz as
        the summation spectrum

        :param new_item: new spectrum to be added
        """
        # convert to same scale
        intensity = np.zeros((1, len(self.products.intensity)))
        new_item.products.ions2array(
            intensity,
            0,
            bin_size=self.products.mass_info.tolerance * 2,
            down_shift=self.products.mass_info.tolerance,
            intensity_norm=1.0,
            channel_first=True,
            take_max=self.take_max
        )
        intensity = np.squeeze(intensity)
        if self.count_spectra:
            intensity[intensity > 0.0] = 1.0

        # assume all spectra start at 0, but may have different max_mz.  len_addition is the size of the smallest
        # mz array
        if len(intensity) >= len(self.products.mz):
            len_addition = len(self.products.mz)
        else:
            len_addition = len(intensity)

        delta = intensity[0:len_addition] - self.products.intensity[0:len_addition]
        # increment the count, dealing with case where new spectrum is longer than the summation spectrum
        self.count[0:len_addition] += 1
        self.products.intensity[0:len_addition] += delta / self.count[0:len_addition]
        delta2 = intensity[0:len_addition] - self.products.intensity[0:len_addition]
        self.products.stddev[0:len_addition] += delta * delta2

    def finalize(self):
        """
        finalize the std deviation after all the the spectra have been added
        """
        self.products.stddev = np.sqrt(self.products.stddev / self.count)

        # delete extra attributes and change class to base class
        del self.count
        del self.count_spectra
        del self.take_max
        self.__class__ = Spectrum


_mkschemas.populate_properties(AccumulatorSpectrum, fields=_mkschemas.spectrum_accumulator_fields)

class HiResSpectrum(Spectrum):
    def __init__(self, precursor_mass_info=None, product_mass_info=None, name=None, id=None, ev=None, nce=None, charge=None, ion_class=HiResIons, mz=None, intensity=None, row=None, precursor_mz=None, precursor_intensity=None, stddev=None, annotations=None, tolerance=None, copy_arrays=False):
        super().__init__(precursor_mass_info=precursor_mass_info, 
                         product_mass_info=product_mass_info, 
                         name=name, 
                         id=id, 
                         ev=ev, 
                         nce=nce, 
                         charge=charge, 
                         ion_class=ion_class, 
                         mz=mz, 
                         intensity=intensity, 
                         row=row, 
                         precursor_mz=precursor_mz, 
                         precursor_intensity=precursor_intensity, 
                         stddev=stddev, 
                         annotations=annotations, 
                         tolerance=tolerance, 
                         copy_arrays=copy_arrays,
                         )

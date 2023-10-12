import copy
import math
import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.stats as sts

from .. import data as _mkdata
from ..data_specs import schemas as _mkschemas

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
            h2o_neutral_loss = (precursor_mz * charge - _mkdata.h2o_mass)/charge
            mask &= (return_ions.mz < h2o_neutral_loss - tolerance) | (return_ions.mz > h2o_neutral_loss + tolerance)
            h2o_neutral_loss = (precursor_mz * charge - 2 * _mkdata.h2o_mass)/charge
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

import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
import re
from masskit.config import EIMLConfig
import scipy.stats as sts
import scipy.special as spc
from pspearman import *
import pandas as pd
import logging
import math
import copy
import random
from scipy.stats import wald
from masskit.utils.common import *

try:
    from numba import jit
except ImportError:
    # when numba is not available, define jit decorator
    def jit(nopython=True):
        def decorator(func):
            def newfn(*args, **kwargs):
                return func(*args, **kwargs)
            return newfn
        return decorator


class MassInfo:
    """
    information about mass measurements of an ion peak
    """
    def __init__(self, tolerance: float = None, tolerance_type: str = None, mass_type: str = None,
                 neutral_loss: str = None, neutral_loss_charge: int = None):
        self.tolerance = tolerance   # mass tolerance.  If 0.5 daltons, this is unit mass
        self.tolerance_type = tolerance_type   # type of tolerance: "ppm", "daltons"
        self.mass_type = mass_type  # "monoisotopic" or "average"

        self.neutral_loss = neutral_loss  # neutral loss chemical formula
        self.neutral_loss_charge = neutral_loss_charge  # sign of neutral loss


class BasePeaks:
    """
    base class for a series of ions
    """
    def __init__(self, mz=None, intensity=None, annotations=None, mass_info: MassInfo = None, jitter=0):
        """
        :param mz: mz values in array-like or scalar
        :param intensity: corresponding intensities in array-like or scalar
        :param annotations: an array of per peak annotations, usually expressed as a dict
        :param mass_info:  dict containing mass type, tolerance, tolerance type
        :param jitter: used to add random jitter value to the mz values.  useful for creating fake data
        """
        if hasattr(mz, "__len__") and hasattr(intensity, "__len__"):  # check to see if both are array-like
            if len(mz) != len(intensity):
                raise ValueError("mz and intensity arrays are of different length")
            self.mz = np.array(mz)
            self.intensity = np.array(intensity)

            # sort by mz.  This is required for spectrum matching
            sorted_indexes = self.mz.argsort()
            self.mz = self.mz[sorted_indexes]
            self.intensity = self.intensity[sorted_indexes]
            if annotations is not None:
                if len(annotations) != len(mz):
                    raise ValueError("mz and annotation arrays are of different length")
                self.annotations = np.array(annotations)
                self.annotations = self.annotations[sorted_indexes]
            else:
                self.annotations = None

            # now rank the peaks by intensity
            self.rank = None
            self.rank_peaks()
        else:
            self.mz = mz
            self.intensity = intensity
            self.annotations = annotations
            self.rank = 1

        self.mass_info = mass_info
        self.jitter = jitter
        self.starts = None  # used for high resolution spectra
        self.stops = None  # used for high resolution spectra
        return

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
        :return: the rank of the peaks by intensity.  could be a numpy array
        1=most intense, rank is integer over the size of the intensity matrix
        """
        return self._rank

    @rank.setter
    def rank(self, value):
        self._rank = value
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

    def rank_peaks(self):
        """
        rank the peaks. intensity rank, 1=most intense, rank is integer over the size of the intensity matrix
        """
        # question:  why is this subtracting from the size of the array?  Obviously to change sign, but otherwise why?
        self.rank = sts.rankdata(self.intensity.shape[0] - self.intensity + 1, method="ordinal")
        self.rank = self.rank.astype(int)
        return

    def copy(self, min_mz=0, max_mz=0, min_intensity=0, max_intensity=0):
        """
        create filtered version of self.  This is essentially a copy constructor
        :param min_mz: minimum mz value
        :param max_mz: maximum mz value.  0 = ignore
        :param min_intensity: minimum intensity value
        :param max_intensity: maximum intensity value.  0 = ignore
        """
        return_peaks = copy.deepcopy(self)
        mask = (return_peaks.mz >= min_mz) & (return_peaks.intensity >= min_intensity)
        if max_mz:
            mask &= return_peaks.mz <= max_mz
        if max_intensity:
            mask &= return_peaks.intensity <= max_intensity
        return_peaks.mz = return_peaks.mz[mask]
        return_peaks.intensity = return_peaks.intensity[mask]
        if hasattr(return_peaks, 'annotations') and return_peaks.annotations is not None and \
                len(return_peaks.annotations) == len(mask):
            return_peaks.annotations = return_peaks.annotations[mask]
        return_peaks.rank_peaks()
        return_peaks.mass_info = return_peaks.mass_info
        return_peaks.jitter = return_peaks.jitter
        if hasattr(return_peaks, 'starts') and return_peaks.starts is not None and \
                len(return_peaks.starts) == len(mask):
            return_peaks.starts = return_peaks.starts[mask]
        if hasattr(return_peaks, 'stops') and return_peaks.stops is not None and \
                len(return_peaks.stops) == len(mask):
            return_peaks.stops = return_peaks.stops[mask]
        return return_peaks

    def normalize(self, max_intensity_in=999, keep_type=True):
        """
        norm the intensities
        :param max_intensity_in: the intensity of the most intense peak
        :param keep_type: keep the type of the intensity array
        :returns: self
        """
        max_intensity = np.max(self.intensity)
        d_type = self.intensity.dtype
        self.intensity = self.intensity/float(max_intensity)*max_intensity_in
        if keep_type:
            self.intensity = self.intensity.astype(d_type)  # cast back to original type
        return self

    def mask(self, indices):
        """
        mask out peaks that are pointed to by the indices
        :param indices: index of peaks to screen out
        :returns: self
        """
        self.mz = np.delete(self.mz, indices)
        self.intensity = np.delete(self.intensity, indices)
        if hasattr(self, 'annotations'):
            self.annotations = np.delete(self.annotations, indices)
        if hasattr(self, 'starts'):
            self.starts = np.delete(self.starts, indices)
        if hasattr(self, 'stops'):
            self.stops = np.delete(self.stops, indices)
        self.rank_peaks()
        return self

    def merge(self, merge_peaks):
        """
        merge another set of peaks into this one.
        will merge in annotations from the merge_peaks if this peaks object has annotations
        :param merge_peaks: the peaks to add in
        """
        if not hasattr(self.mz, "__len__") or not hasattr(merge_peaks.mz, "__len__"):
            raise ValueError("merging peaks without mz arrays is not supported")
        original_size = len(self.mz)

        self.mz = np.concatenate((self.mz, merge_peaks.mz))
        # get indices to sort the mz array
        sorted_indexes = self.mz.argsort()
        self.mz = self.mz[sorted_indexes]

        self.intensity = np.concatenate((self.intensity, merge_peaks.intensity))
        self.intensity = self.intensity[sorted_indexes]

        # recompute the starts and stops arrays
        if hasattr(self, 'starts') and hasattr(self, 'stops'):
            starts = []
            stops = []
            for peak in self.mz:
                start, stop = self.tolerance_interval(peak)
                starts.append(start)
                stops.append(stop)
            self.starts = np.array(starts)
            self.stops = np.array(stops)

        self.rank = None
        self.rank_peaks()

        # copy in the annotations and reorder
        #todo: for now, ignore annotations in merge.  somewhere there is code that creates a numpy array of python list objects for the annotation.  It should be a numpy array of arrays containing objects.
        # if hasattr(self, 'annotations') and self.annotations is not None:
        #     if len(self.annotations) != original_size:
        #         raise ValueError("mz and annotation arrays are of different length")
        #     if hasattr(merge_peaks, 'annotations') and merge_peaks.annotations is not None and \
        #             len(merge_peaks.annotations) == len(merge_peaks.mz):
        #         self.annotations = np.concatenate((self.annotations, merge_peaks.annotations))
        #     else:
        #         annotations = np.empty((len(merge_peaks.mz), 1), dtype=object)
        #         self.annotations = np.concatenate((self.annotations, np.array(annotations)))
        #     self.annotations = self.annotations[sorted_indexes]

    def tolerance_interval(self, mz):
        """
        use the mass tolerance to return the tolerance around a peak
        :param mz: the peak mz in daltons
        :return: the start and stop of the tolerance interval
        """
        if self.mass_info.tolerance_type == 'ppm':
            start = mz - mz * self.mass_info.tolerance / 1000000.0
            stop = mz + mz * self.mass_info.tolerance / 1000000.0
        elif self.mass_info.tolerance_type == 'daltons':
            start = mz - self.mass_info.tolerance
            stop = mz + self.mass_info.tolerance
        else:
            raise ValueError(f'mass tolerance type {self.mass_info.tolerance_type} not supported')
        return start, stop

    def cosine_score(self, peaks2, index1, index2):
        """
        calculate the cosine score between this set of peaks and ion2
        :param peaks2: the peaks to compare agains
        :param index1: matched peaks in this set of peaks
        :param index2: matched peaks in ion2
        :return: cosine score, scaled to 999
        """
        return cosine_score_calc(self.mz, self.intensity, peaks2.mz, peaks2.intensity, index1, index2)

    def peaks2array(self, array, channel, bin_size=1, precursor=0, intensity_norm=1000.0, insert_mz=False,
                    mz_norm=2000.0):
        """
        fill out an array of fixed size with the peaks.  note that this func assumes spectra sorted by mz
        :param array: the array to fill out
        :param channel: which channel to fill out in the array
        :param bin_size: the size of each bin in the array
        :param precursor: if nonzero, use this value to invert the spectra by subtracting mz from this value
        :param intensity_norm: value to norm the intensity
        :param insert_mz: instead of putting the normalized intensity in the array, put in the normalized mz
        :param mz_norm: the value to use to norm the mz values inserted
        """
        last_which_bin = -1
        last_intensity = -1
        for i in range(self.mz.size):
            mz = self.mz[i]
            if precursor != 0:
                mz = precursor - mz
            which_bin = int(mz / bin_size)
            if 0 <= which_bin < array.shape[0]:
                # compare to the previous peak and skip if the last peak was in the same mz bin and was more intense
                if which_bin == last_which_bin and self.intensity[i] <= last_intensity:
                    continue
                last_which_bin = which_bin
                last_intensity = self.intensity[i]

                if insert_mz:
                    array[which_bin, channel] = self.mz[i] / mz_norm
                else:
                    array[which_bin, channel] = self.intensity[i] / intensity_norm

    def score_rank(self, peaks2, index1, index2):
        """
        score a match between two spectra using rank statistics
        :param peaks2: the spectrum to rank against
        :param index1: indexes of matched peaks in the current spectrum
        :param index2: indexes of matched peaks in the input spectrum
        :return: the p value
        """
        # we use spearman's rank correlation to estimate the probability that the peaks matched between two
        # spectra have the same intensity ranks
        rank1 = np.take(self.rank, index1)
        rank2 = np.take(peaks2.rank, index2)
        max_rank = np.max([rank1, rank2])
        if max_rank > 22:
            max_rank = 22
        rank1 = rank1.tolist()
        rank2 = rank2.tolist()
        new_rank1 = []
        new_rank2 = []
        # scan through all ranks in self, up to max_rank
        for i_rank1 in range(1, max_rank + 1):
            # if found, store in new_rank1, along with corresponding spectrum rank in rank2 (place in new_rank2)
            new_rank1.append(i_rank1)
            try:
                index = rank1.index(i_rank1)
                if rank2[index] > max_rank:
                    new_rank2.append(-1)
                else:
                    new_rank2.append(rank2[index])
            except ValueError:
                # if not found, store in new_rank, and put -1 in new_rank2
                new_rank2.append(-1)
        # once done, scan through new_rank1 in reverse order of rank
        for i_rank1 in range(1, max_rank + 1):
            # if corresponding new_rank2 is -1
            index = new_rank1.index(i_rank1)
            if new_rank2[index] == -1:
                for i_rank2 in range(max_rank, 0, -1):
                    try:
                        new_rank2.index(i_rank2)
                    except ValueError:
                        new_rank2[index] = i_rank2
                        break
        s = float(np.sum(np.square(np.array(new_rank1) - np.array(new_rank2))))
        n = int(np.max([new_rank1, new_rank2]))
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.926.6415&rep=rep1&type=pdf
        if n == 0:
            return 1.0
        p = pspearman(s, n)
        return p

    def create_fingerprint(self, min_intensity=20, max_mz=2000):
        """
        create a rdkit ExplicitBitVect that corresponds to the spectra.  Each bit position
        corresponds to an integer mz value and is set if the intensity is above min_intensity
        :param min_intensity: the minimum intensity to set the fingerprint bit
        :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
        :return: bit vector
        """
        bv = DataStructs.ExplicitBitVect(max_mz)
        for i in range(self.mz.size):
            mz = int(self.mz[i] + 0.5)
            if mz < max_mz and self.intensity[i] >= min_intensity:
                bv.SetBit(mz)
        return bv


class HiResPeaks(BasePeaks):
    """
    for containing high mass resolution peaks
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'mass_info' not in kwargs or kwargs['mass_info'] is None:  # set up some reasonable defaults
            self.mass_info = MassInfo(10.0, "ppm", "monoisotopic", "", 1)

        starts = []
        stops = []
        if hasattr(self.mz, "__len__"):
            for peak in self.mz:
                start, stop = self.tolerance_interval(peak)
                starts.append(start)
                stops.append(stop)
        else:
            start, stop = self.tolerance_interval(self.mz)
            starts.append(start)
            stops.append(stop)

        # create mass interval starts and stops
        self.starts = np.array(starts)
        self.stops = np.array(stops)

    def intersect(self, comparison_peaks):
        """
        find the intersections between two high resolution ion series.  calls standalone function to allow use of
        numba
        :param comparison_peaks: the ion series to compare to
        :return: mz values of matches, matched peak indexes in self, matched peak indexes in comparison_peaks
        """
        return intersect_hires(self.starts, self.stops, self.mz, comparison_peaks.starts, comparison_peaks.stops)

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


def qtof_create_noise(max_mz=1000, min_mz=50, max_intensity=10000):
    """
    generates a noise peak set that models the noise found in hires qtof spectra
    :param max_mz: maximum mz value of the noise generated
    :param min_mz: minimum mz value of the noise generated
    :param max_intensity: the max intensity of the spectra that the noise is being generated for
    :return: HiResPeaks
    """
    mzs = []
    intensities = []
    # note that this noise spectrum is normalized to an max intensity of 10000
    median = wald.rvs(loc=0.041090271470796476, scale=12.736191241902917)
    # 0.041090271470796476 12.736191241902917
    # iterate through mz range in steps of 10
    for mz in range(min_mz, max_mz, 10):
        fit_scale_intensity = max(1.08019131*median + 0.03144227*mz + 10.36699926, 0.0)
        fit_loc_intensity = 0.34127992*median + 0.00388558*mz - 5.61148618
        fit_scale_count = max(-0.14607159*median + -0.10885245*mz + 40.66113861, 0.0)
        fit_loc_count = 0.00999616*median + -0.0020208*mz - 0.47845968
        peak_num = max(int(wald.rvs(loc=fit_loc_count, scale=fit_scale_count)+0.5), 0)
        for i in range(peak_num):
            intensities.append(max(wald.rvs(loc=fit_loc_intensity, scale=fit_scale_intensity), 0) * max_intensity/10000.0)
            mzs.append(random.uniform(mz, mz + 10.0))
    return HiResPeaks(mzs, intensities)


class UnitMassPeaks(BasePeaks):
    """
    class for containing ions with unit mass resolution
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'mass_info' not in kwargs:  # set up some reasonable defaults
            self.mass_info = MassInfo(0.5, "daltons", "monoisotopic", "", 1)

    def intersect(self, comparison_peaks):
        """
        find the intersections between two unit mass ion series.  calls standalone function to allow use of
        numba
        :param comparison_peaks: the ion series to compare to
        :return: mz values of matches, matched peak indexes in self, matched peak indexes in comparison_peaks
        """
        return my_intersect1d(self.mz, comparison_peaks.mz)

    @staticmethod
    def cast_mz(mz):
        """
        cast a single mz value
        """
        return int(float(mz)+0.5)

    @staticmethod
    def cast_intensity(intensity):
        """
        cast a single mz value
        """
        return int(float(intensity)+0.5)

    def match(self, peaks2, common_mz, index1, index2, match_ratio=0.7):
        """
        look for matching peaks between this spectrum and the input spectrum
        :param peaks2: spectrum to match
        :param common_mz: matched mz
        :param index1: index into self of matched peaks
        :param index2: index into spectrum to match of matched peaks
        :param match_ratio: the max/min difference in intensity ratio of two peaks needed to declare a match
        :return: common mz values, indexes of matched peaks into this spectrum,
                 indexes of matched peaks into the input spectrum
        """
        new_common_mz = []
        new_index1 = []
        new_index2 = []
        # scan through array and make sure intensities are close if we declare this to be a match
        for i in range(len(index1)):
            if (1.0 - match_ratio) < self.intensity[index1[i]] / \
                    float(peaks2.intensity[index2[i]]) < (1.0 + match_ratio):
                new_common_mz.append(common_mz[i])
                new_index1.append(index1[i])
                new_index2.append(index2[i])
        return np.array(new_common_mz), np.array(new_index1), np.array(new_index2)

    def score_match(self, common_mz, new_hist, min_match=3):
        """
        score the peak matches from match()
        :param common_mz: the mz values of the matched peaks
        :param new_hist: dict holding the probability of a match at each mz value
        :param min_match: minimum number of peaks to match
        :return: the match p value
        """
        # prob = np.array([new_hist[x] for x in common_mz])
        # get probabilities.  assume a probability of 0.1 for values not found
        within = []
        for x in self.mz:
            if new_hist is not None:
                within.append(new_hist.get(x, 0.1))
            else:
                within.append(0.1)
        prob = np.array(within)
        # we model the peak matching process as set of bernoulli processes, where each m/z bin has a different
        # probability of a match.
        # https://stats.stackexchange.com/questions/177199/success-of-bernoulli-trials-with-different-probabilities
        # We assume a normal distribution, which is a rough approximation.  Might want to consider using
        # a multivariate hypergeometric distribution, e.g.
        # https://pymc-devs.github.io/pymc/distributions.html#multivariate-discrete-distributions
        # but will need code to do the integral.

        mu = np.sum(prob)
        sigma = math.sqrt(np.sum(prob * (1.0 - prob)))
        matches = len(common_mz)
        #        print(f"match mu={mu}, sigma={sigma}, matches={matches}")
        # integrate the normal distribution to find the probability of a random match
        # equal to or better than the current match
        # http://mathworld.wolfram.com/NormalDistribution.html
        # possibly consider using poisson, but will require integration
        # https://stats.stackexchange.com/questions/177199/success-of-bernoulli-trials-with-different-probabilities
        if sigma != 0.0 and matches >= min_match:
            match_prob = 0.5 - 0.5 * spc.erf((matches - mu) / (math.sqrt(2.0) * sigma))
            # match_prob = sts.poisson.sf(matches, mu)
            logging.debug(f"mu={mu}, sigma={sigma}, matches={matches}, match_prob={match_prob}")
        else:
            match_prob = 1.0
        return match_prob

    def unmatched_score2(self, index2):
        """
        score for not matching peaks in library spectrum
        :param index2: matches into library spectrum
        :return: probability
        """
        # peaks not in index2
        prob = []
        not_in = 0
        # create list of unmatched peaks in library spectrum
        for i in range(len(self.mz)):
            if self.intensity[i] > 100:
                prob.append(unmatch_prob_by_intensity[int(self.intensity[i] / 50)])
                if i not in index2:
                    not_in += 1
        prob = np.array(prob)
        mu = np.sum(prob)
        sigma = math.sqrt(np.sum(prob * (1.0 - prob)))
        # integrate the normal distribution to find the probability of a random match
        # equal to or better than the current match
        # http://mathworld.wolfram.com/NormalDistribution.html
        logging.debug(f"unmatches = {not_in}, mu={mu}, sigma={sigma}")
        if sigma != 0.0:
            unmatch_prob = 0.5 + 0.5 * spc.erf((not_in - mu) / (math.sqrt(2.0) * sigma))
            # unmatch_prob = sts.poisson.cdf(not_in, mu)
        else:
            unmatch_prob = 0.5
        return unmatch_prob


@jit(nopython=True)
def my_intersect1d(ar1, ar2):
    """
    simplified version of numpy intersect1d.  Pull outside of class so it can be jit compiled by numba (numba has only
    experimental class support).
    Note: this function does not work if there are peaks in each spectra with identical mz!
    :param ar1: mz values for one spectra
    :param ar2: mz values for another spectra
    :return: matched values of mz, index of matches into array 1, index of matches into array 2
    """
    aux = np.concatenate((ar1, ar2))
    aux_sort_indices = np.argsort(aux, kind='mergesort')
    aux = aux[aux_sort_indices]

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    ar1_indices = aux_sort_indices[:-1][mask]
    ar2_indices = aux_sort_indices[1:][mask] - ar1.size
    return int1d, ar1_indices, ar2_indices


@jit(nopython=True)
def cosine_score_calc(spectrum1_mz, spectrum1_intensity, spectrum2_mz, spectrum2_intensity, index1, index2):
    """
    the Stein and Scott 94 cosine score.  By convention, sqrt of score is taken and
    multiplied by 999.  separated out from class and placed here so that can be jit compiled by numba.
    :param spectrum1_mz: query spectrum mz
    :param spectrum1_intensity: query spectrum intensity
    :param spectrum2_mz: the comparison spectrum2 mz
    :param spectrum2_intensity: the comparison spectrum2 intensity
    :param index1: matched peaks in spectrum1
    :param index2: matched peaks in spectrum2
    :return: the cosine score
    """
    # calculate numerator
    a = np.zeros(len(index1))
    b = np.zeros(len(index1))
    for i in range(len(index1)):
        a[i] = spectrum1_intensity[index1[i]] ** 0.6 * spectrum1_mz[index1[i]] ** 3
        b[i] = spectrum2_intensity[index2[i]] ** 0.6 * spectrum2_mz[index2[i]] ** 3
    score = np.sum(np.multiply(a, b)) ** 2
    # calculate denominator
    a = spectrum1_intensity ** 0.6 * spectrum1_mz ** 3
    b = spectrum2_intensity ** 0.6 * spectrum2_mz ** 3
    score /= (np.sum(np.square(a)) * np.sum(np.square(b)))

    return math.sqrt(score) * 999


@jit(nopython=True)
def intersect_hires(peaks1_starts, peaks1_stops, peaks1_mz, peaks2_starts, peaks2_stops):
    """
    find the intersections between two high resolution ion series
    :param peaks1_starts: start positions of the first ion series to compare
    :param peaks1_stops: stop positions of the first ion series to compare
    :param peaks1_mz: mz positions of the first ion series to compare
    :param peaks2_starts: start positions of the second ion series to compare
    :param peaks2_stops: stop positions of the second ion series to compare
    :return: mz values of matches, matched peak indexes in peaks1, matched peak indexes in ion2
    """
    index1 = []  # matched peaks in peaks1
    index2 = []  # matched peaks in ion2
    common_mz = []  # matched mz
    i = 0  # index where search should start in peaks1
    j = 0  # index where the search should start in the comparison peaks
    while i < len(peaks1_starts) and j < len(peaks2_starts):
        if peaks1_starts[i] > peaks2_stops[j]:
            j = j + 1
            continue
        elif peaks1_stops[i] < peaks2_starts[j]:
            i = i + 1
            continue
        index1.append(i)
        index2.append(j)
        common_mz.append(peaks1_mz[i])
        i = i + 1
        j = j + 1
    return np.array(common_mz), np.array(index1), np.array(index2)


# probability of not matching a peak, in intensity bins of 50
unmatch_prob_by_intensity = [0.3464970330778065, 0.049835007699640686, 0.027764136701769703, 0.01835799799049744,
                             0.01275316798440377, 0.010326475494479308, 0.008347667712233128, 0.0072523829258184834,
                             0.005810370631217537, 0.004280196262657897, 0.005089706069474488, 0.00413856529736358,
                             0.003952569169960474, 0.004219409282700422, 0.0029455081001472753, 0.0029585798816568047,
                             0.002119285498032092, 0.0010359116022099447, 0.0011489850631941786, 0.001004915940139602]


class SpectralSearchConfig:
    """
    configuration for spectral similarity search
    """
    cosine_threshold = 200  # the minimum cosine score to place in results
    minimum_match = 2  # the minimum number of matched peaks for matching two spectra
    minimum_mz = 50  # when filtering, accept no mz value below this setting
    minimum_intensity = 10  # when filtering, the minimum intensity value allowed
    identity_name = False  # for identity matching, require the name to match in addition to the inchi_key
    identity_energy = False  # for identity matching, require the collision_energy to match in addition to the inchi_key
    fp_tanimoto_threshold = 0.0  # use the spectra fingerprint and this tanimoto cutoff to speed up searching. 0=none


class BaseSpectrum:
    """
    Base class for spectrum with called peaks.
    The properties attribute is a dict that contains any structured data
    """

    def __init__(self, precursor_mass_info=None, product_mass_info=None):
        self.precursor = None
        self.products = None
        self.filtered = None  # filtered version of self.products
        self.properties = {}
        self.id = None  # unique identifier
        self.name = None  # friendly name
        self.column = None  # column used for retention index
        self.experimental_ri = None  # measured retention index
        self.experimental_ri_error = None  # error on measured retention index
        self.experimental_ri_data = None  # number of data points used to measure retention index
        self.stdnp = None
        self.stdnp_error = None
        self.stdnp_data = None
        self.stdpolar = None
        self.stdpolar_error = None
        self.stdpolar_data = None
        self.estimated_ri = None  # estimated retention index
        self.estimated_ri_error = None  # error on estimated retention index
        self.retention_time = None  # retention time in seconds
        self.inchi_key = None  # inchi key
        self.formula = None  # chemical formula
        self.synonyms = None  # list of synonyms
        self.precursor_class = BasePeaks
        self.precursor_mass_info = precursor_mass_info
        self.product_class = BasePeaks
        self.product_mass_info = product_mass_info
        self.exact_mass = None
        self.ion_mode = None
        self.charge = None
        self.instrument = None
        self.instrument_type = None
        self.ionization = None
        self.collision_energy = None
        self.nce = None
        self.ev = None
        self.insource_voltage = None
        self.collision_gas = None
        self.sample_inlet = None
        self.spectrum_type = None
        self.precursor_type = None
        self.vial_id = None

    def from_arrays(self, mz, intensity, row=None, annotations=None):
        """
        Initialize from a series of arrays and the information in rows

        :param mz: mz array
        :param intensity: intensity array
        :param row: dict containing parameters
        :param annotations: annotations on the peaks
        """
        self.precursor = self.precursor_class(row.get('precursor_mz', None), mass_info=self.precursor_mass_info)
        self.name = row.get("name", None)
        self.id = row.get("id", None)
        self.retention_time = row.get("retention_time", None)
        # make a copy of the properties
        self.properties.update(row)
        # numpy array of peak intensity
        self.products = self.product_class(mz, intensity, mass_info=self.product_mass_info, annotations=annotations)
        return

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
                logging.debug(f"Invalid unicode character in property {prop_name} for spectrum {self.id} with error {err}")
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
                    logging.debug(f"No float in property {prop_name} for spectrum {self.id}")
                    prop = None
            except UnicodeDecodeError as err:
                logging.debug(f"Invalid unicode character in property {prop_name} for spectrum {self.id} with error {err}")
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
                    logging.debug(f"No int in property {prop_name} for spectrum {self.id}")
                    prop = None
            except UnicodeDecodeError as err:
                logging.debug(f"Invalid unicode character in property {prop_name} for spectrum {self.id} with error {err}")
                prop = None
        return prop

    def from_mol(self, mol, skip_expensive=False, id_field='NISTNO', id_field_type='int'):
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
            self.precursor = self.precursor_class(self.precursor_class.cast_mz(precursor_mz),
                                                  mass_info=self.precursor_mass_info)
        self.name = self.get_string_prop(mol, "NAME")
        self.synonyms = self.get_string_prop(mol, "SYNONYMS")
        if self.synonyms is not None:
            self.synonyms = self.synonyms.splitlines()
        if id_field_type == 'int':
            self.id = self.get_int_prop(mol, id_field)
        else:
            self.id = self.get_string_prop(mol, id_field)

        if mol.HasProp("EXPERIMENTAL RI MEDIAN/DEVIATION/#DATA"):
            ri_string = mol.GetProp("EXPERIMENTAL RI MEDIAN/DEVIATION/#DATA")
            ris = ri_string.split()
            for ri_string in ris:
                ri = re.split('[=/]', ri_string)
                if len(ri) == 4:
                    if ri[0] == 'SemiStdNP':
                        self.column = ri[0]
                        self.experimental_ri = float(ri[1])
                        self.experimental_ri_error = float(ri[2])
                        self.experimental_ri_data = int(ri[3])
                    elif ri[0] == 'StdNP':
                        self.stdnp = float(ri[1])
                        self.stdnp_error = float(ri[2])
                        self.stdnp_data = int(ri[3])
                    elif ri[0] == 'StdPolar':
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
            try:
                self.collision_energy = int(ce)
            except ValueError:
                match = re.search(r'NCE=(\d+)%', ce)
                if match:
                    self.nce = int(match.group(1))
                match = re.search(r'(\d+)eV', ce)
                if match:
                    self.ev = int(match.group(1))
        self.insource_voltage = self.get_int_prop(mol, "IN-SOURCE VOLTAGE")

        # populate spectrum properties with mol properties
        for k in mol.GetPropNames():
            # skip over properties handled elsewhere
            if k in ["MASS SPECTRAL PEAKS", "NISTNO", "NAME", "MW", "EXPERIMENTAL RI MEDIAN/DEVIATION/#DATA",
                     "ESTIMATED KOVATS RI", "INCHIKEY", "RI ESTIMATION ERROR", "FORMULA", "SYNONYMS", "RIDATA_01",
                     "RIDATA_15", "PRECURSOR M/Z", "EXACT MASS", "ION MODE", "CHARGE", "INSTRUMENT", "INSTRUMENT TYPE",
                     "IONIZATION", "COLLISION ENERGY", "COLLISION GAS", "SAMPLE INLET", "SPECTRUM TYPE",
                     "PRECURSOR TYPE", "NOTES", "IN-SOURCE VOLTAGE", "ID"]:
                continue
            self.properties[k] = self.get_string_prop(mol, k)

        # get spectrum string and break into list of lines
        if mol.HasProp("MASS SPECTRAL PEAKS"):
            spectrum = mol.GetProp("MASS SPECTRAL PEAKS").splitlines()
            mz_in = []
            intensity_in = []
            annotations_in = []
            has_annotations = False
            for peak in spectrum:
                values = peak.replace("\n", "").split(maxsplit=2)  # get rid of any newlines then strip
                intensity = self.product_class.cast_intensity(values[1])
                # there are intensity 0 peaks in some spectra
                if intensity != 0:
                    mz_in.append(self.product_class.cast_mz(values[0]))
                    intensity_in.append(intensity)
                    annotations = []
                    if len(values) > 2 and not skip_expensive:
                        has_annotations = True
                        end = "".join(values[2:])
                        for match in re.finditer(r'(\?|((\w+\d*)+((\+?-?)((\w|\d)+))?(=((\w+)(\+?-?))?(\w+\d*)+((\+?-?)((\w|\d)+))?)?(\/(-?\d+.?\d+))(\w*)))*(;| ?(\d+)\/(\d+))', end):
                            annotation = {"begin": match.group(10), "change": match.group(11), "loss": match.group(12),
                                          "mass_diff": match.group(18), "diff_units": match.group(19),
                                          "fragment": match.group(3)}
                            annotations.append(annotation)
                            # group 10 is the beginning molecule, e.g. "p" means precursor
                            # group 11 indicates addition or subtraction from the beginning molecule
                            # group 12 is the chemical formula of the rest of the beginning molecule
                            # group 18 is the mass difference, e.g. "-1.3"
                            # group 19 is the mass difference unit, e.g. "ppm"
                            # group 3 is the chemical formula of the peak
                    annotations_in.append(annotations)
            if has_annotations and not skip_expensive:
                self.products = self.product_class(mz_in, intensity_in, mass_info=self.product_mass_info,
                                                   annotations=annotations_in)
            else:
                self.products = self.product_class(mz_in, intensity_in, mass_info=self.product_mass_info)
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

    def single_match(self, spectrum, results, settings=SpectralSearchConfig()):
        """
        try to match two spectra and calculate the probability of a match
        :param spectrum: the spectrum to match against
        :param settings: search settings
        :param results: dict containing search results
        :return: success of search
        filtering = cut peaks (a) with intensity below 10 and
           (b) below min of 50 Da or max of min mz of either spectra.
        """

        # intersect the spectra
        common_mz, index1, index2 = self.products.intersect(spectrum.products)
        matches = len(common_mz)
        if matches >= settings.minimum_match:
            cosine_score = self.products.cosine_score(spectrum.products, index1, index2)
            if cosine_score >= settings.cosine_threshold:
                max_min_mz = min([max([np.min(spectrum.products.mz), np.min(self.products.mz)]), settings.minimum_mz])
                # create filtered versions of the spectra
                self.filtered = self.products.copy(min_mz=max_min_mz, min_intensity=settings.minimum_intensity)
                spectrum.filtered = spectrum.products.copy(min_mz=max_min_mz, min_intensity=settings.minimum_intensity)
                filtered_common_mz, filtered_index1, filtered_index2 = self.filtered.intersect(spectrum.filtered)
                if len(filtered_common_mz) < settings.minimum_match:
                    return False
                results['query_id'].append(self.id)
                results['query_name'].append(self.name)
                results['query_formula'].append(self.formula)
                results['query_inchi_key'].append(self.inchi_key)
                results['hit_id'].append(spectrum.id)
                results['hit_name'].append(spectrum.name)
                results['hit_formula'].append(spectrum.formula)
                results['hit_inchi_key'].append(spectrum.inchi_key)
                results['matched_rank_prob'].append(self.products.score_rank(spectrum.products, index1, index2))
                results['cosine_score'].append(cosine_score)
                results['matches'].append(matches)
                identity = 1 if self.identity(spectrum, settings.identity_name, settings.identity_energy) else 0
                results['identical'].append(identity)
                results['matched_rank_prob_filtered'].append(self.filtered.score_rank(spectrum.filtered,
                                                                                      filtered_index1, filtered_index2))
                results['cosine_score_filtered'].append(self.filtered.cosine_score(spectrum.filtered, filtered_index1,
                                                                                   filtered_index2))
                return True
        return False

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
            return_val = return_val and (self.collision_energy == spectrum.collision_energy)
        return return_val

    def search_spectra(self, query_inchi_key=None, settings=SpectralSearchConfig(), spectrum_fp=None,
                       spectrum_fp_count=None, spectra=None, inchi_keys=None, spectra_fp=None, results=None):
        """
        search this spectrum against a standard dataframe containing spectra
        :param query_inchi_key: the inchi_key of the query.  If none, use the field in the spectrum
        :param settings: search settings
        :param spectrum_fp: fingerprint for the query spectrum
        :param spectrum_fp_count: number of set bits in spectrum_fp
        :param spectra: spectra to search
        :param inchi_keys: inchi_keys of the search set
        :param spectra_fp: spectral fingerprints of the search set
        :param results: dictionary containing search results
        :return results
        """

        if results is None:
            results = {'query_name': [], 'query_formula': [], 'query_inchi_key': [], 'hit_name': [], 'hit_formula': [],
                       'hit_inchi_key': [], 'matched_rank_prob': [], 'matched_rank_prob_filtered': [], 'cosine_score': [],
                       'cosine_score_filtered': [], 'matches': [], 'identical': [], 'query_id': [], 'hit_id': []}

        if settings.fp_tanimoto_threshold != 0.0:
            hits = np.array(DataStructs.BulkTanimotoSimilarity(spectrum_fp, spectra_fp))
            hits_index = np.nonzero(hits >= settings.fp_tanimoto_threshold)[0]
            for hit in hits_index:
                spectra[hit].inchi_key = inchi_keys[hit]
                self.single_match(spectra[hit], results, settings=settings)
            # hits = search_tanimoto(spectrum_fp, spectrum_fp_count, df['spectrum_fp'].values,
            #                        df['spectrum_fp_count'].values,
            #                        settings.fp_tanimoto_threshold)
        else:
            for spectrum, inchi_key in zip(spectra, inchi_keys):
                spectrum.inchi_key = inchi_key  # force the inchikey in the spectrum to match the dataframe
                self.single_match(spectrum, results, settings=settings)
        return results


class HiResSpectrum(BaseSpectrum):
    """
    class for a high resolution spectrum
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precursor_class = HiResPeaks
        self.product_class = HiResPeaks


class UnitMassSpectrum(BaseSpectrum):
    """
    class for containing the unit mass spectra
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precursor_class = UnitMassPeaks
        self.product_class = UnitMassPeaks

    def combinations(self, rank_filter=0):
        """
        generate sorted double and triple combinations of peaks
        :param rank_filter: filter out peaks with ranks above a value.  0 means turn off
        :return: triplets, doublets
        """
        triplet_mz = []
        for i in range(self.products.mz.size - 2):
            for j in range(i + 1, self.products.mz.size - 1):
                for k in range(j + 1, self.products.mz.size):
                    if rank_filter != 0:
                        if self.products.rank[i] > rank_filter or self.products.rank[j] > rank_filter or \
                                self.products.rank[k] > rank_filter:
                            continue
                    triplets = [self.products.mz[i], self.products.mz[j], self.products.mz[k]]
                    triplets.sort()
                    triplet_mz.append(tuple(triplets))
        #        triplet.products.mz = np.array(triplet.products.mz)
        #        triplet_rank = np.array(triplet_rank)

        doublet_mz = []
        for i in range(self.products.mz.size - 1):
            for j in range(i + 1, self.products.mz.size):
                if rank_filter != 0:
                    if self.products.rank[i] > rank_filter or self.products.rank[j] > rank_filter:
                        continue
                doublets = [self.products.mz[i], self.products.mz[j]]
                doublets.sort()
                doublet_mz.append(tuple(doublets))
        #        doublet.products.mz = np.array(doublet.products.mz)
        #        doublet_rank = np.array(doublet_rank)

        return doublet_mz, triplet_mz

    @staticmethod
    def fix_ranks(ranks):
        """
        rerank a numpy array
        :param ranks:
        :return:
        """
        order = ranks.argsort()
        ranks = len(order) - order.argsort()
        return ranks

    def simplealg_mw(self, lowest_ab=3):
        """
        Arun's algorithm to guess mw
        :param lowest_ab:
        :return: molecular weight
        """
        n = self.products.mz.shape[0]

        if n <= 0:
            return 0
        if n == 1:
            return self.products.mz[0]

        a2 = 0

        for i in range(n-1, 0, -1):
            m0 = int(self.products.mz[i])
            a0 = int(self.products.intensity[i])

            m1 = int(self.products.mz[(i-1)])
            a1 = int(self.products.intensity[(i-1)])

            md1 = m0 - m1

            if i-2 < 0:
                md2 = 0
            else:
                m2 = int(self.products.mz[(i - 2)])
                md2 = m0 - m2
                a2 = int(self.products.intensity[(i - 2)])

            if a0 < lowest_ab:
                continue  # ignore trace peaks

            if md1 > 2:
                return m0  # no peaks within 2 Da

            if md1 == 2:  # possible Cl Br peaks
                if a1 < a0 / 2:
                    return m0
            elif md1 == 1:  # check for c13 isotope peaks
                isocalc = float(a1) * .011 * float(m1) / 14.  # (a1*0.011*m1) / 14

                if int(3. * isocalc) > a0 or (a0 - int(isocalc)) < lowest_ab:
                    if i == 1:
                        return m1
                    continue

                if md2 > 2:
                    return m0
                else:
                    if i - 2 >= 0:
                        if a2 < a0 / 2:
                            return m0  # end else if md1==1
        return self.products.mz[(n - 1)]


def init_spectrum(hi_res):
    """
    spectrum factory
    :param hi_res: should it be a hi resolution spectrum?
    :return: spectrum object
    """
    if hi_res:
        spectrum = HiResSpectrum(precursor_mass_info=MassInfo(10.0, "ppm", "monoisotopic", "", 1),
                                 product_mass_info=MassInfo(10.0, "ppm", "monoisotopic", "", 1))
    else:
        spectrum = UnitMassSpectrum(precursor_mass_info=MassInfo(0.5, "daltons", "monoisotopic", "", 1),
                                    product_mass_info=MassInfo(0.5, "daltons", "monoisotopic", "", 1))
    return spectrum


def count_elements(seq, hist):
    """
    make a histogram
    :param seq: the sequence that is histogram
    :param hist: the histogram
    """
    for i in seq:
        hist[i] = hist.get(i, 0) + 1


if __name__ == "__main__":
    import unittest


    class TestSpectrumMethods(unittest.TestCase):
        """
        unit tests for the NISTSpectrum and NISTPeaks classes
        """
        config = EIMLConfig()

        def test_peaks(self):
            precursor = UnitMassPeaks(456, 123,)

            precursor.monoisotopic = True
            self.assertEqual(precursor.intensity, 123)
            self.assertEqual(precursor.mass_info.mass_type, "monoisotopic")
            self.assertEqual(precursor.mz, 456)
            return

        unit_mass_spectrum = UnitMassSpectrum()
        unit_mass_spectrum.precursor = unit_mass_spectrum.precursor_class(2, 1)
        unit_mass_spectrum.products = unit_mass_spectrum.product_class([1, 3, 2], [4, 6, 5],
                                                                       annotations=[{"peak": 0}, {"peak": 1},
                                                                                    {"peak": 2}])

        def test_spectrum(self):
            self.assertEqual(self.unit_mass_spectrum.precursor.intensity, 1)
            self.assertEqual(self.unit_mass_spectrum.precursor.mz, 2)
            self.assertEqual(self.unit_mass_spectrum.products.intensity.tolist(), [4, 5, 6])
            self.assertEqual(self.unit_mass_spectrum.products.mz.tolist(), [1, 2, 3])
            self.assertSequenceEqual(self.unit_mass_spectrum.products.annotations.tolist(), [{"peak": 0}, {"peak": 2},
                                                                                             {"peak": 1}])
            return

        def test_load_spectrum(self):
            spectrum = HiResSpectrum()
            spectrum.from_arrays([100.1, 200.2], [999, 1], row={"id": 1234, "retention_time": 4.5, "name": "hello",
                                                                "precursor_mz": 500.5})
            self.assertEqual(spectrum.precursor.mz, 500.5)
            self.assertEqual(spectrum.products.mz[1], 200.2)
            self.assertEqual(spectrum.products.intensity[1], 1)
            self.assertEqual(spectrum.products.rank[1], 2)
            self.assertEqual(spectrum.id, 1234)
            return

        def test_rdkit(self):
            suppl = Chem.SDMolSupplier("test.new.sdf", sanitize=False)
            for mol in suppl:
                spectrum = UnitMassSpectrum()
                spectrum.from_mol(mol)
                self.assertEqual(spectrum.precursor.mz, 180)
                self.assertEqual(spectrum.products.mz.size, 74)
                self.assertEqual(spectrum.formula, "C9H8O4")
                break

        def test_intersect_spectrum(self):
            spectrum1 = UnitMassSpectrum()
            spectrum1.from_arrays([100, 200, 300], [999, 1, 50],
                                  row={"id": 1234, "retention_time": 4.5, "name": "hello", "precursor_mz": 500})

            spectrum2 = UnitMassSpectrum()
            spectrum2.from_arrays([100, 200, 500, 300], [999, 1, 50, 120],
                                  row={"id": 1234, "retention_time": 4.5, "name": "hello", "precursor_mz": 500})

            common_mz, index1, index2 = spectrum1.products.intersect(spectrum2.products)
            self.assertSequenceEqual(index1.tolist(), [0, 1, 2])
            self.assertSequenceEqual(index2.tolist(), [0, 1, 2])
            return

        hi_res1 = HiResSpectrum()
        hi_res1.from_arrays([100.0001, 200.0002, 300.0003], [999, 1, 50],
                            row={"id": 1234, "retention_time": 4.5, "name": "hello", "precursor_mz": 500.5},
                            annotations=[['annot11'], ['annot12'], ['annot13']])

        hi_res2 = HiResSpectrum()
        hi_res2.from_arrays([100.0002, 200.0062, 500.0, 300.0009], [999, 1, 50, 120],
                            row={"id": 1234, "retention_time": 4.5, "name": "hello", "precursor_mz": 500.5},
                            annotations=[['annot21'], ['annot22'], ['annot24'], ['annot23']])

        def test_intersect_hires_spectrum(self):
            common_mz, index1, index2 = self.hi_res1.products.intersect(self.hi_res2.products)
            self.assertSequenceEqual(index1.tolist(), [0, 2])
            self.assertSequenceEqual(index2.tolist(), [0, 2])
            return

        def test_filter_peaks(self):
            filtered = self.hi_res1.products.copy(min_mz=150, max_mz=250)
            self.assertSequenceEqual(filtered.mz.tolist(), [200.0002])
            filtered = self.hi_res1.products.copy(min_intensity=15, max_intensity=100)
            self.assertSequenceEqual(filtered.mz.tolist(), [300.0003])
            return

        def test_normalize(self):
            filtered = self.hi_res1.products.copy()  # make a copy
            filtered.normalize(10000)
            self.assertSequenceEqual(filtered.intensity.tolist(), [10000, 10, 500])
            return

        def test_mask(self):
            filtered = self.hi_res1.products.copy()  # make a copy
            filtered.mask([1, 2])
            self.assertSequenceEqual(filtered.intensity.tolist(), [999])
            self.assertSequenceEqual(filtered.mz.tolist(), [100.0001])
            return

        def test_merge(self):
            merge1 = self.hi_res1.products.copy()  # make a copy
            merge2 = self.hi_res2.products.copy()  # make a copy
            merge1.merge(merge2)
            self.assertSequenceEqual(merge1.intensity.tolist(), [999, 999, 1, 1, 50, 120, 50])
            self.assertSequenceEqual(merge1.annotations.tolist(),
                                     [['annot11'], ['annot21'], ['annot12'], ['annot22'], ['annot13'], ['annot23'],
                                      ['annot24']])
            return

    unittest.main()

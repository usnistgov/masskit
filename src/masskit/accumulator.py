import math
from abc import ABC, abstractmethod

import pyarrow as pa


class Accumulator(ABC):
    """
    accumulator class used to take the mean and standard deviation of a set of data
    """

    def __init__(self, *args, **kwargs):
        """
        initialize values
        """
        super().__init__(*args, **kwargs)

    @abstractmethod
    def add(self, new_item):
        """
        accumulated a new item

        :param new_item: new item to be accumulated
        """
        pass

    @abstractmethod
    def finalize(self):
        """
        finalize the accumulation
        """
        pass


class AccumulatorProperty(Accumulator):
    """
    used to calculate the mean and standard deviation of a property
    """

    def __init__(self, *args, **kwargs):
        """
        initialize predicted spectrum

        :param mz: array of mz values
        :param tolerance: mass tolerance in daltons
        :param take_max: when converting new_spectrum to evenly spaced bins, take max value per bin, otherwise sum
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        # keep count of the number samples
        self.count = 0
        self.predicted_mean = 0.0
        self.predicted_stddev = 0.0

    def add(self, new_item):
        """
        add an item to the average.  Keeps running total of average and std deviation using
        Welford's algorithm. 

        :param new_spectrum: new item to be added
        """
        delta = new_item - self.predicted_mean
        # increment the count, dealing with case where new spectrum is longer than the summation spectrum
        self.count += 1
        self.predicted_mean += delta / self.count
        delta2 = new_item - self.predicted_mean
        self.predicted_stddev += delta * delta2

    def finalize(self):
        """
        finalize the std deviation after all the the spectra have been added
        """
        self.predicted_stddev = math.sqrt(self.predicted_stddev / self.count)


# populate_properties(AccumulatorProperty, fields=accumulator_property_fields)

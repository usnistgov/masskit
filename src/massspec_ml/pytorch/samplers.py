from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Optional, Iterator

import numpy as np
from torch.utils.data import Dataset, Sampler, DistributedSampler


class BaseSampler(ABC):
    """
    base class for sampler. used by each worker thread to select which records should be included in the epoch
    data for that worker.
    """
    def __init__(self, dataset):
        """
        initialize with dataset.  The columns available to this dataset are set in
        config.ms.dataset_columns.

        :param dataset: the input dataset
        """
        self.dataset = dataset
        self.rng = np.random.default_rng()

    @abstractmethod
    def probability(self):
        """
        method to compute the probability of sampling a particular record

        :return: numpy array with the probability of sampling, from [0,1]
        """
        pass


class DatasetFromSampler(Dataset):
    """
    Dataset to create indexes from `Sampler`
    """

    def __init__(self, sampler: Sampler):
        """
        Initialisation for DatasetFromSampler

        :param sampler: PyTorch sampler
        """
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """
        Gets element of the dataset

        :param index: index of the element in the dataset
        :return: Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns length

        :return: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        initialize distributed sampler wrapper

        :param sampler: Sampler used for subsampling
        :param num_replicas (int, optional): Number of processes participating in distributed training
        :param rank (int, optional): Rank of the current process within ``num_replicas``
        :param shuffle (bool, optional): If true (default), sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over sampler

        :return: python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

#class LengthSampler(BaseSampler):
#    """
#    sampler based on length of a peptide.  borrowed from alphafold 2
#    """
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#
#    def probability(self):
#        """
#        method to compute the probability of sampling a particular record
#
#        :return: numpy array with the probability of sampling, from [0,1]
#        """
#        return np.minimum(
#                np.maximum(np.vectorize(len)(self.dataset.data.table[2]),
#                            self.dataset.config.ml.sampler.min_length) * self.dataset.config.ml.sampler.scale,
#                            self.dataset.config.ml.sampler.max_length) / self.dataset.config.ml.sampler.max_length


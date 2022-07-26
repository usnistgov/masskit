import datetime
import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset

from massspec.utils.general import class_for_name
from massspec_ml.pytorch.base_objects import ModelInput


class BaseDataset(Dataset, ABC):
    """
    abstract base class for NIST Dataset
    Notes:
        - only one of these is created per GPU per epoch (or per entire run?)
    """

    def __init__(self, store_in, config_in, set_to_load=None, output_column=None) -> None:
        """
        :param store_in: name of data store
        :param config_in: configuration data
        :param set_to_load: which set to use, e.g. train, valid, test
        :param output_column: the name of the column to use for output
        """
        super().__init__()
        self.store = store_in
        self.config = config_in
        self.set_to_load = set_to_load
        self.worker_id = 0  # worker rank.  can be reset by worker_init_fn
        self.num_workers = (
            1  # number of worker processes.  can be reset by worker_init_fn
        )
        # retrieve the embedding class
        self.embedding = class_for_name(
            self.config.paths.modules.embeddings, self.config.ml.embedding.embedding_type
        )(self.config)

        if output_column is None:
            self.output_column = config_in.ml.output_column
        else:
            self.output_column = output_column
                    
        logging.debug(
            f"Dataset object in pid {os.getpid()} using db {store_in}"
            f" starting at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def to_pandas(self):
        """
        return data as pandas dataframe

        :raises NotImplementedError: not implemented
        """
        raise NotImplementedError('to_pandas is not implemented')
    
    def __getitem__(self, index: int):
        """
        gets item and outputs input (x) and target (y) tensors, as well as the record index and data store
        Note that the model steps (e.g. training_step) has to handle the extra record index field. get_data_row()
        can be used to retrieve data for the index
        Args:
            index (int): Index
        """
        data_row = self.get_data_row(index)
        return ModelInput(x=self.get_x(data_row), y=self.get_y(data_row), index=index)

    @abstractmethod
    def get_data_row(self, index):
        """
        given the index, return corresponding data for the index
        """
        pass

    def get_x(self, data_row):
        """
        given the data row, return the input to the network
        """
        return self.embedding.embed(data_row)

    @abstractmethod
    def get_y(self, data_row):
        """
        given the data row, return the target of the network
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def init_copy(self, worker_id=0, num_workers=1):
        """
        initialize a copy of a Dataset.
        Pytorch lightning constructs Datasets in the main thread, then forks, so
        to initialize a Dataset, this funcion is called in the worker_init_fn

        :param worker_id: worker id of the thread, defaults to 0
        :param num_workers: number of workers total, defaults to 1
        """
        
        self.worker_id = worker_id
        self.num_workers = num_workers
        

class DataframeDataset(BaseDataset):
    """
    dataset for a dataframe
    """
    def __init__(self, store_in, config_in, set_to_load=None, output_column=None) -> None:
        """
        :param store_in: data store
        :param config_in: configuration data
        :param set_to_load: which set to load, e.g. train, valid, test
        :param output_column: the name ouf the column to use for output
        """
        super().__init__(store_in, config_in, set_to_load=set_to_load, output_column=output_column)
        self.data = store_in

    def to_pandas(self):
        return self.data

    def get_data_row(self, index):
        """
        given the index, return corresponding data for the index
        """
        return self.store.iloc[index]

    def get_y(self, data_row):
        """
        given the data row, return the target of the network
        """
        output = data_row[self.output_column]
        return torch.from_numpy(np.asarray(output))

    def __len__(self) -> int:
        return len(self.store.index)
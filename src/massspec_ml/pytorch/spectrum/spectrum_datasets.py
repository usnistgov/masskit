from massspec.utils.arrow import save_to_plasma
from pyarrow import plasma
import pyarrow.parquet as pq
import logging
import numpy as np
import torch
from massspec.utils.index import ArrowLibraryMap
from massspec_ml.pytorch.base_datasets import BaseDataset, DataframeDataset
from massspec.data_specs.spectral_library import *
from massspec.utils.general import class_for_name
from massspec_ml.pytorch.lightning import get_pytorch_ranks
import builtins

"""
pytorch datasets for spectra.
note on terminology: dataloaders iterate over datasets.  samplers are used by dataloaders to sample from datasets.
datamodules set up and use dataloaders, samplers, and datasets.
"""


class SpectrumDataset(BaseDataset):
    """
    Base spectrum dataset
    """
    def __init__(self, store_in, config_in, set_to_load, columns=None) -> None:
        """
        :param store_in: data store
        :param config_in: configuration data
        :param set_to_load: which set to load, e.g. train, valid, test
        :param columns: columns to load.  otherwise, use ms.columns
        """
        super().__init__(store_in, config_in, set_to_load)

        if columns:
            self.columns = columns
        else:
            self.columns = self.config.ms.columns

    def get_y(self, data_row):
        shape = (1, int(self.config.ms.max_mz / self.config.ms.bin_size))
        spectra = np.zeros(shape, dtype=np.float32)
        query = data_row[self.output_column]
        query.products.ions2array(
            spectra,
            0,
            bin_size=self.config.ms.bin_size,
            down_shift=self.config.ms.down_shift,
            intensity_norm=np.max(query.products.intensity),
            channel_first=self.config.ml.embedding.channel_first,
            take_max=self.config.ms.take_max,
            take_sqrt=self.config.ms.get('take_sqrt', False),
        )
        # spectra = np.squeeze(spectra)
        return torch.from_numpy(np.asarray(spectra))


class TandemDataframeDataset(SpectrumDataset, DataframeDataset):
    
    def __init__(self, store_in, config_in, set_to_load, columns=None) -> None:
        """
        :param store_in: data store
        :param config_in: configuration data
        :param set_to_load: which set to load, e.g. train, valid, test
        :param columns: columns to load.  otherwise, use ms.columns
        """
        super().__init__(store_in, config_in, set_to_load)


class TandemDataset(SpectrumDataset):
    """
    base map Dataset for operating on a pandas dataframe which is loaded from sql.  index2id in this class
    are indices into the dataframe.  The list is assumed to have
    been already prefiltered by query, e.g. only the training set and by the GPU rank if using
    DistributedDataParallel.

    How workers are set up requires some explanation:
    - if there is more than one gpu, each gpu has a corresponding main process.
    - the numbering of this gpu within a node is given by the environment variable LOCAL_RANK
    - if there is more than one node, the numbering of the node is given by the NODE_RANK environment variable
    - the number of nodes times the number of gpus is given by the WORLD_SIZE environment variable
    - the number of gpus on the current node can be found by parsing the PL_TRAINER_GPUS environment variable
    - these environment variables are only available when doing ddp.  Otherwise sharding should be done using
      id and num_workers in torch.utils.data.get_worker_info()

    - each main process creates an instance of Dataset.  This instance is NOT initialized by worker_init_fn, only
      the constructor.
    - each worker is created by forking the main process, giving each worker a copy of Dataset, already constructed.
      - each of these forked Datasets is initialized by worker_init_fn
      - the global torch.utils.data.get_worker_info() contains a reference to the forked Dataset and other info
      - these workers then take turns feeding minibatches into the training process
      - ***important*** since each worker is a copy, __init__() is only called once, only in the main process
    - the dataset in the main processes is used by other steps, such as the validation callback
      - this means that if there is any important initialization done in worker_init_fn, it must explicitly be done
        to the main process Dataset

    - alternative sources of parameters:
       - global_rank = trainer.node_rank * trainer.num_processes + process_idx
       - world_size = trainer.num_nodes * trainer.num_processes
    """

    def __init__(self, store_in, config_in, set_to_load, columns=None) -> None:
        """
        :param store_in: data store
        :param config_in: configuration data
        :param set_to_load: which set to load, e.g. train, valid, test
        :param columns: columns to load.  otherwise, use ms.columns
        """
        super().__init__(store_in, config_in, set_to_load, columns=columns)

        self.lib_type = class_for_name(
            ["massspec.data_specs.base_library"], self.config.input.lib_type
        )
        # query database to get length
        self.engine = sqlalchemy.create_engine(f"sqlite:///{store_in}"
                                               f"?check_same_thread=false&uri=true")
        # use the correct where clause
        where = self.config.input[self.set_to_load].where

        # if multiple nodes and gpus, slice the data with equal slices to each gpu
        is_parallel, world_rank, world_size, num_gpus, num_nodes, node_rank, local_rank, worker_id = get_pytorch_ranks()
        if is_parallel:
            where += f" AND ROWID % {world_size} = {world_rank}"

        # we load in the entire data set for this gpu.  this will be subset in the worker processes
        self.data = LibraryAccessor.read_sql(
            self.engine,
            where=where,
            lib_type=self.lib_type,
            columns=self.columns,
        )
        # make a copy of the dataframe index so we can convert batch indexes to
        self.index2id = self.data.index.copy()
        logging.debug(
            f"TandemDataset created with where clause={where} and set_to_load={set_to_load}"
        )
        self.length = len(self.data.index)

        # self.length = session.query(sqlalchemy.func.count(self.lib_type.id)).filter(sqlalchemy.text(where)).scalar()

    def __len__(self) -> int:
        """
        This returns the number of rows of data for the *entire* gpu process, not necessarily the number of rows
        in self.data

        :return: number of rows
        """
        return self.length

    def get_data_row(self, index):
        """
        given the index, return corresponding data for the index
        """
        # index = int((index - self.worker_id)/self.num_workers)
        row_id = self.index2id[index]
        i = self.data.index.get_loc(row_id)
        return self.data.iloc[i]


class TandemQueryDataset(TandemDataset):
    """
    dataset that retrieves each row by query, rather than from an in memory pandas dataframe.  The
    index, however, is from an in memory pandas dataframe
    """
    def __init__(self, store_in, config_in, set_to_load) -> None:
        """
        :param store_in: data store
        :param config_in: configuration data
        :param set_to_load: which set to load, e.g. train, valid, test
        :param columns: columns to load.  otherwise, use ms.columns
        """
        super().__init__(store_in, config_in, set_to_load, columns=config_in.ms.dataset_columns)

    def get_data_row(self, index):
        """
        given the index, return corresponding data for the index
        """
        session = sessionmaker(self.engine)()
        row_id = self.index2id[index]
        u = session.query(self.lib_type).options(load_only(*self.config.ms.columns))\
            .filter(sqlalchemy.text(f'id = {row_id}')).first()
        # log ids to a file.  opens file each time as this object is copied and not constructed in each thread
        if "log_ids" in self.config.input.train and self.config.input.train.log_ids and self.set_to_load == 'train':
            with open(f"log_worker_{self.worker_id}.txt", "a+") as logfile:
                logfile.write(f"{row_id}\n")
        session.close()
        return u.__dict__


class TandemArrowDataset(SpectrumDataset):
    """
    class for accessing a tandem dataframe of spectra

    How workers are set up requires some explanation:
    - if there is more than one gpu, each gpu has a corresponding main process.
    - the numbering of this gpu within a node is given by the environment variable LOCAL_RANK
    - if there is more than one node, the numbering of the node is given by the NODE_RANK environment variable
    - the number of nodes times the number of gpus is given by the WORLD_SIZE environment variable
    - the number of gpus on the current node can be found by parsing the PL_TRAINER_GPUS environment variable
    - these environment variables are only available when doing ddp.  Otherwise sharding should be done using
      id and num_workers in torch.utils.data.get_worker_info()

    - each main process creates an instance of Dataset.  This instance is NOT initialized by worker_init_fn, only
      the constructor.
    - each worker is created by forking the main process, giving each worker a copy of Dataset, already constructed.
      - each of these forked Datasets is initialized by worker_init_fn
      - the global torch.utils.data.get_worker_info() contains a reference to the forked Dataset and other info
      - these workers then take turns feeding minibatches into the training process
      - ***important*** since each worker is a copy, __init__() is only called once, only in the main process
    - the dataset in the main processes is used by other steps, such as the validation callback
      - this means that if there is any important initialization done in worker_init_fn, it must explicitly be done
        to the main process Dataset

    - alternative sources of parameters:
       - global_rank = trainer.node_rank * trainer.nudatasetm_processes + process_idx
       - world_size = trainer.num_nodes * trainer.num_processes
    """

    def __init__(self, store_in, config_in, set_to_load, columns=None) -> None:
        """
        :param store_in: data store
        :param config_in: configuration data
        :param set_to_load: which set to load, e.g. train, valid, test
        :param columns: columns to load.  otherwise, use ms.columns
        """
        super().__init__(store_in, config_in, set_to_load, columns=columns)
        
        # set up where clause
        
        if self.config.input[self.set_to_load].where is not None:
            self.filters = eval(self.config.input[self.set_to_load].where)
        else:
            self.filters = None

        # plasma client
        self.client = None

        # if multiple nodes and gpus, slice the data with equal slices to each gpu
        is_parallel, world_rank, world_size, num_gpus, num_nodes, node_rank, local_rank, worker_id = get_pytorch_ranks()
        if is_parallel:
            raise NotImplementedError('distributed training not yet implemented')
        #    where += f" AND ROWID % {world_size} = {world_rank}"

    def init_plasma(self):
        """
        used for lazy initialization of plasma client.  Since lightning copies dataset objects on forking, if a client is in
        in the original process, then it will be deleted in the copied process when replaced with a new copy of the client
        causing the any corresponding objects in the plasma store to be deleted as they are refcounted by connection.  So
        we only create the client when we need to get the data.
        """
        if "instance_settings" in dir(builtins) and 'plasma' in builtins.instance_settings and 'socket' in builtins.instance_settings['plasma']:
            self.client  = plasma.connect(builtins.instance_settings['plasma']['socket'])

            data_out = save_to_plasma(self.client, self.store, self.columns, self.filters)
            if data_out:
                self.data = data_out
            else:
                logging.info("init TandemArrowDataset without loading from plasma")
                            
        else:
            self.client = self.store
            self.data = ArrowLibraryMap.from_parquet(self.store, columns=self.columns, filters=self.filters)

    def get_column(self, column):
        """
        retrieve a column from the parquet file

        :param column: the column to retrieve
        """
        table = pq.read_table(self.store, columns=[column], filters=self.filters)
        return table[column].to_numpy()
        
    def __len__(self) -> int:
        """
        This returns the number of rows of data for the *entire* gpu process, not necessarily the number of rows
        in self.data

        :return: number of rows
        """
        if self.client is None:
            id_list = pq.read_table(self.store, columns=['id'], filters=self.filters)
            return len(id_list)

        return len(self.data)

    @property
    def data(self):
        if self.client is None:
            self.init_plasma()
        return self._data

    @data.setter
    def data(self, value):
        if self.client is None:
            self.init_plasma()
        self._data = value
        
    def to_pandas(self):
        return self.data.to_pandas()
    
    def get_data_row(self, index):
        """
        given the index, return corresponding data for the index
        """
        return self.data.getitem_by_row(index)


# collate_fn can be used to pad out minibatches
# https://www.speechmatics.com/wp-content/uploads/2019/10/Speechmatics_Dataloader_Pytorch_Ebook_2019.pdf


"""
Notes on pyarrow usage:
- I believe the sampler automatically selects for each thread a list of rows(index) for each thread.  index2id is used
to turn this into an id for sql retrieval.
- can get individual row using row = mytable.slice(100000,1) or an id using mytable.filter(bool expression)
  - then convert row = row.to_pydict()
  - then {key: value[0] for key, value in dd.items()} to get rid of arrays
  - need to create spectrum out of arrays of mz, intensity
   - if using mol, convert json to mol.
- either we have to write a client/server where the server holds the pyarrow table or override the pytorch lightning 
machinery to distribute a set of ids or rows to each thread.  each thread then loads in the entire pyarrow table, 
but then quickly filters it down to the subset that is needed.
- filtering by row at load time is only supported through the experimental pa.dataset api.  In the lightning Dataset
  code, we'd first load in the column of ids from the parquet file, then create a filter based on the list
  of ids given to the Dataset.  Then we'd use the filter using the pa.dataset filter functionality to read in a subset
  of the data.  get_data_row() would be modified using the above logic for getting a single row.
- note that gpu level slicing of the data is done in the constructor
- problem: it looks like lightning gives each thread a subset of record indices (not ids) at the time of batch
  creation.  This makes it not possible to subset the data beforehand as the list of ids is not know.  It may be
  necessary to subset and shuffle the data in worker_init_fn() by hand, including subsetting by gpu and worker, and 
  not using the weighted, random, or distributed sampler.  It depends on how these samplers work.
  - it appears that RandomSampler just randomizes a list and takes batches:
    https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/torch/csrc/api/src/data/samplers/random.cpp#L12
    implying that the sampler is applied after the list of indices is sliced up
  - in DistributedRandomSampler, it appears that each instance gets a subset of the indices. 
    https://github.com/pytorch/pytorch/blob/e3d75b8475a1668a02ac4a23c160b7aee8ebb3d3/torch/csrc/api/src/data/samplers/distributed.cpp#L15
    https://github.com/pytorch/pytorch/blob/b2e79ed5ecabcf4be299dc2ed085223ab5c22fd7/torch/csrc/api/include/torch/data/samplers/distributed.h#L54
  - it doesn't appear that there is an easy way for the dataset to get the information in the sampler (or the
   dataloader). This can be worked around in SpectrumDataModule.create_loader() where the sampler and dataset is created
   by creating a dict that matches worker_id to index start stop by examining the contents of each sampler
   private variable indices_.  This is suboptimal as this is a private value and may not even be available in python.
   dict is then passed to the sampler on construction. 
  - it's still not clear where the sampler is set up to get the right indices per worker. I don't see how it can be in
    create_loader().  Is the sampler adjusted after the fact?
- due to the above problem, we likely have to resort to a plasma store, see
  https://github.com/apache/arrow/blob/master/python/examples/plasma/sorting/sort_df.py
  for an example.  The plasma store will hold a shared memory copy of the pyarrow table.
  - note that there has to be one plasma store per compute node (or perhaps per GPU)
  - see https://github.com/apache/arrow/blob/72c71a1b44ec35f1c5575cac6c8e096f12b40973/python/pyarrow/plasma.py#L82
    how to start plasma from python.  Note that you should use a with statement to ensure cleanup
  - table can be apparently loaded directly from plasma, e.g. table = plasma_client.get(table_id, timeout_ms=4000)
  - plan
    - create a subclass of BaseDataset that works the same way as TandemDataset
    - create a variant of SpectrumDataModule in spectrum_lightning.py that has a with clause that starts up a plasma
      server.  Put the plasma socket into config.
    - question: is createloader called once?  if not, the plasma creation need to be moved elsewhere.
"""


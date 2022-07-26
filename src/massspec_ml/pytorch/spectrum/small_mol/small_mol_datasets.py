import hashlib
from arrow import load_from_plasma, save_to_plasma
from fingerprints import ECFPFingerprint
from hitlist import CosineScore, TanimotoScore
from pyarrow import plasma
from pyarrow.plasma import ObjectID
from torch.utils.data import DataLoader
import logging
import numpy as np
import torch
from massspec.utils.index import ArrowLibraryMap, DescentIndex, TanimotoIndex
from massspec_ml.pytorch.base_datasets import BaseDataset
from massspec.data_specs.spectral_library import *
from massspec.utils.general import class_for_name
from massspec_ml.pytorch.lightning import get_pytorch_ranks
import builtins
from massspec_ml.pytorch.spectrum.spectrum_datasets import SpectrumDataset

"""
pytorch datasets for spectra.
note on terminology: dataloaders iterate over datasets.  samplers are used by dataloaders to sample from datasets.
datamodules set up and use dataloaders, samplers, and datasets.
"""


class TandemArrowSearchDataset(SpectrumDataset):
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

    # todo: the where clauses come out of the config, but the filenames come from the constructor
    # it should be one or the other.  The complication is that the DataModule tries to resolve the file path
    # so the configuration can't be sent directly.  
    def __init__(self, store_in, config_in, set_to_load, columns=None, store_search=None) -> None:
        """
        :param store_in: filename of data store for queries
        :param config_in: configuration data
        :param set_to_load: which set to load, e.g. train, valid, test
        :param columns: columns to load.  otherwise, use ms.columns
        :param store_search: filename of data store for searching
        """
        super().__init__(store_in, config_in, set_to_load, columns=columns)
        
        # filename of the search datastore
        self.store_search = store_search
        # the search TableMap
        self.data_search = None
        # set up where clause
        
        if self.config.input[self.set_to_load].where is not None:
            self.filters = eval(self.config.input[self.set_to_load].where)
        else:
            self.filters = None
        if self.config.input[self.set_to_load].where_search is not None:
            self.filters_search = eval(self.config.input[self.set_to_load].where_search)
        else:
            self.filters_search = None

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
            
            data_out = save_to_plasma(self.client, self.store_search, self.columns, self.filters_search)
            if data_out:
                self.data_search = data_out
            else:
                logging.info("init TandemArrowDataset without loading from plasma")    
                
        else:
            self.data = ArrowLibraryMap.from_parquet(self.store, columns=self.columns, filters=self.filters)
            self.data_search = ArrowLibraryMap.from_parquet(self.store_search, columns=self.columns, filters=self.filters_search)

        # initialize the search index
        
        # ${env:HOME}/data/nist/ei/2020/mainlib_2020.ecfp4.tani.npy
        # mainlib_2020.ecfp4.tani
        self.index = TanimotoIndex(fingerprint_factory=ECFPFingerprint)
        self.index.load(self.config.input[self.set_to_load].spectral_library_search_index)
        
        # need id list without filters to convert row hits found in DescentIndex into ids
        self.row2id = pq.read_table(self.store, columns=['id'])['id'].to_numpy()
        self.row2id_search = pq.read_table(self.store_search, columns=['id'])['id'].to_numpy()

    def __len__(self) -> int:
        """
        This returns the number of rows of data for the *entire* gpu process, not necessarily the number of rows
        in self.data

        :return: number of rows (which is the number of queries)
        """
        # we do lazy initialization of the plasma client due to the copying lightning does of datasets
        if self.client is None:
            id_list = pq.read_table(self.store, columns=['id'], filters=self.filters)
            return len(id_list)

        return len(self.data)

    def get_data_row(self, index):
        """
        given the index, return corresponding data for the index
        
        - input is index in query db (on self.data)
        - do search on index and score.  return n hits, where n is set in config somewhere
        - if hits are below threshold, set to 0
        - output is dict, query, hits, taninoto, where each is an array of length hitlist
        - query and hit are spectra, tanimoto is float
        - 
        """
        # we do lazy initialization of the plasma client due to the copying lightning does of datasets
        if self.client is None:
            self.init_plasma()
        # get fingerprint from store
        query = self.data.getitem_by_row(index)
        fp = query['ecfp4']
        # nce = query['nce']
        # ion_mode = query['ion_mode']
        hitlist_size = self.config.ms.search.hitlist_size
        if self.set_to_load == 'train':
            predicate = pa.compute.equal(self.data_search.to_arrow()['set'], 'train')
            # predicate = pa.compute.equal(self.data_search.to_arrow()['nce'], nce)
            # predicate = pa.compute.and_(predicate, pa.compute.equal(self.data_search.to_arrow()['ion_mode'], ion_mode))
            predicate = predicate.to_numpy().astype(np.bool_).astype(np.uint8)
        else:
            predicate = None

        # do the search.  note that row to id conversion for the hits come from the parquet file as that is synced to the
        # index (no filters are applied), while the row to id conversion for the query comes from the in memory table
        # where filters have been applied
        # 
        hitlist = self.index.search(fp, hitlist_size=hitlist_size, epsilon=self.config.ms.search.epsilon, predicate=predicate,
                                    id_list=self.row2id_search, id_list_query=np.array([query['id']]))
        # TanimotoScore(self.data_search, query_table_map=self.data).score(hitlist)
        ic = hitlist.hitlist.columns.get_loc('tanimoto')
        for i in range(len(hitlist.hitlist.index)):    
            if self.data_search.getitem_by_id(hitlist.hitlist.index.get_level_values(1)[i])['inchi_key'] == query['inchi_key']:
                hitlist.hitlist.iat[i, ic] = hitlist.hitlist.iat[i, ic] + 0.01
        hitlist.sort('tanimoto')
        results = {'query_spectrum': [None] * hitlist_size, 'hit_spectrum': [None] * hitlist_size, 
                   'tanimoto': np.full(hitlist_size, -1.0, dtype=np.float32),
                   'query_index': np.full(hitlist_size, -1, dtype=np.int64), 
                   'query_id': np.full(hitlist_size, -1, dtype=np.int64), 
                   'hit_id': np.full(hitlist_size, -1, dtype=np.int64)}
        count = 0
        for row in hitlist.hitlist.itertuples():
            if count >= hitlist_size:
                break
            try:
                results['hit_spectrum'][count] = self.data_search.getspectrum_by_id(row.Index[1])
            except IndexError:
                # not all hits are in self.data_search as the index contains records that are filtered out
                continue

            results['query_index'][count] = index            
            results['query_id'][count] = row.Index[0]
            results['hit_id'][count] = row.Index[1]
            results['query_spectrum'][count] = self.data.getspectrum_by_row(index)
            results['hit_spectrum'][count] = self.data_search.getspectrum_by_id(row.Index[1])
            results['tanimoto'][count] = row.tanimoto
            count += 1
        return results
    
    def spectrum2array(self, spectrum):
        """
        given a spectrum, create a spectrum array

        :param spectrum: the spectrum
        :return: the numpy spectrum array
        """
        shape = (1, int(self.config.ms.max_mz / self.config.ms.bin_size))
        spectrum_array = np.zeros(shape, dtype=np.float32)
        if spectrum is not None:
            spectrum.products.ions2array(
                spectrum_array,
                0,
                bin_size=self.config.ms.bin_size,
                down_shift=self.config.ms.down_shift,
                intensity_norm=np.max(spectrum.products.intensity),
                channel_first=self.config.ml.embedding.channel_first,
                take_max=self.config.ms.take_max,
                take_sqrt=self.config.ms.get('take_sqrt', False),
            )
        return spectrum_array

    def get_x(self, data_row):
        """
        given the data row, return the input to the network
        
        the input in this case is a pairwise tensor of query spectrum, hit spectrum with length hitlist
        create embedding that turns two arrays of spectra into a tensor of size 2, mz_max
        """
        pairs = []
        for i in range(len(data_row['query_index'])):
            query_spectrum = self.spectrum2array(data_row['query_spectrum'][i])
            hit_spectrum = self.spectrum2array(data_row['hit_spectrum'][i])
            pairs.append(np.concatenate([query_spectrum, hit_spectrum]))
        return torch.from_numpy(np.stack(pairs))

    def get_y(self, data_row):
        """
        returns tanimoto scores, of size hitlist size

        :param data_row: the data row
        :return: torch tensor
        """
        return torch.from_numpy(data_row['tanimoto'])

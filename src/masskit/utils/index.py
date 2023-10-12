import logging
import pickle
import timeit
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pynndescent
from scipy import sparse

from . import fingerprints as mkfingerprints
from . import hitlist as mkhitlist

# try:
#     from numba import jit, prange
# except ImportError:
#     # when numba is not available, define jit decorator
#     def jit(nopython=True, parallel=False):
#         def decorator(func):
#             def newfn(*args, **kwargs):
#                 return func(*args, **kwargs)

#             return newfn

#         return decorator
    
#     def prange(value):
#         return range(value)

"""
todos

- search() sometimes needs access to the tablemaps
- tablemaps should not be spectrum centric, e.g. supports different types of complex objects
- complex objects (spectra, mol) columns should support basic indexing apis.
- create and create_from_column ought to be merged and deduced from column type
- make it easy to use indices from script
- column object that supports type of indexing and indexing apis (shared with complex object?)

"""

class Index(ABC):
    """
    Index used for searching a library

    :param dimension: the number of features in the fingerprint
    :param fingerprint_factory: class used to encapsulate the
    :param index_name: name of index
    """

    def __init__(self, index_name=None, dimension=None, fingerprint_factory=None):
        super().__init__()
        self.dimension = dimension
        self.index = None
        self.fingerprint_factory = fingerprint_factory
        if index_name:
            self.index_name = index_name
        else:
            self.index_name = "default"

    @abstractmethod
    def create(self, table_map, column_name=None):
        """
        create index from a TableMap

        :param table_map: the library to index
        :param column_name: name of the column containing the objects to index.  None='spectrum'
        """
        raise NotImplementedError

    @abstractmethod
    def optimize(self):
        """
        optimize the index
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, objects, hitlist_size=50, epsilon=0.2, with_raw_score=False, id_list=None, id_list_query=None, predicate=None):
        """
        search a series of query objects

        :param objects: the query objects to be searched or their fingerprints
        :param hitlist_size: the size of the hitlist returned
        :param epsilon: max search accuracy error
        :param with_raw_score: calculate and return the raw_score
        :param id_list: array-like for converting row numbers in search results to hit ids
        :param id_list_query: array-like for converting row numbers in search results to query ids
        :param predicate: filter array that is the result of a predicate query
        """
        raise NotImplementedError

    def spectrum2array(self, spectrum, spectral_array_in, channel_in=0, dtype=np.float32, cutoff=0.0):
        if dtype == np.uint8:
            spectral_array = np.zeros((1, self.dimension), dtype=np.float32)
            channel = 0
        else:
            spectral_array = spectral_array_in
            channel = channel_in
        spectrum.products.ions2array(
            spectral_array,
            channel,
            bin_size=1,
            down_shift=0.5,
            intensity_norm=np.max(spectrum.products.intensity),
            channel_first=True,
            take_max=True
        )
        max_value = np.max(spectral_array[channel])
        if max_value > 0.0:
            spectral_array[channel] = spectral_array[channel] / max_value
            spectral_array[channel][spectral_array[channel] < cutoff] = 0.0
            spectral_array[channel] = np.sqrt(spectral_array[channel])
        else:
            logging.warning(f"zero peaks for spectrum {spectrum.id}")
        if dtype == np.uint8:
            spectral_array[channel] = spectral_array[channel] * 255
            spectral_array[channel][(spectral_array[channel] < 1.0) & (spectral_array[channel] > 0.0)] = 1.0
            spectral_array_in[channel_in] = spectral_array[channel].astype(np.uint8)

    @abstractmethod
    def save(self, file=None):
        """
        save the index

        :param file: name of file to save fingerprint in
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, file=None):
        """
        load the index

        :param file: name of the file to load the fingerprint in
        """
        raise NotImplementedError


"""
Code to use ngtpy library from yahoo.  has issues with missing hits
import ngtpy

class ONNIndex(Index):
    def __init__(self, index_name, dimension=2000, ann_index_name=None):
        super().__init__(dimension)
        self.index_name = f"{index_name}.onng".encode('UTF-8')
        if ann_index_name is None:
            self.ann_index_name = f"{index_name}.anng".encode('UTF-8')
        else:
            self.ann_index_name = ann_index_name.encode('UTF-8')

    def create(self, table_map):
        objects = []
        for i in range(len(table_map)):
            spectrum = init_spectrum(True).from_arrow(table_map.table, i)
            spectral_array = np.zeros((1, self.dimension), dtype=np.uint8)
            self.spectrum2array(spectrum, spectral_array, dtype=np.uint8)
            objects.append(spectral_array[0])

        ngtpy.create(path=self.ann_index_name, dimension=self.dimension, distance_type="Cosine",
                     object_type='Byte',
                     edge_size_for_creation=100, edge_size_for_search=-2)  # object_type='Byte'

        # open index.
        self.index = ngtpy.Index(self.ann_index_name)
        self.index.set(epsilon=0.2, edge_size=100)
        # insert the objects.
        self.index.batch_insert(objects)

        # save the index.
        self.index.save()

        # close the index.
        self.index.close()

        self.optimize()
        self.index = None

    def optimize(self):
        optimizer = ngtpy.Optimizer()
        optimizer.set(num_of_outgoings=10, num_of_incomings=120)
        optimizer.execute(self.ann_index_name, self.index_name)

    def search(self, objects, hitlist_size=100, epsilon=0.2, with_raw_score=False, id_list=None, predicate=None):
        if self.index is None:
            self.load()

        if len(np.shape(objects)) < 2:
            objects = [objects]

        query_ids = []
        hit_ids = []
        for spectrum in objects:
            spectral_array = np.zeros((1, self.dimension), dtype=np.uint8)
            self.spectrum2array(spectrum, spectral_array,  dtype=np.uint8)
            results = self.index.search(spectral_array, size=hitlist_size, epsilon=epsilon,
                                        with_distance=with_raw_score)
            hit_id = None
            for result in results:
                query_ids.append(spectrum.id)
                # dimensionality of result changes if raw score is requested
                if not with_raw_score:
                    hit_id = result
                else:
                    hit_id = result[0]

                # if given a row to id lookup table, use it
                if id_list is not None:
                    assert(0 <= hit_id < len(id_list))
                    hit_id = id_list[hit_id]
                hit_ids.append(hit_id)
                # print('{}\t{}'.format(*result))

        mux = pd.MultiIndex.from_arrays([query_ids, hit_ids], names=["query_id", "hit_id"])
        result_df = pd.DataFrame(index=mux)  # {"query_id": query_ids, "hit_id": hit_ids},
        return result_df

    def save(self, file=None):
        # save the index
        self.index.save()

    def load(self, file=None):
        # load the index
        self.index = ngtpy.Index(self.ann_index_name)

    def get_object(self, i):

        if self.index is None:
            self.load()

        return self.index.get_object(i)
"""


class DescentIndex(Index):
    """
    pynndescent index for a fingerprint

    :param dimension: the number of features in the fingerprint
    :param fingerprint_factory: class used to encapsulate the fingerprint
    """
    def __init__(self, index_name=None, dimension=2000, fingerprint_factory=mkfingerprints.SpectrumFloatFingerprint):
        super().__init__(index_name=index_name, dimension=dimension, fingerprint_factory=fingerprint_factory)

    def create(self, table_map, metric=None, column_name=None):
        if column_name is None:
            column_name = 'spectrum'
        if metric is None:
            metric = "cosine"
        feature_array = np.zeros((len(table_map), self.dimension), dtype=np.float32)
        factory = self.fingerprint_factory(dimension=self.dimension)

        for i in range(len(table_map)):
            feature_array[i] = factory.object2fingerprint(table_map[i][column_name])

        feature_array = sparse.csr_matrix(feature_array)

        self.index = pynndescent.NNDescent(feature_array, metric=metric, n_neighbors=120, n_jobs=None,
                                           low_memory=True, compressed=True)
        self.index.prepare()
        
    def create_from_fingerprint(self, table_map, fingerprint_column='ecfp4', fingerprint_count_column='ecfp4_count',
                           metric=None):
        """
        create the index from a table column containing a binary fingerprint or feature vector

        :param table_map: TableMap that contains the arrow table
        :param fingerprint_column: name of the fingerprint column, defaults to 'ecfp4'
        :param fingerprint_count_column: name of the fingerprint count column, defaults to 'ecfp4_count'
        """
        table = table_map.to_arrow()
        # read the fingerprint size from the column metadata
        self.dimension = int.from_bytes(table.field(fingerprint_column).metadata[b'fp_size'], byteorder='big')
        # is this a packed array of bits?
        is_packed = False

        # the datatype of the fingerprint
        dtype = table.schema.field(fingerprint_column).type.value_type.to_pandas_dtype()
        if dtype == np.float32:
            if metric is None:
                metric = 'cosine'
        elif dtype == np.uint8:
            if self.dimension == len(table.slice(0, 1)[fingerprint_column].to_numpy()[0]):
                if metric is None:
                    metric = 'cosine'
            else:
                is_packed = True
                if metric is None:
                    metric = 'jaccard'
        else:
            raise ValueError('unknown type for indexing')

        
        data = []
        row_ind = [] 
        col_ind = []

        for i in range(len(table_map)):
            intermediate = table.slice(i, 1)[fingerprint_column].to_numpy()[0]
            if is_packed:
                intermediate = np.unpackbits(intermediate)[0:self.dimension]
            nonzero = np.nonzero(intermediate)
            for j in nonzero[0]:
                data.append(intermediate[j])
                row_ind.append(i)
                col_ind.append(j)
                    
        feature_array = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(table_map), self.dimension), dtype=np.float32)
        del data
        del row_ind
        del col_ind
    
        self.index = pynndescent.NNDescent(feature_array, metric=metric, n_neighbors=120, n_jobs=None,
                                           low_memory=True, compressed=True)
        self.index.prepare()
 
    def optimize(self):
        pass

    def search(self, objects, hitlist_size=50, epsilon=0.3, with_raw_score=False, id_list=None, id_list_query=None, predicate=None):
        if self.index is None:
            self.load()

        if type(objects) != list:
            if type(objects) == np.ndarray:
                if objects.ndim == 1:
                    objects = [objects]
            else:
                objects = [objects]

        feature_array = np.zeros((len(objects), self.dimension), dtype=np.float32)
        query_ids = np.zeros((len(objects), hitlist_size), dtype=np.int64)
        factory = self.fingerprint_factory(dimension=self.dimension)

        for i in range(feature_array.shape[0]):
            if isinstance(objects[i], np.ndarray):
                feature_array[i] = np.unpackbits(objects[i])[0:self.dimension].astype(np.float32)
                query_ids[i, :] = i
            else:
                feature_array[i] = factory.object2fingerprint(objects[i])
                query_ids[i, :] = objects[i].id
        
        # adjust hitlist size upwards if there is a predicate
        original_hitlist_size = hitlist_size
        if predicate is not None:
            rows_to_keep = np.nonzero(predicate)[0]
            hitlist_size = 2 * hitlist_size * len(predicate) // len(rows_to_keep)
         
        indices, distances = self.index.query(feature_array, k=hitlist_size, epsilon=epsilon)

        # screen out hits not in predicate
        if predicate is not None:
            new_indices = np.full((indices.shape[0], original_hitlist_size), -1, dtype=np.int64)
            new_distances = np.full((indices.shape[0], original_hitlist_size), -1.0, dtype=np.float32)
            for i in range(indices.shape[0]):
                values, ind1, ind2 = np.intersect1d(indices[i], rows_to_keep, return_indices=True)
                length = min(len(ind1), original_hitlist_size)
                new_indices[i, :length] = indices[i][ind1][:length]
                new_distances[i, :length] = distances[i][ind1][:length]
            indices = new_indices
            distances = new_distances
        
        if id_list is not None:
            indices = id_list[indices]
        if id_list_query is not None:
            query_ids = id_list_query[query_ids]

        assert (query_ids.shape == indices.shape)

        mux = pd.MultiIndex.from_arrays([query_ids.flatten(), indices.flatten()], names=["query_id", "hit_id"])
        result_df = pd.DataFrame({"raw_score": distances.flatten()}, index=mux)
        return mkhitlist.Hitlist(result_df)

    def save(self, file=None):
        if file is None:
            filename = f"{self.index_name}.pynndescent"
        else:
            filename = file
        with open(filename, "wb") as file:
            pickle.dump(self.index, file, protocol=5)

    def load(self, file=None):
        if file is None:
            filename = f"{self.index_name}.pynndescent"
        else:
            filename = file
        with open(filename, "rb") as file:
            self.index = pickle.load(file)
        return self


class BruteForceIndex(Index):
    """
    search a library by brute force spectrum matching

    :param dimension: the number of features in the fingerprint (ignored)
    :param fingerprint_factory: class used to encapsulate the fingerprint
    """
    def __init__(self, index_name=None, dimension=None, fingerprint_factory=None):
        super().__init__(index_name=index_name, dimension=dimension, fingerprint_factory=fingerprint_factory)
        self.index = []

    def create(self, table_map, column_name=None):
        if column_name is None:
            column_name = 'spectrum'
        for i in range(len(table_map)):
            self.index.append(table_map[i][column_name])

    def optimize(self):
        pass

    def search(self, objects, hitlist_size=50, epsilon=0.3, with_raw_score=False, id_list=None, id_list_query=None, predicate=None):
        if len(np.shape(objects)) < 1:
            objects = [objects]

        if predicate is not None:
            assert len(predicate) == len(self.index)

        query_ids = []
        hit_ids = []
        cosine_scores = []
        matches = []
        specnum=0
        for spectrum in objects:
            start_time = timeit.default_timer()
            for i in range(len(self.index)):
                if predicate is not None:
                    if not predicate[i]:
                        continue
                query_id, hit_id, cosine_score, match = spectrum.single_match(self.index[i],
                                                                              cosine_score_scale=1.0,
                                                                              cosine_threshold=epsilon)
                if query_id is not None:
                    query_ids.append(query_id)
                    hit_ids.append(hit_id)
                    cosine_scores.append(cosine_score)
                    matches.append(match)
            elapsed = timeit.default_timer() - start_time
            logging.info(f"Finished spectrum {specnum}, elapsed time = {elapsed}")
            specnum += 1

        mux = pd.MultiIndex.from_arrays([query_ids, hit_ids], names=["query_id", "hit_id"])
        result_df = pd.DataFrame({"raw_score": matches, "cosine_score": cosine_scores}, index=mux)
        return mkhitlist.Hitlist(result_df)

    def save(self, file=None):
        if file is None:
            filename = f"{self.index_name}.brute"
        else:
            filename = file
        with open(filename, "wb") as file:
            pickle.dump(self.index, file, protocol=5)

    def load(self, file=None):
        if file is None:
            filename = f"{self.index_name}.brute"
        else:
            filename = file
        with open(filename, "rb") as file:
            self.index = pickle.load(file)
        return self


# @jit(parallel=True)
# can only be efficiently numba compiled if objects is a single numpy array
def dot_product(objects, column, hit_ids, cosine_scores, hitlist_size):
    for i in range(len(objects)):
        dist = np.sum(objects[i] * column, axis=-1)
        hit_ids[i] = np.argsort(dist)[::-1][:hitlist_size]
        cosine_scores[i] = dist[hit_ids[i]]


class DotProductIndex(Index):
    """
    brute force search of feature vectors using cosine score
    """
    def __init__(self, index_name=None, dimension=None, fingerprint_factory=None):
        super().__init__(index_name=index_name, dimension=dimension, fingerprint_factory=fingerprint_factory)
        
    def create(self, table_map):
        fingerprints = table_map.to_arrow()[self.index_name]
        self.index = np.zeros((len(fingerprints), len(fingerprints[0])))
        for i in range(len(fingerprints)):
            self.index[i] = fingerprints[i].values.to_numpy()
                     
    def optimize(self):
        pass

    def search(self, objects, hitlist_size=30, epsilon=0.1, id_list=None, id_list_query=None, predicate=None):
        """
        search feature vectors

        :param objects: feature vector or list of feature vectors to be queried
        :param hitlist_size: _description_, defaults to 50
        :param epsilon: _description_, defaults to 0.1
        :return: _description_
        """

        if type(objects) != list:
            if type(objects) == np.ndarray:
                if objects.ndim == 1:
                    if type(objects[0]) != np.ndarray:
                        objects = [objects]
            else:
                objects = [objects]

        query_ids = np.repeat(np.arange(len(objects), dtype=np.int64), hitlist_size).reshape((len(objects), hitlist_size))
        cosine_scores = np.zeros((len(objects), hitlist_size), dtype=np.float32)
        hit_ids = np.zeros((len(objects), hitlist_size), dtype=np.int64)

        dot_product(objects, self.index, hit_ids, cosine_scores, hitlist_size)

        if id_list is not None:
            hit_ids = id_list[hit_ids]

        if id_list_query is not None:
            query_ids = id_list_query[query_ids]

        mux = pd.MultiIndex.from_arrays([query_ids.flatten(), hit_ids.flatten()], names=["query_id", "hit_id"])
        result_df = pd.DataFrame({"hybrid_score": cosine_scores.flatten()}, index=mux)
        # for some reason, pandas insists on converting float32 to object, so convert it back
        result_df = result_df.astype({"hybrid_score": np.float32})
        return mkhitlist.Hitlist(result_df)

    def save(self, file=None):
        if file is None:
            filename = f'{self.index_name}.npy'
        else:
            filename = file
        np.save(filename, self.index)

    def load(self, file=None):
        if file is None:
            filename = f'{self.index_name}.npy'
        else:
            filename = file
        self.index = np.load(filename)
        return self


class TanimotoIndex(Index):
    """
    brute force search index of binary fingerprints

    :param dimension: the number of features in the fingerprint
    :param fingerprint_factory: class used to encapsulate the fingerprint
    """
    def __init__(self, index_name=None, dimension=4096, fingerprint_factory=mkfingerprints.SpectrumTanimotoFingerPrint):
        super().__init__(index_name=index_name, dimension=dimension, fingerprint_factory=fingerprint_factory)
        self.index_count = None

    def create(self, table_map, column_name=None):
        if column_name is None:
            column_name = 'spectrum'
        fingerprint = self.fingerprint_factory(dimension=self.dimension)
        self.index = np.zeros((len(table_map), fingerprint.size()), dtype=np.uint8)
        self.index_count = np.zeros((len(table_map),), dtype=np.int32)
        for i in range(len(table_map)):
            fingerprint.object2fingerprint(table_map[i][column_name])
            self.index[i] = fingerprint.to_numpy()
            self.index_count[i] = fingerprint.to_bitvec().GetNumOnBits()
            
    def create_from_fingerprint(self, table_map, fingerprint_column='ecfp4', fingerprint_count_column='ecfp4_count'):
        """
        create the index from columns in a table_map encoded as binary fingerprints

        :param table_map: TableMap that contains the arrow table
        :param fingerprint_column: name of the fingerprint column, defaults to 'ecfp4'
        :param fingerprint_count_column: name of the fingerprint count column, defaults to 'ecfp4_count'
        """
        table = table_map.to_arrow()
        # read the fingerprint size from the column metadata
        self.dimension = int.from_bytes(table.field(fingerprint_column).metadata[b'fp_size'], byteorder='big')
        # unfortunately, arrow converts the fingerprint column to a numpy array of objects, each of which is a 1d uint8 array
        # call vstack to pack them into a 2D numpy array
        self.index = np.vstack(table[fingerprint_column].to_numpy())
        self.index_count = table[fingerprint_count_column].to_numpy()

    def optimize(self):
        pass

    def search(self, objects, hitlist_size=50, epsilon=0.1, with_raw_score=False, id_list=None, id_list_query=None, predicate=None):
        if self.index is None:
            self.load()

        if type(objects) != list:
            if type(objects) == np.ndarray:
                if objects.ndim == 1 and type(objects[0]) != np.ndarray:
                    objects = [objects]
            else:
                objects = [objects]

        query_ids = np.zeros((len(objects), hitlist_size), dtype=np.int64)
        tanimotos = np.zeros((len(objects), hitlist_size), dtype=np.float32)
        hit_ids = np.zeros((len(objects), hitlist_size), dtype=np.int64)
        fingerprint = self.fingerprint_factory(dimension=self.dimension)

        start_time = timeit.default_timer()
        for i in range(len(objects)):
            if isinstance(objects[i],np.ndarray):
                fingerprint_numpy = objects[i]
                query_ids[i, :] = i
                spectrum_fp_count = np.unpackbits(fingerprint_numpy).sum()
            else:
                fingerprint.object2fingerprint(objects[i])
                fingerprint_numpy = fingerprint.to_numpy()
                query_ids[i, :] = objects[i].id
                spectrum_fp_count = fingerprint.to_bitvec().GetNumOnBits()
            # hits_out, tanimotos_out = tanimoto_search(fingerprint_numpy, self.index, spectrum_fp_count,
            #                                           self.index_count, epsilon, predicate=predicate)
            raise NotImplementedError('tanimoto_search no longer supported')
            if hits_out is None:
                logging.warning(f'No hits from search for query {query_ids[i, 0]}')
            else:
                this_hitlist_size = min(hitlist_size, len(hits_out))
                hit_ids[i, 0:this_hitlist_size] = hits_out[0:this_hitlist_size]
                tanimotos[i, 0:this_hitlist_size] = tanimotos_out[0: this_hitlist_size]

        elapsed = timeit.default_timer() - start_time
        # logging.info(f"search time per query = {elapsed / len(objects)}")

        if id_list is not None:
            hit_ids = id_list[hit_ids]

        if id_list_query is not None:
            query_ids = id_list_query[query_ids]

        mux = pd.MultiIndex.from_arrays([query_ids.flatten(), hit_ids.flatten()], names=["query_id", "hit_id"])
        result_df = pd.DataFrame({"tanimoto": tanimotos.flatten()}, index=mux)
        # for some reason, pandas insists on converting float32 to object, so convert it back
        result_df = result_df.astype({"tanimoto": np.float32})
        return mkhitlist.Hitlist(result_df)

    def save(self, file=None):
        if file is None:
            filename = f'{self.index_name}.npy'
            filename_count = f'{self.index_name}.count.npy'
        else:
            filename = file
            filename_count = f'{file}.count'
        np.save(filename, self.index)
        np.save(filename_count, self.index_count)

    def load(self, file=None):
        if file is None:
            filename = f'{self.index_name}.npy'
            filename_count = f'{self.index_name}.count.npy'
        else:
            filename = file
            filename_count = f'{file}.count.npy'
        self.index = np.load(filename)
        self.index_count = np.load(filename_count)
        return self

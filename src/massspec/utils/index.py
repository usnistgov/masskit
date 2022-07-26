import logging
import timeit
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy import sparse
from massspec.spectrum.spectrum import init_spectrum
from massspec.data_specs.schemas import min_spectrum_fields, molecule_experimental_fields, peptide_fields, \
    base_experimental_fields, molecule_annotation_fields
import massspec.utils.files as msuf
from massspec.utils.general import open_if_filename
from massspec.utils.search import tanimoto_search
from massspec.utils.tables import row_view, arrow_to_pandas
import pynndescent
import pickle
from massspec.utils.fingerprints import SpectrumFloatFingerprint, SpectrumTanimotoFingerPrint
from massspec.utils.hitlist import Hitlist

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
    def create(self, table_map):
        """
        create index from a TableMap

        :param table_map: the library to index
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
    def __init__(self, index_name=None, dimension=2000, fingerprint_factory=SpectrumFloatFingerprint):
        super().__init__(index_name=index_name, dimension=dimension, fingerprint_factory=fingerprint_factory)

    def create(self, table_map, metric=None):
        if metric is None:
            metric = "cosine"
        feature_array = np.zeros((len(table_map), self.dimension), dtype=np.float32)
        factory = self.fingerprint_factory(dimension=self.dimension)

        for i in range(len(table_map)):
            feature_array[i] = factory.object2fingerprint(table_map[i])

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
        return Hitlist(result_df)

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

    def create(self, table_map):
        for i in range(len(table_map)):
            self.index.append(table_map[i])

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
        return Hitlist(result_df)

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
        return Hitlist(result_df)

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
    def __init__(self, index_name=None, dimension=4096, fingerprint_factory=SpectrumTanimotoFingerPrint):
        super().__init__(index_name=index_name, dimension=dimension, fingerprint_factory=fingerprint_factory)
        self.index_count = None

    def create(self, table_map):
        fingerprint = self.fingerprint_factory(dimension=self.dimension)
        self.index = np.zeros((len(table_map), fingerprint.size()), dtype=np.uint8)
        self.index_count = np.zeros((len(table_map),), dtype=np.int32)
        for i in range(len(table_map)):
            fingerprint.object2fingerprint(table_map[i])
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
            hits_out, tanimotos_out = tanimoto_search(fingerprint_numpy, self.index, spectrum_fp_count,
                                                      self.index_count, epsilon, predicate=predicate)
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
        return Hitlist(result_df)

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


class TableMap(ABC):
    """
    collections.abc.Sequence wrapper for a library.  Allows use of different stores, e.g. arrow or pandas
    """
    def __init__(self, column_name=None, *args, **kwargs):
        """
        :param column_name: the column name for the spectrum. "spectrum" is default
        """
        super().__init__(*args, **kwargs)
        if column_name is None:
            self.column_name = "spectrum"
        else:
            self.column_name = column_name
        # fields to report
        self.field_list = molecule_experimental_fields + peptide_fields + base_experimental_fields + molecule_annotation_fields

    def __getitem__(self, key):
        """
        get spectrum from the library by row number

        :param key: row number
        :return: spectrum at row number
        """
        return self.getspectrum_by_row(key)

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def create_dict(self, idx):
        """
        create dict for row

        :param idx: row number
        """
        raise NotImplementedError

    @abstractmethod
    def get_ids(self):
        """
        get the ids of all records, in row order

        :return: array of ids
        """
        raise NotImplementedError

    @abstractmethod
    def getrow_by_id(self, key):
        """
        given an id, return the corresponding row number in the table

        :param key: the id
        :return: the row number of the id in the table
        """
        return NotImplementedError

    @abstractmethod
    def getitem_by_id(self, key):
        """
        get an item from the library by id

        :param key: id
        :return: the row dict
        """
        raise NotImplementedError

    @abstractmethod
    def getitem_by_row(self, key):
        """
        get an item from the library by id
        :param key: row number
        :return: the row dict
        """
        raise NotImplementedError

    @abstractmethod
    def getspectrum_by_id(self, key):
        """
        get a particular spectrum from the library by id

        :param key: id
        :return: the row
        """
        raise NotImplementedError

    @abstractmethod
    def getspectrum_by_row(self, key):
        """
        get a particular spectrum from the library by row number

        :param key: row number
        :return: the the spectrum
        """
        raise NotImplementedError
    
    def to_msp(self, file):
        """
        write out spectra in msp format

        :param file: file or filename to write to
        """
        with open_if_filename(file, 'w+') as fp:
            for i in range(len(self)):
                spectrum = self.getspectrum_by_row(i)
                output = spectrum.to_msp()
                print(output, file=fp)


class ArrowLibraryMap(TableMap):
    """
    wrapper for an arrow library

    """

    def __init__(self, table_in, column_name=None, num=0, *args, **kwargs):
        """
        :param table_in: parquet table
        :param num: number of rows to use
        """
        super().__init__(column_name=column_name, *args, **kwargs)
        self.table = table_in
        if num:
            self.table = self.table.slice(0, num)
        self.row = row_view(self.table)
        self.length = len(self.table['id'])
        self.ids = self.table['id'].combine_chunks().to_numpy()
        self.sort_indices = np.argsort(self.ids)
        self.sorted_ids = self.ids[self.sort_indices]

    def __len__(self):
        return self.length

    def get_ids(self):
        return self.ids

    def getrow_by_id(self, key):
        pos = np.searchsorted(self.sorted_ids, key)
        if pos > self.length or key != self.sorted_ids[pos]:
            raise IndexError(f'unable to find key {key}')
        return self.sort_indices[pos]

    def create_dict(self, idx):
        self.row.idx = idx
        return_val = {}
        # put interesting fields in the dictionary
        for field in self.field_list:
            attribute = self.row.get(field.name)
            if attribute is not None:
                return_val[field.name] = attribute()

        # create the spectrum
        return_val[self.column_name] = init_spectrum().from_arrow(self.row)
        return return_val

    def getitem_by_id(self, key):
        pos = self.getrow_by_id(key)
        return self.create_dict(pos)

    def getitem_by_row(self, key):
        # need to expand to deal with slice object
        if isinstance(key, slice):
            print(key.start, key.stop, key.step)
            raise NotImplementedError
        else:
            assert (0 <= key < len(self))
            return self.create_dict(key)

    def getspectrum_by_id(self, key):
        return self.getitem_by_id(key)[self.column_name]

    def getspectrum_by_row(self, key):
        assert (0 <= key < len(self))
        self.row.idx = key
        return init_spectrum().from_arrow(self.row)

    def to_arrow(self):
        return self.table
    
    def to_pandas(self):
        return arrow_to_pandas(self.table)

    def to_parquet(self, file):
        """
        save spectra to parquet file

        :param file: filename or stream
        """
        msuf.write_parquet(file, self.table)
        
    def to_mzxml(self, file, use_id_as_scan=True):
        """
        save spectra to mzxml format file

        :param file: filename or stream
        :param use_id_as_scan: use spectrum.id instead of spectrum.scan
        """
        msuf.spectra_to_mzxml(file, self, use_id_as_scan=use_id_as_scan)
            
    def to_mgf(self, file):
        """
        save spectra to mgf file

        :param file: filename or file pointer
        """
        msuf.spectra_to_mgf(file, self)

    @staticmethod
    def from_parquet(file, columns=None, num=None, combine_chunks=False, filters=None):
        """
        create an ArrowLibraryMap from a parquet file

        :param file: filename or stream
        :param columns: list of columns to read.  None=all, []=minimum set
        :param num: number of rows
        :param combine_chunks: dechunkify the arrow table to allow zero copy
        :param filters: parquet predicate as a list of tuples
        """
        if columns is not None:
            columns = list(set(columns + min_spectrum_fields))
        input_table = msuf.read_parquet(file, columns=columns, num=num, filters=filters)
        if len(input_table) == 0:
            raise IOError(f'Parquet file {file} read in with zero rows when using filters {filters}')
        if combine_chunks:
            input_table = input_table.combine_chunks()
        return ArrowLibraryMap(input_table)

    @staticmethod
    def from_msp(file, num=None, comment_fields=None, min_intensity=0.0, max_mz=2000):
        """
        read in an msp file and create an ArrowLibraryMap

        :param file: filename or stream
        :param num: number of rows.  None means all
        :param comment_fields: a Dict of regexes used to extract fields from the Comment field.  Form of the Dict is
        { comment_field_name: (regex, type, field_name)}.  For example {'Filter':(r'@hcd(\d+\.?\d* )', float, 'nce')}
        :param min_intensity: the minimum intensity to set the fingerprint bit
        :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
        :return: ArrowLibraryMap
        """
        return ArrowLibraryMap(msuf.load_msp2array(file, num=num, comment_fields=comment_fields,
                                              min_intensity=min_intensity, max_mz=max_mz))

    @staticmethod
    def from_mgf(file, num=None, title_fields=None, min_intensity=0.0, max_mz=2000):
        """
        read in an mgf file and create an ArrowLibraryMap

        :param file: filename or stream
        :param num: number of rows.  None means all
        :param title_fields: dict containing column names with corresponding regex to extract field values from the TITLE
        :param min_intensity: the minimum intensity to set the fingerprint bit
        :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
        :return: ArrowLibraryMap
        """
        return ArrowLibraryMap(msuf.load_mgf2array(file, num=num, title_fields=title_fields,
                                                   min_intensity=min_intensity, max_mz=max_mz))


    @staticmethod
    def from_sdf(file,
                 num=None,
                 skip_expensive=True,
                 max_size=0,
                 source=None,
                 id_field=None,
                 min_intensity=0.0,
                 max_mz=2000,
                 set_probabilities=(0.01, 0.97, 0.01, 0.01),
        ):
        """
        read in an sdf file and create an ArrowLibraryMap

        :param file: filename or stream
        :param num: number of rows.  None means all
        :param skip_expensive: don't compute fields that are computationally expensive
        :param max_size: the maximum bounding box size (used to filter out large molecules. 0=no bound)
        :param source: where did the sdf come from?  pubchem, nist, ?
        :param id_field: field to use for the mol id, such as NISTNO, ID or _NAME (the sdf title field). if an integer,
          use the integer as the starting value for an assigned id
        :param min_intensity: the minimum intensity to set the fingerprint bit
        :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
        :param set_probabilities: how to divide into dev, train, valid, test
        :return: ArrowLibraryMap
        """
        return ArrowLibraryMap(msuf.load_sdf2array(file, num=num, skip_expensive=skip_expensive, max_size=max_size,
                                              source=source, id_field=id_field, min_intensity=min_intensity,
                                              max_mz=max_mz, set_probabilities=set_probabilities))


class PandasLibraryMap(TableMap):
    """
    wrapper for a pandas spectral library

    """

    def __init__(self, df, column_name=None, *args, **kwargs):
        """
        :param df: pandas dataframe
        :param column_name: name of the spectrum column
        """
        super().__init__(column_name=column_name, *args, **kwargs)
        self.df = df
        self.ids = self.df.index.values
        self.length = len(self.ids)
        self.sort_indices = np.argsort(self.ids)
        self.sorted_ids = self.ids[self.sort_indices]

    def __len__(self):
        return self.length

    def create_dict(self, idx):
        return_val = self.df.iloc[[idx]].to_dict(orient='records')[0]
        return_val[self.df.index.name] = self.df.index.values[idx]
        return return_val

    def get_ids(self):
        return self.ids

    def getrow_by_id(self, key):
        pos = np.searchsorted(self.sorted_ids, key)
        if pos > self.length or key != self.sorted_ids[pos]:
            raise IndexError(f'unable to find key {key}')
        return self.sort_indices[pos]

    def getitem_by_id(self, key):
        pos = self.getrow_by_id(key)
        return self.create_dict(pos)

    def getitem_by_row(self, key):
        return self.create_dict(key)

    def getspectrum_by_id(self, key):
        return self.getitem_by_id(key)[self.column_name]

    def getspectrum_by_row(self, key):
        return self.df[self.column_name].values[key]


class ListLibraryMap(TableMap):
    """
    wrapper for a spectral library using python lists

    """

    def __init__(self, list_in,  *args, **kwargs):
        """
        :param list_in: list of spectra
        """
        super().__init__(*args, **kwargs)
        self.list_in = list_in
        self.length = len(self.list_in)

    def __len__(self):
        return self.length

    def create_dict(self, idx):
        spectrum = self.list_in[idx]
        return_val = {}
        # put interesting fields in the dictionary
        for field in self.field_list:
            try:
                attribute = getattr(spectrum, field.name)
                if attribute is not None:
                    return_val[field.name] = attribute
            except AttributeError:
                pass

        return_val[self.column_name] = spectrum
        return return_val

    def get_ids(self):
        return [x for x in range(self.length)]

    def getrow_by_id(self, key):
        return key

    def getitem_by_id(self, key):
        return self.create_dict(key)

    def getitem_by_row(self, key):
        return self.create_dict(key)

    def getspectrum_by_id(self, key):
        return self.list_in[key]

    def getspectrum_by_row(self, key):
        return self.list_in[key]


"""
implementation notes for learn to rank:

# need to pass in hitlist to pytorch training
# this would be two float32 tensors of size [batch, hit_list_size].  allow setting of pad value
# one tensor are the true values
# another tensor has the predicted values
# both sorted by predicted values
# this implies that the comparehitlist can return a join of the two hit lists

# how to calculate roc:
# do multiple searches with different thresholds
# do a hitlist compare, where one hitlist is the true hitlist
# in comparehitlist, calculate true_positive_rate, false_positive_rate

# look at pytorch libraries and maybe tensorflow learn to rank
# allRank probably better place to start.  approxNDCG is best from this set per 
# https://ris.utwente.nl/ws/portalfiles/portal/6420086/ipm2015-preprint.pdf
# best overall is https://www.sciencedirect.com/science/article/pii/S0167923612002011?via%3Dihub#bb0015
# which compares to losses from 
# https://www.sciencedirect.com/science/article/pii/S0950705110001772/pdfft?md5=ee0d5d6212c4e02df48ae90c48a0f8c0&pid=1-s2.0-S0950705110001772-main.pdf
# losses are of dim [batch, slate_length] where slate_length is the hitlist.
"""

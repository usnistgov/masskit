# distutils: extra_compile_args=-fopenmp -march=native -mavx
# distutils: extra_link_args=-fopenmp -march=native
# cython: language_level=3

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange, parallel, threadid
cimport openmp
from libc.stdint cimport uint64_t, uint8_t, int64_t, int32_t
from libc.stdlib cimport calloc, free, malloc
cdef extern from *:
    """
    #ifdef _MSC_VER
      #include <intrin.h>
      #define my_popcount(m) __popcnt64(m)
    #else
       #define my_popcount(m) __builtin_popcountll(m) 
    #endif
    """
    int my_popcount(unsigned long long) nogil

"""
to build: python setup.py build_ext --inplace

notes:
- this code requires that the input be uint8 arrays whose size is a multiple of 8
- overhead to call tanimoto_64 is 0.22 seconds!
"""

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef tuple tanimoto_64(uint64_t[::1] query, const uint64_t[:,::1] fingerprint_array, int64_t query_fingerprint_count,
                        const int32_t[::1] fingerprint_array_count, double tanimoto_cutoff, const uint8_t[::1] predicate):
    cdef Py_ssize_t j
    cdef int sum_and
    cdef int sum_or
    cdef int integer_cutoff
    cdef float calculated_tanimoto
    cdef Py_ssize_t k
    cdef Py_ssize_t num_hits
    cdef int max_threads
    cdef int* num_per_thread
    cdef float * intermediate_tanimotos
    cdef int[::1] hit_ids
    cdef float[::1] tanimotos_out
    cdef int * intermediate_hit_ids
    cdef float * tanimotos

    max_threads = openmp.omp_get_max_threads()
    num_per_thread = <int *> calloc(sizeof(int), max_threads)
    if not num_per_thread:
        raise MemoryError
    intermediate_tanimotos = <float *> calloc(sizeof(float), fingerprint_array.shape[0])
    if not intermediate_tanimotos:
        raise MemoryError

    # query_fingerprint_count is a minimum value of sum_or, so multiplied by tanimoto, that has to be the
    # minimum value of sum_and allowable
    integer_cutoff = <int> (query_fingerprint_count * tanimoto_cutoff)
    num_hits = 0

    with nogil, parallel():
        for j in prange(fingerprint_array.shape[0]):
            if predicate is not None and predicate[j] == 0:
                continue
            sum_and = 0
            for k in range(query.shape[0]):
                # explicitly breaking this operation into and followed by popcount does not improve speed
                # either because vectorization didn't happen or it didn't improve vectorization
                sum_and = sum_and + my_popcount(query[k] & fingerprint_array[j, k])
            if sum_and >= integer_cutoff and sum_and >= fingerprint_array_count[j] * tanimoto_cutoff:
                sum_or = 0
                for k in range(query.shape[0]):
                    sum_or = sum_or + my_popcount(query[k] | fingerprint_array[j, k])
                # inserting a filter here for sum_and >= tanimoto * sum_or to avoid division doesn't speed things up
                calculated_tanimoto = (<float>sum_and) / sum_or
                if calculated_tanimoto >= tanimoto_cutoff:
                    intermediate_tanimotos[j] = calculated_tanimoto
                    num_per_thread[threadid()] += 1

    for i in range(max_threads):
        num_hits += num_per_thread[i]
    free(<void *>num_per_thread)
    if num_hits == 0:
        free(<void *>intermediate_tanimotos)
        return None, None
    intermediate_hit_ids = <int *> malloc(sizeof(int) * num_hits)
    if not intermediate_hit_ids:
        raise MemoryError
    hit_ids = <int [:num_hits]> intermediate_hit_ids
    tanimotos = <float *> malloc(sizeof(float) * num_hits)
    if not tanimotos:
        raise MemoryError
    tanimotos_out = <float [:num_hits]> tanimotos

    k = 0
    for j in range(fingerprint_array.shape[0]):
        if intermediate_tanimotos[j] >= tanimoto_cutoff:
            hit_ids[k] = j
            tanimotos_out[k] = intermediate_tanimotos[j]
            k += 1
    free(<void *>intermediate_tanimotos)
    return hit_ids, tanimotos_out

def tanimoto_search(query, fingerprint_array, query_fingerprint_count, fingerprint_array_count, tanimoto_cutoff,
                    predicate=None):
    if query.dtype == np.uint8 and fingerprint_array.dtype == np.uint8 and query.nbytes % 8 == 0 \
            and fingerprint_array.nbytes % 8 == 0:
        query = query.astype(np.uint64)
        fingerprint_array = fingerprint_array.astype(np.uint64)
        target, tanimotos = tanimoto_64(query, fingerprint_array, query_fingerprint_count, fingerprint_array_count,
                                        tanimoto_cutoff, predicate)
        if target is None or tanimotos is None:
            return None, None
        target = np.asarray(target)
        tanimotos = np.asarray(tanimotos)
        indices = np.argsort(tanimotos)
        target = target[indices[::-1]]
        tanimotos = tanimotos[indices[::-1]]
        return np.asarray(target), np.asarray(tanimotos)
    else:
        raise ValueError("input arrays must be of type uint8 and multiple of 8 bytes")

"""
# sample search
import pyarrow.parquet as pq
from search import tanimoto_search
import numpy as np

table = pq.read_table('hr_msms_nist2020_v42_0.parquet')
fingerprints_list = table['spectrum_fp'].combine_chunks()
fingerprints = fingerprints_list.values.to_numpy()
fingerprints = np.reshape(fingerprints, (-1, fingerprints_list.offsets[1].as_py()))
predicate[:fingerprints.shape[0]//2] = False
tanimoto_search(fingerprints[0], fingerprints, table['spectrum_fp_count'][0].as_py(), table['spectrum_fp_count'].to_numpy().astype(np.uint64), 0.1, predicate=predicate)
"""
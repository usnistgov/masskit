import logging
import random

import numpy as np
import pytest
from pytest import approx

import masskit.spectra
import masskit.spectra.ions as mkions
from masskit.utils.fingerprints import (ECFPFingerprint,
                                        SpectrumFloatFingerprint)
from masskit.utils.hitlist import (CompareRecallDCG, CosineScore, Hitlist,
                                   TanimotoScore)
from masskit.utils.index import *
from masskit.utils.tablemap import ArrowLibraryMap


@pytest.fixture
def hi_res3():
    spectrum = masskit.spectra.Spectrum()
    spectrum.from_arrays(
        np.array([100.0, 100.1, 300.5]),
        np.array([100, 200, 999]),
        row={
            "id": 1234,
            "retention_time": 4.5,
            "name": "hello",
            "precursor_mz": 500.5,
        },
        precursor_mz=300.0,
        product_mass_info=mkions.MassInfo(10.0, "ppm", "monoisotopic", "", 1),
        copy_arrays=False
    )
    return spectrum

class Helpers:
    @staticmethod
    def do_queries(index, table_in, num_queries=None, queries=None, hitlist_size=120, column_name=None):
        if column_name is None:
            column_name = 'spectrum'
        if queries is None:
            if num_queries is not None:
                queries = random.sample(range(len(table_in)), num_queries)
            else:
                queries = range(len(table_in))
        else:
            # convert id list into row ids
            queries = [table_in.getrow_by_id(x) for x in queries]

        objects = []
        for i in queries:
            objects.append(table_in[i][column_name])

        hitlist = index.search(objects, id_list=table_in.get_ids(), hitlist_size=hitlist_size, epsilon=0.1)

        if "cosine_score" not in hitlist.to_pandas().columns:
            CosineScore(table_in).score(hitlist)
        print(f"match count = "
                f"{(hitlist.to_pandas().index.get_level_values(0) == hitlist.to_pandas().index.get_level_values(1)).sum()}")
        return hitlist
    
@pytest.fixture
def helpers():
    return Helpers

def test_to_msp(cho_uniq_short_table, tmpdir: str):
    lib = ArrowLibraryMap(cho_uniq_short_table, num=3)
    msp_file = (tmpdir / 'test.msp')
    lib.to_msp(msp_file.open("w+"))
    assert msp_file.read().startswith("Name: AAAACALTPGPLADLAAR/2_1(4,C,CAM)")

def test_to_mgf(cho_uniq_short_table, tmpdir: str):
    lib = ArrowLibraryMap(cho_uniq_short_table, num=3)
    msp_file = (tmpdir / 'test.mgf')
    lib.to_mgf(msp_file.open("w+"))

def test_nnd_index(cho_uniq_short_table, tmpdir: str):
    index = DescentIndex(tmpdir / 'tandem_newfp', dimension=2000)
    lib = ArrowLibraryMap(cho_uniq_short_table, num=100)
    index.create(lib)
    logging.info('saving index')
    index.save()

def test_nnd_query(cho_uniq_short_table, helpers: type[Helpers], tmpdir: str):
    index = DescentIndex(tmpdir / '../test_nnd_indexcurrent/tandem_newfp')
    index.load()
    lib = ArrowLibraryMap(cho_uniq_short_table, num=100)
    helpers.do_queries(index, lib, num_queries=3)

def test_recall():
    mux = pd.MultiIndex.from_arrays([[100, 100, 100, 100, 200, 200, 200, 200, 200],
                                        [10, 20, 100, 200, 30, 100, 200, 20, 70]], names=["query_id", "hit_id"])
    compare_df = pd.DataFrame({"cosine_score": [0.300, 0.800, 0.999, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400]},
                                index=mux)
    mux2 = pd.MultiIndex.from_arrays([[100, 100, 100, 100, 100, 200, 200, 200, 200, 200],
                                        [10, 20, 100, 50, 600, 200, 30, 100, 200, 20]],
                                        names=["query_id", "hit_id"])
    truth_df = pd.DataFrame({"cosine_score": [0.300, 0.998, 0.800, 0.999, 0.999, 0.400, 0.5500, 0.999, 0.4400,
                                                0.700]}, index=mux2)
    comparison = CompareRecallDCG()(Hitlist(compare_df), Hitlist(truth_df))

def test_tanimoto_index(cho_uniq_short_table, tmpdir: str):
    index = TanimotoIndex(tmpdir / 'spectrum_fp_10000')
    lib = ArrowLibraryMap(cho_uniq_short_table, num=100)
    index.create(lib)
    index.save()

@pytest.mark.skip(reason="tanimoto search currently unimplemented")
def test_tanimoto_query(cho_uniq_short_table, helpers: type[Helpers], tmpdir: str):
    index = TanimotoIndex(tmpdir / '../test_tanimoto_indexcurrent/spectrum_fp_10000')
    lib = ArrowLibraryMap(cho_uniq_short_table, num=100)
    index.load()
    helpers.do_queries(index, lib, num_queries=3, hitlist_size=100)

@pytest.mark.skip(reason="tanimoto search currently unimplemented")
def test_tanimoto_from_fingerprint(SRM1950_lumos_table, tmpdir: str):
    index = TanimotoIndex(tmpdir / '../test_tanimoto_indexcurrent/tanimoto_from_column',
                          fingerprint_factory=ECFPFingerprint)
    lib = ArrowLibraryMap(SRM1950_lumos_table, num=10)
    index.create_from_fingerprint(lib)
    hitlist = index.search(lib[0]['ecfp4'], id_list=lib.get_ids(),
                           id_list_query=lib.get_ids(), hitlist_size=10, epsilon=0.01)
    assert hitlist.hitlist.index[0][0] == hitlist.hitlist.index[0][1]
    assert hitlist.hitlist.iloc[0]['tanimoto'] == 1.0

def test_nnd_from_fingerprint(SRM1950_lumos_table):
    index = DescentIndex(fingerprint_factory=ECFPFingerprint)
    lib = ArrowLibraryMap(SRM1950_lumos_table)
    index.create_from_fingerprint(lib)
    hitlist = index.search(lib[0]['ecfp4'], id_list=lib.get_ids(),
                           id_list_query=lib.get_ids(), hitlist_size=30, epsilon=0.1)
    TanimotoScore(lib, query_table_map=lib).score(hitlist)
    hitlist.sort('tanimoto')
    assert lib.getitem_by_id(hitlist.hitlist.index[0][0])['spectrum'].name == lib.getitem_by_id(hitlist.hitlist.index[0][1])['spectrum'].name
    assert hitlist.hitlist.iloc[0]['tanimoto'] == 1.0
    
    predicate = np.zeros(len(lib), dtype=np.uint8)
    predicate[::2] = 1
    hitlist = index.search(lib[0]['ecfp4'], id_list=lib.get_ids(),
                           id_list_query=lib.get_ids(), hitlist_size=30, epsilon=0.1, predicate=predicate)
    TanimotoScore(lib, query_table_map=lib).score(hitlist)
    hitlist.sort('tanimoto')
    pass


def test_a_fingerprint(hi_res3: masskit.spectra.Spectrum):
    fp = SpectrumFloatFingerprint()
    fp.object2fingerprint(hi_res3)

@pytest.mark.skip(reason="tanimoto search currently unimplemented")
def test_compare_tani_fast_search(cho_uniq_short_table, helpers: type[Helpers], tmpdir: str):
    tani = TanimotoIndex(tmpdir / '../test_tanimoto_indexcurrent/spectrum_fp_10000')
    tani.load()    
    lib = ArrowLibraryMap(cho_uniq_short_table, num=100)
    tani_hitlist = helpers.do_queries(tani, lib, num_queries=10, hitlist_size=300)
    descent = DescentIndex(tmpdir / '../test_nnd_indexcurrent/tandem_newfp')
    descent.load()
    descent_hitlist = helpers.do_queries(descent, lib, queries=tani_hitlist.get_query_ids(), hitlist_size=300)
    comparison = CompareRecallDCG()(descent_hitlist, tani_hitlist)

def test_brute_force_index(cho_uniq_short_table, tmpdir: str):
    index = BruteForceIndex(tmpdir / "brute_force")
    lib = ArrowLibraryMap(cho_uniq_short_table, num=100)
    index.create(lib)
    index.save()
    
def test_brute_force_query(cho_uniq_short_table, helpers: type[Helpers], tmpdir: str):
    index = BruteForceIndex(tmpdir / '../test_brute_force_indexcurrent/brute_force')
    index.load()
    lib = ArrowLibraryMap(cho_uniq_short_table, num=100)
    hitlist = helpers.do_queries(index, lib, num_queries=10)

def test_compare_brute_fast_search(cho_uniq_short_table, helpers: type[Helpers], tmpdir: str):
    brute = BruteForceIndex(tmpdir / '../test_brute_force_indexcurrent/brute_force').load()
    lib = ArrowLibraryMap(cho_uniq_short_table, num=100)
    brute_hitlist = helpers.do_queries(brute, lib, num_queries=2)
    descent = DescentIndex(tmpdir / '../test_nnd_indexcurrent/tandem_newfp').load()
    descent_hitlist = helpers.do_queries(descent, lib, queries=brute_hitlist.get_query_ids(), hitlist_size=300)
    comparison = CompareRecallDCG()(descent_hitlist, brute_hitlist)

def test_table(cho_uniq_short_table):
    item = ArrowLibraryMap(cho_uniq_short_table)[0]

# @pytest.fixture
# def table_ei():
#     return pq.read_table('/home/lyg/data/nist/ei/2020/mainlib_2020_hybrid_1.parquet')

@pytest.mark.skip(reason='need example table')
def test_dotproduct_index(table_ei, tmpdir: str):
    index = DotProductIndex('hybrid_fp')
    lib = ArrowLibraryMap(table_ei)
    index.create(lib)
    index.save(str(tmpdir / '../hybrid_fp.npy'))

@pytest.mark.skip(reason='need example table')
def test_dotproduct_query(table_ei, helpers: type[Helpers], tmpdir: str):
    index = DotProductIndex('hybrid_fp')
    # index.load(str(tmpdir / '../hybrid_fp.npy'))
    index.load('/tmp/hybrid_fp.npy')
    lib = ArrowLibraryMap(table_ei)
    hitlist = index.search([table_ei['hybrid_fp'][1000].values.to_numpy(), table_ei['hybrid_fp'][1001].values.to_numpy()], hitlist_size=1000)
    hitlist

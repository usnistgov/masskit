import pytest
from pytest import approx
import numpy as np
import pyarrow.parquet as pq
from pyarrow import plasma
from massspec_ml.pytorch.spectrum.small_mol.small_mol_datasets import TandemArrowSearchDataset
from massspec_ml.pytorch.spectrum.spectrum_datasets import TandemArrowDataset
import builtins
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def start_plasma():
    with plasma.start_plasma_store(1000000000) as ps:
        builtins.instance_settings = {'plasma': {'socket': ps[0], 'pid': ps[1].pid}}
        yield ps

@pytest.fixture()
def config():
    # return_val = {}
    # return_val['input'] = {'dev': {'where': ''}}
    GlobalHydra.instance().clear()
    initialize(config_path="../../../../../apps/ml/peptide/conf", job_name="test_app")
    cfg = compose(config_name="config_search", overrides=["input=2022_tandem_search_test"])
    return cfg

@pytest.mark.skip(reason="needs data/nist/tandem/SRM1950/SRM1950_lumos.ecfp4.pynndescent")
def test_TandemArrowSearchDataset(config, start_plasma):
    ds = TandemArrowSearchDataset('libraries/tests/data/SRM1950_lumos.parquet', config, 'test',
                                  store_search='libraries/tests/data/SRM1950_lumos.parquet')
    row_with_hits = ds.get_data_row(0)
    assert row_with_hits['hit_spectrum'][0].name == 'N-Acetyl-L-alanine'
    assert ds.get_x(ds.get_data_row(0)).shape == (30, 2, 20000)
    pass

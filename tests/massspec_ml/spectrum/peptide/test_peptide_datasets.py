import pytest
from pytest import approx
import numpy as np
import pyarrow.parquet as pq
from pyarrow import plasma
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
    cfg = compose(config_name="config") # overrides=["db=mysql", "db.user=me"])
    return cfg

def test_TandemArrowDataset(config, start_plasma):
    ds = TandemArrowDataset('libraries/tests/data/cho_uniq_short.parquet', config, 'train')
    pass

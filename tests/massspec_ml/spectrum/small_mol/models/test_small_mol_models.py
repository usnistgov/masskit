import pytest
from pytest import approx
import numpy as np
import pyarrow.parquet as pq
from pyarrow import plasma
from massspec_ml.pytorch.spectrum.small_mol.small_mol_datasets import TandemArrowSearchDataset
from massspec_ml.pytorch.spectrum.small_mol.models.small_mol_models import *
import torch
from massspec_ml.pytorch.base_objects import ModelInput
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
    initialize(config_path="../../../../../../apps/ml/peptide/conf", job_name="test_app")
    cfg = compose(config_name="config_search", overrides=["input=2022_tandem_search_test", "ml/model=ResNetBaseline"])
    return cfg

@pytest.fixture()
def ds(config, start_plasma):
    ds = TandemArrowSearchDataset('libraries/tests/data/SRM1950_lumos.parquet', config, 'test',
                                store_search='libraries/tests/data/SRM1950_lumos.parquet')
    return ds

@pytest.mark.skip(reason="uses gpu")
def test_SimpleModel(config, ds):
    y = torch.unsqueeze(ds.get_y(ds.get_data_row(0))[1], 0)
    x = ModelInput(x=ds.get_x(ds.get_data_row(0))[1], y=None, index=None)
    model = SimpleNet(config)
    y_prime = model(x).y_prime
    assert tuple(y.shape) == (1,)
    assert tuple(x.x.shape) == (2, model.bins)
    assert tuple(y_prime.shape) == (2, config.ml.model.SimpleModel.fp_size)

@pytest.mark.skip(reason="uses gpu")
def test_AIMSNet(config, ds):
    y = torch.unsqueeze(ds.get_y(ds.get_data_row(0))[1], 0)
    x = ModelInput(x=ds.get_x(ds.get_data_row(0))[1], y=None, index=None)
    model = AIMSNet(config)
    y_prime = model(x).y_prime
    assert tuple(y.shape) == (1,)
    assert tuple(x.x.shape) == (2, model.bins)
    assert tuple(y_prime.shape) == (2, config.ml.model.AIMSNet.fp_size)

@pytest.mark.skip(reason="needs data/nist/tandem/SRM1950/SRM1950_lumos.ecfp4.pynndescent")
def test_ResNetBaseline(config, ds):
    y = torch.unsqueeze(ds.get_y(ds.get_data_row(0))[1], 0)
    x = ModelInput(x=ds.get_x(ds.get_data_row(0))[1], y=None, index=None)
    model = ResNetBaseline(config)
    y_prime = model(x).y_prime
    assert tuple(y.shape) == (1,)
    assert tuple(x.x.shape) == (2, model.bins)
    assert tuple(y_prime.shape) == (2, config.ml.model.ResNetBaseline.fp_size)


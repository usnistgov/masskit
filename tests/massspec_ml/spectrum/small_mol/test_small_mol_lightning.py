import pytest
from pytest import approx
import numpy as np
import pyarrow.parquet as pq
from pyarrow import plasma
import torch
import pytorch_lightning as pl
from massspec_ml.pytorch.spectrum.small_mol.small_mol_datasets import TandemArrowSearchDataset
from massspec_ml.pytorch.spectrum.small_mol.small_mol_lightning import SearchLightningModule, SmallMolSearchDataModule
from massspec_ml.pytorch.base_objects import ModelInput
from massspec_ml.pytorch.spectrum.spectrum_datasets import TandemArrowDataset
import builtins
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def start_plasma():
    with plasma.start_plasma_store(8000000000) as ps:
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

@pytest.mark.skip(reason="need to set up to use test data.  also, uses gpu")
def test_SearchLightningModule(config, start_plasma):
    model = SearchLightningModule(config)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1,
        limit_train_batches=2,
    )
    dm = SmallMolSearchDataModule(config)
    trainer.fit(model, datamodule=dm)


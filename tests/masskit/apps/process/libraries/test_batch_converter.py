import pytest
from masskit.apps.process.libraries.batch_converter import batch_converter_app

def test_batch_converter_sdf(config_batch_converter_sdf):
    batch_converter_app(config_batch_converter_sdf)

def test_batch_converter_csv(config_batch_converter_csv):
    batch_converter_app(config_batch_converter_csv)

def test_batch_converter_pubchem(config_batch_converter_pubchem_sdf):
    batch_converter_app(config_batch_converter_pubchem_sdf)

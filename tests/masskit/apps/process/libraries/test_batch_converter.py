import pytest
import os
from masskit.apps.process.libraries.batch_converter import batch_converter_app

def test_catch_converter(config_batch_converter):
    batch_converter_app(config_batch_converter)

def test_catch_converter_sdf(config_batch_converter_sdf):
    batch_converter_app(config_batch_converter_sdf)

def test_catch_converter_smiles(config_batch_converter_smiles):
    batch_converter_app(config_batch_converter_smiles)

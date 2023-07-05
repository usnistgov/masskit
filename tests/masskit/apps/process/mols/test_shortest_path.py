import pytest
import os
from masskit.apps.process.mols.shortest_path import path_generator_app

@pytest.mark.dependency(
    depends=["tests/masskit/apps/process/libraries/test_batch_converter.py::test_batch_converter_smiles"],
    scope='session'
)
def test_path_generator(config_shortest_path_smiles):
    path_generator_app(config_shortest_path_smiles)

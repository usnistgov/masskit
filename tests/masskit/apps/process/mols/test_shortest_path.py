import pytest
import os
from masskit.apps.process.mols.shortest_path import path_generator_app

def test_path_generator(config_shortest_path_csv):
    path_generator_app(config_shortest_path_csv)

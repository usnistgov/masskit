import pytest
from hydra import compose, initialize
from masskit.apps.process.libraries import fasta2peptides
import os
from pathlib import Path

"""
pytest fixtures

Placed in the package so that they can be used as plugins for pytest unit tests in
other packages.  To use in other packages, put
pytest_plugins = ("masskit.test_fixtures",)
in the conftest.py file at the root of the package unit tests

"""

@pytest.fixture(scope="session")
def data_dir():
    """
    the directory containing the test data files
    """
    test_dir, _ = os.path.splitext(__file__)
    #return Path(__file__).parents[1] / Path("../../tests/data")
    return Path("data")

@pytest.fixture(scope="session")
def human_uniprot_trunc_parquet(tmpdir_factory):
    return tmpdir_factory.mktemp('fasta2peptides') / 'human_uniprot_trunc.parquet'

@pytest.fixture(scope="session")
def human_uniprot_trunc_fasta(data_dir):
    return data_dir / "human_uniprot_trunc.fasta"

@pytest.fixture(scope="session")
def config_fasta2peptides(human_uniprot_trunc_parquet, human_uniprot_trunc_fasta):
    with initialize(version_base=None, config_path="../apps/process/libraries/conf"):
        cfg = compose(config_name="config_fasta2peptides",
                      overrides=[f"input.file={human_uniprot_trunc_fasta}",
                                 f"output.file={human_uniprot_trunc_parquet}"])
        return cfg

@pytest.fixture(scope="session")
def create_peptide_parquet_file(config_fasta2peptides):
    fasta2peptides.main(config_fasta2peptides)
    return config_fasta2peptides.output.file

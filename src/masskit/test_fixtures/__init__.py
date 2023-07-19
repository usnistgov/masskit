import os
import pytest
from hydra import compose, initialize
from masskit.apps.process.libraries import fasta2peptides
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from masskit.apps.process.libraries.batch_converter import batch_converter_app

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
    if Path("tests/data").exists():
        return Path("tests/data")
    elif Path("data").exists():
        return Path("data")
    else:
        raise FileNotFoundError(f'Unable to find test data directory, cwd={os.getcwd()}')

@pytest.fixture(scope="session")
def SRM1950_lumos_short_sdf(data_dir):
    return data_dir / "SRM1950_lumos_short.sdf"

@pytest.fixture(scope="session")
def SRM1950_lumos_short_parquet(SRM1950_lumos_short_sdf, tmpdir_factory):
    out = tmpdir_factory.mktemp('batch_converter') / 'SRM1950_lumos_short.parquet'
    with initialize(version_base=None, config_path="../apps/process/libraries/conf"):
        cfg = compose(config_name="config_batch_converter",
                      overrides=[f"input.file.names={SRM1950_lumos_short_sdf}",
                                 f"output.file.name={out}",
                                ])
        batch_converter_app(cfg)
        return out
    assert False

@pytest.fixture(scope="session")
def cho_uniq_short_msp(data_dir):
    return data_dir / "cho_uniq_short.msp"

@pytest.fixture(scope="session")
def cho_uniq_short_parquet(cho_uniq_short_msp, tmpdir_factory):
    out = tmpdir_factory.mktemp('batch_converter') / 'cho_uniq_short.parquet'
    with initialize(version_base=None, config_path="../apps/process/libraries/conf"):
        cfg = compose(config_name="config_batch_converter",
                      overrides=[f"input.file.names={cho_uniq_short_msp}",
                                 f"output.file.name={out}",
                                 "conversion/msp=msp_peptide"
                                 ])
        batch_converter_app(cfg)
        return out
    assert False

@pytest.fixture(scope="session")
def cho_uniq_short_table(cho_uniq_short_parquet):
    table = pq.read_table(cho_uniq_short_parquet)
    return table

@pytest.fixture(scope="session")
def SRM1950_lumos_table(SRM1950_lumos_short_parquet):
    table = pq.read_table(SRM1950_lumos_short_parquet)
    return table

@pytest.fixture(scope="session")
def cho_uniq_short_recordbatch(cho_uniq_short_table):
    return cho_uniq_short_table.to_batches()

@pytest.fixture(scope="session")
def SRM1950_lumos_recordbatch(SRM1950_lumos_table):
    return SRM1950_lumos_table.to_batches()

@pytest.fixture(scope="session")
def cho_uniq_short_structarray(cho_uniq_short_recordbatch):
    return pa.StructArray.from_arrays(
        cho_uniq_short_recordbatch[0].columns, 
        names=cho_uniq_short_recordbatch[0].schema.names)

@pytest.fixture(scope="session")
def SRM1950_lumos_structarray(SRM1950_lumos_recordbatch):
    return pa.StructArray.from_arrays(
        SRM1950_lumos_recordbatch[0].columns, 
        names=SRM1950_lumos_recordbatch[0].schema.names)

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

@pytest.fixture(scope="session")
def batch_converted_sdf_files(tmpdir_factory):
    return tmpdir_factory.mktemp('batch_converter') / 'batch_converted_sdf'

@pytest.fixture(scope="session")
def test_new_sdf(data_dir):
    return data_dir / "test.new.sdf"

@pytest.fixture(scope="session")
def config_batch_converter_sdf(test_new_sdf, batch_converted_sdf_files):
    with initialize(version_base=None, config_path="../apps/process/libraries/conf"):
        cfg = compose(config_name="config_batch_converter",
                      overrides=[f"input.file.names={test_new_sdf}",
                                 f"output.file.name={batch_converted_sdf_files}",
                                 f"output.file.types=[parquet]",
                                 f"conversion.row_batch_size=100",
                                ])
        return cfg

@pytest.fixture(scope="session")
def batch_converted_csv_files(tmpdir_factory):
    return tmpdir_factory.mktemp('batch_converter') / 'batch_converted_csv'

@pytest.fixture(scope="session")
def test_csv(data_dir):
    return data_dir / "test.csv"

@pytest.fixture(scope="session")
def config_batch_converter_csv(test_csv, batch_converted_csv_files):
    with initialize(version_base=None, config_path="../apps/process/libraries/conf"):
        cfg = compose(config_name="config_batch_converter",
                      overrides=[f"input.file.names={test_csv}",
                                 f"output.file.name={batch_converted_csv_files}",
                                 f"output.file.types=[parquet]",
                                 f"conversion.row_batch_size=100",
                                ])
        return cfg
    
@pytest.fixture(scope="session")
def batch_converted_pubchem_sdf_files(tmpdir_factory):
    return tmpdir_factory.mktemp('batch_converter') / 'batch_converted_sdf'

@pytest.fixture(scope="session")
def pubchem_sdf(data_dir):
    return data_dir / "pubchem.sdf"

@pytest.fixture(scope="session")
def config_batch_converter_pubchem_sdf(pubchem_sdf, batch_converted_pubchem_sdf_files):
    with initialize(version_base=None, config_path="../apps/process/libraries/conf"):
        cfg = compose(config_name="config_batch_converter",
                      overrides=[f"input.file.names={pubchem_sdf}",
                                 f"output.file.name={batch_converted_pubchem_sdf_files}",
                                 f"output.file.types=[parquet]",
                                 f"conversion.row_batch_size=100",
                                 f"conversion/sdf=sdf_pubchem_mol"])
        return cfg

    
@pytest.fixture(scope="session")
def batch_converted_csv_path_file(tmpdir_factory):
    return tmpdir_factory.mktemp('batch_converter') / 'batch_converted_csv_path_file'

# configurations are kept here so that the config_path can resolve correctly

@pytest.fixture(scope="session")
def config_shortest_path_csv(batch_converted_csv_files, batch_converted_csv_path_file):
    with initialize(version_base=None, config_path="../apps/process/mols/conf"):
        cfg = compose(config_name="config_path",
                      overrides=[f"input.file.name={batch_converted_csv_files}.parquet",
                                 f"output.file.name={batch_converted_csv_path_file}.parquet",
                                 ])
        return cfg

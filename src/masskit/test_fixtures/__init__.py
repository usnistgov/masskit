import os
import pytest
from hydra import compose, initialize
from masskit.apps.process.libraries.fasta2peptides import fasta2peptides_app
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from masskit.apps.process.libraries.batch_converter import batch_converter_app
import pandas as pd

"""
pytest fixtures

Placed in the package so that they can be used as plugins for pytest unit tests in
other packages.  To use in other packages, put
pytest_plugins = ("masskit.test_fixtures",)
in the conftest.py file at the root of the package unit tests

"""


@pytest.fixture(scope="session")
def test_molecules():
    """
    SMILES of molecules to be derivatized
    """
    return [
            "O=C(C)Oc1ccccc1C(=O)O",
            "CCCCCC1=CC2=C([C@@H]3C=C(CC[C@H]3C(O2)(C)C)C)C(=C1C(=O)O)O",
            "c1ccc(cc1)O",  # phenol
            "c1ccc(cc1)C(=O)N",  # benzamide
            "OCC(N)(CO)CO",  # tris
            "Cc1ccc(cc1)S(=O)(=O)NC(=O)NCCC",
        ]

@pytest.fixture(scope="session")
def test_reactants():
    """
    derivatives for test_molecules
    """
    return [
            [None],
            ["trimethylsilylation"],
            ["trimethylsilylation"],
            ["trimethylsilylation"],
            ["trimethylsilylation"],
            ["trimethylsilylation"],
        ]

@pytest.fixture(scope="session")
def test_num_tautomers():
    """
    num of tautomers to generate
    """
    return [
            0,
            0,
            0,
            0,
            0,
            5,
        ]

@pytest.fixture(scope="session")
def test_products():
    """
    the products of test_molecules derivatized with test_reactants
    """
    return [
            [
                "CC(=O)Oc1ccccc1C(=O)OC(=O)C(F)(F)F",
                "COC(=O)c1ccccc1OC(C)=O",
                "CC(=O)Oc1ccccc1C(=O)O[Si](C)(C)C(C)(C)C",
                "CC(=O)OC(=O)c1ccccc1OC(C)=O",
                "CC(=O)Oc1ccccc1C(=O)O[Si](C)(C)C",
            ],
            [
                "CCCCCc1cc2c(c(O[Si](C)(C)C)c1C(=O)O)[C@@H]1C=C(C)CC[C@H]1C(C)(C)O2",
                "CCCCCc1cc2c(c(O)c1C(=O)O[Si](C)(C)C)[C@@H]1C=C(C)CC[C@H]1C(C)(C)O2",
                "CCCCCc1cc2c(c(O[Si](C)(C)C)c1C(=O)O[Si](C)(C)C)[C@@H]1C=C(C)CC[C@H]1C(C)(C)O2",
            ],
            [
                "C[Si](C)(C)Oc1ccccc1",
            ],
            [
                "C[Si](C)(C)NC(=O)c1ccccc1", 
                "C[Si](C)(C)N(C(=O)c1ccccc1)[Si](C)(C)C",
            ],
            [
                "C[Si](C)(C)NC(CO)(CO)CO",
                "C[Si](C)(C)OCC(N)(CO)CO",
                "C[Si](C)(C)OCC(N)(CO)CO[Si](C)(C)C",
                "C[Si](C)(C)N(C(CO)(CO)CO)[Si](C)(C)C",
                "C[Si](C)(C)NC(CO)(CO)CO[Si](C)(C)C",
                "C[Si](C)(C)OCC(N)(CO[Si](C)(C)C)CO[Si](C)(C)C",
                "C[Si](C)(C)NC(CO)(CO[Si](C)(C)C)CO[Si](C)(C)C",
                "C[Si](C)(C)OCC(CO)(CO)N([Si](C)(C)C)[Si](C)(C)C",
                "C[Si](C)(C)NC(CO[Si](C)(C)C)(CO[Si](C)(C)C)CO[Si](C)(C)C",
                "C[Si](C)(C)OCC(CO)(CO[Si](C)(C)C)N([Si](C)(C)C)[Si](C)(C)C",
                "C[Si](C)(C)OCC(CO[Si](C)(C)C)(CO[Si](C)(C)C)N([Si](C)(C)C)[Si](C)(C)C",
            ],
            [
                "CCCN(C(=O)NS(=O)(=O)c1ccc(C)cc1)[Si](C)(C)C",
                "CCCN=C(O)N([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1",
                "CCCN(C(O)=NS(=O)(=O)c1ccc(C)cc1)[Si](C)(C)C",
                "CCCN=C(NS(=O)(=O)c1ccc(C)cc1)O[Si](C)(C)C",
                "CCCNC(=O)N([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1",
                "CCCNC(=NS(=O)(=O)c1ccc(C)cc1)O[Si](C)(C)C",
                "CCCN(C(=O)N([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1)[Si](C)(C)C",
                "CCCN=C(O[Si](C)(C)C)N([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1",
                "CCCN(C(=NS(=O)(=O)c1ccc(C)cc1)O[Si](C)(C)C)[Si](C)(C)C",
            ],
        ]


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
        raise FileNotFoundError(
            f'Unable to find test data directory, cwd={os.getcwd()}')

def make_react_parquet(test_molecules, test_reactants, tmpdir_factory, slice_start=None, slice_end=None, file_prefix=None):
    df = pd.DataFrame(list(zip(test_molecules, [x[0] for x in test_reactants])),
               columns =['SMILES', 'reactants'])
    df.index.name='id'
    tmpdir = tmpdir_factory.mktemp('react') 
    df = df[slice_start:slice_end] 
    df.to_csv(tmpdir / f'{file_prefix}.csv')
    with initialize(version_base=None, config_path="../apps/process/libraries/conf"):
        cfg = compose(config_name="config_batch_converter",
                      overrides=[f"input.file.names={tmpdir / file_prefix}.csv",
                                 f"output.file.name={tmpdir / file_prefix}",
                                 f"output.file.types=[parquet]",
                                 f"conversion.row_batch_size=100",
                                 ])

        batch_converter_app(cfg)
        return tmpdir / f'{file_prefix}.parquet'
    assert False  

@pytest.fixture(scope="session")
def reactants(test_molecules, test_reactants, tmpdir_factory):
    """
    create a parquet file of molecules to be reacted
    skips over the first example
    """
    return make_react_parquet(test_molecules, test_reactants, tmpdir_factory, slice_start=1, slice_end=-1, file_prefix='reactants')

@pytest.fixture(scope="session")
def reactants_tautomers(test_molecules, test_reactants, tmpdir_factory):
    """
    create a parquet file of molecules to be reacted
    skips over the first example
    """
    return make_react_parquet(test_molecules, test_reactants, tmpdir_factory, slice_start=-1, slice_end=None, file_prefix='reactants_tautomer')


def make_reactor_config(reactants, tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp('reactor') 
    out = tmpdir / 'reactor_reactants.parquet'
    with initialize(version_base=None, config_path="../apps/process/mols/conf"):
        cfg = compose(config_name="config_reactor",
                      overrides=[f"input.file.name={reactants}",
                                 f"output.file.name={out}",
                                 f"conversion.include_original_molecules=False",
                                 ])
        return cfg
    assert False


@pytest.fixture(scope="session")
def config_reactor(reactants, tmpdir_factory):
    """
    configuration for running reactor_app on test data
    """
    cfg = make_reactor_config(reactants, tmpdir_factory)
    return cfg


@pytest.fixture(scope="session")
def config_reactor_tautomer(reactants_tautomers, tmpdir_factory):
    """
    configuration for running reactor_app on test data
    """
    cfg = make_reactor_config(reactants_tautomers, tmpdir_factory)
    cfg.conversion.num_tautomers = 5
    return cfg


@pytest.fixture(scope="session")
def SRM1950_lumos_short_sdf(data_dir):
    return data_dir / "SRM1950_lumos_short.sdf"


@pytest.fixture(scope="session")
def SRM1950_lumos_short_parquet(SRM1950_lumos_short_sdf, tmpdir_factory):
    out = tmpdir_factory.mktemp('batch_converter') / \
        'SRM1950_lumos_short.parquet'
    with initialize(version_base=None, config_path="../apps/process/libraries/conf"):
        cfg = compose(config_name="config_batch_converter",
                      overrides=[f"input.file.names={SRM1950_lumos_short_sdf}",
                                 f"output.file.name={out}",
                                 f"conversion/sdf=sdf_nist_mol"
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
    assert False


@pytest.fixture(scope="session")
def create_peptide_parquet_file(config_fasta2peptides):
    fasta2peptides_app(config_fasta2peptides)
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
                                 f"conversion/sdf=sdf_nist_mol"
                                 ])
        return cfg
    assert False


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
    assert False


@pytest.fixture(scope="session")
def batch_converted_pubchem_sdf_files(tmpdir_factory):
    return tmpdir_factory.mktemp('batch_converter') / 'batch_converted_pubchem_sdf'


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
    assert False


@pytest.fixture(scope="session")
def batch_converted_plain_sdf_files(tmpdir_factory):
    return tmpdir_factory.mktemp('batch_converter') / 'batch_converted_plain_sdf'


@pytest.fixture(scope="session")
def plain_sdf(data_dir):
    return data_dir / "plain.sdf"


@pytest.fixture(scope="session")
def config_batch_converter_plain_sdf(plain_sdf, batch_converted_plain_sdf_files):
    with initialize(version_base=None, config_path="../apps/process/libraries/conf"):
        cfg = compose(config_name="config_batch_converter",
                      overrides=[f"input.file.names={plain_sdf}",
                                 f"output.file.name={batch_converted_plain_sdf_files}",
                                 f"output.file.types=[parquet]",
                                 f"conversion.row_batch_size=100"])
        return cfg
    assert False


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
    assert False

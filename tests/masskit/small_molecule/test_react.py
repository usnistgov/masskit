from rdkit import Chem

from masskit.apps.process.mols.reactor import reactor_app
import masskit.small_molecule.react as react
from masskit.small_molecule.utils import standardize_mol
import pyarrow.parquet as pq


def do_standardization(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = standardize_mol(mol)
    return mol

def test_reactants(config_reactor, test_products):
    """
    test reactions with TMS and no tautomers
    """
    cfg = config_reactor(slice_start=1, slice_end=-1, file_prefix="react")
    reactor_app(cfg)
    t = pq.read_table(cfg.output.file.name)
    results = [Chem.MolToSmiles(x.as_py()) for x in t['mol']]
    flat_list = [item for sublist in test_products[1:-1] for item in sublist]
    assert set(results) == set(flat_list)

def test_reactants_tautomers(config_reactor, test_products):
    """
    test reaction with TMS and tautomers
    """
    cfg = config_reactor(slice_start=-1, slice_end=None, file_prefix="react_tautomers")
    cfg.conversion.num_tautomers = 5
    reactor_app(cfg)
    t = pq.read_table(cfg.output.file.name)
    results = [Chem.MolToSmiles(x.as_py()) for x in t['mol']]
    flat_list = [item for sublist in test_products[-1:None] for item in sublist]
    assert set(results) == set(flat_list)

def test_reactants_all(config_reactor, test_products):
    """
    test reaction with all possible derivatives and no tautomers
    """
    cfg = config_reactor(slice_start=0, slice_end=1, file_prefix="react_all")
    cfg.conversion.reactant_names = [None]
    reactor_app(cfg)
    t = pq.read_table(cfg.output.file.name)
    results = [Chem.MolToSmiles(x.as_py()) for x in t['mol']]
    flat_list = [item for sublist in test_products[0:1] for item in sublist]
    assert set(results) == set(flat_list)

def do_reaction(
    molecule,
    comparison,
    num_tautomers=0,
    reactant_names=None,
    functional_group_names=None,
):
    # logging.info(f"reactant={Chem.MolToSmiles(molecule)}")
    reactor = react.Reactor()
    products = reactor.react(
        molecule,
        reactant_names=reactant_names,
        functional_group_names=functional_group_names,
        num_tautomers=num_tautomers,
    )
    smiles = [Chem.MolToSmiles(x) for x in products]
    assert set(comparison) == set(smiles)


def test_react_finalM(test_molecules, test_products, test_reactants, test_num_tautomers):
    finalM = do_standardization(test_molecules[0])
    do_reaction(
        finalM,
        test_products[0],
        num_tautomers=test_num_tautomers[0],
        reactant_names=test_reactants[0],
    )


def test_react_m(test_molecules, test_products, test_reactants,test_num_tautomers):
    m = do_standardization(test_molecules[1])
    do_reaction(
        m,
        test_products[1],
        num_tautomers=test_num_tautomers[1],
        reactant_names=test_reactants[1],
)


def test_react_phenol(test_molecules, test_products, test_reactants,test_num_tautomers):
    phenol = do_standardization(test_molecules[2])
    do_reaction(
        phenol, test_products[2], reactant_names=test_reactants[2], num_tautomers=test_num_tautomers[2],

    )


def test_react_benzamide(test_molecules, test_products, test_reactants, test_num_tautomers):
    benzamide = do_standardization(test_molecules[3])
    do_reaction(
        benzamide,
        test_products[3],
        num_tautomers=test_num_tautomers[3],
        reactant_names=test_reactants[3],
    )


def test_react_tris(test_molecules, test_products, test_reactants, test_num_tautomers):
    tris = do_standardization(test_molecules[4])
    do_reaction(
        tris,
        test_products[4],
        num_tautomers=test_num_tautomers[4],
        reactant_names=test_reactants[4],
    )


def test_react_from_sandy(test_molecules, test_products, test_reactants, test_num_tautomers):
    from_sandy = do_standardization(test_molecules[5])
    do_reaction(
        from_sandy,
        test_products[5],
        num_tautomers=test_num_tautomers[5],
        reactant_names=test_reactants[5],
    )


def test_standard_derivatization():
    mol = Chem.MolFromSmiles("O=C(C)Oc1ccccc1C(=O)O")
    mol = standardize_mol(mol)
    reactor = react.Reactor()
    products = reactor.react(
        mol,
        reactant_names=[
            "methylation",
            "acetylation",
            "trifluoroacetylation",
            "t-butyldimethylsilylation",
            "trimethylsilylation",
        ],
        functional_group_names=[
            "alcohol",
            "carboxyl",
            "amine",
            "pyrrole",
            "amide",
            "thiol",
        ],
        num_tautomers=0,
    )
    assert len(products) == 5

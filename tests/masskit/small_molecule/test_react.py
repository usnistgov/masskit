import masskit.small_molecule.react as react
from masskit.small_molecule.utils import standardize_mol
from rdkit import Chem


def do_standardization(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = standardize_mol(mol)
    return mol


def do_reaction(molecule, comparison, num_tautomers=0, reactant_names=None, functional_group_names=None):
    # logging.info(f"reactant={Chem.MolToSmiles(molecule)}")
    reactor = react.Reactor()
    if reactant_names is None:
        reactant_names = reactor.reactant_names
    if functional_group_names is None:
        functional_group_names = reactor.functional_group_names
    products = reactor.react(molecule, 
                             reactant_names=reactant_names,
                             functional_group_names=functional_group_names,
                             num_tautomers=num_tautomers)
    smiles = [Chem.MolToSmiles(x) for x in products]
    assert set(comparison) == set(smiles)


def test_react_finalM():
    finalM = do_standardization('O=C(C)Oc1ccccc1C(=O)O')
    do_reaction(finalM,
                ['CC(=O)Oc1ccccc1C(=O)OC(=O)C(F)(F)F',
                 'COC(=O)c1ccccc1OC(C)=O',
                 'CC(=O)Oc1ccccc1C(=O)O[Si](C)(C)C(C)(C)C',
                 'CC(=O)OC(=O)c1ccccc1OC(C)=O',
                 'CC(=O)Oc1ccccc1C(=O)O[Si](C)(C)C',
                 ])


def test_react_m():
    m = do_standardization(
        'CCCCCC1=CC2=C([C@@H]3C=C(CC[C@H]3C(O2)(C)C)C)C(=C1C(=O)O)O')
    do_reaction(m,
                ['CCCCCc1cc2c(c(O[Si](C)(C)C)c1C(=O)O)[C@@H]1C=C(C)CC[C@H]1C(C)(C)O2',
                 'CCCCCc1cc2c(c(O)c1C(=O)O[Si](C)(C)C)[C@@H]1C=C(C)CC[C@H]1C(C)(C)O2',
                 'CCCCCc1cc2c(c(O[Si](C)(C)C)c1C(=O)O[Si](C)(C)C)[C@@H]1C=C(C)CC[C@H]1C(C)(C)O2',
                 ],
                reactant_names=['trimethylsilylation'])


def test_react_phenol():
    phenol = do_standardization('c1ccc(cc1)O')
    do_reaction(phenol,
                ['C[Si](C)(C)Oc1ccccc1'],
                reactant_names=['trimethylsilylation'])


def test_react_benzamide():
    benzamide = do_standardization('c1ccc(cc1)C(=O)N')
    do_reaction(benzamide,
                ['C[Si](C)(C)NC(=O)c1ccccc1',
                 'C[Si](C)(C)N(C(=O)c1ccccc1)[Si](C)(C)C'
                 ],
                reactant_names=['trimethylsilylation']
                )


def test_react_tris():
    tris = do_standardization('OCC(N)(CO)CO')
    do_reaction(tris,
                ['C[Si](C)(C)NC(CO)(CO)CO',
                 'C[Si](C)(C)OCC(N)(CO)CO',
                 'C[Si](C)(C)OCC(N)(CO)CO[Si](C)(C)C', 'C[Si](C)(C)N(C(CO)(CO)CO)[Si](C)(C)C',
                 'C[Si](C)(C)NC(CO)(CO)CO[Si](C)(C)C',
                 'C[Si](C)(C)OCC(N)(CO[Si](C)(C)C)CO[Si](C)(C)C',
                 'C[Si](C)(C)NC(CO)(CO[Si](C)(C)C)CO[Si](C)(C)C',
                 'C[Si](C)(C)OCC(CO)(CO)N([Si](C)(C)C)[Si](C)(C)C',
                 'C[Si](C)(C)NC(CO[Si](C)(C)C)(CO[Si](C)(C)C)CO[Si](C)(C)C',
                 'C[Si](C)(C)OCC(CO)(CO[Si](C)(C)C)N([Si](C)(C)C)[Si](C)(C)C',
                 'C[Si](C)(C)OCC(CO[Si](C)(C)C)(CO[Si](C)(C)C)N([Si](C)(C)C)[Si](C)(C)C'],
                reactant_names=['trimethylsilylation']
                )


def test_react_from_sandy():
    from_sandy = do_standardization('Cc1ccc(cc1)S(=O)(=O)NC(=O)NCCC')
    do_reaction(from_sandy,
                ['CCCN(C(=O)NS(=O)(=O)c1ccc(C)cc1)[Si](C)(C)C',
                 'CCCN=C(O)N([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1',
                 'CCCN(C(O)=NS(=O)(=O)c1ccc(C)cc1)[Si](C)(C)C',
                 'CCCN=C(NS(=O)(=O)c1ccc(C)cc1)O[Si](C)(C)C',
                 'CCCNC(=O)N([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1',
                 'CCCNC(=NS(=O)(=O)c1ccc(C)cc1)O[Si](C)(C)C',
                 'CCCN(C(=O)N([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1)[Si](C)(C)C',
                 'CCCN=C(O[Si](C)(C)C)N([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1',
                 'CCCN(C(=NS(=O)(=O)c1ccc(C)cc1)O[Si](C)(C)C)[Si](C)(C)C'],
                num_tautomers=5,
                reactant_names=['trimethylsilylation'])

def test_standard_derivatization():
    mol = Chem.MolFromSmiles('O=C(C)Oc1ccccc1C(=O)O')
    mol = standardize_mol(mol)
    reactor = react.Reactor()
    products = reactor.react(mol, 
                             reactant_names=['methylation',
                                             'acetylation',
                                             'trifluoroacetylation',
                                             't-butyldimethylsilylation',
                                             'trimethylsilylation'],
                             functional_group_names=['alcohol',
                                                     'carboxyl',
                                                     'amine',
                                                     'pyrrole',
                                                     'amide',
                                                     'thiol'],
                             num_tautomers=0)
    assert len(products) == 5
    
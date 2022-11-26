from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from massspec.small_molecule.utils import standardize_mol
import logging
from rdkit.Chem.MolStandardize import rdMolStandardize
# import molvs


class Reactor:
    """
    class for doing several reactions on molecules with multiple steps
    """

    def __init__(self, standardization_function=standardize_mol):
        """
        initialize

        :param standardization_function: the standardization function to use (takes Mol, returns Mol)
        """
        self.reactants = {
            'methylation': 'C',
            'acetylation': 'C(=O)C',
            'trifluoroacetylation': 'C(=O)(C(F)(F)F)',
            't-butyldimethylsilylation': '[Si](C)(C)C(C)(C)(C)',
            'trimethylsilylation': '[Si](C)(C)C'
        }

        self.functional_groups = {
            'alcohol': '[#6;H3,H2,H1,H0;!$(C=O):1][OX2:2][#1]>>[C:1][OX:2]({replacement_group})',
            'carboxyl': '[CX3:1](=[O:2])[OX2:3][#1]>>[CX3:1](=[O:2])[OX2:3]({replacement_group})',
            'amine': '[NX3;H2,H1;!$(NC=O):1][#1]>>[N:1]{replacement_group}',
            'pyrrole': '[n;H1:1][#1]>>[n:1]{replacement_group}',
            'amide': '[C:1](=[OX1:2])([NX3;H2,H1,H0:3][#1])>>[C:1](=[OX1:2])[NX3:3]({replacement_group})',
            'thiol': '[#16X2H:1][#1]>>[SX2:1]({replacement_group})'
        }

        # functional subgroups for future use
        self.other = {
            'primary_alcohol': '[C;H3,H2;!$(C=O):1][OX2:2][#1]>>[C:1][OX:2]({replacement_group})',
            'secondary_alcohol': '[C;H1;!$(C=O):1][OX2:2][#1]>>[C:1][OX:2]({replacement_group})',
            'tertiary_alcohol': '[C;H0;!$(C=O):1][OX2:2][#1]>>[C:1][OX:2]({replacement_group})',
            'aromatic_alcohol': '[c:1][OX2:2][#1]>>[c:1][O:2]({replacement_group})',
            'thiol': '[SX2H:1][#1]>>[SX2:1]({replacement_group})',
            'aromatic_thiol': '[sX2H:1][#1]>>[sX2:1]({replacement_group})'
        }

        self.reactions = []

        # self.reaction_library = {
        #     'TMS_alcohol_primary':
        #         AllChem.ReactionFromSmarts('[C;H3,H2;!$(C=O):1][OX2:2][#1]>>[C:1][OX:2]([Si](C)(C)C)'),
        #     # alcohol, not phenol, not carboxyl
        #     'TMS_alcohol_secondary':
        #         AllChem.ReactionFromSmarts('[C;H1;!$(C=O):1][OX2:2][#1]>>[C:1][OX:2]([Si](C)(C)C)'),
        #     # alcohol, not phenol, not carboxyl
        #     'TMS_alcohol_tertiary':
        #         AllChem.ReactionFromSmarts('[C;H0;!$(C=O):1][OX2:2][#1]>>[C:1][OX:2]([Si](C)(C)C)'),
        #     # alcohol, not phenol, not carboxyl
        #     'TMS_aromatic_alcohol':
        #         AllChem.ReactionFromSmarts('[c:1][OX2:2][#1]>>[c:1][O:2]([Si](C)(C)C)'),
        #     'TMS_carboxyl':
        #         AllChem.ReactionFromSmarts('[CX3:1](=[O:2])[OX2:3][#1]>>[CX3:1](=[O:2])[OX2:3]([Si](C)(C)C)'),
        #     'TMS_amine':
        #         AllChem.ReactionFromSmarts('[NX3;H2,H1;!$(NC=O):1][#1]>>[N:1][Si](C)(C)C'),
        #     'TMS_pyrrole':
        #         AllChem.ReactionFromSmarts('[n;H1:1][#1]>>[n:1][Si](C)(C)C'),
        #     'TMS_amide':
        #         AllChem.ReactionFromSmarts(
        #             '[C:1](=[OX1:2])([NX3;H2,H1,H0:3][#1])>>[C:1](=[OX1:2])[NX3:3]([Si](C)(C)C)'),
        #     'TMS_thiol':
        #         AllChem.ReactionFromSmarts('[SX2H:1][#1]>>[SX2:1]([Si](C)(C)C)'),
        #     'TMS_thiol_aromatic':
        #         AllChem.ReactionFromSmarts('[sX2H:1][#1]>>[sX2:1]([Si](C)(C)C)')
        # }
        # some important points about reaction SMARTS
        # - the reactants are forced to have explicit hydrogens.  So if you want one of those hydrogens to disappear,
        #   it has to be explicitly included [#1] in the LHS of the reaction

        self.standardization_function = standardization_function

    @property
    def reactant_names(self):
        """
        returns a list of replacement group names

        :return: list of replacement group names
        """
        return list(self.reactants.keys())

    @property
    def functional_group_names(self):
        """
        returns a list of functional group names

        :return: list of functional group names
        """
        return list(self.functional_groups.keys())

    def create_reactions(self, reactant_names=None, functional_group_names=None):
        """
        create a set of reactions using functional groups that are modified by the addition of replacement groups

        :param reactant_names: list of replacement group names
        :param functional_group_names: list of functional group names
        """
        if reactant_names is None:
            reactant_names = self.reactant_names
        if functional_group_names is None:
            functional_group_names = self.functional_group_names

        self.reactions = []
        for replacement_group in reactant_names:
            for functional_group in functional_group_names:
                self.reactions.append(AllChem.ReactionFromSmarts(self.functional_groups[functional_group].format(
                    replacement_group=self.reactants[replacement_group])))

    def apply_reactions(self, molecules, maxProducts=1000, mass_range=None):
        """
        apply a series of reactions to a list of molecules

        :param molecules: list of rdkit Mol
        :param maxProducts: maximum number of products per reaction
        :param mass_range: tuple containing low and high value of allowed mass of product
        :return: list of product molecules
        """
        # add hydrogen
        molecules = [(Chem.AddHs(x)) for x in molecules]
        all_products = []
        # run the reaction
        for reaction in self.reactions:
            for molecule in molecules:
                products = reaction.RunReactants((molecule,), maxProducts=maxProducts)
                all_products.extend([x[0] for x in products])
        # dedup molecules
        deduplicated_products = set()
        for molecule in all_products:
            try:
                molecule = self.standardization_function(molecule)
                molecule = Chem.RemoveHs(molecule)
                # mass filter
                if mass_range:
                    if mass_range[0] <= Descriptors.ExactMolWt(molecule) <= mass_range[1]:
                        deduplicated_products.add(Chem.MolToSmiles(molecule, isomericSmiles=True))
            except:  # unable to standardize
                continue
        return [Chem.MolFromSmiles(x) for x in deduplicated_products]

    def react(self, molecules, reactant_names=None, functional_group_names=None, maxProducts=1000,
              max_passes=100, include_original_molecules=False, num_tautomers=0, mass_range=None):
        """
        Given a list of molecules, react them using the named reactions

        :param molecules: standardized molecule or list of molecules to be reacted
        :param reactant_names: list of names of replacement groups added in the reaction.  [] means all
        :param functional_group_names: list of names of the functional groups where reactions take place. [] means all
        :param maxProducts: maximum number of products per reaction and overall.  an approximate bound
        :param max_passes: iteratively apply the reactions to reaction products up to max_passes
        :param include_original_molecules: add the original molecules to the returned products
        :param num_tautomers: create up to this this number of tautomers from the input structures
        :param mass_range: tuple containing low and high value of allowed mass of product
        :return: list of reaction products as rdkit Mol
        """
        if type(molecules) is not list:
            molecules = [molecules]

        # create tautomers if asked
        if num_tautomers > 0:
            new_molecules = []
            # enumerator = molvs.tautomer.TautomerEnumerator(max_tautomers=num_tautomers)
            enumerator = rdMolStandardize.TautomerEnumerator()
            enumerator.SetMaxTautomers(num_tautomers)
            for molecule in molecules:
                # new_molecules.extend(enumerator.enumerate(molecule))
                new_molecules.extend(enumerator.Enumerate(molecule))
            molecules = new_molecules

        self.create_reactions(reactant_names=reactant_names,
                              functional_group_names=functional_group_names)

        products = self.apply_reactions(molecules, maxProducts=maxProducts, mass_range=mass_range)
        new_products = products
        for i in range(max_passes):
            new_products = self.apply_reactions(new_products, maxProducts=maxProducts, mass_range=mass_range)
            if not new_products:
                break
            products.extend(new_products)
            if len(products) > maxProducts:
                break
        if include_original_molecules:
            molecules.extend(products)  # use molecules to keep the correct order
            return molecules
        else:
            return products


if __name__ == '__main__':

    import unittest

    class TestReactionMethods(unittest.TestCase):
        """
        unit tests for the NISTSpectrum and NISTPeaks classes
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.maxDiff = None
            self.reaction = Reactor()

            self. finalM = self.do_standardization('O=C(C)Oc1ccccc1C(=O)O')
            self.m = self.do_standardization('CCCCCC1=CC2=C([C@@H]3C=C(CC[C@H]3C(O2)(C)C)C)C(=C1C(=O)O)O')
            self.phenol = self.do_standardization('c1ccc(cc1)O')
            self.benzamide = self.do_standardization('c1ccc(cc1)C(=O)N')
            self.tris = self.do_standardization('OCC(N)(CO)CO')
            self.from_sandy = self.do_standardization('Cc1ccc(cc1)S(=O)(=O)NC(=O)NCCC')

        def do_standardization(self, smiles):
            mol = Chem.MolFromSmiles(smiles)
            mol = self.reaction.standardization_function(mol)
            return mol

        def test_react(self):
            self.do_reaction(self.finalM, ['CC(=O)Oc1ccccc1C(=O)OC(=O)C(F)(F)F', 'COC(=O)c1ccccc1OC(C)=O',
                                           'CC(=O)Oc1ccccc1C(=O)O[Si](C)(C)CC(C)C', 'CC(=O)OC(=O)c1ccccc1OC(C)=O',
                                           'CC(=O)Oc1ccccc1C(=O)O[Si](C)(C)C'])
            self.do_reaction(self.phenol, ['C[Si](C)(C)Oc1ccccc1'], reactant_names=['trimethylsilylation'])
            self.do_reaction(self.m, ['CCCCCc1cc2c(c(O[Si](C)(C)C)c1C(=O)O)[C@@H]1C=C(C)CC[C@H]1C(C)(C)O2',
                                      'CCCCCc1cc2c(c(O)c1C(=O)O[Si](C)(C)C)[C@@H]1C=C(C)CC[C@H]1C(C)(C)O2',
                                      'CCCCCc1cc2c(c(O[Si](C)(C)C)c1C(=O)O[Si](C)(C)C)[C@@H]1C=C(C)CC[C@H]1C(C)(C)O2'],
                             reactant_names=['trimethylsilylation']
                             )
            self.do_reaction(self.benzamide, ['C[Si](C)(C)NC(=O)c1ccccc1', 'C[Si](C)(C)N(C(=O)c1ccccc1)[Si](C)(C)C'],
                             reactant_names=['trimethylsilylation']
                             )
            self.do_reaction(self.tris, ['C[Si](C)(C)NC(CO)(CO)CO', 'C[Si](C)(C)OCC(N)(CO)CO',
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
            self.do_reaction(self.from_sandy, ['CCCN(C(=O)NS(=O)(=O)c1ccc(C)cc1)[Si](C)(C)C',
                                               'CCCN=C(O)N([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1',
                                               'CCCN(C(O)=NS(=O)(=O)c1ccc(C)cc1)[Si](C)(C)C',
                                               'CCCN=C(NS(=O)(=O)c1ccc(C)cc1)O[Si](C)(C)C',
                                               'CCCNC(=O)N([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1',
                                               'CCCNC(=NS(=O)(=O)c1ccc(C)cc1)O[Si](C)(C)C',
                                               'CCCN(C(=O)N([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1)[Si](C)(C)C',
                                               'CCCN=C(O[Si](C)(C)C)N([Si](C)(C)C)S(=O)(=O)c1ccc(C)cc1',
                                               'CCCN(C(=NS(=O)(=O)c1ccc(C)cc1)O[Si](C)(C)C)[Si](C)(C)C'],
                             num_tautomers=5, reactant_names=['trimethylsilylation'])

            return

        def do_reaction(self, molecule, comparison, num_tautomers=0, reactant_names=None):
            # logging.info(f"reactant={Chem.MolToSmiles(molecule)}")
            products = self.reaction.react(molecule, reactant_names=reactant_names,
                                           num_tautomers=num_tautomers)
            smiles = [Chem.MolToSmiles(x) for x in products]
            self.assertCountEqual(smiles, comparison)
            logging.info(f"products={smiles}")

    logging.getLogger().setLevel(logging.INFO)
    unittest.main()

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from . import utils as _mksmutils


class Reactor:
    """
    class for doing several reactions on molecules with multiple steps
    """

    def __init__(self, standardization_function=_mksmutils.standardize_mol):
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
                products = reaction.RunReactants(
                    (molecule,), maxProducts=maxProducts)
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
                        deduplicated_products.add(
                            Chem.MolToSmiles(molecule, isomericSmiles=True))
                else:
                    deduplicated_products.add(
                        Chem.MolToSmiles(molecule, isomericSmiles=True))
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

        products = self.apply_reactions(
            molecules, maxProducts=maxProducts, mass_range=mass_range)
        new_products = products
        for i in range(max_passes):
            new_products = self.apply_reactions(
                new_products, maxProducts=maxProducts, mass_range=mass_range)
            if not new_products:
                break
            products.extend(new_products)
            if len(products) > maxProducts:
                break
        if include_original_molecules:
            # use molecules to keep the correct order
            molecules.extend(products)
            return molecules
        else:
            return products

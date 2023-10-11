try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:
    pass
import copy
import unittest

import numpy as np
from .. import data as _mkdata

def yang_polarizability(mol):
    """
    calculate group additivity values per atom

    :param mol: the molecule
    :return: dict that maps atom number to group additivity value, calculated polarizability
    """
    return_value = {}
    # go through the smarts patterns
    for pattern in _mkdata.yang_model_2e:
        hits = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
        for i in range(len(hits)):
            # don't allow duplicates
            if hits[i][0] not in return_value:
                return_value[hits[i][0]] = _mkdata.yang_model_2e[pattern]
    # now fill in missing values using the measured atomic polarizability
    for atomic_num in _mkdata.atom_num_dipole:
        hits = mol.GetSubstructMatches(Chem.MolFromSmarts(f"[#{atomic_num}]"))
        for i in range(len(hits)):
            # don't allow duplicates
            if hits[i][0] not in return_value:
                return_value[hits[i][0]] = _mkdata.atom_num_dipole[atomic_num]
    polarizability = sum(return_value.values()) - 1.529
    return polarizability, return_value


# the following functions taken from
# https://github.com/abradle/rdkitserver/blob/master/MYSITE/src/testproject/mol_parsing/sanifix.py
# https://sourceforge.net/p/rdkit/mailman/message/27552085/
# used to clean up the molecules when adding explicit H's


def _FragIndicesToMol(oMol,indices):
    # indices are the atom indicies of the fragment
    # create a new editable mol
    em = Chem.EditableMol(Chem.Mol())

    # create map from fragment to indices in old molecule
    newIndices={}
    for i,idx in enumerate(indices):
        em.AddAtom(oMol.GetAtomWithIdx(idx))
        newIndices[idx]=i

    for i,idx in enumerate(indices):
        at = oMol.GetAtomWithIdx(idx)
        for bond in at.GetBonds():
            if bond.GetBeginAtomIdx()==idx:
                oidx = bond.GetEndAtomIdx()
            else:
                oidx = bond.GetBeginAtomIdx()
            # make sure every bond only gets added once:
            if oidx<idx:
                continue
            em.AddBond(newIndices[idx],newIndices[oidx],bond.GetBondType())
    res = em.GetMol()
    res.ClearComputedProps()
    Chem.GetSymmSSSR(res)
    res.UpdatePropertyCache(False)
    # set up the reverse map to the old molecule
    res._idxMap=newIndices
    return res


def _recursivelyModifyNs(mol,matches,indices=None):
    # take fragment mol and matches to aromatic nitrogens matching pattern
    if indices is None:
        indices=[]
    res=None
    while len(matches) and res is None:
        tIndices=indices[:]
        # take the first aromatic nitrogen
        nextIdx = matches.pop(0)
        tIndices.append(nextIdx)
        nm = Chem.Mol(mol)
        # set nitrogen with no implicit H and one explicit H
        nm.GetAtomWithIdx(nextIdx).SetNoImplicit(True)
        nm.GetAtomWithIdx(nextIdx).SetNumExplicitHs(1)
        cp = Chem.Mol(nm)
        try:
            Chem.SanitizeMol(cp)
        except ValueError:
            # keep on trying to fix the nitrogens
            res,indices = _recursivelyModifyNs(nm,matches,indices=tIndices)
        else:
            indices=tIndices
            res=cp
    # returns fragment and remaining indices
    return res, indices


def AdjustAromaticNs(m, nitrogenPattern='[n&D2&H0;r5,r6]'):
    """
       default nitrogen pattern matches Ns in 5 rings and 6 rings in order to be able
       to fix: O=c1ccncc1
       nitrogenPattern is aromatic n in ring of size 5 or 6 with 2 further explict connections, with 0 further H
    """
    Chem.GetSymmSSSR(m)
    # regenerate computed properties like implicit valence and ring information.
    m.UpdatePropertyCache(False)

    # break non-ring bonds linking rings.  start with creating an editable mol
    em = Chem.EditableMol(m)
    # match ring atom, non-ring atom, ring atom
    linkers = m.GetSubstructMatches(Chem.MolFromSmarts('[r]!@[r]'))
    plsFix=set()
    # go thru the pairs of atoms, break bonds and add them to plsfix list
    for a,b in linkers:
        em.RemoveBond(a,b)
        plsFix.add(a)
        plsFix.add(b)
    # get a regular mol object
    nm = em.GetMol()
    # get the atoms to fix.  if aromatic nitrogen, set 1 explicit hyrogen, no implicit Hs
    # I guess the idea is to ignore the nitrogens in rings that are just linkers to other rings.
    for at in plsFix:
        at=nm.GetAtomWithIdx(at)
        if at.GetIsAromatic() and at.GetAtomicNum()==7:
            at.SetNumExplicitHs(1)
            at.SetNoImplicit(True)

    # build molecules from the fragments:
    fragLists = Chem.GetMolFrags(nm)
    # creates the fragment mols
    frags = [_FragIndicesToMol(nm,x) for x in fragLists]

    # loop through the fragments in turn and try to aromatize them:
    ok=True
    for i,frag in enumerate(frags):
        cp = Chem.Mol(frag)
        try:
            Chem.SanitizeMol(cp)
        except ValueError:
            # presumably this catches the nitrogen valence error.  matches contain N
            matches = [x[0] for x in frag.GetSubstructMatches(Chem.MolFromSmarts(nitrogenPattern))]
            lres,indices=_recursivelyModifyNs(frag,matches)
            # if lres is not None, then there is a fragment that didn't pass sanitization
            if not lres:
                #print 'frag %d failed (%s)'%(i,str(fragLists[i]))
                ok=False
                break
            else:
                # use reverse map of fragment in
                revMap={}
                for k,v in frag._idxMap.iteritems():
                    revMap[v]=k
                for idx in indices:
                    oatom = m.GetAtomWithIdx(revMap[idx])
                    oatom.SetNoImplicit(True)
                    oatom.SetNumExplicitHs(1)
    if not ok:
        return None
    return m


def get_unspec_double_bonds(m):
    """
    get list of double bonds with undefined stereochemistry.  Copied from
    https://github.com/DrrDom/rdkit-scripts/blob/master/sanitize_rdkit.py

    :param m: molecule
    :return: list of bonds
    """

    def check_nei_bonds(bond):
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        a1_bonds_single = [b.GetBondType() == Chem.BondType.SINGLE for b in a1.GetBonds() if b.GetIdx() != bond.GetIdx()]
        a2_bonds_single = [b.GetBondType() == Chem.BondType.SINGLE for b in a2.GetBonds() if b.GetIdx() != bond.GetIdx()]

        # if there are two identical substituents in one side then the bond is unsteric (no stereoisomers possible)
        ranks = list(Chem.CanonicalRankAtoms(m, breakTies=False))
        a1_nei = [a.GetIdx() for a in a1.GetNeighbors() if a.GetIdx() != a2.GetIdx()]
        if len(a1_nei) == 2 and \
                all(m.GetBondBetweenAtoms(i, a1.GetIdx()).GetBondType() == Chem.BondType.SINGLE for i in a1_nei) and \
                ranks[a1_nei[0]] == ranks[a1_nei[1]]:
            return False
        a2_nei = [a.GetIdx() for a in a2.GetNeighbors() if a.GetIdx() != a1.GetIdx()]
        if len(a2_nei) == 2 and \
                all(m.GetBondBetweenAtoms(i, a2.GetIdx()).GetBondType() == Chem.BondType.SINGLE for i in a2_nei) and \
                ranks[a2_nei[0]] == ranks[a2_nei[1]]:
            return False

        # if list is empty this is a terminal atom, e.g. O in C=O
        if a1_bonds_single and a2_bonds_single and \
                all(a1_bonds_single) and all(a2_bonds_single):
            return True
        else:
            return False

    res = []
    for b in m.GetBonds():
        if b.GetBondType() == Chem.BondType.DOUBLE and \
           b.GetStereo() == Chem.BondStereo.STEREONONE and \
           (not b.IsInRing() or not (b.IsInRingSize(3) or b.IsInRingSize(4) or b.IsInRingSize(5) or
                                     b.IsInRingSize(6) or b.IsInRingSize(7))) and \
           check_nei_bonds(b):
            res.append(b.GetIdx())
    return res

def is_transition_metal(at):
    n = at.GetAtomicNum()
    return (n>=22 and n<=29) or (n>=40 and n<=47) or (n>=72 and n<=79)

def set_dative_bonds(mol, fromAtoms=(7,8)):
    """ convert some bonds to dative

    Replaces some single bonds between metals and atoms with atomic numbers in fomAtoms
    with dative bonds. The replacement is only done if the atom has "too many" bonds.

    Returns the modified molecule.

    """
    pt = Chem.GetPeriodicTable()
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]
    for metal in metals:
        for nbr in metal.GetNeighbors():
            if nbr.GetAtomicNum() in fromAtoms and \
               nbr.GetExplicitValence()>pt.GetDefaultValence(nbr.GetAtomicNum()) and \
               rwmol.GetBondBetweenAtoms(nbr.GetIdx(),metal.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                rwmol.RemoveBond(nbr.GetIdx(),metal.GetIdx())
                rwmol.AddBond(nbr.GetIdx(),metal.GetIdx(),Chem.BondType.DATIVE)
    return rwmol

def standardize_mol(mol):
    """
    standardize molecule

    :param mol: rdkit mol to standardize
    :return: standardized mol
    """
    # disconnector = rdMolStandardize.MetalDisconnector()
    normalizer = rdMolStandardize.Normalizer()
    reionizer = rdMolStandardize.Reionizer()
    # mol_in = copy.deepcopy(mol)  # copy as AdjustAromaticNs return None on failure
    mol_props = copy.deepcopy(mol.GetPropsAsDict())
    # mol = AdjustAromaticNs(mol_in)  # allows for explicit H's
    # if mol is None:
    #     mol = mol_in
    try:
        # molvs standarizer
        # s = Standardizer()
        # mol = s.standardize(mol)  # versions of rdkit older than 2019_09_3 arbitrarily delete properties
        # molvs standardizer found in rdkit
        # mol = rdMolStandardize.Cleanup(mol)

        # only v3000 sdf files support coordinate/dative bonds.  Attempt to detect organometallics, e.g.
        # CCN1#C[Mo]12(Br)(C#O)(C1C=CC=C1)C#N2CC and CCN1#C[W]1(C#O)(C#O)(C1C=CC=C1)[Ge](Cl)(Cl)Cl
        # and convert appropriate bonds to coordinate bonds
        # doesn't yet deal with elements other than C and N nor fix bond order (I think).  So for now,
        # treat coordinate bonds as convalent.
        # mol = set_dative_bonds(mol)

        # to turn off valence errors, ask rdkit not to call updatePropertyCache on atoms with strict=True, which enforces
        # the valencies in PeriodTable.  To do this, turn off SANITIZE_PROPERTIES
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)

        # remove Hs but don't sanitize as that will exclude diborane, etc. due to valence checking
        mol = Chem.RemoveHs(mol, sanitize=False)

        # for now, don't disconnect metals as it cause exceptions to be thrown on organometallics
        # molvs calls SanitizeMol, which throws an exception on valences it doesn't like
        # mol = s.disconnect_metals(mol)
        # mol = disconnector.Disconnect(mol)

        # mol = s.normalize(mol)
        # note that normalize may remove properties
        mol = normalizer.normalize(mol)

        # mol = s.reionize(mol)
        # skip reionization from rdkit as it cleans out the property cache.  Also, very much doubt
        # that there are any free metal ions in the structures, although there should be a check.
        # rebalancing which atoms are most acidic may be arbitrary in some instances, too.
        mol = reionizer.reionize(mol)

        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        # this is the solution rdkit itself uses to preserve properties.  otherwise properties might be
        # removed in normalize or reionize
        for k, v in mol_props.items():
            mol.SetProp(k, str(v))
    except:
        raise
    return mol


if __name__ == "__main__":

    class TestUtils(unittest.TestCase):
        """
        unit tests for utility functions
        """
        def test_yang_polarizability(self):

            m = Chem.MolFromSmiles('C(CC)N(=O)=O')
            m = Chem.AddHs(m)
            polarizability, per_atom = yang_polarizability(m)
            self.assertEqual(polarizability, 58.416)
            m = Chem.MolFromSmiles('C1CCCCCCC1')
            m = Chem.AddHs(m)
            polarizability, per_atom = yang_polarizability(m)
            self.assertEqual(polarizability, 98.34300000000005)
            m = Chem.MolFromSmiles('C(=S)=S')
            m = Chem.AddHs(m)
            polarizability, per_atom = yang_polarizability(m)
            self.assertEqual(polarizability, 53.355000000000004)
            return

        def test_get_unspec_double_bonds(self):
            m = Chem.MolFromSmiles("CC=CC")
            Chem.AssignStereochemistry(m)
            test = get_unspec_double_bonds(m)
            self.assertEqual(len(test), 1)
            return

    unittest.main()

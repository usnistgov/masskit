from rdkit import Chem
import numpy as np
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import unittest

# various Van der Waals atomic radii.  Recipe for creation is
# from mendeleev import get_table
# df = get_table('elements')
# element2vdw = dict(zip(df.symbol, df.vdw_radius/100.0))
# symbol_radius = {k.upper(): v for k, v in element2vdw.items()}

symbol_radius = {'H': 1.1, 'HE': 1.4, 'LI': 1.82, 'BE': 1.53, 'B': 1.92, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47,
                 'NE': 1.54, 'NA': 2.27, 'MG': 1.73, 'AL': 1.84, 'SI': 2.1, 'P': 1.8, 'S': 1.8, 'CL': 1.75, 'AR': 1.88,
                 'K': 2.75, 'CA': 2.31, 'SC': 2.15, 'TI': 2.11, 'V': 2.07, 'CR': 2.06, 'MN': 2.05, 'FE': 2.04,
                 'CO': 2.0, 'NI': 1.97, 'CU': 1.96, 'ZN': 2.01, 'GA': 1.87, 'GE': 2.11, 'AS': 1.85, 'SE': 1.9,
                 'BR': 1.85, 'KR': 2.02, 'RB': 3.03, 'SR': 2.49, 'Y': 2.32, 'ZR': 2.23, 'NB': 2.18, 'MO': 2.17,
                 'TC': 2.16, 'RU': 2.13, 'RH': 2.1, 'PD': 2.1, 'AG': 2.11, 'CD': 2.18, 'IN': 1.93, 'SN': 2.17,
                 'SB': 2.06, 'TE': 2.06, 'I': 1.98, 'XE': 2.16, 'CS': 3.43, 'BA': 2.68, 'LA': 2.43, 'CE': 2.42,
                 'PR': 2.4, 'ND': 2.39, 'PM': 2.38, 'SM': 2.36, 'EU': 2.35, 'GD': 2.34, 'TB': 2.33, 'DY': 2.31,
                 'HO': 2.3, 'ER': 2.29, 'TM': 2.27, 'YB': 2.26, 'LU': 2.24, 'HF': 2.23, 'TA': 2.22, 'W': 2.18,
                 'RE': 2.16, 'OS': 2.16, 'IR': 2.13, 'PT': 2.13, 'AU': 2.14, 'HG': 2.23, 'TL': 1.96, 'PB': 2.02,
                 'BI': 2.07, 'PO': 1.97, 'AT': 2.02, 'RN': 2.2, 'FR': 3.48, 'RA': 2.83, 'AC': 2.47, 'TH': 2.45,
                 'PA': 2.43, 'U': 2.41, 'NP': 2.39, 'PU': 2.43, 'AM': 2.44, 'CM': 2.45, 'BK': 2.44, 'CF': 2.45,
                 'ES': 2.45, 'FM': 2.45, 'MD': 2.46, 'NO': 2.46, 'LR': 2.46}

# use same recipe as above
# atom_num_dipole = dict(zip(df.atomic_number, df.dipole_polarizability))
# 115 (Lv) is unmeasured, so we use the value for Polonium.
atom_num_dipole = {1: 4.50711, 2: 1.38375, 3: 164.1125, 4: 37.74, 5: 20.5, 6: 11.3, 7: 7.4, 8: 5.3, 9: 3.74, 10: 2.6611,
                   11: 162.7, 12: 71.2, 13: 57.8, 14: 37.3, 15: 25.0, 16: 19.4, 17: 14.6, 18: 11.083, 19: 289.7,
                   20: 160.8, 21: 97.0, 22: 100.0, 23: 87.0, 24: 83.0, 25: 68.0, 26: 62.0, 27: 55.0, 28: 49.0, 29: 46.5,
                   30: 38.67, 31: 50.0, 32: 40.0, 33: 30.0, 34: 28.9, 35: 21.0, 36: 16.78, 37: 319.8, 38: 197.2,
                   39: 162.0, 40: 112.0, 41: 98.0, 42: 87.0, 43: 79.0, 44: 72.0, 45: 66.0, 46: 26.14, 47: 55.0,
                   48: 46.0, 49: 65.0, 50: 53.0, 51: 43.0, 52: 38.0, 53: 32.9, 54: 27.32, 55: 400.9, 56: 272.0,
                   57: 215.0, 58: 205.0, 59: 216.0, 60: 208.0, 61: 200.0, 62: 192.0, 63: 184.0, 64: 158.0, 65: 170.0,
                   66: 163.0, 67: 156.0, 68: 150.0, 69: 144.0, 70: 139.0, 71: 137.0, 72: 103.0, 73: 74.0, 74: 68.0,
                   75: 62.0, 76: 57.0, 77: 54.0, 78: 48.0, 79: 36.0, 80: 33.91, 81: 50.0, 82: 47.0, 83: 48.0,
                   84: 44.0, 85: 42.0, 86: 35.0, 87: 317.8, 88: 246.0, 89: 203.0, 90: 217.0, 91: 154.0, 92: 129.0,
                   93: 151.0, 94: 132.0, 95: 131.0, 96: 144.0, 97: 125.0, 98: 122.0, 99: 118.0, 100: 113.0, 101: 109.0,
                   102: 110.0, 103: 320.0, 104: 112.0, 105: 42.0, 106: 40.0, 107: 38.0, 108: 36.0, 109: 34.0, 110: 32.0,
                   111: 32.0, 112: 28.0, 113: 29.0, 114: 31.0, 115: 71.0, 116: 44.0, 117: 76.0, 118: 58.0}

# group additivity method for polarizability.
# from https://pubs.acs.org/doi/pdfplus/10.1021/jp068423w
# nitro group query returns ((1, 0),)
# sulfone query returns ((1,),)
# note that value for N with three bonded atoms is unnecessary for model 2e
# more specific patterns have to be listed first so that the calculation works (dicts keep order in python >=3.6)
yang_model_2e = {"[#6^1]": 10.152, "[#6^2]": 8.765, "[#6^3]": 5.702, "[#1]": 3.391, "[#9]": 3.833, "[#17]": 16.557,
                 "[#35]": 24.123, "[#53]": 38.506, "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]": 10.488, "[#7]": 6.335,
                 "[#8]": 4.307, "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]": 15.726, "[#16]": 22.366,
                 "[#15]": 11.173}


def yang_polarizability(mol):
    """
    calculate group additivity values per atom
    :param mol: the molecule
    :return: dict that maps atom number to group additivity value, calculated polarizability
    """
    return_value = {}
    # go through the smarts patterns
    for pattern in yang_model_2e:
        hits = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
        for i in range(len(hits)):
            # don't allow duplicates
            if hits[i][0] not in return_value:
                return_value[hits[i][0]] = yang_model_2e[pattern]
    # now fill in missing values using the measured atomic polarizability
    for atomic_num in atom_num_dipole:
        hits = mol.GetSubstructMatches(Chem.MolFromSmarts(f"[#{atomic_num}]"))
        for i in range(len(hits)):
            # don't allow duplicates
            if hits[i][0] not in return_value:
                return_value[hits[i][0]] = atom_num_dipole[atomic_num]
    polarizability = sum(return_value.values()) - 1.529
    return polarizability, return_value


def create_fingerprint(mol):
    """
    create a MACCS fingerprint from an rdkit mol and put in a packed numpy array
    :param mol: rdkit mol
    :return: packed numpy array
    """
    bit_string = MACCSkeys.GenMACCSKeys(mol).ToBitString()
    bit_array = np.fromiter(map(int, bit_string), int)
    bit_array = np.packbits(bit_array)
    return bit_array


def fingerprint2numpy(fingerprint):
    """
    convert rdkit fingerprint to numpy array
    :param fingerprint: rdkit fingerprint
    :return: numpy array
    """
    return_value = np.array((fingerprint.GetNumBits()))
    DataStructs.ConvertToNumpyArray(fingerprint, return_value)
    return return_value


def fingerprint_search(query, column, cutoff):
    """
    return a list of hits to a query
    :param query: query fingerprint
    :param column: pandas Series (returned by subsetting a pandas dataframe by column)
    :param cutoff: Tanimoto cutoff
    :return: list of hits, where each hit is a tuple of record id and score
    """
    return_value = []
    for row, val in column.iteritems():
        score = DataStructs.FingerprintSimilarity(query, val)
        if score >= cutoff:
            return_value.append((row, score))
    # sort
        return_value.sort(key=lambda s: s[1], reverse=True)
    return return_value


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
    return res,indices

def AdjustAromaticNs(m,nitrogenPattern='[n&D2&H0;r5,r6]'):
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


def similarity_search(fingerprints, query, threshold, skip_query=True):
    """
    search a list of chemical structures.
    :param fingerprints: list of the fingerprints
    :param query: either index in the list has the query fingerprint or the query fingerprint itself
    :param threshold: the tanimoto threshold for allowing hits
    :param skip_query: don't include the query in the hits if querying by index
    :return: a numpy bool array that points to the hits
    """
    if type(query) is int:
        query_fingerprint = fingerprints[query]
        query_row = query
    elif type(query) is DataStructs.cDataStructs.ExplicitBitVect:
        query_fingerprint = query
        query_row = -1
    else:
        raise ValueError("query has to be an index or a fingerprint object")
    hits = np.full(len(fingerprints), False)
    for i in range(0, len(fingerprints)):
        if skip_query and i == query_row:
            continue
        tanimoto = DataStructs.FingerprintSimilarity(query_fingerprint, fingerprints[i])
        if tanimoto >= threshold:
            hits[i] = True
    return hits


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


if __name__ == "__main__":
    unittest.main()
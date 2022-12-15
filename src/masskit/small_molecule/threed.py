try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    # from rdkit.Chem import rdFreeSASA
    from rdkit.Chem.rdMolTransforms import GetBondLength
except ImportError:
    pass
import numpy as np
from masskit.small_molecule import utils
import unittest
from math import *
import logging
import copy
from random import randrange


def create_conformer(mol, num_conformers=10):
    """
    create a conformer for a rdkit mol

    :param mol: rdkit molecule
    :param num_conformers: max number of 3D conformers created per molecule
    :return: molecule, conformer ids, success/failure
    ***note that an rdkit molecule is passed by value, not reference***
    """
    if mol.GetNumAtoms() < 1:
        logging.info("Cannot create conformer for molecule with no atoms")
        return mol, [], -1
    # add explicit hydrogens
    mol = Chem.AddHs(mol)
    # Chem.AssignAtomChiralTagsFromStructure(mol)
    # calculate a conformer
    # return_value = AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, clearConfs=True)
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, numThreads=0)
    # for some reason sanitization removes stereo flags and AssignAtomChiralTagsFromStructure doesn't replace them
    # also, EmbedMultipleConfs gets rid of stereo flags
    Chem.AssignStereochemistry(mol)
    if mol.GetNumConformers() < 1:
        logging.info("No conformers created")
        return mol, [], -1
    # minimization by MMFF94 is left out as it causes sanitization of the molecule
    # AllChem.MMFFOptimizeMoleculeConfs(mol_copy, numThreads=0)
    return mol, list(ids), 1
    # center the conformer
    # AllChem.CanonicalizeMol(mol)


def bounding_box(mol, conformer_id=-1):
    """
    given a mol with a conformer, return the bounding box of the conformer

    :param mol: the rdkit mol
    :param conformer_id: the conformer to use
    :return: the bounding box as an np array
    """
    x = []
    y = []
    z = []
    for i in range(0, mol.GetNumAtoms()):
        #  if mol.GetAtomWithIdx(i).GetAtomicNum() != 1:  # skip hydrogen
        pos = mol.GetConformer(conformer_id).GetAtomPosition(i)
        x.append(pos.x)
        y.append(pos.y)
        z.append(pos.z)
    return np.array(((min(x), max(x)), (min(y), max(y)), (min(z), max(z))))

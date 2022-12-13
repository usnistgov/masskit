from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit.Chem import rdFreeSASA
from rdkit.Chem.rdMolTransforms import GetBondLength
import numpy as np
from masskit.small_molecule import utils
import unittest
from masskit.config import EIMLConfig, SmallMolMLConfig
from math import *
import quaternion
from skimage.draw import line
import logging
import copy
from typing import List
from random import randrange
from molvs import Standardizer
# from rdkit.Chem.MolStandardize import rdMolStandardize


try:
    from numba import jit
except ImportError:
    # when numba is not available, define jit decorator
    def jit(nopython=True):
        def decorator(func):
            def newfn(*args, **kwargs):
                return func(*args, **kwargs)
            return newfn
        return decorator


def random_quaternion():
    """
    generate a random rotation quaternion
    :return: quaternion
    """
    r1, r2, r3 = np.random.random(3)
    q1 = sqrt(1.0 - r1) * (sin(2 * pi * r2))
    q2 = sqrt(1.0 - r1) * (cos(2 * pi * r2))
    q3 = sqrt(r1) * (sin(2 * pi * r3))
    q4 = sqrt(r1) * (cos(2 * pi * r3))
    r = quaternion.quaternion(q1, q2, q3, q4)
    return r

# following functions modified from
# https://stackoverflow.com/questions/33190042/how-to-calculate-all-24-rotations-of-3d-array


def A_(m, i):  # i in (0, 1, 2)
    idx = np.array([[0, 1, 2, 3], [1, 2, 0, 3], [2, 0, 1, 3]])
    return np.transpose(m, idx[i])


def B_(m, j):  # j in (0, 1, 2, 3)
    idx = np.array([[1, 1, 1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1]])
    return m[::idx[j, 0], ::idx[j, 1], ::idx[j, 2], :]


def C_(m, k):  # k in (1, -1)
    y = [x for x in range(0, 3)][::k]
    y.append(3)
    return np.transpose(m, y)[::k, ::k, ::k, :]


def random_flip(m):
    """
    randomly flip first 3 dimensions of 4d tensor in 90 degree increments without changing chirality
    :param m: tensor to flip
    :return: flipped tensor
    """
    i = randrange(3)
    j = randrange(4)
    k = randrange(-1, 2, 2)
    return C_(B_(A_(m, i), j), -k)


def standardize_mol(mol):
    """
    standardize molecule.  correct valences on aromatic N's and do molvs standardization
    :param mol: rdkit mol to standardize
    :return: standardized mol
    """
    mol = utils.AdjustAromaticNs(mol)  # allows for explicit H's
    try:
        s = Standardizer()
        # molvs standarizer
        # mol = s.standardize(mol)  # versions of rdkit older than 2019_09_3 arbitrarily delete properties

        # molvs standardizer found in rdkit
        # mol = rdMolStandardize.Cleanup(mol)

        # copy of molvs standardizer, modified to ignore valence checks
        # mol = copy.deepcopy(mol)  # no need to make a copy
        # to turn off valence errors, ask rdkit not to call updatePropertyCache on atoms with strict=True, which enforces
        # the valencies in PeriodTable.  To do this, turn off SANITIZE_PROPERTIES
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
        mol = Chem.RemoveHs(mol)
        mol = s.disconnect_metals(mol)
        mol = s.normalize(mol)
        mol = s.reionize(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    except:
        logging.warning(f"Unable to sanitize molecule {mol.id}")
        raise
    return mol


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


@jit(nopython=True)
def pos2index(position, cube_scale, cube_dim):
    """
    calculate position in 3d grid
    :param position: position along one axis
    :param cube_scale: number of grid positions per angstrom
    :param cube_dim: size of grid along one axis
    :return: the grid position
    """
    return int(int(position * cube_scale + 0.5) + cube_dim / 2)


@jit(nopython=True)
def np_pos2index(position, cube_scale, cube_dim):
    """
    same as above, but for numpy array
    :param position: numpy array with position along one axis
    :param cube_scale: number of grid positions per angstrom
    :param cube_dim: size of grid along one axis
    :return: the grid position in numpy array
    """
    return_val = position * cube_scale + 0.5
    return_val = return_val.astype(int) + cube_dim / 2
    return return_val.astype(int)


def mol2tensor(mol, config: SmallMolMLConfig, cube: np.ndarray, rotation: float = None,
               partial_charges=None, conformer_id: int = -1):
    """
    convert the 3d coordinates in an rdkit mol to a numpy 3d grid
    :param mol: the rdkit molecule
    :param config: configuration information
    :param cube: array to contain representation of molecule
    :param rotation: quaternion
    :param partial_charges: array of partial charges for each atom
    :param conformer_id: which conformer to use
    :return: a boolean the indicates if there was any clipping performed
    """
    if partial_charges is None:
        partial_charges = []
    cube_scale = config.cube_dim/config.cube_size  # scale from angstroms to 3d cube
    clipped = False  # has the molecule been clipped?
    # if config.sasa and not config.zero_extra_channels:
    #     # extract atomic radii used to calculate sasa
    #     radii = []
    #     opts = rdFreeSASA.SASAOpts()
    #     opts.probeRadius = config.probeRadius
    #     for atom in mol.GetAtoms():
    #         radii.append(utils.symbol_radius[atom.GetSymbol().upper()])
    #     rdFreeSASA.CalcSASA(mol, radii=radii, opts=opts)
    polarizability = {}
    if config.polarizability:
        _, polarizability = utils.yang_polarizability(mol)
    for i in range(0, mol.GetNumAtoms()):
        pos = mol.GetConformer(conformer_id).GetAtomPosition(i)
        element = mol.GetAtomWithIdx(i).GetAtomicNum()
        one_hot_position = np.where(config.atomic_nums == element)
        if len(one_hot_position[0]) == 1:  # if there is one match.  note that np.where() returns tuple of np arrays
            vector = np.array([pos.x, pos.y, pos.z])
            if rotation is not None:
                vector = quaternion.rotate_vectors(rotation, vector)
            x = pos2index(vector[0], cube_scale, config.cube_dim)
            y = pos2index(vector[1], cube_scale, config.cube_dim)
            z = pos2index(vector[2], cube_scale, config.cube_dim)
            if 0 <= x < config.cube_dim and 0 <= y < config.cube_dim and 0 <= z < config.cube_dim:
                if config.abbreviated:
                    period = config.atomic_period[one_hot_position[0]][0]
                    group = config.atomic_group[one_hot_position[0]][0]
                    if element == 1:  # hydrogen
                        cube[x, y, z, 0 + config.element_pos] = 1.0
                    # hydrogen is channel 0
                    # period from 2-5, set channel 1-4
                    # group from 13-17, set channel 5-9
                    # otherwise, set channel 10 as "other" element
                    elif 2 <= period <= 5 and 13 <= group <= 17:
                        cube[x, y, z, period - 1 + config.element_pos] = 1.0
                        cube[x, y, z, group - 8 + config.element_pos] = 1.0
                    else:
                        cube[x, y, z, 10 + config.element_pos] = 1.0
                else:
                    cube[x, y, z, one_hot_position[0] + config.element_pos] = 1.0
                if config.implicit_hydrogens:
                    implicit_hydrogens = mol.GetAtomWithIdx(i).GetNumImplicitHs()
                    if implicit_hydrogens == 1:
                        cube[x, y, z, 0] = 1.0
                    elif implicit_hydrogens == 2:
                        cube[x, y, z, 1] = 1.0
                    elif implicit_hydrogens == 3:
                        cube[x, y, z, 2] = 1.0
                    elif implicit_hydrogens >= 4:
                        cube[x, y, z, 3] = 1.0
                if config.partial_charge and not config.zero_extra_channels and partial_charges:
                    cube[x, y, z, config.partial_charge_pos] = partial_charges[i]/config.partial_charge_norm
                    #     float(mol.GetAtomWithIdx(i).GetProp('_partial_chargeCharge'))  for gasteiger
                if config.sasa and not config.zero_extra_channels:
                    atomic_props = mol.GetAtomWithIdx(i).GetPropsAsDict()
                    cube[x, y, z, config.sasa_pos] = atomic_props["SASA"]/config.sasa_norm
                if config.vdw_radius and not config.zero_extra_channels:
                    radius = utils.symbol_radius.get(mol.GetAtomWithIdx(i).GetSymbol().upper(), 1.7)
                    cube[x, y, z, config.vdw_radius_pos] = radius/config.vdw_radius_norm
                if config.polarizability and not config.zero_extra_channels:
                    cube[x, y, z, config.polarizability_pos] = polarizability[i]/config.polarizability_norm
            else:
                clipped = True
        else:
            clipped = True
    if config.bonds:
        for j in range(mol.GetNumBonds()):
            bond = mol.GetBondWithIdx(j)
            pos1 = mol.GetConformer(conformer_id).GetAtomPosition(bond.GetBeginAtomIdx())
            pos2 = mol.GetConformer(conformer_id).GetAtomPosition(bond.GetEndAtomIdx())
            x = (pos1.x + pos2.x)/2.0
            y = (pos1.y + pos2.y)/2.0
            z = (pos1.z + pos2.z)/2.0
            x = pos2index(x, cube_scale, config.cube_dim)
            y = pos2index(y, cube_scale, config.cube_dim)
            z = pos2index(z, cube_scale, config.cube_dim)
            if 0 <= x < config.cube_dim and 0 <= y < config.cube_dim and 0 <= z < config.cube_dim:
                if config.abbreviated:
                    # sigma and pi channel.  0.5 means complete bond
                    # single is sigma = 0.5, pi = 0
                    # double is sigma = 0.5, pi = 0.5
                    # triple is sigma = 0.5, pi = 1.0
                    # aromatic is sigma = 0.5, pi = 0.25
                    # if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    #     cube[x, y, -2] = 1.0
                    # else:
                    #     cube[x, y, -3] = 1.0
                    if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                        cube[x, y, z, -1] = 0.5
                        cube[x, y, z, -2] = 0.0
                    elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        cube[x, y, z, -1] = 0.5
                        cube[x, y, z, -2] = 0.5
                    elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                        cube[x, y, z, -1] = 0.5
                        cube[x, y, z, -2] = 1.0
                    elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                        cube[x, y, z, -1] = 0.5
                        cube[x, y, z, -2] = 0.25
                else:
                    if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                        cube[x, y, z, -4] = 1.0
                    elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        cube[x, y, z, -3] = 1.0
                    elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                        cube[x, y, z, -2] = 1.0
                    elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                        cube[x, y, z, -1] = 1.0
                    else:  # anything else encoded as triple
                        cube[x, y, z, -2] = 1.0
                if config.bond_length:
                    cube[x, y, z, config.bond_length_pos] = \
                        GetBondLength(mol.GetConformer(conformer_id), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) / \
                        config.bond_length_norm
                if config.bond_dipole and partial_charges:
                    # get partial charges of atoms
                    # use charge that has absolute min charge
                    charge = min([partial_charges[bond.GetBeginAtomIdx()], partial_charges[bond.GetEndAtomIdx()]],
                                 key=abs)
                    distance = GetBondLength(mol.GetConformer(conformer_id), bond.GetBeginAtomIdx(),
                                             bond.GetEndAtomIdx())
                    # calculate the norm of the dipole vector (it doesn't have a direction)
                    cube[x, y, z, config.bond_dipole_pos] = abs(charge * distance)/config.bond_dipole_norm
            else:
                clipped = True

    return clipped


def rotate_2d(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    return qx, qy


def calculate_third_dim(config: SmallMolMLConfig):
    """
    calculate the third dimension (channels) of a 2D molecular depiction
    :param config: configuration
    :return: size of the third dimension
    """
    # atoms:  if config.abbreviated:
    #           if implicit_hydrogens: 14
    #           else: 11
    #         else: config.atomic_nums.shape[0]
    # partial charges: if config.partial_charge: 1
    #         else: 0
    # bonds: if config.bonds: if abbreviated_bonds: 2
    #                         else: 4
    #
    # namespace:
    #   abbreviated:
    #       implicit h:
    #           [0] 1 hydrogen
    #           [1] 2 hydrogen
    #           [2] 3 hydrogen
    #           [3] 4 or more hydrogen
    #       explicit h:
    #           [0] is hydrogen
    #       -- config.element_pos
    #       [elementpos, elementpos+3] period
    #       [elementpos+4, elementpos+8] group
    #       [elementpos+9] other atom
    #   not abbreviated:
    #       [1,16] one hot encoding of atomic number.  Note that channel 0 is not used, but could be
    #   partial_charge:
    #       [partial_charge_pos]
    #   sasa:
    #       [sasa_pos]
    #   vdw_radius:
    #       [vdw_radius_pos]
    #   bond_length:
    #       [bond_length_pos]
    #   bond_dipole:
    #       [bond_dipole_pos]
    #   polarizability:
    #       [polarizability_pos]
    #   abbreviated bonds:
    #       [-2] sigma
    #       [-1] pi
    #   not abbreviated bonds:
    #       [-4] single
    #       [-3] double
    #       [-2] triple or other
    #       [-1] aromatic

    return_value = 0
    if config.abbreviated:
        if config.implicit_hydrogens:
            return_value += 14
            config.element_pos = 4
        else:
            return_value += 11
            config.element_pos = 0
    else:
        return_value += config.atomic_nums.shape[0]
        config.element_pos = 0
    if config.partial_charge:
        config.partial_charge_pos = return_value
        return_value += 1
    if config.sasa:
        config.sasa_pos = return_value
        return_value += 1
    if config.vdw_radius:
        config.vdw_radius_pos = return_value
        return_value += 1
    if config.polarizability:
        config.polarizability_pos = return_value
        return_value += 1
    if config.xyz:
        config.xyz_pos = return_value
        return_value += 3
    if config.bonds:
        if config.bond_length:
            config.bond_length_pos = return_value
            return_value += 1
        if config.bond_dipole:
            config.bond_dipole_pos = return_value
            return_value += 1
        config.bond_pos = return_value
        if config.abbreviated_bonds:
            return_value += 2
        else:
            return_value += 4
    return return_value


def get_midpoint(config: SmallMolMLConfig, cube_scale, pos1, pos2):
    """
    calculate 2d midpoint of two atoms
    :param config: configuration
    :param cube_scale: scale from angstroms to cube
    :param pos1: position of atom 1
    :param pos2: position of atom 2
    :return: midpoint x, midpoint y
    """
    x = (pos1.x + pos2.x) / 2.0
    y = (pos1.y + pos2.y) / 2.0
    x = pos2index(x, cube_scale, config.cube_dim)
    y = pos2index(y, cube_scale, config.cube_dim)
    return x, y


def mol22d(mol, config: SmallMolMLConfig, rotation=0.0, flip_x=1, flip_y=1, partial_charges=None):
    """
    generate 2d coordinates in an rdkit mol and place in a numpy array.
    size is (config.cube_dim, config.cube_dim, config.atomic_nums.shape[0])
    last dimension is config.atomic_nums.shape[0] + 1 if requesting partial charges.
    Note that the 3d conformer is likely overwritten by this operation
    :param mol: the rdkit molecule
    :param config: configuration information
    :param rotation: angle of rotation in radians
    # todo: note that rotation is currently broken as bond positions are not rotated
    :param flip_x: multiply the x coord
    :param flip_y: multiply the y coord
    :param partial_charges: array of partial charges for each atom
    :return: the grid, and a boolean the indicates if there was any clipping performed
    """
    if partial_charges is None:
        partial_charges = []
    cube_scale = config.cube_dim/config.cube_size  # scale from angstroms to 3d cube
    third_dim = calculate_third_dim(config)
    cube = np.zeros((config.cube_dim, config.cube_dim, third_dim), dtype=np.float32)
    clipped = False  # has the molecule been clipped?
    # how to remove hydrogens:  Chem.RemoveHs(mol), followed by Chem.AddHs(mol).  both return *new* mol
    # then atom.GetNumImplicitHs()
    # create 4 channels: no H, one H, two H, three or more H
    # delete the existing H channel
    # convert any explicit Hs to "other" atom.  Maybe other atom should be atomic_num *1/max(atomic_num)
    #
    # more bond info: draw in 2d using skimage.draw.line, but omit start and stop coordinates
    #
    # consider changing bond encoding to single, double, triple, aromatic
    #
    AllChem.Compute2DCoords(mol)
    polarizability = {}
    if config.polarizability:
        _, polarizability = utils.yang_polarizability(mol)
    if config.sasa:
        raise AssertionError("sasa does not work with 2d")
    if config.implicit_hydrogens:
        mol = Chem.RemoveHs(mol)  # remove hs, both implicit and explicit
        mol = Chem.AddHs(mol)  # adds implicit and explicit hs
    for i in range(0, mol.GetNumAtoms()):
        pos = mol.GetConformer().GetAtomPosition(i)
        element = mol.GetAtomWithIdx(i).GetAtomicNum()
        # make version that does CNO and other atoms, plus partial_charge, plus bond and aromatic  (7 channels)
        one_hot_position = np.where(config.atomic_nums == element)
        if len(one_hot_position[0]) == 1:  # if there is one match.  note that np.where() returns tuple of np arrays
            if rotation != 0.0:
                x, y = rotate_2d((0.0, 0.0), (pos.x, pos.y), rotation)
            else:
                x = flip_x * pos.x
                y = flip_y * pos.y
            x = pos2index(x, cube_scale, config.cube_dim)
            y = pos2index(y, cube_scale, config.cube_dim)
            if 0 <= x < config.cube_dim and 0 <= y < config.cube_dim:
                if config.abbreviated:
                    period = config.atomic_period[one_hot_position[0]]
                    group = config.atomic_group[one_hot_position[0]]
                    if element == 1:  # hydrogen
                        cube[x, y, 0 + config.element_pos] = 1.0
                    # elif element == 6:
                    #     cube[x, y, 1] = 1.0
                    # elif element == 7:
                    #     cube[x, y, 2] = 1.0
                    # elif element == 8:
                    #     cube[x, y, 3] = 1.0
                    # else:
                    #     cube[x, y, 4] = 1.0
                    # hydrogen is channel 0
                    # period from 2-5, set channel 1-4
                    # group from 13-17, set channel 5-9
                    # otherwise, set channel 10 as "other" element
                    elif 2 <= period <= 5 and 13 <= group <= 17:
                        cube[x, y, period - 1 + config.element_pos] = 1.0
                        cube[x, y, group - 8 + config.element_pos] = 1.0
                    else:
                        cube[x, y, 10 + config.element_pos] = 1.0
                else:
                    cube[x, y, one_hot_position[0] + config.element_pos] = 1.0
                if config.implicit_hydrogens:
                    implicit_hydrogens = mol.GetAtomWithIdx(i).GetNumImplicitHs()
                    if implicit_hydrogens == 1:
                        cube[x, y, 0] = 1.0
                    elif implicit_hydrogens == 2:
                        cube[x, y, 1] = 1.0
                    elif implicit_hydrogens == 3:
                        cube[x, y, 2] = 1.0
                    elif implicit_hydrogens >= 4:
                        cube[x, y, 3] = 1.0
                if config.partial_charge and not config.zero_extra_channels and partial_charges:
                    cube[x, y, config.partial_charge_pos] = partial_charges[i]/config.partial_charge_norm
                if config.vdw_radius and not config.zero_extra_channels:
                    radius = utils.symbol_radius.get(mol.GetAtomWithIdx(i).GetSymbol().upper(), 1.7)
                    cube[x, y, config.vdw_radius_pos] = radius/config.vdw_radius_norm
                if config.polarizability and not config.zero_extra_channels:
                    cube[x, y, config.polarizability_pos] = polarizability[i]/config.polarizability_norm
            else:
                clipped = True
        else:
            clipped = True
    if config.bonds:
        for j in range(mol.GetNumBonds()):
            bond = mol.GetBondWithIdx(j)
            pos1 = mol.GetConformer().GetAtomPosition(bond.GetBeginAtomIdx())
            pos2 = mol.GetConformer().GetAtomPosition(bond.GetEndAtomIdx())
            if config.abbreviated_bonds:
                x, y = get_midpoint(config, cube_scale, pos1, pos2)
                x = x * flip_x
                y = y * flip_y
                if 0 <= x < config.cube_dim and 0 <= y < config.cube_dim:
                    # sigma and pi channel.  0.5 means complete bond
                    # single is sigma = 0.5, pi = 0
                    # double is sigma = 0.5, pi = 0.5
                    # triple is sigma = 0.5, pi = 1.0
                    # aromatic is sigma = 0.5, pi = 0.25
                    # if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    #     cube[x, y, -2] = 1.0
                    # else:
                    #     cube[x, y, -3] = 1.0
                    if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                        cube[x, y, -1] = 0.5
                        cube[x, y, -2] = 0.0
                    elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        cube[x, y, -1] = 0.5
                        cube[x, y, -2] = 0.5
                    elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                        cube[x, y, -1] = 0.5
                        cube[x, y, -2] = 1.0
                    elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                        cube[x, y, -1] = 0.5
                        cube[x, y, -2] = 0.25
                    if config.bond_length:
                        cube[x, y, config.bond_length_pos] = \
                            GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) / \
                            config.bond_length_norm
                    if config.bond_dipole and partial_charges:
                        # get partial charges of atoms
                        # use charge that has absolute min charge
                        charge = min([partial_charges[bond.GetBeginAtomIdx()], partial_charges[bond.GetEndAtomIdx()]],
                                     key=abs)
                        distance = GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        # calculate the norm of the dipole vector (it doesn't have a direction)
                        cube[x, y, config.bond_dipole_pos] = abs(charge * distance) / config.bond_dipole_norm
                else:
                    clipped = True
            else:
                if config.full_bonds:
                    # get bond coordinate
                    rr, cc = line(pos2index(pos1.x, cube_scale, config.cube_dim),
                                  pos2index(pos1.y, cube_scale, config.cube_dim),
                                  pos2index(pos2.x, cube_scale, config.cube_dim),
                                  pos2index(pos2.y, cube_scale, config.cube_dim))
                    # delete coordinates that overlap atoms
                    # scale to cube
                    rr = rr[1:-2] * flip_x
                    cc = cc[1:-2] * flip_y
                else:
                    x, y = get_midpoint(config, cube_scale, pos1, pos2)
                    rr = np.array([x]) * flip_x
                    cc = np.array([y]) * flip_y
                filter_out = (rr >= 0) & (rr < config.cube_dim) & (cc >= 0) & (cc < config.cube_dim)
                rr = rr[filter_out]
                cc = cc[filter_out]
                # note: sometimes the lines are very long, e.g. 32 pixels, in 2d depictions
                if len(rr) > 0:
                    if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                        cube[rr, cc, -4] = 1.0
                    elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        cube[rr, cc, -3] = 1.0
                    elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                        cube[rr, cc, -2] = 1.0
                    elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                        cube[rr, cc, -1] = 1.0
                    else:  # anything else is encoded as a triple
                        cube[rr, cc, -2] = 1.0
                else:
                    clipped = True

    return cube, clipped


def mol21D(mol, config: SmallMolMLConfig, seq: np.ndarray, rotation: float = None,
           partial_charges=None, conformer_id: int = -1):
    """
    convert the 3d coordinates in an rdkit mol to a 1D record
    :param mol: the rdkit molecule
    :param config: configuration information
    :param seq: array to contain representation of molecule
    :param rotation: quaternion
    :param partial_charges: array of partial charges for each atom
    :param conformer_id: which conformer to use
    :return: a boolean the indicates if there was any clipping performed
    """
    if partial_charges is None:
        partial_charges = []
    clipped = False  # has the molecule been clipped?
    polarizability = {}
    if config.polarizability:
        _, polarizability = utils.yang_polarizability(mol)
    # if the encoding is supposed to include bonds, halve the max number of bonds and atoms
    max_size = config.max_atoms_bonds//2 if config.bonds else config.max_atoms_bonds

    # iterate over atoms up to the configured max
    for i in range(0, min(max_size, mol.GetNumAtoms())):
        pos = mol.GetConformer(conformer_id).GetAtomPosition(i)
        element = mol.GetAtomWithIdx(i).GetAtomicNum()
        one_hot_position = np.where(config.atomic_nums == element)
        if len(one_hot_position[0]) == 1:  # if there is one match.  note that np.where() returns tuple of np arrays
            vector = np.array([pos.x, pos.y, pos.z])
            if rotation is not None:
                vector = quaternion.rotate_vectors(rotation, vector)
            if config.xyz:
                seq[i, config.xyz_pos] = vector[0] / config.max_bound
                seq[i, config.xyz_pos + 1] = vector[1] / config.max_bound
                seq[i, config.xyz_pos + 2] = vector[2] / config.max_bound
            if config.abbreviated:
                period = config.atomic_period[one_hot_position[0]][0]
                group = config.atomic_group[one_hot_position[0]][0]
                if element == 1:  # hydrogen
                    seq[i, 0 + config.element_pos] = 1.0
                elif 2 <= period <= 5 and 13 <= group <= 17:
                    seq[i, period - 1 + config.element_pos] = 1.0
                    seq[i, group - 8 + config.element_pos] = 1.0
                else:
                    seq[i, 10 + config.element_pos] = 1.0
            else:
                seq[i, one_hot_position[0] + config.element_pos] = 1.0
            if config.implicit_hydrogens:
                implicit_hydrogens = mol.GetAtomWithIdx(i).GetNumImplicitHs()
                if implicit_hydrogens == 1:
                    seq[i, 0] = 1.0
                elif implicit_hydrogens == 2:
                    seq[i, 1] = 1.0
                elif implicit_hydrogens == 3:
                    seq[i, 2] = 1.0
                elif implicit_hydrogens >= 4:
                    seq[i, 3] = 1.0
            if config.partial_charge and not config.zero_extra_channels and partial_charges:
                seq[i, config.partial_charge_pos] = partial_charges[i] / config.partial_charge_norm
            if config.sasa and not config.zero_extra_channels:
                atomic_props = mol.GetAtomWithIdx(i).GetPropsAsDict()
                seq[i, config.sasa_pos] = atomic_props["SASA"] / config.sasa_norm
            if config.vdw_radius and not config.zero_extra_channels:
                radius = utils.symbol_radius.get(mol.GetAtomWithIdx(i).GetSymbol().upper(), 1.7)
                seq[i, config.vdw_radius_pos] = radius / config.vdw_radius_norm
            if config.polarizability and not config.zero_extra_channels:
                seq[i, config.polarizability_pos] = polarizability[i] / config.polarizability_norm
        else:
            clipped = True
    if config.bonds:
        bond_start = max_size  # starting position of bonds in sequence
        for j in range(min(max_size, mol.GetNumBonds())):
            bond = mol.GetBondWithIdx(j)
            pos1 = mol.GetConformer(conformer_id).GetAtomPosition(bond.GetBeginAtomIdx())
            pos2 = mol.GetConformer(conformer_id).GetAtomPosition(bond.GetEndAtomIdx())
            x = (pos1.x + pos2.x)/2.0
            y = (pos1.y + pos2.y)/2.0
            z = (pos1.z + pos2.z)/2.0
            if config.xyz:
                seq[j + bond_start, config.xyz_pos] = x / config.max_bound
                seq[j + bond_start, config.xyz_pos + 1] = y / config.max_bound
                seq[j + bond_start, config.xyz_pos + 2] = z / config.max_bound
            if config.abbreviated:
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    seq[j + bond_start, -1] = 0.5
                    seq[j + bond_start, -2] = 0.0
                elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    seq[j + bond_start, -1] = 0.5
                    seq[j + bond_start, -2] = 0.5
                elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                    seq[j + bond_start, -1] = 0.5
                    seq[j + bond_start, -2] = 1.0
                elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    seq[j + bond_start, -1] = 0.5
                    seq[j + bond_start, -2] = 0.25
            else:
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    seq[j + bond_start, -4] = 1.0
                elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    seq[j + bond_start, -3] = 1.0
                elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                    seq[j + bond_start, -2] = 1.0
                elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    seq[j + bond_start, -1] = 1.0
                else:  # anything else encoded as triple
                    seq[j + bond_start, -2] = 1.0
            if config.bond_length:
                seq[j + bond_start, config.bond_length_pos] = \
                    GetBondLength(mol.GetConformer(conformer_id), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) / \
                    config.bond_length_norm
            if config.bond_dipole and partial_charges:
                # get partial charges of atoms
                # use charge that has absolute min charge
                charge = min([partial_charges[bond.GetBeginAtomIdx()], partial_charges[bond.GetEndAtomIdx()]],
                             key=abs)
                distance = GetBondLength(mol.GetConformer(conformer_id), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                # calculate the norm of the dipole vector (it doesn't have a direction)
                seq[j + bond_start, config.bond_dipole_pos] = abs(charge * distance) / config.bond_dipole_norm

    return clipped


def mol2seq(mol_in, config: SmallMolMLConfig, seq: np.ndarray, rotation: float = None,
            partial_charges=None, conformer_id: int = -1, add_matrices=True):
    """
    convert the 3d coordinates in an rdkit mol to a sequence record
    :param mol_in: the rdkit molecule
    :param config: configuration information
    :param seq: array to contain representation of molecule
    :param rotation: quaternion
    :param partial_charges: array of partial charges for each atom
    :param conformer_id: which conformer to use
    :param add_matrices: add in the distance and topological matrices
    :return: a boolean the indicates if there was any clipping performed
    """
    mol = Chem.RemoveHs(mol_in)
    clipped = False  # has the molecule been clipped?
    # get 3D distance matrix
    if add_matrices:
        dm = Chem.Get3DDistanceMatrix(mol, confId=conformer_id)
        dm = dm / 50.0
        # norm
        # invert?
        # get topological distance matrix using bond order
        tdm = Chem.GetDistanceMatrix(mol, useBO=True)
        tdm = tdm / 50.0

    atomic_num_pos = 0
    h_pos = len(config.atomic_nums)
    matrix3d_pos = h_pos + 4
    matrix1d_pos = matrix3d_pos + config.max_atoms_bonds

    max_atoms = min(config.max_atoms_bonds, mol.GetNumAtoms())

    # iterate over atoms up to the configured max
    for i in range(0, max_atoms):
        element = mol.GetAtomWithIdx(i).GetAtomicNum()
        one_hot_position = np.where(config.atomic_nums == element)
        # encode using one hot
        # encode up to 4 implicit hydrogens
        # encode topological distance matrix
        # encode 3d distance matrix
        if len(one_hot_position[0]) == 1:  # if there is one match.  note that np.where() returns tuple of np arrays
            seq[i, one_hot_position[0] + atomic_num_pos] = 1.0
            hydrogens = min(4, mol.GetAtomWithIdx(i).GetNumImplicitHs() + mol.GetAtomWithIdx(i).GetNumExplicitHs())
            if hydrogens:
                seq[i, h_pos + hydrogens - 1] = 1.0
            if add_matrices:
                seq[i, matrix1d_pos:matrix1d_pos + max_atoms] = tdm[i, 0:max_atoms]
                seq[i, matrix3d_pos:matrix3d_pos + max_atoms] = dm[i, 0:max_atoms]
        else:
            seq[i, atomic_num_pos] = 1.0
            clipped = True
    if config.max_atoms_bonds < mol.GetNumAtoms():
        clipped = True

    return clipped


class Test3DMethods(unittest.TestCase):
    """
    unit tests for 3D functions
    """
    config = EIMLConfig()

    def test_peaks(self):
        self.config.cube_dim = 16
        self.config.cube_size = 4.08
        self.config.partial_charge = True
        self.config.sasa = True
        self.config.vdw_radius = True
        self.config.polarizability = True
        self.config.bond_length = True
        self.config.bond_dipole = True
        m = Chem.MolFromSmiles('CO')  # use CO for partial charges to be nonzero
        # atoms at 8,8,8; 12,7,8; 6,8,12; 8,12,8; 5,7,6; 14,9,8
        # bonds at 9,8,8; 7,8,10; 8,10,8; 6,8,8; 13,8,8
        m, ids, success = create_conformer(m)
        self.assertNotEqual(success, -1)

        partial_charges = []
        mol_copy = copy.deepcopy(m)  # the MMFF calculation sanitizes the molecule
        fps = AllChem.MMFFGetMoleculeProperties(mol_copy)
        if fps is not None:
            for atom_num in range(0, mol_copy.GetNumAtoms()):
                partial_charges.append(fps.GetMMFFPartialCharge(atom_num))

        radii = []
        for atom in m.GetAtoms():
            radii.append(utils.symbol_radius[atom.GetSymbol().upper()])
        # rdFreeSASA.CalcSASA(m, radii=radii)  # todo: currently broken in rdkit

        third_dim = calculate_third_dim(self.config)
        cube = np.zeros((self.config.cube_dim, self.config.cube_dim, self.config.cube_dim, third_dim), dtype=np.float32)
        clipped = mol2tensor(m, self.config, cube, partial_charges=partial_charges)
        self.assertFalse(clipped)
        # look for one of the carbons at the right position
        self.assertEqual(cube[8, 8, 8, 1], 1.0)
        self.assertEqual(cube[8, 8, 8, self.config.sasa_pos], 0.08952131364154355)
        self.assertEqual(cube[8, 8, 8, self.config.partial_charge_pos], 0.14)
        self.assertEqual(cube[8, 8, 8, self.config.vdw_radius_pos], 0.4857142857142857)
        self.assertEqual(cube[8, 8, 8, self.config.polarizability_pos], 5.702/self.config.polarizability_norm)
        # check bond properties
        self.assertEqual(cube[13, 8, 8, self.config.bond_length_pos], 0.3273826906748564)
        self.assertEqual(cube[13, 8, 8, self.config.bond_dipole_pos], 0.0654765381349713)
        return

    def test_mol2seq(self):
        self.config.max_bound = 16.32
        self.config.xyz = True

        m = Chem.MolFromSmiles('CO')  # use CO for partial charges to be nonzero
        # atoms at 8,8,8; 12,7,8; 6,8,12; 8,12,8; 5,7,6; 14,9,8
        # bonds at 9,8,8; 7,8,10; 8,10,8; 6,8,8; 13,8,8
        m, ids, success = create_conformer(m)
        self.assertNotEqual(success, -1)

        third_dim = calculate_third_dim(self.config)
        seq = np.zeros((self.config.max_atoms_bonds, third_dim), dtype=np.float32)
        clipped = mol21D(m, self.config, seq)
        return


if __name__ == "__main__":
    unittest.main()

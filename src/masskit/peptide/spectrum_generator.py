import logging
import random

import pandas as pd

from .. import data as mkdata
from ..spectra import theoretical_spectrum as mktheoretical_spectrum
from . import encoding as mkencoding

"""
The amino acids used to generate the theoretical peptides
"""
generator_alphabet = 'ARNDCQEGHILKMFPSTWYV'

def create_peptide_name(peptide, precursor_charge, mod_names=None, mod_positions=None, ev=None):
    """_
    create the name of a peptide spectrum

    :param peptide: the peptide string
    :param precursor_charge: the precursor charge
    :param mod_names: list of modification names (integer)
    :param mod_positions: position of modifications, 0 based
    :param ev: collision energy in ev
    """
    output = peptide + '/' + str(precursor_charge)
    if mod_names is not None and mod_positions is not None:
        for mod in range(len(mod_names)):
            if mod == 0:
                output += f'_{len(mod_names)}'
            output += f'({mod_positions[mod]+1},{peptide[mod_positions[mod]]},{mkdata.mod_masses.id2row[mod_names[mod]]})'
    if ev is not None:
        output += f'_{ev}'
    return output
    
def generate_mods(peptide, mod_list, n_peptide=False, c_peptide=False, mod_probability=None):
    """
    Given a peptide and a list of modifications expressed as tuples, place the
    allowable modifications on the peptide.

    :param mod_list: the list of allowed modifications, expressed as a string (see encoding.py)
    :param peptide: the peptide
    :param n_peptide: is the peptide at the N terminus of the protein?
    :param c_peptide: is the peptide at the C terminus of the protein?
    :param mod_probability: the probability of a modification at a particular site.  None=1.0
    :return: list of modification name, list of modification positions
    """
    if mod_list is None or peptide is None:
        return [], []

    def add_mod(mod_name_in, mod_pos_in, mod_names_in, mod_positions_in):
        mod_names_in.append(mkdata.mod_masses.df.at[mod_name_in, 'id'])
        mod_positions_in.append(mod_pos_in)
    
    mod_names = []
    mod_positions = []
    length = len(peptide)

    for mod in mod_list:
        if not mod[1].isalpha():
            # N term peptide, no specificity
            if mod[1] == '0' and mod[2] == '0' and (mod_probability is None or random.random() < mod_probability):
                add_mod(mod[0], 0, mod_names, mod_positions)
            # C term peptide, no specificity
            elif mod[1] == '.' and mod[2] == '.' and (mod_probability is None or random.random() < mod_probability):
                add_mod(mod[0], length - 1, mod_names, mod_positions)
            # N or C term protein
            elif n_peptide and mod[1] == '0' and mod[2] == '^' and (mod_probability is None or random.random() < mod_probability):
                add_mod(mod[0], 0, mod_names, mod_positions)
            elif c_peptide and mod[1] == '.' and mod[2] == '$' and (mod_probability is None or random.random() < mod_probability):
                add_mod(mod[0], length - 1, mod_names, mod_positions)

        # specific amino acids
        else:
            for i, aa in enumerate(peptide):
                if aa == mod[1] and (mod_probability is None or random.random() < mod_probability):
                    if mod[2] == "":
                        add_mod(mod[0], i, mod_names, mod_positions)
                    # N or C peptide term specific amino acid mods
                    elif mod[2] == '0' and i == 0:
                        add_mod(mod[0], i, mod_names, mod_positions)
                    elif mod[2] == '.' and i == length-1:
                        add_mod(mod[0], i, mod_names, mod_positions)
                    # N or C protein term specific amino acid mods
                    elif n_peptide and mod[2] == '^' and i == 0:
                        add_mod(mod[0], i, mod_names, mod_positions)
                    elif c_peptide and mod[2] == '$' and i == length-1:
                        add_mod(mod[0], i, mod_names, mod_positions)
    return mod_names, mod_positions

def generate_peptide_library(num=100, min_length=5, max_length=30, min_charge=1, max_charge=8, min_ev=10, max_ev=60,
                             mod_list=None, set='train', mod_probability=0.1):
    """
    Generate a theoretical peptide library

    :param set: which set to create, e.g. train, valid, test
    :param num: the number of peptides
    :param min_length: minimum length of the peptides
    :param max_length: maximum length of the peptides
    :param min_charge: the minimum charge of the peptides
    :param max_charge: the maximum charge of the peptides
    :param min_ev: the minimum eV (also used for nce)
    :param max_ev: the maximum eV (also used for nce)
    :param mod_list: the list of allowed modifications, expressed as a string (see encoding.py)
    :param mod_probability: the probability of a modification at a particular site
    :return: the dataframe
    """

    mod_list = mkencoding.parse_modification_encoding(mod_list)
    data = {'peptide': [], 'peptide_len': [], 'charge': [], 'ev': [], 'nce': [], 'mod_names': [], 'mod_positions': [], 'spectrum': [], 'id': [], 'name':[]}
    for j in range(num):
        length = random.randint(min_length, max_length)
        peptide = ''.join(random.choices(generator_alphabet, k=length))
        data['peptide'].append(peptide)
        data['peptide_len'].append(length)
        charge = random.randint(min_charge, max_charge)
        data['charge'].append(charge)
        ev = random.uniform(min_ev, max_ev)
        data['ev'].append(ev)
        data['nce'].append(ev)
        data['set'] = set

        # go through list of allowed modifications
        mod_names, mod_positions = generate_mods(peptide, mod_list, mod_probability=mod_probability)

        data['mod_names'].append(mod_names)
        data['mod_positions'].append(mod_positions)
        name = create_peptide_name(peptide, charge, mod_names, mod_positions, ev)
        data['spectrum'].append(mktheoretical_spectrum.TheoreticalPeptideSpectrum(peptide, charge=charge,
                                                           ion_types=[('b', 1), ('y',1)], mod_names=mod_names,
                                                           mod_positions=mod_positions, id=j, 
                                                           name=name, ev=ev, nce=ev))
        data['id'].append(j)
        data['name'].append(name)
        if j % 100 == 0:
            logging.info(f'creating theoretical spectrum {j}')
    df = pd.DataFrame.from_dict(data)
    df.index.name = "id"
    return df


def add_theoretical_spectra(df, theoretical_spectrum_column=None, ion_types=None, num_isotopes=2):
    """
    add theoretical spectra to a column

    :param df: dataframe containing spectra
    :param theoretical_spectrum_column:  name of the column to hold theoretical spectra
    :param ion_types: ion types to generate. None is default for TheoreticalPeptideSpectrum
    :param num_isotopes: number of c-13 isotopes to calculate
    """
    if theoretical_spectrum_column is None:
        theoretical_spectrum_column = 'theoretical_spectrum'

    df[theoretical_spectrum_column] = None

    for j in range(len(df.index)):
        charge = df['charge'].iat[j]
        peptide = df['peptide'].iat[j]
        mod_names = df['mod_names'].iat[j]
        mod_positions = df['mod_positions'].iat[j]
        df[theoretical_spectrum_column].iat[j] = mktheoretical_spectrum.TheoreticalPeptideSpectrum(peptide,
                                                                            ion_types=ion_types,
                                                                            charge=charge,
                                                                            mod_names=mod_names,
                                                                            mod_positions=mod_positions,
                                                                            analysis_annotations=True,
                                                                            num_isotopes=num_isotopes)

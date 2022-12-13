import logging
import random
import pandas as pd

from masskit.peptide.encoding import calc_ions_mz, mod_sites, mod_masses
from masskit.spectrum.theoretical_spectrum import TheoreticalPeptideSpectrum


"""
The amino acids used to generate the theoretical peptides
"""
generator_alphabet = 'ARNDCQEGHILKMFPSTWYV'

def create_peptide_name(peptide, precursor_charge, mod_names, mod_positions, ev):
    """_
    create the name of a peptide spectrum

    :param peptide: the peptide string
    :param precursor_charge: the precursor charge
    :param mod_names: list of modification names (integer)
    :param mod_positions: position of modifications, 0 based
    :param ev: collision energy in ev
    """
    output = peptide + '/' + str(precursor_charge)
    for mod in range(len(mod_names)):
        if mod == 0:
            output += f'_{len(mod_names)}'
        output += f'({mod_positions[mod]+1},{peptide[mod_positions[mod]]},{mod_names[mod]})'
    output += f'_eV{ev}'
    return output
    

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
    :param mod_list: the list of allowed modifications
    :param mod_probability: the probability of a modification at a particular site
    :return: the dataframe
    """

    data = {'peptide': [], 'peptide_len': [], 'charge': [], 'ev': [], 'nce': [], 'mod_names': [], 'mod_positions': [], 'spectrum': [], 'id': [], 'name':[]}
    for j in range(num):
        length = random.randint(min_length, max_length)
        peptide = ''.join(random.choices(generator_alphabet, k=length))
        data['peptide'].append(peptide)
        data['peptide_len'].append(len(peptide))
        charge = random.randint(min_charge, max_charge)
        data['charge'].append(charge)
        ev = random.uniform(min_ev, max_ev)
        data['ev'].append(ev)
        data['nce'].append(ev)
        data['set'] = set
        mod_names = []
        mod_positions = []
        # go through list of allowed modifications
        for mod in mod_list:
            # for each mod, go through allowed sites
            for site in mod_sites[mod]['sites']:
                # N term
                if site == '0' and random.random() < mod_probability:
                    mod_names.append(mod_masses.df.at[mod, 'id'])
                    mod_positions.append(0)
                # C term
                elif site == '-1'and random.random() < mod_probability:
                    mod_names.append(mod_masses.df.at[mod, 'id'])
                    mod_positions.append(length - 1)
                # specific amino acids
                else:
                    for i, aa in enumerate(peptide):
                        if aa == site and random.random() < mod_probability:
                            mod_names.append(mod_masses.df.at[mod, 'id'])
                            mod_positions.append(i)
        data['mod_names'].append(mod_names)
        data['mod_positions'].append(mod_positions)
        name = create_peptide_name(peptide, charge, mod_names, mod_positions, ev)
        data['spectrum'].append(TheoreticalPeptideSpectrum(peptide, charge=charge,
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
        df[theoretical_spectrum_column].iat[j] = TheoreticalPeptideSpectrum(peptide,
                                                                            ion_types=ion_types,
                                                                            charge=charge,
                                                                            mod_names=mod_names,
                                                                            mod_positions=mod_positions,
                                                                            analysis_annotations=True,
                                                                            num_isotopes=num_isotopes)

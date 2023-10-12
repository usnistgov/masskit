import string
from collections import OrderedDict

import numpy as np
import pyarrow as pa

from .. import data as mkdata
from ..data_specs import schemas as mkschemas


def parse_ion_type_tuple(tuple_in, precursor_charge):
    """
    split ion_type tuple into ion type and neutral loss, if specified

    :param tuple_in: ion type tuple
    :raises ValueError: more than one neutral loss
    :return: ion type, neutral loss
    """
    charge = tuple_in[1]
    if charge == "z":
        charge = int(precursor_charge)
    if not isinstance(charge, int):
        raise ValueError("Non integer charge")
    split_type = tuple_in[0].split('-')
    if len(split_type) == 1:
        return split_type[0], None, charge
    elif len(split_type) == 2:
        return split_type[0], split_type[1], charge
    else:
        raise ValueError("Too many neutral losses specified")

def calc_ion_series(ion_type, num_isotopes, cumulative_masses, arrays, peptide, mod_names, mod_positions, neutral_loss, charge_in,
                    analysis, positions, start_offset=0, max_internal_size=7):
    
    # extract ion type info column by column to preserve type
    ion_type_start = mkdata.ion_types.df['start'].loc[ion_type]
    ion_type_stop = mkdata.ion_types.df['stop'].loc[ion_type]
    ion_type_direction = mkdata.ion_types.df['direction'].loc[ion_type]
    ion_type_id = mkdata.ion_types.df['id'].loc[ion_type]
    ion_type_offset = mkdata.ion_types.df['offset'].loc[ion_type]
    
    # tack on neutral mass if specified
    if neutral_loss is None:
        neutral_loss_offset = 0.0
        neutral_loss_id = None
    else:
        neutral_loss_offset = mkdata.named_ions.df['offset'].loc[neutral_loss]
        neutral_loss_id = mkdata.named_ions.df['id'].loc[neutral_loss]
    
    # if there is a start offset (usually for internal ions) calculate the mass offset
    if start_offset != 0:
        start_offset_mass = cumulative_masses[ion_type_direction][ion_type_start:ion_type_stop:ion_type_direction][start_offset-1]
    else:
        start_offset_mass = 0.0
        
    # if an internal ion, set the maximum size for an internal ion
    if mkdata.ion_types.df['is_internal'].loc[ion_type]:
        end_offset = start_offset + max_internal_size
    else:
        end_offset = None
    
    # iterate through the number of carbon 13
    for num_carbon_13 in range(num_isotopes):
        # add in offset for particular ion series and slice cumulative masses for correct ion series direction
        # also subtract off the offset for internal ions
        mz = ion_type_offset + cumulative_masses[ion_type_direction][ion_type_start:ion_type_stop:ion_type_direction][start_offset:end_offset] \
             - neutral_loss_offset - start_offset_mass
        mz += num_carbon_13 * mkdata.delta_c_13

        arrays['ion_mz'].append(protonate_mass(mz, charge_in))
        arrays['ion_intensity'].append(np.full_like(mz, 999.0))
        arrays['ion_type_array'].append(np.full_like(mz, ion_type_id))
        arrays['charge_array'].append(np.full_like(mz, charge_in))
        arrays['isotope_array'].append(np.full_like(mz, num_carbon_13))
        # pos is the number in the ion series (1 based)
        if neutral_loss_id is None:
            arrays['ion_subtype_array'].append(pa.array(np.zeros(len(mz)), mask=np.ones(len(mz), dtype=np.bool_), type=pa.int32()))
        else:
            arrays['ion_subtype_array'].append(pa.array(np.full_like(mz, neutral_loss_id), type=pa.int32()))
        # end position of ion (used for internal fragments)
        if mkdata.ion_types.df['is_internal'].loc[ion_type]:
            arrays['position_array'].append(pa.array(np.full_like(mz, positions[ion_type_direction][ion_type_start:ion_type_stop:ion_type_direction][start_offset] - 1), type=pa.uint16()))
            arrays['end_position_array'].append(pa.array(positions[ion_type_direction][ion_type_start:ion_type_stop:ion_type_direction][start_offset:start_offset + len(mz)], type=pa.uint16()))
        else:
            arrays['position_array'].append(pa.array(positions[ion_type_direction][ion_type_start:ion_type_stop:ion_type_direction][start_offset: start_offset + len(mz)], type=pa.uint16()))
            arrays['end_position_array'].append(pa.array(np.zeros(len(mz)), mask=np.ones(len(mz), dtype=np.bool_), type=pa.uint16()))
        if analysis is not None:
            aa_after = np.zeros(len(mz))
            aa_after_mask = np.ones(len(mz), dtype=np.bool_)
            aa_before = np.zeros(len(mz))
            aa_before_mask = np.ones(len(mz), dtype=np.bool_)
            ptm_before = np.zeros(len(mz))
            ptm_before_mask = np.ones(len(mz), dtype=np.bool_)
            ptm_after = np.zeros(len(mz))
            ptm_after_mask = np.ones(len(mz), dtype=np.bool_)
            for i in range(len(mz)):
                # todo: need to add starting offset to negative direction.  
                if ion_type_direction == 1:
                    before_cleavage = i + ion_type_start
                    after_cleavage = i + 1 + ion_type_start
                else:
                    before_cleavage = i - 1 + ion_type_stop
                    after_cleavage = i + ion_type_stop
                if after_cleavage < len(peptide):
                    aa_after[i] = ord(peptide[after_cleavage]) - ord('A')
                    aa_after_mask[i] = 0
                if before_cleavage > 0:
                    aa_before[i] = ord(peptide[before_cleavage]) - ord('A')
                    aa_before_mask[i] = 0

                # add in ptms if they exist
                if mod_positions is not None:
                    mod_index = np.where(mod_positions == before_cleavage)
                    if len(mod_index[0]):
                        ptm_before[i] = mod_names[mod_index[0][0]]
                        ptm_before_mask[i] = 0
                    mod_index = np.where(mod_positions == after_cleavage)
                    if len(mod_index[0]):
                        ptm_after[i] = mod_names[mod_index[0][0]]
                        ptm_after_mask[i] = 0
            analysis['aa_before_array'].append(pa.array(aa_before, type=pa.int32(), mask=aa_before_mask))
            analysis['aa_after_array'].append(pa.array(aa_after, type=pa.int32(), mask=aa_after_mask))
            analysis['ptm_before_array'].append(pa.array(ptm_before, type=pa.int32(), mask=ptm_before_mask))
            analysis['ptm_after_array'].append(pa.array(ptm_after, type=pa.int32(), mask=ptm_after_mask))


def calc_named_ions(arrays, analysis=None, named_ion=None, precursor_mass=None, precursor_charge=None,
                    charge_in=None, neutral_loss=None, num_isotopes=2):
    if named_ion is None:
        return
    ion_df = mkdata.named_ions.df[mkdata.named_ions.df['ion_type'] == named_ion]

    if len(ion_df) != 0:
        ion_subtype = ion_df['id'].to_numpy(na_value=np.nan)
        offset = ion_df['offset'].to_numpy(na_value=np.nan)
        ion_subtype_mask = None
    else:
        ion_subtype = np.zeros((1,))
        ion_subtype_mask = np.ones((1,), dtype=np.bool_)
        offset = np.array([mkdata.ion_types.df.loc[named_ion, 'offset']])

    if neutral_loss is not None:
        neutral_loss_offset = np.array([mkdata.named_ions.df.loc[neutral_loss, 'offset']])
        ion_subtype = np.array([mkdata.named_ions.df.loc[neutral_loss, 'id']])
        ion_subtype_mask = None
    else:
        neutral_loss_offset = np.zeros((1,))
      
    # don't bother doing C-13 isotopes for some ion types
    if not mkdata.ion_types.df.at[named_ion, 'calc_c_13']:
        num_isotopes = 1

    for num_carbon_13 in range(num_isotopes):
        offsets = offset - neutral_loss_offset + num_carbon_13 * mkdata.delta_c_13
        if named_ion == 'parent':
            offsets = protonate_mass(precursor_mass + offsets, precursor_charge)

        arrays['ion_mz'].append(offsets)
        arrays['ion_intensity'].append(np.full_like(offsets, 999.0))
        arrays['ion_type_array'].append(np.full_like(offsets, mkdata.ion_types.df.loc[named_ion]['id']))
        arrays['charge_array'].append(np.full_like(offsets, charge_in))
        arrays['isotope_array'].append(np.full_like(offsets, num_carbon_13))
        arrays['ion_subtype_array'].append(pa.array(ion_subtype, type=pa.int32(), mask=ion_subtype_mask))
        arrays['position_array'].append(pa.array(np.zeros_like(offsets), mask=np.ones_like(offsets, dtype=np.bool_), type=pa.uint16()))
        arrays['end_position_array'].append(pa.array(np.zeros_like(offsets), mask=np.ones_like(offsets, dtype=np.bool_), type=pa.uint16()))
        if analysis is not None:
            analysis['aa_before_array'].append(pa.array(np.zeros_like(offsets), mask=np.ones_like(offsets, dtype=np.bool_), type=pa.int32()))
            analysis['aa_after_array'].append(pa.array(np.zeros_like(offsets), mask=np.ones_like(offsets, dtype=np.bool_), type=pa.int32()))
            analysis['ptm_before_array'].append(pa.array(np.zeros_like(offsets), mask=np.ones_like(offsets, dtype=np.bool_), type=pa.int32()))
            analysis['ptm_after_array'].append(pa.array(np.zeros_like(offsets), mask=np.ones_like(offsets, dtype=np.bool_), type=pa.int32()))

    
def mod_mass_pos(mod_positions, mod_names, i):
    """
    at a given pos in the sequence, find any matching modification positions in mod_positions
    and sum up the masses of the modifications

    :param mod_positions: mod positions
    :param mod_names: mod names
    :param i: position in peptide
    :return: masses of matching modifications
    """

    ret_value = 0.0
    mod_index = np.where(mod_positions == i)
    # add in the masses of any modifications that match the position
    for j in range(len(mod_index[0])):
        row_index = mkdata.mod_masses.id2row[mod_names[mod_index[0][j]]]
        ret_value += mkdata.mod_masses.df.at[row_index, 'mono_mass']
    return ret_value


def calc_precursor_mz(peptide, charge, mod_names=None, mod_positions=None):
    """
    calculate m/z of modified peptide
    
    :param peptide: the peptide
    :param charge: the charge of the peptide
    :param mod_names: the modification ids
    :param mod_positions: the positions of the modifications
    :return: the mass
    """
    return protonate_mass(calc_precursor_mass(peptide, mod_names=mod_names, mod_positions=mod_positions), charge)


def calc_precursor_mass(peptide, mod_names=None, mod_positions=None):
    """
    calculate mass of modified peptide
    
    :param peptide: the peptide
    :param mod_names: the modification ids
    :param mod_positions: the positions of the modifications
    :return: the mass
    """
    ret_value = 0.0
    mod_positions = np.array(mod_positions)
    for i in range(len(peptide)):
        # the mass of any matching modification, for forward ion series and reverse ion series
        ret_value += mkdata.aa_masses[peptide[i]]['mono_mass']
        if mod_names is not None:
            ret_value += mod_mass_pos(mod_positions, mod_names, i)

    # TODO: h2o_mass may not be correct if there are blocking PTMs at N or C terminus
    ret_value += mkdata.h2o_mass      
    return ret_value


def calc_ions_mz(peptide, ion_types_in, mod_names=None, mod_positions=None,
                 analysis_annotations=False, precursor_charge=2, num_isotopes=2, max_internal_size=7):
    """
    calculate the mz values of an ion type
    default values are taken from the HCD values in https://pubs.acs.org/doi/full/10.1021/pr3007045

    :param peptide: the peptide sequence
    :param ion_types_in: tuple or array of tuple of ion type and charge
    :param mod_names: any modifications
    :param mod_positions: the positions of the modifications
    :param analysis_annotations: add additional annotations useful for analyzing spectra
    :param precursor_charge: used to filter out ion types with charge greater than the precursor
    :param num_isotopes: number of carbon 13 isotopes to calculate
    :return: a numpy arrays of the mz values for the ion series, ion intensities, annotations as an arrow list, precursor mass, fields used for analysing ion peaks
    """
    # if not array like, make into list
    if not hasattr(ion_types_in, "__len__") or (type(ion_types_in) == tuple and type(ion_types_in[0]) != tuple):
        ion_types_in = [ion_types_in]

    arrays = OrderedDict([
        ("ion_type_array", []),
        ("charge_array", []),
        ("isotope_array", []),
        ("ion_subtype_array", []),
        ("position_array", []),
        ("end_position_array", []),
        ("ion_mz", []),
        ("ion_intensity", []),
   ])

    # analysis annotations
    if analysis_annotations:
        analysis = OrderedDict([
            ("aa_before_array", []),
            ("aa_after_array", []),
            ("ptm_before_array", []),
            ("ptm_after_array", []),
        ])
    else:
        analysis = None

    # peptide mass at which monoisotopic and one carbon 13 peaks have some abundance: 1446.94
    # begin by generating cumulative mass arrays that can be used for all ion series except immonium
    peptide_len = len(peptide)
    # dictionary that contains the forward (N to C) mass ladder (1) and the reverse (C to N) mass ladder (-1)
    cumulative_masses = {1: np.zeros(peptide_len), -1: np.zeros(peptide_len)}
    cumulative_masses[1][0] = mkdata.aa_masses[peptide[0]]['mono_mass'] + mod_mass_pos(mod_positions, mod_names, 0)
    # note that the reverse cumulative masses are still in N to C order
    cumulative_masses[-1][-1] = mkdata.aa_masses[peptide[-1]]['mono_mass']  + mod_mass_pos(mod_positions, mod_names, peptide_len - 1)
    # ion position ordering for both forward and reverse.  starts with 1
    positions = {1: np.arange(1, peptide_len + 1), -1: np.arange(peptide_len, 0, -1)}

    for i in range(1, peptide_len):
        # the mass of any matching modification, for forward ion series and reverse ion series
        mod_mass_forward = 0.0
        mod_mass_reverse = 0.0

        if mod_positions is not None:
            # add in the masses of any modifications that match the position
            mod_mass_forward += mod_mass_pos(mod_positions, mod_names, i)
            mod_mass_reverse += mod_mass_pos(mod_positions, mod_names, peptide_len - i - 1)

        cumulative_masses[1][i] = cumulative_masses[1][i-1] + mkdata.aa_masses[peptide[i]]['mono_mass'] + mod_mass_forward
        cumulative_masses[-1][peptide_len - i - 1] = cumulative_masses[-1][peptide_len - i] \
            + mkdata.aa_masses[peptide[peptide_len - i - 1]]['mono_mass'] + mod_mass_reverse
    
    # add on the last bit of the reverse cumulative mass array to get last AA plus modifications
    # todo: h2o_mass may not be correct if there are blocking PTMs at N or C terminus
    precursor_mass = cumulative_masses[-1][0] + mkdata.h2o_mass

    for ion_type_in in ion_types_in:
        ion_type, neutral_loss, charge_in = parse_ion_type_tuple(ion_type_in, precursor_charge)

        if ion_type == "immonium" or ion_type == "parent":
            calc_named_ions(arrays, analysis, ion_type, precursor_mass, precursor_charge, charge_in, neutral_loss, num_isotopes)
        else:
            # skip product charges that exceed the precursor charge
            if charge_in > precursor_charge:
                continue
            if mkdata.ion_types.df['is_internal'].loc[ion_type]:
                start = 1
                stop = peptide_len - 1
            else:
                start = 0
                stop = 1
            for start_offset in range(start, stop):
                calc_ion_series(ion_type, num_isotopes, cumulative_masses, arrays, peptide, mod_names, mod_positions, neutral_loss, charge_in,
                                analysis, positions, start_offset=start_offset, max_internal_size=max_internal_size)

    arrays['ion_mz'] = np.concatenate(arrays['ion_mz'])
    arrays['ion_intensity'] = np.concatenate(arrays['ion_intensity'])
    arrays['ion_type_array'] = np.concatenate(arrays['ion_type_array'])
    arrays['ion_subtype_array'] = pa.concat_arrays(arrays['ion_subtype_array'])
    arrays['charge_array'] = np.concatenate(arrays['charge_array'])
    arrays['isotope_array'] = np.concatenate(arrays['isotope_array'])
    arrays['position_array'] = pa.concat_arrays(arrays['position_array'])
    arrays['end_position_array'] = pa.concat_arrays(arrays['end_position_array'])
    
    # create the array
    arrays['ion_type_array'] = pa.DictionaryArray.from_arrays(
        indices=pa.array(arrays['ion_type_array'], type=pa.int32()),
        dictionary=mkdata.ion_types.dictionary)
    arrays['charge_array'] = pa.array(arrays['charge_array'], type=pa.int16())
    arrays['isotope_array'] = pa.array(arrays['isotope_array'], type=pa.uint8())
    arrays['ion_subtype_array'] = pa.DictionaryArray.from_arrays(
        indices=arrays['ion_subtype_array'],
        dictionary=mkdata.named_ions.dictionary)
    fields = mkschemas.ion_annot_fields[0:6]
    arrays_out = list(arrays.values())[0:6] 
    if analysis is not None:
        analysis['aa_before_array'] = pa.DictionaryArray.from_arrays(
            indices=pa.concat_arrays(analysis['aa_before_array']),
            dictionary=pa.array(string.ascii_uppercase))
        analysis['aa_after_array'] = pa.DictionaryArray.from_arrays(
            indices=pa.concat_arrays(analysis['aa_after_array']),
            dictionary=pa.array(string.ascii_uppercase))
        analysis['ptm_before_array'] = pa.DictionaryArray.from_arrays(
            indices=pa.concat_arrays(analysis['ptm_before_array']),
            dictionary=mkdata.mod_masses.dictionary)
        analysis['ptm_after_array'] = pa.DictionaryArray.from_arrays(
            indices=pa.concat_arrays(analysis['ptm_after_array']),
            dictionary=mkdata.mod_masses.dictionary)
        fields += mkschemas.ion_annot_fields[6:]
        arrays_out += list(analysis.values())

    annotations = pa.StructArray.from_arrays(arrays=arrays_out, fields=fields)
    return arrays['ion_mz'], arrays['ion_intensity'], annotations, precursor_mass


def protonate_mass(mass, z):
    """
    Given a neutral mass and charge of an ion, calculate the m/z of the ion

    :param mass: mass
    :param z: charge
    :return: m/z
    """
    return (mass + z * mkdata.atom_masses['p']) / z


def expand_mod_string(mod_string):
    """
    decode modification string into site and position
    
    :param mod_string: the standard modification string, e.g. "A" or "A0" or "$"
    :return: tuple of site, position
    """
    if len(mod_string) == 2:
        return (mod_string[0], mod_string[1])
    elif mod_string == "0":
        return ("0","0")
    elif mod_string == ".":
        return (".",".")
    elif mod_string == "^":
        return ("0","^")
    elif mod_string == "$":
        return (".","$")
    else:
        return (mod_string[0], "")

def parse_modification_encoding(modification_encoding):
    """
    Takes a string containing a set of modification strings and creates a list of tuples.
    The tuples contain the modification name, the site, and the position of the modification.
    The string has the following format:

    Site encoding of a modification:
    A-Y amino acid

    which can be appended with a modification position encoding:
    0 peptide N-terminus
    . peptide C-terminus
    ^ protein N-terminus
    $ protein C-terminus

    So that 'K.' means lysine at the C-terminus of the peptide.
    The position encoding can be used separately, e.g. '^' means apply to any protein N-terminus,
    regardless of amino acid

    A list of modifications is separated by hashes:
    Phospho{S}#Methyl{0/I}#Carbamidomethyl#Deamidated{F^/Q/N}

    An optional list of sites is specified within the {} for each modification. 
    If there are no '{}' then a default set of sites is used.  
    Multiple sites are separated by a '/'.

    "0" by itself implies "00"
    "." by itself implies ".."
    "^" by itself implies "0^"
    "$" by itself implies ".$"

    :param modification_encoding: a string containing the above format
    :return: list of tuples, each tuple has modification name, site, and position
    """
    ret_values = []
    # take out any spaces
    modification_encoding = modification_encoding.replace(" ", "")
    # Check for empty list
    if modification_encoding == '':
        return ret_values   
    # split by hashes (commas confuse hydra)
    mod_list = modification_encoding.split('#')
    # split by curly brace
    for mod_string in mod_list:
        mod_string = mod_string.replace("}", '')
        mod_substrings = mod_string.split("{")
        # use default if nothing provided
        if len(mod_substrings) < 2:
            values = mkdata.mod_masses.df['default_sites'].loc[mod_substrings[0]]
        else:
            # split by forward slash
            values = mod_substrings[1].split('/')

        for value in values:
            value = expand_mod_string(value)
            ret_values.append((mod_substrings[0], *value))
        
    # return list of tuples where tuples are 
    # (mod, site, position)
    return ret_values



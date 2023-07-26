import copy
import csv
from functools import partial
from io import StringIO
import logging
from multiprocessing import Pool
import os
from pathlib import Path
import pkgutil
import re
from typing import Dict, List, Union
import masskit.small_molecule
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import csv as pacsv
import json
from omegaconf import DictConfig, ListConfig, OmegaConf
from masskit.constants import SET_NAMES
from masskit.data_specs.schemas import mod_names_field, \
    hitlist_schema, table2structarray, table_add_structarray
from masskit.data_specs.file_schemas import schema_groups
from masskit.data_specs.arrow_types import MolArrowType, SpectrumArrowType
from masskit.peptide.encoding import mod_masses
from masskit.small_molecule import threed, utils
from masskit.spectrum.spectrum import Spectrum
from masskit.utils.spectrum_writers import spectra_to_mgf, spectra_to_msp
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions, EnumerateStereoisomers
    from rdkit import RDLogger
except ImportError:
    pass
from masskit.utils.fingerprints import ECFPFingerprint
from masskit.utils.general import open_if_filename
from masskit.utils.hitlist import Hitlist
import rich.progress as rprogress


float_match = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'  # regex used for matching floating point numbers

# convert mspepsearch column names to standard search results
# do not include 'uNIST r.n.': 'query_id', 'NIST r.n.': 'hit_id' as this is decided by program logic
pepsearch_names = {'Unknown': 'query_name', 'Peptide': 'hit_name', 'Dot Product': 'cosine_score'}

# types of various mspepsearch columns
pepsearch_types = {'uNIST r.n.': int, 'NIST r.n.': int, 'Unknown': str, 'Charge': int,
                   'Peptide': str, 'Dot Product': float, 'Id': int, 'Num': int, 'Rank': int}


def add_row_to_records(records, row):
    for i in records.keys():
        records[i].append(row.get(i))


def empty_records(schema):
    return schema.empty_table().to_pydict()

import io
def seek_size(fp):
    pos = fp.tell()
    fp.seek(0, io.SEEK_END)
    size = fp.tell()
    fp.seek(pos) # back to where we were
    return size


def get_progress(fp):
    """
    Given a fp pointer, return best possible progress meter

    :param fp: name of the file to read or a file object
    """
    if fp.seekable:
        prog = rprogress.Progress(
            rprogress.TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            rprogress.BarColumn(bar_width=None),
            "[prog.percentage]{task.percentage:>3.1f}%",
            "•",
            rprogress.DownloadColumn(),
            "•",
            rprogress.TransferSpeedColumn(),
            "•",
            rprogress.TimeRemainingColumn(),
            transient=False,
            #console=global_console,
        )
        total=seek_size(fp)
    else:
        prog = rprogress.Progress(
            rprogress.TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            rprogress.SpinnerColumn(),
            rprogress.TimeElapsedColumn(),
        )
        total = 0
    return prog, total


def write_parquet(fp, table):
    """
    save a PyArrow table to a parquet file.

    :param table: the dataframe
    :param fp: stream or filename
    """
    fp = open_if_filename(fp, 'wb')
    pq.write_table(table, where=fp, row_group_size=5000)


def read_parquet(fp, columns=None, num=None, filters=None):
    """
    reads a PyArrow table from a parquet file.

    :param fp: stream or filename
    :param columns: list of columns to remy_sdf.parquetad, None=all
    :param filters: parquet predicate as a list of tuples
    :return: PyArrow table
    """
    fp = open_if_filename(fp, 'rb')
    if type(columns) == ListConfig:
        columns = list(columns)
    metadata = pq.read_metadata(fp)
    # if columns is not None:
    #     columns = [x for x in metadata.schema.names if x in columns]
    table = pq.read_table(fp, columns=columns, filters=filters)
    if num is None:
        return table
    else:
        return table.slice(0, num)


def spectra_to_array(spectra,
                     min_intensity=0,
                     write_starts_stops=False,
                     schema_group=None
                     ):
    """
    convert an array-like of spectra to an arrow_table

    :param spectra: iteratable containing the spectrums
    :param min_intensity: the minimum intensity to set the fingerprint bit
    :param write_starts_stops: put the starts and stops arrays into the arrow Table
    :param schema_group: the schema group of spectrum file
    :return: arrow table of spectra
    """
    if schema_group is None:
        schema_group = 'peptide'
    
    records_schema = schema_groups[schema_group]['flat_schema']
    records = empty_records(records_schema)

    for s in spectra:
        row = {}
        row['id'] = s.id
        row['charge'] = s.charge
        row['ev'] = s.ev
        row['instrument'] = s.instrument
        row['instrument_type'] = s.instrument_type
        row['instrument_model'] = s.instrument_model
        row['ion_mode'] = s.ion_mode
        row['ionization'] = s.ionization
        row['name'] = s.name
        row['scan'] = s.scan
        row['nce'] = s.nce
        row['intensity'] = s.products.intensity
        # row['stddev'] = s.stddev
        row['product_massinfo'] = s.product_mass_info.__dict__
        row['mz'] = s.products.mz
        if write_starts_stops:
            row['starts'] = s.products.starts
            row['stops'] = s.products.stops
        row['precursor_intensity'] = s.precursor.intensity
        # Temp hack because PyArrow can't convert empty structs to pandas
        if (s.precursor_mass_info):
            row['precursor_massinfo'] = s.precursor_mass_info.__dict__
        else:
            row['precursor_massinfo'] = s.product_mass_info.__dict__
        row['precursor_mz'] = s.precursor.mz
        fingerprint = s.filter(min_intensity=min_intensity).create_fingerprint(max_mz=2000)
        row['spectrum_fp'] = fingerprint.to_numpy()
        row['spectrum_fp_count'] = fingerprint.get_num_on_bits()
        row['peptide'] = s.peptide
        row['peptide_len'] = s.peptide_len
        row['protein_id'] = s.protein_id
        row['mod_names'] = s.mod_names
        row['mod_positions'] = s.mod_positions
        add_row_to_records(records, row)
    table = pa.table(records, records_schema)
    return table


# NIST modification names to unimod names
# note the the NIST mod "TMT" does not match the unimod "TMT", rather "TMT6plex"
nist_mod_2_unimod = { 
    "Oxidation": "Oxidation",
    "Carbamidomethyl ": "Carbamidomethyl",
    "ICAT_light": "ICAT-C",
    "ICAT_heavy": "ICAT-C:13C(9)",
    "AB_old_ICATd0": "ICAT-D",
    "AB_old_ICATd8": "ICAT-D:2H(8)",
    "Acetyl": "Acetyl",
    "Deamidation": "Deamidated",
    "Pyro-cmC": "Pyro-carbamidomethyl",
    "Pyro-glu": "Gln->pyro-Glu",
    "Pyro_glu": "Glu->pyro-Glu",
    "Amide": "Amidated",
    "Phospho": "Phospho",
    "Methyl": "Methyl",
    "Carbamyl": "Carbamyl",
    'CAM': "Carbamidomethyl",
    'iTRAQ': "iTRAQ4plex",
    'TMT6ple': "TMT6plex",
    'TMT': 'TMT6plex',
}


class BatchLoader:
    def __init__(self, file_type, row_batch_size=5000, format=None, num=None):
        """
        initialize stream batch loader
        
        :param row_batch_size: the number of rows for a single batch
        :param file_type: the type of file, e.g. mgf, msp, csv, etc.
        :param format: the specific format of the file_type, e.g. msp_mol, msp_peptide, sdf_mol, sdf_nist_mol, sdf_pubchem_mol. If none, use default.
        :param num: number of records to read per batch
        """
        self.row_batch_size = row_batch_size
        self.num = num
        if format is None:
            self.format = load_default_config_schema(file_type)
        else:
            self.format = format

        self.schema_group = schema_groups[self.format['schema_group']]


    def setup(self):
        """
        set up loading
        """
        self.tables = []
        self.records = empty_records(self.schema_group['flat_schema'])

        if self.format['id']['field_type'] == 'int':
            self.current_id = self.format['id']['initial_value']
        else:
            self.current_id = 0

    def loop_end(self):
        """
        end of loop to read in one record
        """
        if not self.format['id']['field']:
            self.current_id += 1

        if len(self.records["id"]) % 25000 == 0:
            self.tables.append(records2table(self.records, self.schema_group))
            logging.info(f"created chunk {len(self.tables)} with {len(self.records['id'])} records")
            self.records = empty_records(self.schema_group['flat_schema'])

    def finalize(self):
        """
        finalized the chunk
        """
        if self.records:
            self.tables.append(records2table(self.records, self.schema_group))
        logging.info(f"created chunk {len(self.tables)} with {len(self.records['id'])} records")
        table = pa.concat_tables(self.tables)
        return table
    
    def load(self, fp):
        """
        read in a chunk from a stream
        
        :param fp: the stream
        """
        raise NotImplementedError
    
    # fingerprint generator
    ecfp4 = ECFPFingerprint()
    # smarts for tms for substructure matching
    tms = Chem.MolFromSmarts("[#14]([CH3])([CH3])[CH3]")
    # only enumerate unique stereoisomers
    opts = StereoEnumerationOptions(unique=True)
    # fingerprint generator that respects counts but not chirality as mass spec tends to be chirality blind

    @classmethod
    def mol2row(cls, mol, max_size:int=0, skip_computed_props=True, skip_expensive:bool=True) -> Dict:
        """
        Convert an rdkit Mol into a row
        
        :param mol: the molecule
        :param max_size: the maximum bounding box size (used to filter out large molecules. 0=no bound)
        :param skip_computed_props: skip computing properties
        :param skip_expensive: skip computing computationally expensive properties
        :return: row as dict, mol
        """
        if mol is None or mol.GetNumAtoms() < 1:
            logging.info(f"Unable to use molecule object")
            return {}, mol

        try:
            mol = masskit.small_molecule.utils.standardize_mol(mol)
        except ValueError as e:
            logging.info(f"Unable to standardize")
            return {}, mol
        
        if len(mol.GetPropsAsDict()) == 0:
            logging.info(f"All molecular props unavailable")

        new_row = {
            "mol": Chem.rdMolInterchange.MolToJSON(mol)
        }
        
        if not skip_computed_props:
            # calculate some identifiers before adding explicit hydrogens.  In particular, it seems that rdkit
            # ignores stereochemistry when creating the inchi key if you add explicit hydrogens

            new_row["has_2d"] = True
            new_row["inchi_key"] = Chem.inchi.MolToInchiKey(mol)
            new_row["isomeric_smiles"] = Chem.MolToSmiles(mol)
            new_row["smiles"] = Chem.MolToSmiles(mol, isomericSmiles=False)

            if not skip_expensive:
                mol, conformer_ids, return_value = threed.create_conformer(mol)
                if return_value == -1:
                    logging.info(f"Not able to create conformer")
                    return {}, mol
                # do not call Chem.AllChem.Compute2DCoords(mol) after this point as it will erase the 3d conformers

                # calculation MMFF94 partial charges
                partial_charges = []
                try:
                    mol_copy = copy.deepcopy(
                        mol
                    )  # the MMFF calculation sanitizes the molecule
                    fps = AllChem.MMFFGetMoleculeProperties(mol_copy)
                    if fps is not None:
                        for atom_num in range(0, mol_copy.GetNumAtoms()):
                            partial_charges.append(fps.GetMMFFPartialCharge(atom_num))
                except ValueError:
                    logging.info(f"unable to run MMFF")
                new_row["num_conformers"] = len(conformer_ids)
                new_row["partial_charges"] = partial_charges
                bounding_box = threed.bounding_box(mol)
                new_row["min_x"] = bounding_box[0, 0]
                new_row["max_x"] = bounding_box[0, 1]
                new_row["min_y"] = bounding_box[1, 0]
                new_row["max_y"] = bounding_box[1, 1]
                new_row["min_z"] = bounding_box[2, 0]
                new_row["max_z"] = bounding_box[2, 1]
                new_row["has_conformer"] = True
                new_row["max_bound"] = np.max(np.abs(bounding_box))
                if max_size != 0 and new_row["max_bound"] > max_size:
                    logging.info(f"larger than the max bound")
                    return {}, mol
                try:
                    new_row["num_stereoisomers"] = len(
                        tuple(EnumerateStereoisomers(mol, options=cls.opts))
                    )  # GetStereoisomerCount(mol)
                except RuntimeError as err:
                    logging.info(
                        f"Unable to create stereoisomer count, error = {err}"
                    )
                    new_row["num_stereoisomers"] = None
            else:
                # Chem.AssignStereochemistry(mol)  # normally done in threed.create_conformer
                new_row["has_conformer"] = False

            # calculate solvent accessible surface area per atom
            # try:
            #     radii = []
            #     for atom in mol.GetAtoms():
            #         radii.append(utils.symbol_radius[atom.GetSymbol().upper()])
            #     rdFreeSASA.CalcSASA(mol, radii=radii)
            # except:
            #     logging.info("unable to create sasa")
            
            new_row["has_tms"] = len(
                mol.GetSubstructMatches(cls.tms)
            )  # count of trimethylsilane matches
            new_row["exact_mw"] = Chem.rdMolDescriptors.CalcExactMolWt(mol)
            new_row["hba"] = Chem.rdMolDescriptors.CalcNumHBA(mol)
            new_row["hbd"] = Chem.rdMolDescriptors.CalcNumHBD(mol)
            new_row["rotatable_bonds"] = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
            new_row["tpsa"] = Chem.rdMolDescriptors.CalcTPSA(mol)
            new_row["aromatic_rings"] = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
            new_row["formula"] = Chem.rdMolDescriptors.CalcMolFormula(mol)
            new_row["num_atoms"] = mol.GetNumAtoms()
            cls.ecfp4.object2fingerprint(mol)  # expressed as a bit vector
            new_row["ecfp4"] = cls.ecfp4.to_numpy()
            new_row["ecfp4_count"] = cls.ecfp4.get_num_on_bits()
            # see https://cactus.nci.nih.gov/presentations/meeting-08-2011/Fri_Aft_Greg_Landrum_RDKit-PostgreSQL.pdf
            # new_row['tt'] = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
            # calc number of stereoisomers.  doesn't work as some bonds have incompletely specified stereochemistry
            # new_row['num_stereoisomers'] = len(tuple(EnumerateStereoisomers(mol)))
            # number of undefined stereoisomers
            new_row[
                "num_undef_stereo"
            ] = Chem.rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
            # get number of unspecified double bonds
            new_row["num_undef_double"] = len(utils.get_unspec_double_bonds(mol))

        return new_row, mol


class MSPLoader(BatchLoader):   
    def __init__(self, row_batch_size=5000, format=None, min_intensity=0, num=None):
        super().__init__('msp', row_batch_size=row_batch_size, format=format, num=num)

        if self.format['comment_fields']:
            self.format['comment_fields'] = eval(self.format['comment_fields'][0])
        self.min_intensity = min_intensity
        
    def load(self, fp):
        self.setup()

        # move forward to first begin ions
        line = fp.readline()
        # skip to first full entry
        while line and not line.lower().startswith("Name: ".lower()):
            line = fp.readline()

        while line:
            row = {}
            mz = []
            intensity = []
            row["id"] = self.current_id

            while line:
                line = line.strip()
                if (
                    line and (line[0].isdigit() or line[0] == "-" or line[0] == ".")
                ):  # read in peak
                    peak_list = line.split(';')
                    for peak in peak_list:
                        values = peak.strip().split(None, 2)  # split on first two whitespaces
                        if (len(values) >= 2):
                            # have not yet dealt with case where there is a charge appended
                            mz.append(float(values[0]))
                            intensity.append(float(values[1]))
                else:  # header
                    values = re.split(r": ", line, 1)
                    if len(values) == 2:
                        # Name: DLPQGFSALEPLVDLPIGINITR/3_1(19,N,G:G2Hx5)  -> peptide, charge, position of mod, mod
                        if values[0] == "Name":
                            m = re.match(
                                r"([A-Z]+)/(\d)+(_\d+)*\((\d+),[A-Z]+,(.*)\)", values[1]
                            )
                            if m is not None:
                                row["name"] = m.group(0)
                                row["peptide"] = m.group(1)
                                row["peptide_len"] = len(m.group(1))
                                # row['glycosylation'] = m.group(5)
                                # row['glycosylation_pos'] = int(m.group(4))
                                row["charge"] = int(m.group(2))
                            else:
                                m = re.match(r"([A-Z]+)/(\d)+", values[1])
                                if m is not None:
                                    row["name"] = m.group(0)
                                    row["peptide"] = m.group(1)
                                    row["peptide_len"] = len(m.group(1))
                                    row["charge"] = int(m.group(2))
                            if self.format['title_fields']:
                                for column_name, regex in self.format['title_fields'].items():
                                    m = re.search(regex, values[1])
                                    if m:
                                        row[column_name] = m.group(1)
                        elif values[0] == "PrecursorMZ" or values[0] == "PRECURSORMZ":
                            row["precursor_mz"] = float(values[1].rstrip())
                        elif values[0] == "NCE":
                            row["nce"] = float(values[1].rstrip())
                        elif values[0] == "eV":
                            row["ev"] = float(values[1].rstrip())
                        elif values[0] == "Ion_mode":
                            row["ion_mode"] = values[1].rstrip()
                        elif values[0] == "Collision_energy":
                            parse_energy(row, values[1])
                        elif values[0] == "Instrument":
                            row["instrument"] = values[1].rstrip()
                        elif values[0] == "Instrument_type":
                            row["instrument_type"] = values[1].rstrip()
                        elif values[0] == "InstrumentModel":
                            row["instrument_model"] = values[1].rstrip()
                        elif values[0] == "Ionization":
                            row["ionization"] = values[1].rstrip()
                        elif values[0] == "Precursor_type":
                            row["precursor_type"] = values[1].rstrip()
                        elif values[0] == "ProteinId":
                                row["protein_id"] = values[1].rstrip().split(',')
                        elif values[0] == "Comment":
                            # iterate through name=value pairs.  If the value is bracketed by quotes, the regex
                            # takes the whole value
                            # alternative mechanism to decode quoted string is csv.reader([match.group(2)])
                            if 'Single ' in values[1]:
                                row['composition'] = 'bestof'
                            elif 'Consensus ' in values[1]:
                                row['composition'] = 'consensus'

                            for match in re.finditer(
                                r'([^= ]+)=(("([^"]+)")|([^" ]+))', values[1].rstrip()
                            ):
                                if match.group(1) == "Parent":
                                    row["precursor_mz"] = float(match.group(2))
                                elif match.group(1) == "PrecursorMonoisoMZ" and "precursor_mz" not in row:
                                    # use the precursor mass instead of parent if parent not available
                                    row["precursor_mz"] = float(match.group(2))
                                elif match.group(1) == "Pep":
                                    if match.group(2).casefold() == 'Tryptic'.casefold():
                                        row["peptide_type"] = 'tryptic'
                                    elif match.group(2).casefold() == 'N-Semitryptic'.casefold():
                                        row["peptide_type"] = 'semitryptic'
                                    elif match.group(2).casefold() == 'C-Semitryptic'.casefold():
                                        row["peptide_type"] = 'semitryptic'
                                    elif match.group(2).casefold() == 'SemiTryptic'.casefold():
                                        row["peptide_type"] = 'semitryptic'
                                    elif match.group(2).casefold() == 'NonTryptic'.casefold():
                                        row["peptide_type"] = 'nontryptic'
                                elif match.group(1) == "NCE":
                                    row["nce"] = float(match.group(2))
                                elif match.group(1) == "CE":
                                    parse_energy(row, match.group(2))
                                elif match.group(1) == "HCD":
                                    parse_energy(row, match.group(2))
                                elif match.group(1) == "Mods":
                                    submatch = re.findall(r'\((\d+),[A-Z]+,([^\)]*)\)', match.group(2))
                                    if not submatch:
                                        # deal with older format of modifications
                                        submatch = re.findall(r'/(\d+),[A-Z]+,([^/]*)', match.group(2))
                                    row["mod_names"] = [mod_masses.dictionary.index(nist_mod_2_unimod.get(x[1], x[1])) for x in submatch]
                                    row["mod_positions"] = [int(x[0]) for x in submatch]
                                elif match.group(1) == "Filter":
                                    submatch = re.search(r'@hcd(\d+\.*\d*) ', match.group(2))
                                    if submatch:
                                        row["nce"] = float(submatch.group(1))
                                elif self.format['comment_fields']:
                                    for subfield_name, (regex, field_type, field_name) in self.format['comment_fields'].items():
                                        if match.group(1) == subfield_name:
                                            submatch = re.search(regex, match.group(2))
                                            if submatch:
                                                #row[field_name] = field_type(submatch.group(1))
                                                pass
                line = fp.readline()
                if not line or re.match("^Name:", line):
                    break

            # finalize the spectrum only if it has peaks
            if len(mz) != 0:
                row['mz'] = mz
                row['intensity'] = intensity
                spectrum = Spectrum(mz=mz, intensity=intensity, row={}, precursor_mz=row['precursor_mz'],
                                        precursor_intensity=row.get('precursor_intensity', None))
                spectrum.charge = row.get('charge', None)
                spectrum.peptide = row.get('peptide', None)
                spectrum.peptide_len = row.get('peptide_len', None)
                spectrum.protein_id = row.get('protein_id')
                spectrum.mod_names = row.get('mod_names', None)
                spectrum.mod_positions = row.get('mod_positions', None)
                row['product_massinfo'] = spectrum.product_mass_info.__dict__
                row['precursor_massinfo'] = spectrum.precursor_mass_info.__dict__
                fingerprint = spectrum.filter(min_intensity=self.min_intensity).create_fingerprint(max_mz=2000)
                row["spectrum_fp"] = fingerprint.to_numpy()
                row["spectrum_fp_count"] = fingerprint.get_num_on_bits()
                add_row_to_records(self.records, row)
            else:
                pass

            if self.num is not None and len(self.records['id']) >= self.num:
                break

            self.loop_end()
        return self.finalize()

class MGFLoader(BatchLoader):   
    def __init__(self, row_batch_size=5000, format=None, num=None, 
                 set_probabilities=(0.00, 0.93, 0.02, 0.05), 
                 min_intensity=0.0):
        super().__init__('mgf', row_batch_size=row_batch_size, format=format, num=num)
        self.min_intensity = min_intensity
        self.set_probabilities = set_probabilities
        
    def load(self, fp):
        self.setup()
        for line in fp:
            mz = []
            intensity = []
            row = {
                "id": self.current_id,
                "set": np.random.choice(SET_NAMES, p=self.set_probabilities),
            }
            if self.format['row_entries']:
                row = {**row, **self.format['row_entries']}  # merge in the passed in row entries
            try:
                while not line.lower().startswith("BEGIN IONS".lower()):
                    line = next(fp)
                # move to first header
                line = next(fp).strip()

                while not line.lower().startswith("END IONS".lower()):
                    if len(line) > 0:
                        if (
                            line[0].isdigit() or line[0] == "-" or line[0] == "."
                        ):  # read in peak
                            values = line.split()
                            if (
                                len(values) == 2
                            ):  # have not yet dealt with case where there is a charge appended
                                mz.append(float(values[0]))
                                intensity.append(float(values[1]))
                        elif line[0] in "#;/!":  # skip comments
                            pass
                        else:  # header
                            values = line.split("=", maxsplit=1)
                            if len(values) == 2:
                                if values[0] == "TITLE":
                                    row["name"] = values[1].rstrip()
                                    if self.format['title_fields']:
                                        for column_name, regex in self.format['title_fields'].items():
                                            m = re.search(regex, values[1])
                                            if m:
                                                row[column_name] = m.group(1)
                                elif values[0] == "PEPMASS":
                                    mass_values = values[1].split(" ")
                                    row["precursor_mz"] = float(mass_values[0])
                                    if len(mass_values) > 1:
                                        row["precursor_intensity"] = float(mass_values[1])
                                elif values[0] == "SEQ":
                                    row["peptide"] = values[1]
                                    row["peptide_len"] = len(values[1])
                                elif values[0] == "RTINMINUTES":
                                    row["retention_time"] = float(values[1]) * 60.0
                                elif values[0] == "RTINSECONDS":
                                    row["retention_time"] = float(values[1])
                                elif values[0] == "CHARGE":
                                    number, sign = re.match(
                                        r"^(\d+)([+\-]*)$", values[1]
                                    ).groups()
                                    row["charge"] = int(sign + number)
                                else:
                                    row[values[0]] = values[1].rstrip()
                    line = next(fp).strip()
            except StopIteration:
                pass
            # finalize the spectrum only if it has peaks
            if len(mz) != 0:      
                row['mz'] = mz
                row['intensity'] = intensity
                spectrum = Spectrum(mz=mz, intensity=intensity, row=row, precursor_mz=row['precursor_mz'],
                                        precursor_intensity=row.get('precursor_intensity', None))
                spectrum.charge = row.get('charge', None)
                spectrum.peptide = row.get('peptide', None)
                spectrum.peptide_len = row.get('peptide_len', None)
                spectrum.mod_names = row.get('mod_names', None)
                spectrum.mod_positions = row.get('mod_positions', None)
                row['product_massinfo'] = spectrum.product_mass_info.__dict__
                row['precursor_massinfo'] = spectrum.precursor_mass_info.__dict__
                fingerprint = spectrum.filter(min_intensity=self.min_intensity).create_fingerprint(max_mz=2000)
                row["spectrum_fp"] = fingerprint.to_numpy()
                row["spectrum_fp_count"] = fingerprint.get_num_on_bits()
                row['precursor_mz'] = spectrum.precursor.mz

                add_row_to_records(self.records, row)

            if self.num is not None and len(self.records['id']) >= self.num:
                break

            self.loop_end()
        return self.finalize()

class SDFLoader(BatchLoader):   
    def __init__(self, row_batch_size=5000, format=None, num=None,
                 set_probabilities=(0.0, 0.93, 0.02, 0.05), 
                 min_intensity=0.0):
        super().__init__('sdf', row_batch_size=row_batch_size, format=format, num=num)
        self.min_intensity = min_intensity
        self.set_probabilities = set_probabilities

        
    def load(self, fp):
        # max_size: the maximum bounding box size (used to filter out large molecules. 0=no bound)
        # precursor_mass_info: mass information for precursor. if None, will use default.
        # product_mass_info: mass information for product. if None, will use default.
        # suppress_rdkit_warnings: don't print out spurious rdkit warnings
        max_size=0
        precursor_mass_info=None
        product_mass_info=None
        suppress_rdkit_warnings=True

        self.setup()
        current_name = None

        # Turn off RDKit error messages
        if suppress_rdkit_warnings:
            RDLogger.DisableLog('rdApp.*')

        while True:
            try:
                mol = next(fp)
            except StopIteration:
                break
            if mol is None:
                logging.info(f'unable to read mol after mol with name {current_name} and id {self.current_id}')
                continue

            if self.format['id']['field'] and mol.HasProp(self.format['id']['field']):
                if self.format['id']['field_type'] == 'int':
                    self.current_id = int(mol.GetProp(self.format['id']['field']))
                else:
                    self.current_id = mol.GetProp(self.format['id']['field'])

            # workaround for getting private props like _NAME
            props = mol.GetPropsAsDict(includePrivate=True)

            for field in self.format['name']['field']:
                if field in props:
                    current_name = props[field]
                    break

            new_row, mol = self.mol2row(mol, 
                                        skip_computed_props=self.format['skip_computed_props'],
                                        skip_expensive=self.format['skip_expensive'], 
                                        max_size=max_size
                                        )
            if new_row is None or not new_row:
                logging.info(f'unable to standardize mol with name {current_name} and id {self.current_id}')
                continue

            # create useful set labels for training
            spectrum = None
            if self.format['source'] == "nist":
                # create the mass spectrum
                spectrum = Spectrum(product_mass_info=product_mass_info,
                                        precursor_mass_info=precursor_mass_info)
                spectrum.from_mol(mol, 
                                  self.format['skip_expensive'], 
                                  id_field=self.format['id']['field'], 
                                  id_field_type=self.format['id']['field_type']
                                  )
                spectrum.id = self.current_id
                spectrum.name = current_name
                try:
                    new_row["set"] = np.random.choice(["dev", "train", "valid", "test"], p=self.set_probabilities)
                    new_row["precursor_mz"] = spectrum.precursor.mz
                    new_row["casno"] = spectrum.casno
                    new_row["synonyms"] = json.dumps(spectrum.synonyms)
                    new_row["spectrum"] = spectrum
                    new_row["column"] = spectrum.column
                    new_row["experimental_ri"] = spectrum.experimental_ri
                    new_row["experimental_ri_error"] = spectrum.experimental_ri_error
                    new_row["experimental_ri_data"] = spectrum.experimental_ri_data
                    new_row["stdnp"] = spectrum.stdnp
                    new_row["stdnp_error"] = spectrum.stdnp_error
                    new_row["stdnp_data"] = spectrum.stdnp_data
                    new_row["stdpolar"] = spectrum.stdpolar
                    new_row["stdpolar_error"] = spectrum.stdpolar_error
                    new_row["stdpolar_data"] = spectrum.stdpolar_data
                    new_row["estimated_ri"] = spectrum.estimated_ri
                    new_row["estimated_ri_error"] = spectrum.estimated_ri_error
                    new_row["exact_mass"] = spectrum.exact_mass
                    new_row["ion_mode"] = spectrum.ion_mode
                    new_row["charge"] = spectrum.charge
                    new_row["instrument"] = spectrum.instrument
                    new_row["instrument_type"] = spectrum.instrument_type
                    new_row["instrument_model"] = spectrum.instrument_model
                    new_row["ionization"] = spectrum.ionization
                    new_row["collision_gas"] = spectrum.collision_gas
                    new_row["sample_inlet"] = spectrum.sample_inlet
                    new_row["spectrum_type"] = spectrum.spectrum_type
                    new_row["precursor_type"] = spectrum.precursor_type
                    new_row["inchi_key_orig"] = spectrum.inchi_key
                    new_row["vial_id"] = spectrum.vial_id
                    new_row["collision_energy"] = spectrum.collision_energy
                    new_row["nce"] = spectrum.nce
                    new_row["ev"] = spectrum.ev
                    new_row["insource_voltage"] = spectrum.insource_voltage
                    new_row["mz"] = spectrum.products.mz
                    new_row["intensity"] = spectrum.products.intensity
                    new_row['product_massinfo'] = spectrum.product_mass_info.__dict__
                    new_row['precursor_massinfo'] = spectrum.precursor_mass_info.__dict__
                    fingerprint = spectrum.filter(min_intensity=self.min_intensity).create_fingerprint(max_mz=2000)
                    new_row["spectrum_fp"] = fingerprint.to_numpy()
                    new_row["spectrum_fp_count"] = fingerprint.get_num_on_bits()
                except AttributeError:
                    raise ValueError('attribute error from spectrum: ' + spectrum.id)
            elif self.format['source'] == "pubchem":
                if mol.HasProp("PUBCHEM_XLOGP3"):
                    new_row["xlogp"] = float(mol.GetProp("PUBCHEM_XLOGP3"))
                if mol.HasProp("PUBCHEM_COMPONENT_COUNT"):
                    new_row["component_count"] = float(
                        mol.GetProp("PUBCHEM_COMPONENT_COUNT")
                    )
            elif self.format['source'] == 'nist_ri':
                new_row["inchi_key_orig"] = mol.GetProp("INCHIKEY")
                new_row["set"] = np.random.choice(["dev", "train", "valid", "test"], p=self.set_probabilities)
                if mol.HasProp("COLUMN CLASS") and mol.HasProp("KOVATS INDEX"):
                    ri_string = mol.GetProp("COLUMN CLASS")
                    if ri_string in ['Semi-standard non-polar', 'All column types', 'SSNP']:
                        new_row['column'] = 'SemiStdNP'
                        new_row['experimental_ri'] = float(mol.GetProp("KOVATS INDEX"))
                        new_row['experimental_ri_error'] = 0.0
                        new_row['experimental_ri_data'] = 1
                    elif ri_string == 'Standard non-polar':
                        new_row['column'] = 'StdNP'
                        new_row['stdnp'] = float(mol.GetProp("KOVATS INDEX"))
                        new_row['stdnp_error'] = 0.0
                        new_row['stdnp_data'] = 1
                    elif ri_string == 'Standard polar':
                        new_row['column'] = 'StdPolar'
                        new_row['stdpolar'] = float(mol.GetProp("KOVATS INDEX"))
                        new_row['stdpolar_error'] = 0.0
                        new_row['stdpolar_data'] = 1

            new_row["id"] = self.current_id
            new_row["name"] = current_name
            add_row_to_records(self.records, new_row)

            if self.num is not None and len(self.records['id']) >= self.num:
                break

            self.loop_end()
        return self.finalize()

class CSVLoader(BatchLoader):   
    def __init__(self, row_batch_size=5000, format=None, num=None):
        super().__init__('csv', row_batch_size=row_batch_size, format=format, num=num)
        
    def load(self, fp):
        self.setup()
        suppress_rdkit_warnings=True

        # Turn off RDKit error messages
        if suppress_rdkit_warnings:
            RDLogger.DisableLog('rdApp.*')

        for next_chunk in fp:
            if next_chunk is None:
                break
            table = pa.Table.from_batches([next_chunk])
            column_names = table.column_names

            # edit possibly overlapping column names
            column_names = [f'{self.format["mol_column_name"]}_original' if self.format["mol_column_name"] == x else x for x in column_names]
            if self.current_id is not None:
                column_names = ['id_original' if 'id' in x else x for x in column_names]
            table = table.rename_columns(column_names)

            ids = []
            mols = []

            for i in range(len(table)):
                try:
                    mol = Chem.MolFromSmiles(table[self.format['smiles_column_name']][i].as_py())
                    mol = masskit.small_molecule.utils.standardize_mol(mol)
                except ValueError as e:
                    logging.info(f"Unable to standardize {table[self.format['smiles_column_name']][i].as_py()}")
                    continue
                mols.append(Chem.rdMolInterchange.MolToJSON(mol))
                if self.current_id is not None:
                    ids.append(self.current_id)
                    self.current_id += 1
            table = table.append_column('mol', pa.array(mols, type=MolArrowType()))
            if self.current_id is not None:
                table = table.append_column('id', pa.array(ids, type=pa.uint64()))
            self.tables.append(table)

        if len(self.tables) == 0:
            table = pa.table([])
        else:
            table = pa.concat_tables(self.tables)
        return table


def records2table(records, schema_group):
    """
    convert a flat table with spectral records into a table with a nested spectrum

    :param records: records to be added to a table
    :param schema_group: the schema group
    """
    table = pa.table(records, schema_group['flat_schema'])
    structarray = table2structarray(table, SpectrumArrowType(storage_type=schema_group['storage_type']))
    table = table.select([x for x in table.column_names if x in schema_group['nested_schema'].names])
    table = table_add_structarray(table, structarray)
    return table


def parse_energy(row, value):
    """
    parse energy field that has nce or collision_energy in it
    """
    match = re.search(r"(\d+)%", value)
    if match:
        row["nce"] = float(match.group(1))
        return
    match = re.search(r"(\d+)eV", value)
    if match:
        row["ev"] = float(match.group(1))
        return
    # for some reason, some eVs are given as "CE=29.00,"
    match = re.search(r"(\d+),", value)
    if match:
        row["ev"] = float(match.group(1))
        return
    try:
        row["ev"] = float(value)
    except ValueError:
        logging.info(f"parse_energy: unable to parse {value}")



def create_table_with_mod_dict(records, schema):
    # row["mod_names"] = [mod_masses.dictionary.index(x[1]) for x in submatch]
    # ptm_before_array = pa.DictionaryArray.from_arrays(
    #     indices=pa.array(ptm_before_array, type=pa.int16()),
    #     dictionary=mod_masses.dictionary)
    mod_names = pa.DictionaryArray.from_arrays(indices=pa.array(records['mod_names'], type=pa.list_(pa.int16())), dictionary=mod_masses.dictionary)
    del records['mod_names']
    table = pa.table(records, schema)
    table.append_column(mod_names_field, mod_names)
    return table


def parse_glycopeptide_annot(annots, peak_index):
    """
    take a glycopeptide annotation string like "Y0-H2O+i/-18.2ppm,Y0-NH3/7.6ppm 22 23" and parse it.
    this function is not complete.

    :param annots: the annotation string
    :param peak_index: the index of the peak with the annotation in the mz array
    :return: the parsed values in an array
    """
    parsed_annots = []
    annot = annots.split(" ")

    penultimate_int = None
    last_int = None

    if len(annot) > 2:
        penultimate_int = int(annot[1])
        last_int = int(annot[2])
    if annot:
        ions = annot[0].split(",")
        for ion in ions:
            charge = None
            neutral_loss = None
            neutral_loss_sign = None
            peptide_ion = None
            peptide_apostrophes = None
            peptide_ion_num = None
            glyco_complete_ion = None
            glyco_ion = None
            glyco_apostrophes = None
            glyco_ion_num = None
            internal_ion = None
            pre_loss = None
            other = None
            error_units = None
            error = None
            sub_ions = ion.split("/")
            if sub_ions:
                # split at addition or subtraction of neutral ion

                m = re.search(r"\^(\d+)", sub_ions[0])
                if m:
                    charge = int(m.group(1))
                    sub_ions[0] = re.sub(r"\^(\d+)", "", sub_ions[0])

                first_split = re.split(r"([+-])", sub_ions[0], 1)
                if len(first_split) > 2:
                    neutral_loss_sign = first_split[1]
                    neutral_loss = first_split[2]

                while True:
                    ion = first_split[0]
                    m = re.fullmatch(r"{.*}", ion)
                    if m:
                        glyco_complete_ion = ion
                        break
                    m = re.fullmatch(r"([abcxyz])('*)(\d+)", ion)
                    if m:
                        peptide_ion = m.group(1)
                        peptide_apostrophes = m.group(2)
                        peptide_ion_num = int(m.group(3))
                        break
                    # match question mark or IKF ions
                    m = re.fullmatch(
                        r"\?|(I[A-Z]+)", ion
                    )  # theoretically would match iodine
                    if m:
                        other = m.group(0)  # this is to fit stuff like IKF, ?
                        break
                    # glyco ions
                    m = re.fullmatch(
                        r"([AXYZ])('*)(\d+)", ion
                    )  # does not include C or B due to ambiguity to chemical formula
                    if m:
                        glyco_ion = m.group(1)
                        glyco_apostrophes = m.group(2)
                        glyco_ion_num = int(m.group(3))
                        break
                    # chemical formula match
                    m = re.fullmatch(
                        r"([A-Z][a-z]?\d*)+", ion
                    )  # matches C1 glyco ions and IKF ions
                    if m:
                        pre_loss = m.group(0)
                        break
                    # internal ions
                    m = re.fullmatch(r"Int", ion)
                    if m:
                        internal_ion = sub_ions[1]
                        break
                    break  # other unmatched peaks

                # do error in sub_ions[-1]
                if len(sub_ions) > 1:
                    m = re.match(r"([-+]?\d*\.\d+|\d+)(\w+)", sub_ions[-1])
                    error = float(m.group(1))
                    error_units = m.group(2)

            parsed_annots.append(
                {
                    "peak_index": peak_index,
                    "charge": charge,
                    "neutral_loss_sign": neutral_loss_sign,
                    "neutral_loss": neutral_loss,
                    "peptide_ion": peptide_ion,
                    "peptide_apostrophes": peptide_apostrophes,
                    "peptide_ion_num": peptide_ion_num,
                    "glyco_complete_ion": glyco_complete_ion,
                    "glyco_ion": glyco_ion,
                    "glyco_apostrophes": glyco_apostrophes,
                    "glyco_ion_num": glyco_ion_num,
                    "internal_ion": internal_ion,
                    "pre_loss": pre_loss,
                    "other": other,
                    "error_units": error_units,
                    "error": error,
                    "penultimate_int": penultimate_int,
                    "last_int": last_int,
                }
            )
    return parsed_annots


mq2unimod = {
    'CHEMMOD:57.0214637236': 4,  # Carbamidomethyl (C)
    'CHEMMOD:15.9949146221': 35, # Oxidation (M)   
    'CHEMMOD:42.0105646863': 1,  # Acetyl (Protein N-term)
    'CHEMMOD:79.9663304084': 21, # Phospho (STY)
}

class MzTab_Reader():
    """
    Class for reading an mzTab file

    :param fp: stream or filename
    :param dedup: Only take the first row with a given hit_id
    """
    def __init__(self, fp, dedup=False, decoy_func=None):
        self.dedup = dedup
        if decoy_func is None:
            self.decoy_func = lambda x: False
        else:
            self.decoy_func = decoy_func
        self.metadata_rows = []
        self.psm_rows = []
        self.read_sections(fp)
        self.parse_metadata()
        self.parse_psm()
        # print(self.psm.to_pandas())

    def read_sections(self, fp):
        fp = open_if_filename(fp, 'r')
        for line in fp:
            if len(line) < 3 or line[0] == 'C':
                # Empty lines and comments are ignored
                None
            elif line[0:3] == "MTD":
                self.metadata_rows.append(line.rstrip())
            elif line[0:2] == "PS":
                self.psm_rows.append(line.rstrip())
            else:
                # Other sections are not supported at this time.
                None

    # Modifications from MaxQuant are hard coded here as a stopgap. To really
    # parse them correctly we should:
    #
    #   1. parse the input name from the comment, 
    #   2. look up the compostion in the MaxQuant modifications.xml file
    #   3. match the composition and partial name(?) to the information in our encoding.py
    #   4. return the hopefully matched Unimod ID

    def parse_metadata(self):
        # Default to mascot ?
        self.raw_score_label = "opt_global_cv_MS:1001171_Mascot:score"
        self.search_engine = 'mascot'

        for row in self.metadata_rows:
            fields = row.split('\t')
            #print(fields)

            if fields[1].startswith('psm_search_engine_score'):
                cv = fields[2].split(',')[1].strip()
                if cv == "MS:1002338":
                    self.raw_score_label = fields[1][4:] # strip "psm_" prefix
                    self.search_engine = 'mq'
                # if cv == "MS:1002995":
                #     self.pep_label = fields[1]
            # elif fields[1].startswith('fixed_mod') or 



    def parse_psm(self):
        reader = csv.DictReader(self.psm_rows, dialect="excel-tab", )
        records = empty_records(hitlist_schema)
        prev_hit_id = None
        for row in reader:
            psm_row = self.parse_psm_row(row)
            if self.dedup:
                if psm_row['hit_id'] != prev_hit_id:
                    add_row_to_records(records, psm_row)
            else:
                add_row_to_records(records, psm_row)
            prev_hit_id = psm_row['hit_id']
        table = pa.table(records, hitlist_schema)
        self.psm = table

    def parse_psm_row(self, row):
        new_row = {}
        new_row['peptide'] = row['sequence']
        # Parse mod list
        mNames = []
        mPos = []
        # hack to deal with mq not encoding fixed modifications in mzTab, but mascot does
        if self.search_engine == 'mq':
            for i in range(len(new_row['peptide'])):
                if new_row['peptide'][i] == 'C':
                    mNames.append(4)
                    mPos.append(i)
        
        # clean out the extra cruft Mascot adds
        row['modifications'] = re.sub(r"\|\[MS, MS:1001524, fragment neutral loss, [+\-\d\.]+\]", '', row['modifications'])

        mods = row["modifications"].split(',')
        for mod in mods:
            (pos, sep, name) = mod.strip().partition('-')
            (db, sep, db_id) = name.strip().partition(':')
            if db == 'UNIMOD':
                mNames.append(int(db_id))
                mPos.append(max(0, int(pos)-1))  # mzTab uses 0 for n-terminus
            elif name in mq2unimod:
                mNames.append(mq2unimod[name])
                mPos.append(max(0, int(pos)-1))
        new_row['mod_names'] = mNames
        new_row['mod_positions'] = mPos
        new_row['charge'] = int(row['charge'])

        if 'opt_global_cv_MS:1000776_scan_number_only_nativeID_format' in row:
            new_row['query_id'] = int(row['opt_global_cv_MS:1000776_scan_number_only_nativeID_format'])
        elif 'opt_global_cv_MS:1000797_peak_list_scans' in row:
            new_row['query_id'] = int(row['opt_global_cv_MS:1000797_peak_list_scans'])
        else:
            new_row['query_id'] = int(row["spectra_ref"].rsplit("=", 1)[1])

        new_row['hit_id'] = int(row['PSM_ID'])
        new_row['accession'] = row['accession']
        if 'opt_global_cv_MS:1002217_decoy_peptide' in row:
            new_row['decoy_hit'] = row['opt_global_cv_MS:1002217_decoy_peptide'] == '1'
            new_row['protein_start'] = 0
            new_row['protein_stop'] = 0
        else:
            #new_row['decoy_hit'] = False
            new_row['decoy_hit'] = (self.decoy_func)(row)
            new_row['protein_start'] = int(row['start'])
            new_row['protein_stop'] = int(row['end'])

        new_row['raw_score'] = float(row[self.raw_score_label])
        # new_row['Andromeda PEP'] = float(row[self.pep_label])
        new_row['source_search'] = row["database"].split(" ", 1)[0]

        return new_row

    def get_hitlist(self):
        df = self.psm.to_pandas()
        mIdx = pd.MultiIndex.from_arrays([df['query_id'], df['hit_id']])

        return Hitlist(df.set_index(['query_id','hit_id']))


def load_mzTab(fp, dedup=True, decoy_func=None):
    """
    Read file in SDF format and return an array.

    :param fp: stream or filename
    :param dedup: Only take the first row with a given hit_id
    """
    
    mztr = MzTab_Reader(fp, dedup, decoy_func)
    return mztr.get_hitlist()


default_formats ={
    'arrow': 'arrow_mol',
    'csv': 'csv_mol',
    'mgf': 'mgf_peptide',
    'msp': 'msp_mol',
    'parquet': 'parquet_mol',
    'sdf': 'sdf_mol',
    }


def load_default_config_schema(file_type, format=None):
    """
    for a given file type and format, return configuration

    :param file_type: the type of file, e.g. mgf, msp, arrow, parquet, sdf
    :param format: the specific format of the file_type, e.g. msp_mol, msp_peptide, sdf_mol, sdf_nist_mol, sdf_pubchem_mol. If none, use default
    :return: config
    """
    if format is None:
        format = default_formats[file_type]

    config_file = pkgutil.get_loader(f'masskit.conf.conversion.{file_type}').get_filename()
    format_config = OmegaConf.load(Path(config_file).parent / f'{format}.yaml')
    return dict(format_config)

class BatchFileReader:
    def __init__(self, 
                 filename: Union[str, os.PathLike],
                 format:Union[Dict, DictConfig]=None, 
                 row_batch_size:int=5000,
                 ) -> None:
        """
        read files in batches of pyarrow Tables

        :param filename: input file
        :param format: format of the input file using configuration dictionary
        :param row_batch_size: size of batches to read (except parquet files, which use existing batches)
        """
        super().__init__()
        self.format = format
        self.row_batch_size = row_batch_size
        self.filename = str(filename)
        if format['format'] == 'parquet':
            # does not support compression, but parquet compresses itself so no need
            self.dataset = pq.ParquetFile(filename)
        elif format['format'] == 'arrow':
            # as a memory map, does not support compression
            self.dataset = pa.ipc.RecordBatchFileReader(pa.memory_map(filename, 'r')).read_all()
        elif format['format'] in ['mgf', 'msp']:
            # supports gz and bz2 compression
            self.dataset = open_if_filename(filename, mode="r")
        elif format['format'] == 'csv':
            # supports gz and bz2 compression
            read_options = pacsv.ReadOptions(autogenerate_column_names=format['no_column_headers'],
                                              block_size=row_batch_size)
            parse_options = pacsv.ParseOptions(delimiter=format['delimiter'])
            self.dataset =  pacsv.open_csv(filename, read_options=read_options, parse_options=parse_options)
        elif format['format'] == 'sdf':
             # supports gz and bz2 compression
            self.dataset = open_if_filename(filename, mode="rb")
            self.dataset = Chem.ForwardSDMolSupplier(self.dataset, sanitize=False)
        else:
            raise ValueError(f'Unknown format {self.format["format"]}')
        
    loaders = {'msp': MSPLoader,
               'sdf': SDFLoader,
               'mgf': MGFLoader,
               'csv': CSVLoader,
               }
    
    def iter_tables(self) -> pa.Table:
        """
        read batch generator, returns a Table
        """
        batch_num = 0
        if self.format['format'] == 'parquet':
            for batch in self.dataset.iter_batches():
                table = pa.Table.from_batches(batch)  # schema is inferred from batch
                logging.info(f'processing batch {batch_num} with size {len(table)}')
                batch_num += 1
                yield table 
        elif self.format['format'] == 'arrow':
            # the entire table is memmapped as self.dataset
            # get the length, which is computed from the number of rows in all batches
            # not clear if this goes through the entire memmap -- need to test
            start = 0
            while True:
                batch = self.dataset.slice(start, self.row_batch_size)
                if len(batch) == 0:
                    break
                start += self.row_batch_size
                logging.info(f'processing batch {batch_num} with size {len(batch)}')
                batch_num += 1
                yield batch
        elif self.format['format'] in ['msp', 'mgf', 'sdf', 'csv']:
            loader = self.loaders[self.format['format']](num=self.row_batch_size, 
                                                         format=self.format,
                                                        )
            while True:
                batch = loader.load(self.dataset)
                if len(batch) == 0:
                    break
                if self.format['id']['field_type'] == 'int':
                    self.format['id']['initial_value'] += len(batch)
                logging.info(f'processing batch {batch_num} with size {len(batch)}')
                batch_num += 1
                yield batch
        else:
            raise ValueError(f'Unknown format {self.format["format"]}')

# processing functions defined at top level of module to allow pickling
# for multiprocessing
        
def spectrum2msp(spectrum, annotate=False):
    output = StringIO()
    spectra_to_msp(output, [spectrum], annotate_peptide=annotate)
    return output.getvalue() 

def spectrum2mgf(spectrum):
    output = StringIO()
    spectra_to_mgf(output, [spectrum])
    return output.getvalue() 
        
class BatchFileWriter:
    def __init__(self, filename: Union[str, os.PathLike], 
                 format:str=None, annotate:bool=False, 
                 row_batch_size:int=5000,
                 num_workers:int=7,
                 column_name:str=None) -> None:
        """
        write batches of pyarrow Tables to files

        :param filename: output file
        :param format: format of the output file
        :param annotate: whether to annotate the output file
        :param row_batch_size: size of batches to write 
        :param num_workers: number of threads for processing
        :param column_name: name of the struct column
        """
        super().__init__()
        self.format = format
        self.row_batch_size = row_batch_size
        self.filename = str(filename)
        self.annotate = annotate
        self.num_workers = num_workers
        if column_name is None:
            self.column_name = 'spectrum'
        else:
            self.column_name = column_name
        if format in ['parquet','arrow']:
            self.dataset = None  # set this up in writer as it needs the schema 
        elif format in ['mgf', 'msp']:
            self.dataset = open(self.filename, mode="w")
        else:
            raise ValueError(f'Unknown format {self.format}')
            
    def write_table(self, table:pa.Table) -> None:
        """
        write a table out

        :param table: table to write
        """

        if self.format == 'parquet':
            if self.dataset == None:
                self.dataset = pq.ParquetWriter(pa.OSFile(self.filename, 'wb'), table.schema)
            self.dataset.write_table(table)
        elif self.format == 'arrow':
            if self.dataset == None:
                self.dataset = pa.RecordBatchFileWriter(pa.OSFile(self.filename, 'wb'), table.schema)
            self.dataset.write_table(table)
        elif self.format == 'msp':
            spectra = table[self.column_name].to_pylist()
            with Pool(self.num_workers) as p:
                spectra = p.map(partial(spectrum2msp, annotate=self.annotate), spectra)
            for spectrum in spectra:
                self.dataset.write(spectrum)
            self.dataset.flush()
        elif self.format == 'mgf':
            spectra = table[self.column_name].to_pylist()
            with Pool(self.num_workers) as p:
                spectra = p.map(spectrum2mgf, spectra)
            for spectrum in spectra:
                self.dataset.write(spectrum)
            self.dataset.flush()
        else:
            raise ValueError(f'Unknown format {self.format}')

    def close(self):
        """
        Close the writer. Essential to avoid race conditions with threaded writers.
        """
        if self.dataset is not None:
            self.dataset.close()

    def __del__(self):
        # explictly close RecordBatchFileWriter which can be threaded
        self.close()



"""
Notes:

MSP
examples:
immonium
84.2	221	"IKC/0.12"
85.1	28	"IKC+i/0.02"
101.0	56	"IKA/-0.11"
102.1	44	"IEA/0.04"

neutral loss
810.3	20	"b8-H2O/-0.07"
811.3	31	"b8-NH3/-0.05"

ion type, including single
828.3	1279	"b8/-0.08"
829.4	383	"b8+i/0.02"
207.1	124	"a2/0.00"
208.1	19	"b4-NO2^2/-0.01"
226.2	28	"y4-CO-NH3^2/0.59"

multiple neutral losses with multiple moieties from parent ion
465.4	47	"p-CO-NH3/0.17,p-CO2/-0.36"

multiple isotopes (up to 5i)
453.2	54	"y8^2+2i/-0.03"
458.2	858	"b4/0.01"
459.3	138	"b4+i/0.11"

internal ions (single charged?)
590.3	1855	"Int/KKFW/0.00"
217.1	136	"Int/TD/0.00"

multiple charges
444.9	109	"b7-NH3^2/0.21"
453.9	160	"b7^2+i/0.18"

losses
-CO-2H2O
-H2O
-NH3
-CO-NH3
-CO2
-2CO-H2O-NH3
-CO-NH3
-NO2
-2H2O


"""

# allowable msp delimiters:   |space|tab|,|;|:|(|)|[|]|}|
# newline is also delimiter.  quotes then signify a peak annotation.
# todo: fix annotations in small_molecule so that they mask by any changes, e.g. partial copy.  change tests

# split at spaces
# last two are ints
# split first at commas
# for each one,
# parse at slashes
# first: split at first +- and then at ^\d$
# first of first: {.*} or [abcxyz]'*\d+ or [ABCXYZ]\d+ or Int or  (\w+\d*)+ or \w
# if last \d, save to charge
# second of first: save
# middle (if exists)
# last: parse float then string

# look for following start: [xyzabc]'*\d+,[XYZABC]\d+,Int,.*
# for Int
# get /.*/ and parse .*

# columns: peptide ion, peptide seq. #, glycan ion, glycan ion#, curlybrace, int sequence,
# other (IKF IRJ), neutral loss,

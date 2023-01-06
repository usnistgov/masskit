import base64
import copy
import csv
import logging
import re
import zlib
from collections import OrderedDict
import masskit.small_molecule
from matplotlib.pyplot import annotate
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
from masskit.constants import SET_NAMES, EPSILON
from masskit.data_specs.schemas import spectrums_schema, molecules_schema, set_field_int_metadata, \
    mod_names_field, hitlist_schema
from masskit.peptide.encoding import mod_masses
from masskit.small_molecule import threed, utils
from masskit.spectrum.spectrum import init_spectrum
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
import masskit.spectrum.theoretical_spectrum as msts

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


def load_mgf2array(
    fp,
    num=0,
    start_spectrum_id=0,
    set_probabilities=(0.01, 0.97, 0.01, 0.01),
    row_entries=None,
    title_fields=None,
    min_intensity=0.0,
    max_mz=2000,
):
    """
    Read stream in MGF format and return as Pandas data frame.

    :param fp: stream or filename
    :param num: the maximum number of records to generate (0=all)
    :param start_spectrum_id: the spectrum id to begin with
    :param set_probabilities: how to divide into dev, train, valid, test
    :param row_entries: dict containing additional row columns
    :param title_fields: dict containing column names with corresponding regex to extract field values from the TITLE
    regex match group 1 is the value
    :param min_intensity: the minimum intensity to set the fingerprint bit
    :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
    :return: one dataframe
    """
    # Scan:3\w  scan
    # \wRT:0.015\w  retention_time
    # \wHCD=10.00%\w  collision_energy
    # sample is from filename  _Urine3667_
    # ionization, energy, date, sample, run#, from filename

    records = empty_records(spectrums_schema)
    tables = []

    fp = open_if_filename(fp, 'r')

    spectrum_id = start_spectrum_id  # a made up integer spectrum id
    # move forward to first begin ions
    for line in fp:
        mz = []
        intensity = []
        row = {
            "id": spectrum_id,
            "set": np.random.choice(SET_NAMES, p=set_probabilities),
        }
        if row_entries:
            row = {**row, **row_entries}  # merge in the passed in row entries
        try:
            while not line.lower().startswith("BEGIN IONS".lower()):
                line = next(fp)
            # move to first header
            line = next(fp).strip()
            spectrum_id += 1  # increment id

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
                                if title_fields:
                                    for column_name, regex in title_fields.items():
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
            spectrum = init_spectrum(mz, intensity, row=row, precursor_mz=row['precursor_mz'],
                                     precursor_intensity=row.get('precursor_intensity', None))
            spectrum.charge = row.get('charge', None)
            spectrum.peptide = row.get('peptide', None)
            spectrum.peptide_len = row.get('peptide_len', None)
            spectrum.mod_names = row.get('mod_names', None)
            spectrum.mod_positions = row.get('mod_positions', None)
            row['product_massinfo'] = spectrum.product_mass_info.__dict__
            row['precursor_massinfo'] = spectrum.precursor_mass_info.__dict__
            fingerprint = spectrum.filter(min_intensity=min_intensity).create_fingerprint(max_mz=max_mz)
            row["spectrum_fp"] = fingerprint.to_numpy()
            row["spectrum_fp_count"] = fingerprint.get_num_on_bits()
            row['precursor_mz'] = spectrum.precursor.mz

            add_row_to_records(records, row)

        # check to see if we have enough records to log
        if len(records["id"]) % 5000 == 0:
            logging.info(f"read record {len(records['id'])}, spectrum_id={spectrum_id}")
        # check to see if we have enough records to add to the pyarrow table
        if len(records["id"]) % 25000 == 0:
            table = pa.table(records, spectrums_schema)
            tables.append(table)
            logging.info(f"created chunk {len(tables)} with {len(records['id'])} records")
            records = empty_records(spectrums_schema)
        if num is not None and len(records['id']) >= num:
            break

    table = pa.table(records, spectrums_schema)
    tables.append(table)
    logging.info(f"created final chunk {len(tables)} with {len(records['id'])} records")
    table = pa.concat_tables(tables)
    return table


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
    :param columns: list of columns to read, None=all
    :param filters: parquet predicate as a list of tuples
    :return: PyArrow table
    """
    fp = open_if_filename(fp, 'rb')
    table = pq.read_table(fp, columns=columns, filters=filters)
    if num is None:
        return table
    else:
        return table.slice(0, num)


def spectra_to_array(spectra, min_intensity=0, max_mz=2000, write_starts_stops=False):
    """
    convert an array-like of spectra to an arrow_table

    :param spectra: iteratable containing the spectrums
    :param min_intensity: the minimum intensity to set the fingerprint bit
    :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
    :param write_starts_stops: put the starts and stops arrays into the arrow Table
    :return: arrow table of spectra
    """
    records = empty_records(spectrums_schema)

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
        fingerprint = s.filter(min_intensity=min_intensity).create_fingerprint(max_mz=int(max_mz))
        row['spectrum_fp'] = fingerprint.to_numpy()
        row['spectrum_fp_count'] = fingerprint.get_num_on_bits()
        row['peptide'] = s.peptide
        row['peptide_len'] = s.peptide_len
        row['mod_names'] = s.mod_names
        row['mod_positions'] = s.mod_positions
        add_row_to_records(records, row)
    table = pa.table(records, spectrums_schema)
    return table


def spectra_to_msp(fp, spectra, annotate=annotate, ion_types=None):
    """
    write out an array-like of spectra  in msp format

    :param fp: stream or filename to write out to.  will append
    :param spectra: map containing the spectrum
    :param annotate: annotate the spectra
    :param ion_types: ion types for annotation
    """
    fp = open_if_filename(fp, 'w+')

    for i in range(len(spectra)):
        if annotate:
            msts.annotate_peptide_spectrum(spectra[i], ion_types=ion_types)
        print(spectra[i].to_msp(), file=fp)
    return


def spectra_to_mgf(fp, spectra, charge_list=None):
    """
    write out an array-like of spectra in mgf format

    :param fp: stream or filename to write out to.  will append
    :param spectra: name of the column containing the spectrum
    :param charge_list: list of charges for Mascot to search, otherwise use the CHARGE field
    """
    fp = open_if_filename(fp, 'w+')
    if charge_list is not None:
        charge_string = "CHARGE="
        for charge in charge_list:
            charge_sign = '+' if charge > 0 else '-'
            charge_string += str(charge) + charge_sign + ' '
        print(charge_string, file=fp)
 
    for i in range(len(spectra)):
        print("BEGIN IONS", file=fp)
        print(f"TITLE={spectra[i].name}", file=fp)
        # note that outputting the charge locks Mascot into searching only that charge
        if charge_list is None:
            charge_sign = '+' if spectra[i].charge > 0 else '-'
            print(f"CHARGE={spectra[i].charge}{charge_sign}", file=fp)
        if spectra[i].id is not None:
            print(f"SCANS={spectra[i].id}", file=fp)
        if spectra[i].precursor is not None:
            print(f"PEPMASS={spectra[i].precursor.mz} 1.0", file=fp)
        if spectra[i].retention_time is not None:
            print(f"RTINSECONDS={spectra[i].retention_time}", file=fp)
#        else:
#            print("RTINSECONDS=0.0", file=fp)
        for j in range(len(spectra[i].products.mz)):
            print(f'{spectra[i].products.mz[j]} {spectra[i].products.intensity[j]}', file=fp)
        print("END IONS\n", file=fp)
    return


def spectra_to_mzxml(fp, spectra, mzxml_attributes=None, min_intensity=EPSILON, compress=True, use_id_as_scan=True):
    """
    write out an array-like of spectra in mzxml format

    :param fp: stream or filename to write out to.  will not append
    :param min_intensity: the minimum intensity value
    :param spectra: name of the column containing the spectrum
    :param mzxml_attributes: dict containing mzXML attributes
    :param use_id_as_scan: use spectrum.id instead of spectrum.scan
    :param compress: should the data be compressed?
    """

    fp = open_if_filename(fp, 'w')

    """
    Notes:
    - MaxQuant has particular requirements for the MZxml that it will accept.  For hints, see
      https://github.com/OpenMS/OpenMS/blob/develop/src/openms/source/FORMAT/HANDLERS/MzXMLHandler.cpp
      - maxquant does not allow empty spectra
      - depending on scan type, may have to force scan type to Full
      - may require lowMz, highMz, basePeakMz, basePeakIntensity, totIonCurrent
      - should have activationMethod=CID?
    - ideally use xmlschema, but using json input.  However, broken xml format for mzXML and maxquant argues for print()
    - http://sashimi.sourceforge.net/schema_revision/mzXML_2.1/Doc/mzXML_2.1_tutorial.pdf
    - https://www.researchgate.net/figure/An-example-mzXML-file-This-figure-was-created-based-on-the-downloaded-data-from_fig3_268397878
    - http://www.codems.de/reports/mzxml_for_maxquant/ issues:
      - must build record offset index at the end of the xml file
      - requires line breaks after attributes
      - manufacturer should be <msManufacturer category=”msManufacturer” value=”Thermo Finnigan” />
      - msResolution appears to be ignored by maxquant                    
    """

    # default values for attributes
    info = {'startTime': 0.0, 'endTime': 1000.0, 'scanCount': 0, 'fileName': 'unknown', 'fileType': 'RAWData',
            'fileSha1':  'fc6ffa16c1a8c2a4794d4fbb0b345d08e73fe577', 'msInstrumentID': '1',
            'msManufacturer': 'Thermo Scientific', 'msModel': 'Orbitrap Fusion Lumos', 'centroided': '1'
            }

    # overwrite default values
    if mzxml_attributes is not None:
        for k, v in mzxml_attributes.items():
            info[k] = v

    if info['scanCount'] == 0:
        for i in range(len(spectra)):
            # MQ doesn't like empty spectra
            if spectra[i].products is not None and len(spectra[i].products.mz) > 0:
                info['scanCount'] += 1

    # find the min max retention time
    min_retention_time = 1.0
    max_retention_time = 0.0
    for i in range(len(spectra)):
        if spectra[i].retention_time is not None:
            if spectra[i].retention_time < min_retention_time:
                min_retention_time = spectra[i].retention_time 
            if spectra[i].retention_time > max_retention_time:
                max_retention_time = spectra[i].retention_time 
    if min_retention_time <= max_retention_time:
        info["startTime"] = min_retention_time
        info["endTime"] = max_retention_time

    index = OrderedDict()

    # create the header
    print('<?xml version="1.0" encoding="ISO-8859-1"?>', file=fp)
    print('<mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2"', file=fp)
    print('       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"', file=fp)
    print('       xsi:schemaLocation="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2 '
          'http://sashimi.sourceforge.net/schema_revision/mzXML_3.2/mzXML_idx_3.2.xsd">', file=fp)
    # MaxQuant requires startTime and endTime
    print(f'  <msRun scanCount="{info["scanCount"]}" startTime="PT{info["startTime"]}S"'
        f' endTime="PT{info["endTime"]}S">', file=fp)

    # singleton children of msRun
    print(f'    <parentFile fileName="{info["fileName"]}"', file=fp)
    print(f'                fileType="{info["fileType"]}"', file=fp)
    print(f'                fileSha1="{info["fileSha1"]}"/>', file=fp)

    print(f'    <msInstrument msInstrumentID="{info["msInstrumentID"]}">', file=fp)
    # MaxQuant requires msManufacturer to be set to 'Thermo Scientific'
    print(f'      <msManufacturer category="msManufacturer" value="{info["msManufacturer"]}"/>', file=fp)
    print(f'      <msModel category="msModel" value="{info["msModel"]}"/>', file=fp)
    print(f'    </msInstrument>', file=fp)

    print(f'    <dataProcessing centroided="1">', file=fp)
    print(f'    </dataProcessing>', file=fp)

    # now the spectra
    scan_number = 1
    for i in range(len(spectra)):
        spectrum = spectra[i].filter(min_mz=1.0, min_intensity=min_intensity).norm(max_intensity_in=999.0)

        if spectrum.products is not None and len(spectrum.products.mz) > 0:
            if use_id_as_scan:
                scan = spectrum.id
            elif spectrum.scan is not None:
                scan = spectrum.scan
            else:
                scan = scan_number
                scan_number += 1

            index[scan] = fp.tell() + 4  # add 4 to get to first character of scan tag

            retentionTime = spectrum.retention_time if spectrum.retention_time is not None else 0.001*i
            collisionEnergy = spectrum.ev if spectrum.ev is not None else spectrum.nce if spectrum.nce is not None \
                else spectrum.collision_energy if spectrum.collision_energy is not None else ""
            if spectrum.ion_mode is None or spectrum.ion_mode == "P":
                polarity = "+"
            elif spectrum.ion_mode == "N":
                polarity = "-"

            print(f'    <scan num="{scan}"', file=fp)
            print('          scanType="Full"', file=fp)  # set to MS2
            print(f'          centroided="{info["centroided"]}"', file=fp)
            print('          msLevel="2"', file=fp)
            print(f'          peaksCount="{len(spectrum.products.mz)}"', file=fp)
            print(f'          polarity="{polarity}"', file=fp)
            print(f'          retentionTime="PT{retentionTime:.4f}S"', file=fp)
            if collisionEnergy != "":
                print(f'          collisionEnergy="{collisionEnergy:.4f}"', file=fp)
            print(f'          lowMz="{min(spectrum.products.mz):.4f}"', file=fp)
            print(f'          highMz="{max(spectrum.products.mz):.4f}"', file=fp)
            basePeak = np.argmax(spectrum.products.intensity)
            print(f'          basePeakMz="{spectrum.products.mz[basePeak]:.4f}"', file=fp)
            print(f'          basePeakIntensity="{spectrum.products.intensity[basePeak]:.4f}"', file=fp)
            print(f'          totIonCurrent="{spectrum.products.intensity.sum():.4f}"', file=fp)
            print(f'          msInstrumentID="1">', file=fp)

            precursorIntensity = spectrum.precursor.intensity if spectrum.precursor is not None and spectrum.precursor.intensity is not None else 999.0
            precursorMz = spectrum.precursor.mz if spectrum.precursor is not None else ""
            precursorCharge = spectrum.charge if spectrum.charge is not None else ""
            activationMethod = spectrum.instrument_type if spectrum.instrument_type is not None else "HCD"

            print(f'      <precursorMz precursorScanNum="{scan}" precursorIntensity="{precursorIntensity:.4f}"'
                  f' precursorCharge="{precursorCharge}" activationMethod="{activationMethod}">{precursorMz:.4f}</precursorMz>', file=fp)

            # create (mz, intensity) pairs
            data = np.ravel([spectrum.products.mz, spectrum.products.intensity], 'F')
            # convert to 32 bit floats, network byte order (big endian)
            data = np.ascontiguousarray(data, dtype='>f4')
            # zlib compress
            if compress:
                data = zlib.compress(data)
                compressed_len = len(data)
            # base64
            data = base64.b64encode(data)
            if compress:
                print('      <peaks compressionType="zlib"', file=fp)
                print(f'             compressedLen="{compressed_len}"', file=fp)
            else:
                print('      <peaks compressionType="none"', file=fp)
                print('             compressedLen="0"', file=fp)
            print('             precision="32"', file=fp)
            print('             byteOrder="network"', file=fp)
            print(f'             contentType="m/z-int">{data.decode("utf-8")}</peaks>', file=fp)  # or "mz-int"

            print('    </scan>', file=fp)

    print('  </msRun>', file=fp)
    indexOffset = fp.tell() + 2  # add 2 to get to first character of index
    print('  <index name="scan">', file=fp)
    for k, v in index.items():
        print(f'    <offset id="{k}">{v}</offset>', file=fp)
    print('  </index>', file=fp)
    print(f'  <indexOffset>{indexOffset}</indexOffset>', file=fp)
    print(f'  <sha1>{info["fileSha1"]}</sha1>', file=fp)
    print('</mzXML>', file=fp)

    return

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

def load_msp2array(
    fp,
    num=None,
    id_field=0,
    set_probabilities=(0.01, 0.97, 0.01, 0.01),
    title_fields=None,
    comment_fields=None,
    min_intensity=20,
    max_mz=2000,
    parse_glyco_annotations=False,
):
    """
    Read stream or filename in MSP format and return as pyarrow Table.

    :param fp: stream or filename
    :param num: the maximum number of records to generate (None=all)
    :param id_field: the spectrum id to begin with
    :param set_probabilities: how to divide into dev, train, valid, test
    :param title_fields: dict containing column names with corresponding regex to extract field values from the TITLE. regex match group 1 is the value
    :param comment_fields: a Dict of regexes used to extract fields from the Comment field.  Form of the Dict is { comment_field_name: (regex, type, field_name)}.  For example {'Filter':(r'@hcd(\d+\.?\d* )', float, 'nce')}
    :param min_intensity: the minimum intensity to set the fingerprint bit
    :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
    :param parse_glyco_annotations: parse glycopeptide annotations
    :return: arrow table
    """

    fp = open_if_filename(fp, 'r')
    tables = []
    records = empty_records(spectrums_schema)

    spectrum_id = id_field  # a made up integer spectrum id
    # move forward to first begin ions
    line = fp.readline()
    # skip to first full entry
    while not line.lower().startswith("Name: ".lower()):
        line = fp.readline()

    while line:
        row = {}
        mz = []
        intensity = []
        annotations = []  # peak annotations
        row["id"] = spectrum_id
        spectrum_id += 1  # increment id

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
                        if title_fields:
                            for column_name, regex in title_fields.items():
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
                            elif comment_fields:
                                for subfield_name, (regex, field_type, field_name) in comment_fields.items():
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
            spectrum = init_spectrum(mz, intensity, row={}, precursor_mz=row['precursor_mz'],
                                     precursor_intensity=row.get('precursor_intensity', None))
            spectrum.charge = row.get('charge', None)
            spectrum.peptide = row.get('peptide', None)
            spectrum.peptide_len = row.get('peptide_len', None)
            spectrum.mod_names = row.get('mod_names', None)
            spectrum.mod_positions = row.get('mod_positions', None)
            row['product_massinfo'] = spectrum.product_mass_info.__dict__
            row['precursor_massinfo'] = spectrum.precursor_mass_info.__dict__
            fingerprint = spectrum.filter(min_intensity=min_intensity).create_fingerprint(max_mz=max_mz)
            row["spectrum_fp"] = fingerprint.to_numpy()
            row["spectrum_fp_count"] = fingerprint.get_num_on_bits()
            add_row_to_records(records, row)
        else:
            pass

        # check to see if we have enough records to log
        if len(records["id"]) % 5000 == 0:
            #print(f"read record {len(records)}, spectrum_id={spectrum_id}")
            logging.info(f"read record {len(records['id'])}, spectrum_id={spectrum_id}")
        # check to see if we have enough records to add to the pyarrow table
        if len(records["id"]) % 25000 == 0:
            table = pa.table(records, spectrums_schema)
            tables.append(table)
            logging.info(f"created chunk {len(tables)} with {len(records['id'])} records")
            records = empty_records(spectrums_schema)
        if num is not None and len(records['id']) >= num:
            break

    table = pa.table(records, spectrums_schema)
    tables.append(table)
    logging.info(f"created final chunk {len(tables)} with {len(records['id'])} records")
    table = pa.concat_tables(tables)
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

def load_sdf2array(
    filename,
    max_size=0,
    num=None,
    source=None,
    skip_expensive=True,
    id_field=None,
    id_field_type=None,
    min_intensity=0,
    max_mz=2000,
    fp_tolerance=0.1,
    precursor_mass_info=None,
    product_mass_info=None,
    set_probabilities=(0.01, 0.97, 0.01, 0.01),
    suppress_rdkit_warnings=True
):
    if source is None:
        source = 'nist'
    if id_field is None:
        id_field = 'NISTNO'
    if type(id_field) is int:
        # generate id using id_field as start
        current_id = id_field
    else:
        current_id = None
    if id_field_type is None:
        id_field_type = 'int'
    """
    Read file in SDF format and return as Lists of Dicts.

    :param filename: name of the file to read
    :param max_size: the maximum bounding box size (used to filter out large molecules. 0=no bound)
    :param num: the maximum number of records to generate (None=all)
    :param source: where did the sdf come from?  pubchem, nist, ?
    :param precursor_mass_info: mass information for precursor. if None, will use default.
    :param product_mass_info: mass information for product. if None, will use default.
    :param skip_expensive: skip expensive calculations for better perf
    :param id_field: field to use for the mol id, such as NISTNO, ID or _NAME (the sdf title field). if integer, use
    the value as the starting value for the id and increment for each spectrum
    :param id_field_type: the id field type, such as int or str
    :param min_intensity: the minimum intensity to set the fingerprint bit
    :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
    :param fp_tolerance: mass tolerance in Daltons used to create interval fingerprint
    :param set_probabilities: how to divide into dev, train, valid, test
    :param suppress_rdkit_warnings: don't print out spurious rdkit warnings
    :return: arrow table with records

    various issues:
    - argon has no structure
    - hydrogen causes CanonicalizeMol to fail
    - some molecules fail parsing
    - some molecules fail AdjustAromaticNs
    - some molecules fail Sanitize
    - some have empty spectra
    """

    # list of batch tables
    tables = []
    # create arrow schema and batch table
    records_schema = molecules_schema
    ecfp4_size = 4096  # size of ecfp4 fingerprint
    spectrum_fp_size = int(max_mz)  # size of spectrum fingerprint
    records_schema = set_field_int_metadata(records_schema, "ecfp4", "fp_size", ecfp4_size)
    records_schema = set_field_int_metadata(records_schema, "spectrum_fp", "fp_size", spectrum_fp_size)
    records = empty_records(records_schema)

    # Turn off RDKit error messages
    if suppress_rdkit_warnings:
        RDLogger.DisableLog('rdApp.*')

    # smarts for tms for substructure matching
    tms = Chem.MolFromSmarts("[#14]([CH3])([CH3])[CH3]")
    # only enumerate unique stereoisomers
    opts = StereoEnumerationOptions(unique=True)
    # fingerprint generator that respects counts but not chirality as mass spec tends to be chirality blind
    ecfp4 = ECFPFingerprint()

    # warning: for some reason, setting sanitize=False in SDMolSupplier can create false stereochemistry information, so
    # there is only one stereoisomer per molecule.
    # 2020-02-27  on the other hand, molvs does call sanitization in standardize_mol, so move to threed.standardize_mol
    for i, mol in enumerate(Chem.SDMolSupplier(filename, sanitize=False)):

        error_string = f"index={i} "

        if mol is None:
            logging.info(f"Unable to create molecule object for {error_string}")
            continue

        # get ids for error reporting
        if type(id_field) is not int and mol.HasProp(id_field):
            error_string += "id=" + mol.GetProp(id_field)

        if mol.HasProp("NAME"):
            error_string += " name=" + mol.GetProp("NAME")
        elif mol.HasProp("_NAME"):
            error_string += " name=" + mol.GetProp("_NAME")
        else:
            logging.info(f"{error_string} does not have a NAME or _NAME property")

        try:
            mol = masskit.small_molecule.utils.standardize_mol(mol)
        except ValueError as e:
            logging.info(f"Unable to standardize {error_string}")
            continue
        if len(mol.GetPropsAsDict()) == 0:
            logging.info(f"All molecular props unavailable {error_string}")
            # likely due to bug in rdMolStandardize.Cleanup()
            continue

        # calculate some identifiers before adding explicit hydrogens.  In particular, it seems that rdkit
        # ignores stereochemistry when creating the inchi key if you add explicit hydrogens
        new_row = {
            "has_2d": True,
            "inchi_key": Chem.inchi.MolToInchiKey(mol),
            "isomeric_smiles": Chem.MolToSmiles(mol),
            "smiles": Chem.MolToSmiles(mol, isomericSmiles=False),
        }

        # todo: note that MolToInchiKey may generate a different value than what is in the spectrum.
        # a replib example is 75992
        # this may be fixed by workarounds with rdkit inchi generation created on 10/23/2019. need to check

        if not skip_expensive:
            mol, conformer_ids, return_value = threed.create_conformer(mol)
            if return_value == -1:
                logging.info(f"Not able to create conformer for {error_string}")
                continue
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
                logging.info(f"unable to run MMFF for {error_string}")
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
                logging.info(f"{error_string} is larger than the max bound")
                continue
            try:
                new_row["num_stereoisomers"] = len(
                    tuple(EnumerateStereoisomers(mol, options=opts))
                )  # GetStereoisomerCount(mol)
            except RuntimeError as err:
                logging.info(
                    f"Unable to create stereoisomer count for {error_string}, error = {err}"
                )
                new_row["num_stereoisomers"] = None
        else:
            Chem.AssignStereochemistry(mol)  # normally done in threed.create_conformer
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
            mol.GetSubstructMatches(tms)
        )  # count of trimethylsilane matches
        new_row["exact_mw"] = Chem.rdMolDescriptors.CalcExactMolWt(mol)
        new_row["hba"] = Chem.rdMolDescriptors.CalcNumHBA(mol)
        new_row["hbd"] = Chem.rdMolDescriptors.CalcNumHBD(mol)
        new_row["rotatable_bonds"] = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        new_row["tpsa"] = Chem.rdMolDescriptors.CalcTPSA(mol)
        new_row["aromatic_rings"] = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
        new_row["formula"] = Chem.rdMolDescriptors.CalcMolFormula(mol)
        new_row["mol"] = Chem.rdMolInterchange.MolToJSON(mol)
        new_row["num_atoms"] = mol.GetNumAtoms()
        ecfp4.object2fingerprint(mol)  # expressed as a bit vector
        new_row["ecfp4"] = ecfp4.to_numpy()
        new_row["ecfp4_count"] = ecfp4.get_num_on_bits()
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

        # create useful set labels for training
        new_row["set"] = np.random.choice(
            ["dev", "train", "valid", "test"], p=set_probabilities
        )

        spectrum = None
        if source == "nist":
            # create the mass spectrum
            spectrum = init_spectrum(product_mass_info=product_mass_info,
                                     precursor_mass_info=precursor_mass_info)
            spectrum.from_mol(
                mol, skip_expensive, id_field=id_field, id_field_type=id_field_type
            )
            new_row["precursor_mz"] = spectrum.precursor.mz
            new_row["name"] = spectrum.name
            new_row["synonyms"] = json.dumps(spectrum.synonyms)
            if current_id is not None:
                spectrum.id = current_id
                current_id += 1
            new_row["id"] = spectrum.id
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
            fingerprint = spectrum.filter(min_intensity=min_intensity).create_fingerprint(max_mz=spectrum_fp_size)
            new_row["spectrum_fp"] = fingerprint.to_numpy()
            new_row["spectrum_fp_count"] = fingerprint.get_num_on_bits()

        elif source == "pubchem":
            if mol.HasProp("PUBCHEM_COMPOUND_CID"):
                new_row["id"] = mol.GetProp("PUBCHEM_COMPOUND_CID")
            if mol.HasProp("PUBCHEM_IUPAC_NAME"):
                new_row["name"] = mol.GetProp("PUBCHEM_IUPAC_NAME")
            if mol.HasProp("PUBCHEM_XLOGP3"):
                new_row["xlogp"] = float(mol.GetProp("PUBCHEM_XLOGP3"))
            if mol.HasProp("PUBCHEM_COMPONENT_COUNT"):
                new_row["component_count"] = float(
                    mol.GetProp("PUBCHEM_COMPONENT_COUNT")
                )

        add_row_to_records(records, new_row)
        if i % 10000 == 0:
            logging.info(f"processed record {i}")
        # check to see if we have enough records to add to the pyarrow table
        if len(records["id"]) % 25000 == 0:
            tables.append(pa.table(records, records_schema))
            logging.info(f"created chunk {len(tables)} with {len(records['id'])} records")
            records = empty_records(records_schema)

        if num is not None and num == i:
            break

    if records:
        tables.append(pa.table(records, records_schema))
    logging.info(f"created final chunk {len(tables)} with {len(records['id'])} records")
    table = pa.concat_tables(tables)

    return table


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

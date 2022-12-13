import numpy as np
import pandas as pd
import re
from masskit.spectrum.small_molecule import init_spectrum


float_match = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'  # regex used for matching floating point numbers


def load_mgf(fp, num=0, start_spectrum_id=0, hi_res=True, set_names=("dev", "train", "valid", "test"),
             set_probabilities=(0.01, 0.97, 0.01, 0.01), row_entries=None, title_fields=None,
             min_intensity=20, max_mz=2000):
    """
    Read stream in MGF format and return as Pandas data frame.

    :param fp: stream
    :param num: the maximum number of records to generate (0=all)
    :param start_spectrum_id: the spectrum id to begin with
    :param hi_res: should we load the spectra as hi res or unit mass?
    :param set_names: names of sets
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

    records = []
    spectrum_id = start_spectrum_id  # a made up integer spectrum id
    # move forward to first begin ions
    for line in fp:
        mz = []
        intensity = []
        row = {'id': spectrum_id, 'mol': None, 'has_2d': False, 'has_conformer': False, 'set':
               np.random.choice(set_names, p=set_probabilities)}
        if row_entries:
            row = {**row, **row_entries}  # merge in the passed in row entries
        try:
            while not line.lower().startswith("BEGIN IONS".lower()):
                line = next(fp)
            # move to first header
            line = next(fp)
            spectrum_id += 1  # increment id

            while not line.lower().startswith("END IONS".lower()):
                if len(line) > 0:
                    if line[0].isdigit() or line[0] == '-' or line[0] == ".":  # read in peak
                        values = line.split()
                        if len(values) == 2:  # have not yet dealt with case where there is a charge appended
                            mz.append(float(values[0]))
                            intensity.append(float(values[1]))
                    elif line[0] in "#;/!":  # skip comments
                        pass
                    else:  # header
                        values = line.split('=', maxsplit=1)
                        if len(values) == 2:
                            if values[0] == "TITLE":
                                row['name'] = values[1].rstrip()
                                if title_fields:
                                    for column_name, regex in title_fields.items():
                                        m = re.search(regex, values[1])
                                        if m:
                                            row[column_name] = m.group(1)
                            elif values[0] == "PEPMASS":
                                row['precursor_mz'] = float(values[1])
                            elif values[0] == "RTINMINUTES":
                                row['retention_time'] = float(values[1]) * 60.0
                            elif values[0] == "RTINSECONDS":
                                row['retention_time'] = float(values[1])
                            elif values[0] == "CHARGE":
                                number, sign = re.match(r'^(\d+)([+\-])$', values[1]).groups()
                                row['charge'] = int(sign+number)
                            else:
                                row[values[0]] = values[1].rstrip()
                line = next(fp)
        except StopIteration:
            pass
        # finalize the spectrum only if it has peaks
        if len(mz) != 0:
            spectrum = init_spectrum(hi_res)
            spectrum.from_arrays(mz, intensity, row)
            row['spectrum'] = spectrum
            row['spectrum_fp'] = spectrum.products.create_fingerprint(min_intensity=min_intensity, max_mz=max_mz)
            row['spectrum_fp_count'] = row['spectrum_fp'].GetNumOnBits()
            records.append(row)
        # check to see if we have enough records
        if num != 0 and len(records) >= num:
            break

    return pd.DataFrame(records).set_index('id')


def to_msp(fp, df_out):
    """
    write out a spectrum dataframe in msp format
    :param fp: stream to write out to
    :param df_out: dataframe containing spectra
    """
    for index, row in df_out.iterrows():
        if "name" in row:
            print(f"Name: {row['name']}", file=fp)
        if "precursor_mz" in row:
            print(f"MW: {row['precursor_mz']}", file=fp)
        if "formula" in row:
            print(f"Formula: {row['formula']}", file=fp)
        print(f"DB#: {index}", file=fp)
        num_peaks = len(row['spectrum'].products.mz)
        print(f"Num Peaks: {num_peaks}", file=fp)
        lines = int(num_peaks/5) + 1
        for i in range(lines):
            if i == lines - 1:
                num_in_line = num_peaks % 5
            else:
                num_in_line = 5
            for j in range(num_in_line):
                print(f"{row['spectrum'].products.mz[i*5+j]} {row['spectrum'].products.intensity[i*5+j]}; ",
                      file=fp, end='')
            print("", file=fp)
        print("", file=fp)
    return
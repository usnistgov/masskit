#!/usr/bin/env python

import argparse
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import gzip
import bz2
from itertools import groupby, combinations
import pyarrow as pa
import pyarrow.parquet as pq
from masskit.peptide.spectrum_generator import generate_mods
from masskit.utils.files import empty_records, add_row_to_records
from masskit.peptide.encoding import allowable_mods, calc_ions_mz, calc_precursor_mz, parse_modification_encoding

# TODO: this ought to come out of schemas.py
schema = pa.schema([
    pa.field("id", pa.uint64()),
    pa.field("charge", pa.int8()),
    pa.field("ev", pa.float64(), metadata={'description': 'collision energy (voltage drop to collision cell)'}),
    pa.field("nce", pa.float64(), metadata={'description': 'normalized collision energy'}),
    pa.field("peptide", pa.string()),
    pa.field("peptide_len", pa.int32()),
    pa.field("peptide_type", pa.dictionary(pa.int8(), pa.string())),  # tryptic, semitryptic
    # note that mod_names should be a list of dictionary arrays but it's not clear how to initialize
    # this properly with arrays of mod_names, such as in library import.  So for now, just using a list of int16
    # which matches the modifications dictionary in encoding.py
    pa.field("mod_names", pa.large_list(pa.int16())),  # should be pa.large_list(pa.dictionary(pa.int16(), pa.string()))
    pa.field("mod_positions", pa.large_list(pa.int32())),
    pa.field("precursor_mz", pa.float64()),
])


def flex_open(filename):
    magic_dict = {
        b"\x1f\x8b\x08": "gzip",
        b"\x42\x5a\x68": "bzip2"
    }

    max_len = max(len(x) for x in magic_dict)

    def get_type(filename):
        with open(filename, "rb") as f:
            file_start = f.read(max_len)
        for magic, filetype in magic_dict.items():
            if file_start.startswith(magic):
                return filetype
        return "text"

    file_type = get_type(filename)

    if file_type == "gzip":
        return gzip.open(filename, "rt")
    elif file_type == "bzip2":
        return bz2.open(filename, "rt")
    else:
        return open(filename)

"""
Iterate over a fasta file, yields tuples of (header, sequence)
"""
def fasta(filename):
    f = flex_open(filename)

    # ditch the boolean (x[0]) and just keep the header or sequence since
    # we know they alternate.
    fasta_iter = (x[1] for x in groupby(f, lambda line: line[0] == ">"))

    for header in fasta_iter:
        # drop the ">"
        headerStr = header.__next__()[1:].strip()

        # join all sequence lines to one.
        seq = "".join(s.strip() for s in fasta_iter.__next__())

        yield (headerStr, seq)


# The cleavage rule for trypsin is: after R or K, but not before P
def trypsin(residues):
    sub = ''
    while residues:
        k, r = residues.find('K'), residues.find('R')
        if k > 0 and r > 0:
            cut = min(k, r)+1 
        elif k < 0 and r < 0:
            yield residues
            return
        else:
            cut = max(k, r)+1
        sub += residues[:cut]
        residues = residues[cut:]
        if not residues or residues[0] != 'P':
            yield sub
            sub = ''

def tryptic(residues, min, max, missed):
    for miss in range(missed+1):
        if miss < 1:
            for pep in trypsin(residues):
                if min <= len(pep) <= max:
                    yield pep
        else:
            peptides = list(trypsin(residues))
            for i in range(len(peptides)+1-miss):
                pep = "".join(peptides[i:i+miss+1])
                if min <= len(pep) <= max:
                    yield pep

# Semi-Tryptic Peptides are peptides which were cleaved at the C-Terminal side of arginine (R) and lysine (K) by trypsin at one end but not the other. The figure below shows some semi-tryptic peptides.
# https://massqc.proteomesoftware.com/help/metrics/percent_semi_tryptic#:~:text=Semi%2DTryptic%20Peptides%20are%20peptides,can%20indicate%20digestion%20preparation%20problems.
def semitryptic(residues, min, max, missed):
    for pep in trypsin(residues, min, max, missed):
        yield pep
        for i in range(1,len(pep)+1-min):
            yield pep[i:]
            yield pep[:-i]

# cleave everywhere, given size constraints
def nonspecific(residues, min, max, missed):
    for sz in range(min,max+1):
        for i in range(len(residues)+1-sz):
            yield residues[i:i+sz]

def extract_peptides(cfg):
    pepset = set()
    fasta_file = fasta(cfg.filename)

    if cfg.protein.cleavage.digest == "tryptic":
        cleavage = tryptic
    elif cfg.protein.cleavage.digest == "semitryptic":
        cleavage = semitryptic
    elif cfg.protein.cleavage.digest == "nonspecific":
        cleavage = nonspecific

    for defline, protein in fasta_file:
        print("protein:", protein)
        for peptide in cleavage(protein, cfg.peptide.length.min, cfg.peptide.length.max, 1):
            print("pep:", peptide)
            pepset.add(peptide)
    peptides = list(pepset)
    peptides.sort()
    return peptides

class pepgen:

    def __init__(self, peptides, cfg):
        self.peptides = peptides
        self.nces = list(map(float, cfg.nce.split(',')))
        if cfg.mods:
            self.mods = parse_modification_encoding(cfg.mods)
        else:
            self.mods = None
        if cfg.mods:
            self.fixed_mods = parse_modification_encoding(cfg.fixed_mods)
        else:
            self.fixed_mods = None
        limits = cfg.charge.split(':')
        self.max_mods = cfg.max_mods
        min = int(limits[0])
        max = int(limits[1])
        self.charges = list(range(min,max+1))
        self.digest = cfg.digest

        # initialize data structs
        self.records = empty_records(schema)
        self.tables = []
        self.table = []

    def add_row(self, row):
        add_row_to_records(self.records, row)
        if len(self.records["id"]) % 25000 == 0:
            table = pa.table(self.records, schema)
            self.tables.append(table)
            self.records = empty_records(schema)

    def finalize_table(self):
        table = pa.table(self.records, schema)
        self.tables.append(table)
        table = pa.concat_tables(self.tables)
        return table

    def enumerate(self):
        spectrum_id = 1
        for pep in self.peptides:
            # for mpep in self.modifications(pep):
            mod_names, mod_positions = generate_mods(pep, self.mods)
            mods = list(zip(mod_positions, mod_names))
            fixed_mods_names, fixed_mods_positions = generate_mods(pep, self.fixed_mods)
            for charge in self.charges:
                for nce in self.nces:
                    row = {
                        "id": spectrum_id,
                        "charge": charge,
                        "ev": nce,
                        "nce": nce,
                        "peptide": pep,
                        "peptide_len": len(pep),
                        "peptide_type": self.digest
                    }
                    for modset in self.permute_mods(pep,mods, max_mods=self.max_mods):
                        row["mod_names"] = fixed_mods_names.copy()
                        row["mod_positions"] = fixed_mods_positions.copy()
                        if modset:
                            row["mod_positions"].extend(list(map(lambda x: x[0], modset)))
                            row["mod_names"].extend(list(map(lambda x: x[1], modset)))
                        row["precursor_mz"] = calc_precursor_mz(pep, charge, mod_names=row["mod_names"], mod_positions=row["mod_positions"])
                        self.add_row(row)
        return self.finalize_table()
    
    def permute_mods(self, pep, mods, max_mods=4):
        for i in range(min(max_mods, len(mods)+1)):
            for m in combinations(mods,i):
                yield m

@hydra.main(config_path="conf", config_name="config_fasta2peptides", version_base=None)
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # return

    # parser = argparse.ArgumentParser(description='',
    #                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('filename', help='fasta filename')
    # parser.add_argument('-c', '--charge', default='2:4', help='charge range of peptides')
    # parser.add_argument('-d', '--digest', default='tryptic', help='enzyme-style digestion (tryptic, semitryptic, or nonspecific) used to generate peptides. ')
    # parser.add_argument('-m', '--mods', type=str, 
    #                     help=f'comma separated list of variable modifications to apply to peptides. Known modifications: {allowable_mods}')
    # parser.add_argument('-f', '--fixed_mods', type=str, 
    #                     help=f'comma separated list of fixed modifications to apply to peptides. Known modifications: {allowable_mods}')
    # parser.add_argument('-n', '--nce', default='30', help='comma separated list of NCE values to apply to peptides')
    # parser.add_argument('--min', default='7', type=int, help='minimum allowable peptide length')
    # parser.add_argument('--max', default='30', type=int, help='maximum allowable peptide length')
    # parser.add_argument('--max_mods', default='4', type=int, help='maximum number of variable modifications per peptide')
    # parser.add_argument('-o', '--outfile', help='name of output file, defaults to {filename}.parquet')

    # cfg = parser.parse_args()

    peptides = extract_peptides(cfg)
    print(peptides)
    return
    pg = pepgen(peptides, cfg)
    table = pg.enumerate()
    #print(table.to_pandas())
    #data = schema.empty_table().to_pydict()
    if cfg.outfile:
        outfile = cfg.outfile
    else:
        outfile = cfg.filename + ".parquet"
    pq.write_table(table,outfile,row_group_size=50000)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from itertools import groupby, combinations
import pyarrow as pa
import pyarrow.parquet as pq
from masskit.data_specs.schemas import min_peptide_schema
from masskit.peptide.spectrum_generator import generate_mods
from masskit.utils.general import open_if_compressed
from masskit.utils.files import empty_records, add_row_to_records
from masskit.peptide.encoding import allowable_mods, calc_ions_mz, calc_precursor_mz, parse_modification_encoding

"""
Iterate over a fasta file, yields tuples of (header, sequence)
"""
def fasta(filename):
    f = open_if_compressed(filename)

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
    cterm = True
    nterm = False
    while residues:
        k, r = residues.find('K'), residues.find('R')
        if k > 0 and r > 0:
            cut = min(k, r)+1 
        elif k < 0 and r < 0:
            nterm = True
            yield (cterm, residues, nterm)
            return
        else:
            cut = max(k, r)+1
        sub += residues[:cut]
        residues = residues[cut:]
        if not residues or residues[0] != 'P':
            if not residues: nterm = True
            yield (cterm, sub, nterm)
            cterm = False
            sub = ''

def tryptic(residues, min, max, missed):
    for miss in range(missed+1):
        if miss < 1:
            for pepTuple in trypsin(residues):
                if min <= len(pepTuple[1]) <= max:
                    yield pepTuple
        else:
            peptides = list(trypsin(residues))
            for i in range(len(peptides)+1-miss):
                tups = peptides[i:i+miss+1]
                #pep = "".join(peptides[i:i+miss+1])
                pep = "".join( [ i[1] for i in tups ] )
                if min <= len(pep) <= max:
                    yield (tups[0][0], pep, tups[-1][2])

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
    ctermset = set()
    pepset = set()
    ntermset = set()
    bothset = set()
    fasta_file = fasta(cfg.filename)

    if cfg.protein.cleavage.digest == "tryptic":
        cleavage = tryptic
    elif cfg.protein.cleavage.digest == "semitryptic":
        cleavage = semitryptic
    elif cfg.protein.cleavage.digest == "nonspecific":
        cleavage = nonspecific

    for defline, protein in fasta_file:
        #print("protein:", protein)
        for (cterm, pep, nterm) in cleavage(protein, 
                                            cfg.peptide.length.min, 
                                            cfg.peptide.length.max, 
                                            cfg.protein.cleavage.max_missed):
            #print("pep:", peptide)
            if cterm and nterm:
                bothset.add(pep)
            elif cterm:
                ctermset.add(pep)
            elif nterm:
                ntermset.add(pep)
            else:
                pepset.add(pep)
    cterm = list(ctermset)
    cterm.sort()
    peptides = list(pepset)
    peptides.sort()
    nterm = list(ntermset)
    nterm.sort()
    both = list(bothset)
    both.sort()
    return (cterm, peptides, nterm, both)

def count_rhk(peptide):
    count = 0
    for res in peptide:
        if res == 'R' or res == 'H' or res == 'K':
            count += 1
    return count

class pepgen:

    def __init__(self, peptides, cfg):
        self.peptides = peptides
        self.nces = list(map(float, cfg.peptide.nce))
        self.mods = parse_modification_encoding(cfg.peptide.mods.variable)
        self.fixed_mods = parse_modification_encoding(cfg.peptide.mods.fixed)
        self.max_mods = cfg.peptide.mods.max
        min = int(cfg.peptide.charge.min)
        max = int(cfg.peptide.charge.max)
        self.charges = list(range(min,max+1))
        self.digest = cfg.protein.cleavage.digest
        self.limit_rhk = cfg.peptide.use_basic_limit
        # initialize data structs
        self.records = empty_records(min_peptide_schema)
        self.tables = []
        self.table = []

    def add_row(self, row):
        add_row_to_records(self.records, row)
        if len(self.records["id"]) % 25000 == 0:
            table = pa.table(self.records, min_peptide_schema)
            self.tables.append(table)
            self.records = empty_records(min_peptide_schema)

    def finalize_table(self):
        table = pa.table(self.records, min_peptide_schema)
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
            if self.limit_rhk:
                num_rhk = count_rhk(pep)
            for charge in self.charges:
                if self.limit_rhk:
                    if num_rhk > charge:
                        continue
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
    # print(list(map(float, cfg.peptide.nce)))

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
    # print(peptides)
    # return
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

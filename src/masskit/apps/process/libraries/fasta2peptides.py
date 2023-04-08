#!/usr/bin/env python
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from itertools import groupby, combinations
from collections import namedtuple
import pyarrow as pa
import pyarrow.parquet as pq
from masskit.data_specs.schemas import peptide_schema
from masskit.peptide.spectrum_generator import generate_mods
from masskit.utils.general import open_if_compressed
from masskit.utils.files import empty_records, add_row_to_records
from masskit.peptide.encoding import calc_precursor_mz, parse_modification_encoding

PepTuple = namedtuple('PepTuple', ['nterm', 'pep', 'cterm'])

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
    nterm = True
    cterm = False
    while residues:
        k, r = residues.find('K'), residues.find('R')
        if k > 0 and r > 0:
            cut = min(k, r)+1 
        elif k < 0 and r < 0:
            cterm = True
            yield PepTuple(nterm=nterm, pep=residues, cterm=cterm)
            return
        else:
            cut = max(k, r)+1
        sub += residues[:cut]
        residues = residues[cut:]
        if not residues or residues[0] != 'P':
            if not residues: 
                cterm = True
            yield PepTuple(nterm=nterm, pep=sub, cterm=cterm)
            nterm = False
            sub = ''

def tryptic(residues, min, max, missed):
    for miss in range(missed+1):
        if miss < 1:
            for pepTuple in trypsin(residues):
                if min <= len(pepTuple.pep) <= max:
                    yield pepTuple
        else:
            peptides = list(trypsin(residues))
            for i in range(len(peptides)+1-miss):
                tups = peptides[i:i+miss+1]
                #pep = "".join(peptides[i:i+miss+1])
                pep = "".join( [ i.pep for i in tups ] )
                if min <= len(pep) <= max:
                    yield PepTuple(nterm=tups[0].nterm, pep=pep, cterm=tups[-1].cterm)

# Semi-Tryptic Peptides are peptides which were cleaved at the C-Terminal side of arginine (R) and lysine (K) by trypsin at one end but not the other. The figure below shows some semi-tryptic peptides.
# https://massqc.proteomesoftware.com/help/metrics/percent_semi_tryptic#:~:text=Semi%2DTryptic%20Peptides%20are%20peptides,can%20indicate%20digestion%20preparation%20problems.
def semitryptic(residues, min, max, missed):
    for peptup in tryptic(residues, min, max, missed):
        yield peptup
        for i in range(1,len(peptup.pep)+1-min):
            yield PepTuple(nterm=False, pep=peptup.pep[i:], cterm=peptup.cterm)
            if i == 1:
                yield PepTuple(nterm=peptup.nterm, pep=peptup.pep[:-i], cterm=False)
            else:
                yield PepTuple(nterm=False, pep=peptup.pep[:-i], cterm=False)

# cleave everywhere, given size constraints
def nonspecific(residues, min, max, missed):
    p = PepTuple(nterm=False, pep="", cterm=False)
    for sz in range(min,max+1):
        last_residue = len(residues)-sz
        for i in range(last_residue+1):
            nterm = False
            cterm = False
            if i == 0: nterm = True
            if i == last_residue: cterm = True
            yield PepTuple(nterm=nterm, pep=residues[i:i+sz], cterm=cterm)

def extract_peptides(cfg):
    peps = {
        'nterm': set(),
        'neither': set(),
        'cterm': set(),
        'both': set()
    }
    fasta_file = fasta(cfg.input.file)

    if cfg.protein.cleavage.digest == "tryptic":
        cleavage = tryptic
    elif cfg.protein.cleavage.digest == "semitryptic":
        cleavage = semitryptic
    elif cfg.protein.cleavage.digest == "nonspecific":
        cleavage = nonspecific

    for defline, protein in fasta_file:
        print("protein:", protein)
        for p in cleavage(protein, 
                          cfg.peptide.length.min, 
                          cfg.peptide.length.max, 
                          cfg.protein.cleavage.max_missed):
            print("pep:", p)
            if p.cterm and p.nterm:
                peps['both'].add(p.pep)
            elif p.cterm:
                peps['cterm'].add(p.pep)
            elif p.nterm:
                peps['nterm'].add(p.pep)
            else:
                peps['neither'].add(p.pep)
    retval = {}
    for k in peps.keys():
        retval[k] = list(peps[k])
        retval[k].sort()
    return retval

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
        self.records = empty_records(peptide_schema)
        self.tables = []
        self.table = []

    def add_row(self, row):
        add_row_to_records(self.records, row)
        if len(self.records["id"]) % 25000 == 0:
            table = pa.table(self.records, peptide_schema)
            self.tables.append(table)
            self.records = empty_records(peptide_schema)

    def finalize_table(self):
        table = pa.table(self.records, peptide_schema)
        self.tables.append(table)
        table = pa.concat_tables(self.tables)
        return table

    def enumerate(self):
        spectrum_id = 1
        for ptype, peps in self.peptides.items():
            # print(ptype, len(peps))
            args = {}
            if ptype == 'nterm':
                args['n_peptide'] = True
            elif ptype == 'cterm':
                args['c_peptide'] = True
            elif ptype == 'both':
                args['n_peptide'] = True
                args['c_peptide'] = True
            for pep in peps:
                mod_names, mod_positions = generate_mods(pep, self.mods, **args)
                mods = list(zip(mod_positions, mod_names))
                fixed_mods_names, fixed_mods_positions = generate_mods(pep, self.fixed_mods, **args)
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
                            "peptide_type": self.digest,
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

    peptides = extract_peptides(cfg)
    # print(peptides)
    # return
    pg = pepgen(peptides, cfg)
    table = pg.enumerate()
    #print(table.to_pandas())
    #data = schema.empty_table().to_pydict()
    if cfg.output.file:
        outfile = cfg.output.file
    else:
        outfile = cfg.input.file + ".parquet"
    pq.write_table(table,outfile,row_group_size=50000)


if __name__ == "__main__":
    main()

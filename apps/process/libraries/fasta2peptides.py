#!/usr/bin/env python

import argparse
import gzip
import bz2
from itertools import groupby, combinations
import pyarrow as pa
import pyarrow.parquet as pq
from masskit.utils.files import empty_records, add_row_to_records
from masskit.peptide.encoding import mod_sites, mod_masses

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
    pa.field("mod_positions", pa.large_list(pa.int32()))
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
def trypsin(bases):
    sub = ''
    while bases:
        k, r = bases.find('K'), bases.find('R')
        cut = min(k, r)+1 if k > 0 and r > 0 else max(k, r)+1
        sub += bases[:cut]
        bases = bases[cut:]
        if not bases or bases[0] != 'P':
            yield sub
            sub = ''

# Semi-Tryptic Peptides are peptides which were cleaved at the C-Terminal side of arginine (R) and lysine (K) by trypsin at one end but not the other. The figure below shows some semi-tryptic peptides.
# https://massqc.proteomesoftware.com/help/metrics/percent_semi_tryptic#:~:text=Semi%2DTryptic%20Peptides%20are%20peptides,can%20indicate%20digestion%20preparation%20problems.
def semitryptic(bases):
    pass

# cleave everywhere, given size constraints
def nonspecific(bases):
    pass

def extract_peptides(args):
    pepset = set()
    fasta_file = fasta(args.filename)
    for defline, protein in fasta_file:
        #print(protein)
        #print(list(trypsin(protein)))
        for peptide in trypsin(protein):
            if args.min <= len(peptide) <= args.max:
                #print(peptide)
                pepset.add(peptide)
    peptides = list(pepset)
    peptides.sort()
    return peptides

class pepgen:

    def __init__(self, peptides, args):
        self.peptides = peptides
        self.nces = list(map(float, args.nce.split(',')))
        if args.mods:
            self.mods = args.mods.split(',')
        else:
            self.mods = None
        limits = args.charge.split(':')
        min = int(limits[0])
        max = int(limits[1])
        self.charges = list(range(min,max+1))
        self.digest = args.digest

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
            mods = self.get_modlist(pep)
            #print(mods)
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
                    for modset in self.permute_mods(pep,mods):
                        if modset:
                            row["mod_positions"] = list(map(lambda x: x[0], modset))
                            row["mod_names"] = list(map(lambda x: x[1], modset))
                        #print(row)
                        self.add_row(row)
        return self.finalize_table()

    def get_modlist(self, pep):
        mods = []
        if self.mods:
            for mod in self.mods:
                # for each mod, go through allowed sites
                for site in mod_sites[mod]['sites']:
                    # N term
                    if site == '0':
                        mods.append( (0, mod_masses.df.at[mod, 'id']) )
                    # C term
                    elif site == '-1':
                        mods.append( (len(pep)-1, mod_masses.df.at[mod, 'id']) )
                    # specific amino acids
                    else:
                        for i, aa in enumerate(pep):
                            if aa == site:
                                mods.append( (i, mod_masses.df.at[mod, 'id']) )
        return mods
    
    def permute_mods(self, pep, mods):
        for i in range(len(mods)+1):
            for m in combinations(mods,i):
                yield m

def main():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('filename', help='fasta filename')
    parser.add_argument('-c', '--charge', default='1:2', help='charge range of peptides')
    parser.add_argument('-d', '--digest', default='tryptic', help='enzyme-style digestion (tryptic, semitryptic, or nonspecific) used to generate peptides. ')
    parser.add_argument('-m', '--mods', type=str, 
                        help=f'comma separated list of post-translational modifications to apply to peptides. Known modifications: {list((mod_sites.keys()))}')
    parser.add_argument('-n', '--nce', default='42', help='comma separated list of NCE values to apply to peptides')
    parser.add_argument('--min', default='6', type=int, help='minimum allowable peptide length')
    parser.add_argument('--max', default='20', type=int, help='maximum allowable peptide length')
    parser.add_argument('-o', '--outfile', help='name of output file, defaults to {filename}.parquet')

    args = parser.parse_args()

    peptides = extract_peptides(args)
    #print(peptides)
    pg = pepgen(peptides, args)
    table = pg.enumerate()
    #print(table.to_pandas())
    #data = schema.empty_table().to_pydict()
    if args.outfile:
        outfile = args.outfile
    else:
        outfile = args.filename + ".parquet"
    pq.write_table(table,outfile,row_group_size=50000)


if __name__ == "__main__":
    main()

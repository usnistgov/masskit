#!/usr/bin/env python
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from itertools import groupby, combinations
from collections import namedtuple
import pyarrow as pa
import pyarrow.parquet as pq
from masskit.data_specs.file_schemas import schema_groups
from masskit.peptide.spectrum_generator import generate_mods
from masskit.utils.general import open_if_filename
from masskit.utils.files import empty_records, add_row_to_records, records2table
from masskit.peptide.encoding import calc_precursor_mz, parse_modification_encoding
from masskit.utils.general import MassKitSearchPathPlugin
from hydra.core.plugins import Plugins


Plugins.instance().register(MassKitSearchPathPlugin)
PepTuple = namedtuple('PepTuple', ['nterm', 'pep', 'cterm'])


def fasta(filename):
    """
    Iterate over a fasta file, yields tuples of (header, sequence)

    :param filename: name of fasta file
    :return: 
    """
    f = open_if_filename(filename, 'rt')

    # ditch the boolean (x[0]) and just keep the header or sequence since
    # we know they alternate.
    fasta_iter = (x[1] for x in groupby(f, lambda line: line[0] == ">"))

    for header in fasta_iter:
        # drop the ">"
        headerStr = header.__next__()[1:].strip()

        # join all sequence lines to one.
        seq = "".join(s.strip() for s in fasta_iter.__next__())

        yield (headerStr, seq)


def fasta_parse_id(header):
    """
    Extract an accession from a UniProt fasta file header string
    The UniProtKB uses the following format:
    >db|UniqueIdentifier|EntryName ProteinName OS=OrganismName OX=OrganismIdentifier \
                                [GN=GeneName ]PE=ProteinExistence SV=SequenceVersion

    :param header: UniProt formated fasta header string
    :return: accession string
    """
    db, uid, remainder = header.split("|", 2)
    return db + '|' + uid


def trypsin(residues):
    """
    Follow the rules for a tryptic digestion enzyme to yield peptides from a given
    protein string. The cleavage rule for trypsin is: after R or K, but not before P

    :param residues: protein string
    :return: yield peptides in order
    """

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
    """
    Control a tryptic digest by limiting the length and allowing for missed cleavages

    :param residues: protein string
    :param min: minimum length for returned peptides
    :param max: maximum length for returned peptides
    :param missed: maximum number of possible missed cleavages
    :return: yield peptides in order
    """
    for miss in range(missed+1):
        if miss < 1:
            for pepTuple in trypsin(residues):
                if min <= len(pepTuple.pep) <= max:
                    yield pepTuple
        else:
            peptides = list(trypsin(residues))
            for i in range(len(peptides)+1-miss):
                tups = peptides[i:i+miss+1]
                # pep = "".join(peptides[i:i+miss+1])
                pep = "".join([i.pep for i in tups])
                if min <= len(pep) <= max:
                    yield PepTuple(nterm=tups[0].nterm, pep=pep, cterm=tups[-1].cterm)


def semitryptic(residues, min, max, missed):
    """
    Yield semi-Tryptic Peptides are peptides which were cleaved at the C-Terminal
    side of arginine (R) and lysine (K) by trypsin at one end but not the other. 

    :param residues: protein string
    :param min: minimum length for returned peptides
    :param max: maximum length for returned peptides
    :param missed: maximum number of possible missed cleavages
    :return: yield peptides in order
    """
    for peptup in tryptic(residues, min, max, missed):
        yield peptup
        for i in range(1, len(peptup.pep)+1-min):
            yield PepTuple(nterm=False, pep=peptup.pep[i:], cterm=peptup.cterm)
            if i == 1:
                yield PepTuple(nterm=peptup.nterm, pep=peptup.pep[:-i], cterm=False)
            else:
                yield PepTuple(nterm=False, pep=peptup.pep[:-i], cterm=False)


def nonspecific(residues, min, max, missed):
    """
    Yield eptides are peptides which were cleaved at every location. 

    :param residues: protein string
    :param min: minimum length for returned peptides
    :param max: maximum length for returned peptides
    :param missed: maximum number of possible missed cleavages
    :return: yield peptides in order
    """
    for sz in range(min, max+1):
        last_residue = len(residues)-sz
        for i in range(last_residue+1):
            nterm = False
            cterm = False
            if i == 0:
                nterm = True
            if i == last_residue:
                cterm = True
            yield PepTuple(nterm=nterm, pep=residues[i:i+sz], cterm=cterm)


def extract_peptides(cfg):
    """
    Breakdown all peptides from all of the proteins in a fasta file according to the
    specified digestion strategy.

    :param cfg: configuration parameters
    :return: non-redundant list of peptides.
    """
    peps = {
        'nterm': set(),
        'neither': set(),
        'cterm': set(),
        'both': set()
    }
    fasta_file = fasta(Path(cfg.input.file).expanduser())

    if cfg.protein.cleavage.digest == "tryptic":
        cleavage = tryptic
    elif cfg.protein.cleavage.digest == "semitryptic":
        cleavage = semitryptic
    elif cfg.protein.cleavage.digest == "nonspecific":
        cleavage = nonspecific

    pep2proteins = {}
    for defline, protein in fasta_file:
        protein_id = fasta_parse_id(defline)
        # print("protein id:", protein_id)
        # print("protein:", protein)
        for p in cleavage(protein,
                          cfg.peptide.length.min,
                          cfg.peptide.length.max,
                          cfg.protein.cleavage.max_missed):
            # print("pep:", p)
            # Add protein to mapping
            if p.pep in pep2proteins:
                pep2proteins[p.pep].add(protein_id)
            else:
                pep2proteins[p.pep] = {protein_id}

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
    return retval, pep2proteins


def count_rhk(peptide):
    """
    Return a count of the basic residues in a peptide

    :param peptide:
    :return: count of basic residues
    """
    count = 0
    for res in peptide:
        if res == 'R' or res == 'H' or res == 'K':
            count += 1
    return count


class pepgen:

    def __init__(self, peptides, pep2proteins, cfg):
        """
        initialize the decorated peptide generator

        :param peptides: list of all peptide strings
        :param pep2proteins: dict mapping pep string to set of protein identifiers
        :return: peptide generator object
        """
        self.peptides = peptides
        self.pep2proteins = pep2proteins
        self.nces = list(map(float, cfg.peptide.nce))
        self.mods = parse_modification_encoding(cfg.peptide.mods.variable)
        self.fixed_mods = parse_modification_encoding(cfg.peptide.mods.fixed)
        self.max_mods = cfg.peptide.mods.max
        min = int(cfg.peptide.charge.min)
        max = int(cfg.peptide.charge.max)
        self.charges = list(range(min, max+1))
        self.digest = cfg.protein.cleavage.digest
        self.limit_rhk = cfg.peptide.use_basic_limit
        # initialize data structs
        self.records = empty_records(schema_groups["peptide"]["flat_schema"])
        self.tables = []
        self.table = []

    def add_row(self, row):
        """
        add row to row cache

        :param row: the new row
        """
        # self.records['annotations']=None
        # self.records['intensity']=None
        # self.records['stddev']=None
        # self.records['product_massinfo']=None
        # self.records['mz']=None
        # self.records['precursor_intensity']=None
        # self.records['precursor_massinfo']=None
        # self.records['starts']=None
        # self.records['stops']=None
        add_row_to_records(self.records, row)
        if len(self.records["id"]) % 25000 == 0:
            self.tables.append(records2table(
                self.records, schema_groups["peptide"]))
            self.records = empty_records(
                schema_groups["peptide"]["flat_schema"])

    def finalize_table(self):
        """
        retrieve the completed table

        :return: a pyarrow table
        """
        self.tables.append(records2table(
            self.records, schema_groups["peptide"]))
        table = pa.concat_tables(self.tables)
        return table

    def enumerate(self):
        """
        Perform the work to generate all of the decorated peptides

        :return: The completed pyarrow table
        """
        spectrum_id = 0
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
                mod_names, mod_positions = generate_mods(
                    pep, self.mods, **args)
                mods = list(zip(mod_positions, mod_names))
                fixed_mods_names, fixed_mods_positions = generate_mods(pep,
                                                                       self.fixed_mods,
                                                                       **args)
                if self.limit_rhk:
                    num_rhk = count_rhk(pep)
                for charge in self.charges:
                    if self.limit_rhk:
                        if num_rhk > charge:
                            continue
                    for nce in self.nces:
                        row = {
                            "charge": charge,
                            "ev": nce,
                            "nce": nce,
                            "peptide": pep,
                            "peptide_len": len(pep),
                            "peptide_type": self.digest,
                        }
                        for modset in self.permute_mods(pep,
                                                        mods,
                                                        max_mods=self.max_mods):
                            row["id"] = spectrum_id
                            row["mod_names"] = fixed_mods_names.copy()
                            row["mod_positions"] = fixed_mods_positions.copy()
                            if modset:
                                row["mod_positions"].extend(
                                    list(map(lambda x: x[0], modset)))
                                row["mod_names"].extend(
                                    list(map(lambda x: x[1], modset)))
                            row["precursor_mz"] = calc_precursor_mz(pep,
                                                                    charge,
                                                                    mod_names=row["mod_names"],
                                                                    mod_positions=row["mod_positions"])
                            row["protein_id"] = list(self.pep2proteins[pep])
                            self.add_row(row)
                            spectrum_id += 1
        return self.finalize_table()

    def permute_mods(self, pep, mods, max_mods=4):
        """
        Yield all of the possible permutations of the set of modifications
        applied to the given peptide

        :param pep: a peptide string
        :param mods: list of mods
        :param max_mods: maximum number of modifications to apply at one time
        :return: yield a peptide with modifications applied
        """
        for i in range(min(max_mods, len(mods)+1)):
            for m in combinations(mods, i):
                yield m


@hydra.main(config_path="conf", config_name="config_fasta2peptides", version_base=None)
def fasta2peptides_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # print(list(map(float, cfg.peptide.nce)))

    peptides, pep2proteins = extract_peptides(cfg)
    # print(peptides)
    # return
    pg = pepgen(peptides, pep2proteins, cfg)
    table = pg.enumerate()
    # print(table.to_pandas())
    # data = schema.empty_table().to_pydict()
    if cfg.output.file:
        outfile = Path(cfg.output.file).expanduser()
    else:
        outfile = Path(cfg.input.file).expanduser().with_suffix(".parquet")
    pq.write_table(table, outfile, row_group_size=50000)


if __name__ == "__main__":
    fasta2peptides_app()

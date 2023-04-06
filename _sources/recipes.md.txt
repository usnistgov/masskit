# Recipes

## Library generation

### Protein sequences to peptide library

To generate a library of peptides, which is typically the first step in generating a peptide
spectral library, use the program `fasta2peptides.py`. This program takes protein sequences
in fasta format and generates a peptide library in parquet format. The configuration for this program is
contained in the `conf/config_fasta2peptides.yaml` file in the same directory as
`fasta2peptides.py`.

* to change the name of the input file, specify `input.file=myfilename.fasta` on the
command line.
* to change the name of the output file, specify `output.file=myfilename.parquet` on
the command line.
* the program supports the following options:
  * `protein.cleavage.digest=tryptic` where the digest can be tryptic, semitryptic, or nonspecific
  * `protein.cleavage.max_missed=1` which is the number of missed cleavages allowed
  * `peptide.charge.min=2` and `peptide.charge.max=4` sets the range of charges generated
  * `peptide.mods.fixed=Carbamidomethyl` is a list of fixed modifications, using a [string format](#modification-specification)
  * `peptide.mods.variable=Phospho{S/T}#Oxidation#Acetyl{^}` is a list of variable modifications, using a [string format](#modification-specification)
  * `peptide.length.min=7` and `peptide.length.max=30` are the minimum and maximum sizes of
  the peptide generated.
  * `peptide.nce=[30]` is a list of NCE values to generate per peptide
  * `peptide.use_basic_limit=True` limits the max charge of a peptide to the number of basic residues

## Modification specification

Modifications in Masskit are taken from [Unimod](https://www.unimod.org) and identified using either
the `Interim name` for naming by string or the `Accession #` for naming by integer.

Site encoding of a modification:

* `A`-`Y` amino acid

which can be appended with a modification position encoding:

* `0` peptide N-terminus
* `.` peptide C-terminus
* `^` protein N-terminus
* `$` protein C-terminus

So that `K.` means lysine at the C-terminus of the peptide.
The position encoding can be used separately, e.g. `^` means apply to any protein N-terminus,
regardless of amino acid

A list of modifications is separated by hashes:
`Phospho{S}#Methyl{0/I}#Carbamidomethyl#Deamidated{F^/Q/N}`

An optional list of sites is specified within the `{}` for each modification.
If there are no `{}` then a default set of sites is used.  
Multiple sites are separated by a `/`.

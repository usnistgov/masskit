# Recipes

## Library generation

### Protein sequences to peptide library

To generate a library of peptides, which is typically the first step in generating a peptide
spectral library, use the program `fasta2peptides`. This program takes protein sequences
in fasta format and generates a peptide library in parquet format. The default configuration for this program can be see by running `fasta2peptides -h`.

* to change the name of the input file, specify `input.file=myfilename.fasta` on the
command line.
* to change the name of the output file, specify `output.file=myfilename.parquet` on
the command line.
* the program supports the following options:
  * `protein.cleavage.digest=tryptic` where the digest can be tryptic, semitryptic, or nonspecific
  * `protein.cleavage.max_missed=1` which is the number of missed cleavages allowed
  * `peptide.charge.min=2` and `peptide.charge.max=4` sets the range of charges generated
  * `peptide.mods.fixed=Carbamidomethyl` is a list of fixed modifications, using a [string format](#modification-specification).
  * `peptide.mods.variable=Phospho{S/T}#Oxidation#Acetyl{^}` is a list of variable modifications, using a [string format](#modification-specification).
  * `peptide.length.min=7` and `peptide.length.max=30` are the minimum and maximum sizes of
  the peptide generated.
  * `peptide.nce=[30]` is a list of NCE values to generate per peptide
  * `peptide.use_basic_limit=True` limits the max charge of a peptide to the number of basic residues

An example command line: `fasta2peptides input.file=uniprot.fasta output.file=uniprot_peptides.parquet`

## Library import

Masskit computational pipelines operate on standardized parquet and arrow files.
These open source columnar data stores allow for high performance from vectorization
and parallelization and, by checking and correcting
data at import and placing it in well specified fields, modularizes and
simplifies computational tasks by avoiding data errors that can arise in
computational pipelines that depend on ill-specified file formats. The
command line program `batch_converter` is used to load different file formats
into standardized parquet and arrow files and to convert these standardized
files into common file formats. For performance, `batch_converter` is
parallelized and operates on batches so that it can handle any size of
file without exhausting memory.

### SDF Molfiles to small molecule libraries

To convert an SDF file (also known as a Molfile) into parquet format, use a
command line of the format `batch_converter input.file.names=my_sdf.sdf output.file.name=my_sdf output.file.types=[parquet]`.

### CSV file to small molecule libraries

To convert an CSV file that includes SMILES molecular specifications into
parquet format, use a command line of the format `batch_converter input.file.names=my_csv.csv output.file.name=my_csv output.file.types=[parquet]`.

#### SDF files with incorrect encoding or pre-v2000 format

Some SDF files include characters encoded using non-ASCII encodings,
such as Latin-1 (ISO-8859-1) while rdkit and python support ASCII and
UTF-8 (unicode).  Other SDF files are written in a pre version v2000
format that does not include 'M  END' section separators.  To address
these issues, use the command line program
`rewrite_sdf input.file.name=my_input.sdf output.file.name=my_output.sdf`.
If the encoding is not latin-1, set the `input.file.encoding` option
to the encoding used.

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

Note that this string may have be escaped when
using a command line like bash, e.g. `peptide.mods.fixed='"TMT6plex#Carbamidomethyl"'`

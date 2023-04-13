import pytest
import os

def test_fasta2peptides_main(create_peptide_parquet_file):
    assert os.path.exists(create_peptide_parquet_file)
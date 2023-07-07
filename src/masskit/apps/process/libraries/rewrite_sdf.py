#!/usr/bin/env python

import argparse
import rich.progress
import re

"""
reformat old fashioned sdf files into those usable by rdkit by appending missing M END lines
also turn any latin-1 characters into the unicode expected by rdkit

"""

parser = argparse.ArgumentParser()
parser.add_argument('--input', default="")
parser.add_argument('--output', default="")
parser.add_argument('--encoding', default="latin-1")
args = parser.parse_args()

# match connectivity block line
connectivity_block = re.compile(r'^\s{0,2}\d{1,3}\s{0,2}\d{1,3}\s{1,2}\d+\s{1,2}(\d{1,3}|\s)\s{1,2}(\d{1,3}|\s)\s{1,2}(\d{1,3}|\s)\s{0,2}(\d{1,3}|\s)?\s*$')
                     
with rich.progress.open(args.input, 'rt', encoding=args.encoding, 
                        description=f"{args.input} -> {args.output}") as fin:
    with open(args.output, 'w') as fout:
        previous_line = ""
        first_line = True
        in_m_block = False
        no_m_end = True
        for line in fin:
            if not first_line:
                if connectivity_block.match(previous_line) and \
                not connectivity_block.match(line):
                    in_m_block = True
                    no_m_end = True
                if in_m_block and line.strip() == 'M  END':
                    no_m_end = False
                if in_m_block and (line.startswith("> ") or line.startswith('$$$$')):
                    in_m_block = False
                    if no_m_end:
                        fout.write('M  END\n')
            first_line = False
            fout.write(line)
            previous_line = line

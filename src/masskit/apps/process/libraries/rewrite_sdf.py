#!/usr/bin/env python

import argparse
import rich.progress

"""
reformat sdf file from NIST library to format usable by rdkit by appending missing M END lines

to create sdf files from nist libraries, use commands like

wine lib2nist64.exe /log9 hr_msms_nist2020_v47.log /OutSDF /ToV2000 z:\\home\\lyg\\data\\nist\\2020\\v47\\hr_msms_nist2020_v42 =hr_msms_nist2020_v47.orig.sdf
# get rid of non unicode characters in chemical names.
iconv -f utf-8 -t utf-8 -c ~/nist/hr_msms_nist2020_v47.orig.sdf | uniq > ~/nist/hr_msms_nist2020_v42.sdf
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input', default="")
parser.add_argument('--output', default="")
args = parser.parse_args()

with rich.progress.open(args.input, 'rt', description=f"{args.input} -> {args.output}") as fin:
    with open(args.output, 'w') as fout:
        previous_line = ""
        for line in fin:
            if line.strip() == '>  <NAME>' and previous_line.strip() != 'M  END':
                fout.write('M  END\n')
            fout.write(line)
            previous_line = line

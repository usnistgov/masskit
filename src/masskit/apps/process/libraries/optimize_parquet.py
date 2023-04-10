#!/usr/bin/env python
import argparse
import pyarrow.parquet as pq

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("source", type=str, help="The name of the Parquet file to read.")
parser.add_argument("target", type=str, help="The name of the new Parquet file to be created.")
parser.add_argument('-m', '--maxsize', type=int, default=5000, help="The maximum number of rows to put in each row group.")
args=parser.parse_args()

if len(sys.argv) < 4:
    print(f"Usage: {sys.argv[0]} infile outfile row_group_size")
    sys.exit(-1)

maxrows = args.maxsize
table = pq.read_table(args.source)
pq.write_table(table, args.target, row_group_size=maxrows)

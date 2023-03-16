#!/usr/bin/env python

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import math
from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel

# Initial parsing geneated with duckview
# filename -c --columns collist -s --start 0 -e --end 100 -i --info
import argparse

class parquet_info:

    def __init__(self, args):
        self.parquet_file = pq.ParquetFile(args.filename)
        self.verbose = args.verbose
    
    def fdt(self, type):
        if pa.types.is_dictionary(type):
            return f"dict<{type.index_type}:{type.value_type}>"
        elif pa.types.is_large_list(type):
            return f"large_list<{type.value_type}>"
        else:
            return type

    def get_subfields(self, field):
        res = ""
        for i in range(field.num_fields):
            res += f"{field[i].name}: {self.fdt(field[i].type)}\n"
        return res

    def format_field(self, field,idx):
        name = f"[b]{field.name}[/b]"
        if pa.types.is_struct(field.type):
            type = f"[blue]struct\n{self.get_subfields(field.type)}"
        elif pa.types.is_large_list(field.type):
            type = f"[blue]{self.fdt(field.type)}"
        elif pa.types.is_dictionary(field.type):
            type = f"[blue]{self.fdt(field.type)}"
        else:
            type = f"[yellow]{field.type}"
        return f"{idx}: {name}\n{type}"

    def print(self):
        metadata = self.parquet_file.metadata
        #print(metadata)
        #print(metadata.num_rows, metadata.num_row_groups, metadata.row_group(0).num_rows)
        schema = metadata.schema.to_arrow_schema()
        #print(schema.names)
        #print(len(schema))
        # for i in range(len(schema)):
        #     print(i, schema.field(i).name)
        if self.verbose:
            renderables = [Panel(self.format_field(field,idx), expand=True) for field, idx in zip(schema,range(len(schema)))]
        else:
            #renderables = [Panel(f"{idx}: {field.name}", expand=True) for field, idx in zip(schema,range(len(schema)))]
            renderables = [f"{idx:#2}: [b]{field.name}[/b]" for field, idx in zip(schema,range(len(schema)))]
        console = Console()
        console.print(Columns(renderables))

class parquet_data:

    def __init__(self, args):
        self.columns = args.columns
        self.start = args.start
        self.end = args.end
        self.verbose = args.verbose
        self.parquet_file = pq.ParquetFile(args.filename)
        self.metadata = self.parquet_file.metadata
        self.calc_row_groups()
        self.select_columns()

    def calc_row_groups(self):
        nrows = self.metadata.num_rows
        ngroups = self.metadata.num_row_groups
        group_size = self.metadata.row_group(0).num_rows
        
        start_group = math.floor(self.start / group_size)
        end_group = math.floor(self.end / group_size)
        self.groups = list(range(start_group, end_group+1))
        self.start_idx = start_group * group_size

    def select_columns(self):
        if self.columns == None:
            self.col_list = None
            return
        # ["id", "charge", "peptide", "peptide_len"]
        self.col_list = []
        names = self.metadata.schema.to_arrow_schema().names
        for col in self.columns.split(','):
            if col.isdigit():
                self.col_list.append(names[int(col)])
            else:
                self.col_list.append(col)
        print(self.col_list)

    def print(self):
        rg = self.parquet_file.read_row_groups(self.groups, columns = self.col_list)
        df = rg.to_pandas()
        df.index = np.arange(self.start_idx, len(df)+self.start_idx)
        with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', None,
                       #'display.precision', None,
                       'display.colheader_justify', 'right'):
            print(df.loc[self.start:self.end])
        # for batch in self.parquet_file.iter_batches(batch_size=50):
        #     print(batch.to_pandas())
        #     return

def main():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('filename', help='parquet filename')
    parser.add_argument('-c', '--columns', default=None, help='list of columns to display')
    parser.add_argument('-s', '--start', default=0, type=int, help='row starting index')
    parser.add_argument('-e', '--end', default=20, type=int, help='row ending index')
    parser.add_argument('-i', '--info', action='store_true', help='display file info')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose, only applies to info')
    args = parser.parse_args()

    # print(args.filename)
    # print(args.columns)
    # print(args.start)
    # print(args.end)
    # print(args.info)


    if args.info:
        pqinfo = parquet_info(args)
        pqinfo.print()
        #print_info(args.filename, args.verbose)
        return
    
    pqdata = parquet_data(args)
    pqdata.print()


if __name__ == "__main__":
    main()

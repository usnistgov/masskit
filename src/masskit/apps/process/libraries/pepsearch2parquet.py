#!/usr/bin/env python
import pandas as pd
import argparse
import glob
import re
import json
import hashlib
import logging
import sys
from masskit.utils.hitlist import Hitlist

"""
load one or more mspepsearch tsv files into a arrow table and put the result in an parquet file
formatted as a pandas dataframe with a multiindex (query_id, hit_id)
note that the pytables package must be installed
read the output of this program via something, something ... pd.read_hdf('filename.h5', 'search_results')
"""

float_match = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'  # regex used for matching floating point numbers

# convert mspepsearch column names to standard search results
# do not include 'uNIST r.n.': 'query_id', 'NIST r.n.': 'hit_id' as this is decided by program logic
pepsearch_names = {'Unknown': 'query_name', 'Peptide': 'hit_name', 'Dot Product': 'cosine_score'}

# types of various mspepsearch columns
pepsearch_types = {'uNIST r.n.': int, 'NIST r.n.': int, 'Unknown': str, 'Charge': int,
                   'Peptide': str, 'Dot Product': float, 'Id': int, 'uId': int, 'Num': int, 'Rank': int, 'MF': float}


def read_pepsearch(file_in, df_in=None, filename_fields_in=None, title_fields_in=None):
    """
    read in a file from mspepsearch and turn it into a standard arrow table
    :param file_in: stream or filename of tsv file to read in
    :param df_in: the dataframe to concatenate to, otherwise a dataframe is created
    :param filename_fields_in: regex for pulling fields from the filename
    :param title_fields_in: regex for pulling fields from the Unknown name
    :return: the dataframe
    """

    new_df = pd.read_csv(file_in, sep='\t', skiprows=3, skipfooter=2,
                         engine='python')  # index_col=['uNIST r.n.', 'NIST r.n.'],
    # new_df = new_df.rename_axis(index=['query_id', 'hit_id'])
    # eliminate spectra without hits
    logging.info(f'number of hits in file: {len(new_df.index)}')
    new_df = new_df[new_df['Id'].notnull()]
    logging.info(f'number of non-null hits in file: {len(new_df.index)}')
    new_df['Charge'] = new_df['Charge'].fillna(-1)

    # recast types
    # first filter column list to those that are in both the psearch_types list and the new datafram
    filtered_pepsearch_types = {k: pepsearch_types[k] for k in pepsearch_types.keys() & new_df.columns.values}
    # print(filtered_pepsearch_types)
    # for k in filtered_pepsearch_types:
    #     if pepsearch_types[k] == int:
    #         print(k, pepsearch_types[k])
    #         print(f"\n{k} NULLs:\n")
    #         print(new_df[new_df[k].isnull()])
    #         print(f"\n{k} NAs:\n")
    #         print(new_df[new_df[k].isna()])
    # return
    new_df = new_df.astype(filtered_pepsearch_types)

    # if the nistno can be found in the results and all are nonzero, use that for the index
    if 'uNIST r.n.' in new_df and (new_df['uNIST r.n.'] == 0).sum() == 0:
        new_df.set_index(['uNIST r.n.', 'NIST r.n.'], inplace=True)
    elif 'Unknown' in new_df and new_df['Unknown'].isnull().sum() == 0:
        # number groups with same Unknown using a hash
        new_df['query_id'] = new_df.apply(lambda x: int(hashlib.md5(x['Unknown'].encode('utf-8')).hexdigest()[:8], 16),
                                          axis=1)
        new_df.set_index(['query_id', 'NIST r.n.'], inplace=True)
    else:
        logging.error("Unable to find query index column")
        sys.exit(2)
    new_df = new_df.rename_axis(index=['query_id', 'hit_id'])

    # convert column names
    new_df = new_df.rename(columns=pepsearch_names)
# Id, NIST r.n., Charge
    if filename_fields_in:
        for column_name, regex in filename_fields_in.items():
            m = re.search(regex, file_name)
            if m:
                new_df[column_name] = m.group(1)
    if title_fields_in:
        # create empty columns for below
        for column_name in title_fields_in:
            new_df[column_name] = None

        for i, row in new_df.iterrows():
            if title_fields_in:
                for column_name, regex in title_fields_in.items():
                    m = re.search(regex, row['query_name'])
                    if m:
                        new_df.at[i, column_name] = m.group(1)
    if df_in is None:
        df_in = new_df
    else:
        df_in = pd.concat([df_in, new_df])
    return df_in


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="convert tab delimited output from MSPepSearch to a parquet file.")
    parser.add_argument('-i', '--input', help="input tsv file(s). can be wildcarded",
                        default="")
    parser.add_argument('-o', '--output', help="name of output parquet", default="")
    parser.add_argument('--filename_fields', help="parse filename fields",
                        default=False, action='store_true')
    parser.add_argument('--filename_fields_regex', help="regex for parsing filename fields",
                        default=json.dumps({'run': r'_(\d+)_',
                                            'sample': r'_\d+_([^_]+)_',
                                            'date': r'(\d+-\d+-\d+)_'}))
    parser.add_argument('--title_fields', help="parse title fields",
                        default=False, action='store_true')
    parser.add_argument('--title_fields_regex', help="regex for parsing title fields",
                        default=json.dumps({'scan': r'^Scan:(\d+)\s',
                                            'retention_time': fr'[^\w]RT:({float_match})\s',
                                            'collision_energy': fr'[^\w]HCD=({float_match}\W+)\s'}))
    args = parser.parse_args()

    file_list = glob.glob(args.input)
    filename_fields = args.filename_fields_regex if args.filename_fields else None
    title_fields = args.title_fields_regex if args.title_fields else None
    df = None
    for file_name in file_list:
        with open(file_name) as fp:
            df = read_pepsearch(fp, df, filename_fields, title_fields)

    if df is None:
        logging.error('no data loaded')
        sys.exit(1)

    if args.output:
        # store = pd.HDFStore(args.output)
        # store['search_results'] = df
        # store.close()
        #df.to_parquet(path=args.output)
        Hitlist(df).save(args.output)
    else:
        print(df)

    sys.exit(0)

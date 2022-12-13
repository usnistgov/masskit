import pandas as pd
import numpy as np
import argparse
import sqlalchemy
from masskit.data_specs.base_library import supported_library_types
from masskit.data_specs.spectral_library import *
from rdkit import DataStructs
from masskit.spectrum.spectrum import SpectralSearchConfig


"""
--queries are the query spectra, e.g. replib
--library is the dataframe containing the library spectra to search, e.g. the mainlib
Note that "true" in this program means exact match.  
--jitter will add a random mz value to each peak in the searched spectrum
to search a mixture of true spectra and false spectra, use --mix

example: python spectra_compare.py --output replib_search_1 --output_queries replib_search_queries_1 --num_chunks 10000 --chunk 1
"""

parser = argparse.ArgumentParser()
parser.add_argument('--library', help="library to search", default="")
parser.add_argument('--input_true', default="",
                    help="file containing the true spectra to search.  only works with --mix")
parser.add_argument('--queries', help="search queries", default="")
parser.add_argument('--query_set', help="query set to search (train, test, valid) if empty, all", default="test")
parser.add_argument('--filter', help="pandas query used to filter both the library and queries, like nce == 35.0", default="")
parser.add_argument('--num_chunks', help='split queries into n chunks (0=none)', default=0, type=int)
parser.add_argument('--chunk', help='select a chunk, number starting with 0', default=0, type=int)
parser.add_argument('--output', help="output file prefix", default="test")
parser.add_argument('--output_queries', help="if set, output pkl containing queries", default="")
parser.add_argument('--num_queries', help="number of queries, randomly selected (0=all)", default=0, type=int)
parser.add_argument('--num_false', help="if doing only_trues, add in some false hits (0=none)", default=0, type=int)
parser.add_argument('--cosine_threshold', help="the minimum cosine threshold to record a hit", default=200, type=int)
parser.add_argument('--fp_tanimoto_threshold', help="use the spectral fingerprint to speed up searching (0=no)",
                    default=0.0, type=float)
parser.add_argument('--fingerprint', help="fingerprint to use", default="ecfp4")
parser.add_argument('--jitter', help="add a random value to the mz of the library spectra",
                    default=False, action='store_true')
parser.add_argument('--jitter_val', help="set a nonrandom jitter value (0=random)", default=0, type=int)
parser.add_argument('--only_trues', help="only search true matches",
                    default=False, action='store_true')
parser.add_argument('--nonrandom', help="don't select the queries and false hits randomly",
                    default=False, action='store_true')
parser.add_argument('--mix', help="creates a mixture of trues and falses to search",
                    default=False, action='store_true')
parser.add_argument('--min_size', help="minimize output size", default=False, action='store_true')
parser.add_argument('--minimum_match', help="the minimum number of peak matches to record a hit", default=1, type=int)
parser.add_argument(
    "--lib_type",
    choices=[x.__name__ for x in supported_library_types],
    default='TandemMolLib',
    help="what library format to read?",
)

args = parser.parse_args()

engine = sqlalchemy.create_engine(f"sqlite:///{args.library}")
df = LibraryAccessor.read_sql(engine, lib_type=globals()[args.lib_type])

engine = sqlalchemy.create_engine(f"sqlite:///{args.queries}")
df_query = LibraryAccessor.read_sql(engine, lib_type=globals()[args.lib_type])

if args.query_set:
    df_query = df_query.query('set == @args.query_set')
if args.filter:
    df_query = df_query.query(args.filter)
    df = df.query(args.filter)

# break query list into chunks and pick one of them
if args.num_chunks > 0 and args.chunk < args.num_chunks:
    df_query = np.array_split(df_query, args.num_chunks)[args.chunk]

if args.num_queries != 0:
    if args.nonrandom:
        # select queries starting with the 100th row
        df_query = df_query.iloc[pd.np.r_[100:100+args.num_queries]]
    else:
        df_query = df_query.sample(n=args.num_queries)

# combine the false hit set with the true queries
if args.mix:
    # get ids of queries
    queries = df_query["name"].values
    # drop them from the library
    df.drop(df.query("name in @queries").index)
    # get them from the true library
    df_true = pd.read_pickle(args.input_true)
    df_true = df_true.query("name in @queries")
    # concatenate them
    df = pd.concat([df, df_true])

#  used for scoring
# with open(os.path.join(data_dir, 'new_hist.pkl'), 'rb') as handle:
#    new_hist = pickle.load(handle)

results = None  # search results
search_settings = SpectralSearchConfig()
search_settings.cosine_threshold = args.cosine_threshold
search_settings.fp_tanimoto_threshold = args.fp_tanimoto_threshold
search_settings.minimum_match = args.minimum_match

# set up search library columns
spectra = df['spectrum'].values
spectra_fp = df['spectrum_fp'].values
inchi_keys = df['inchi_key'].values

# search_settings2 = SpectralSearchConfig()
# search_settings2.cosine_threshold = args.cosine_threshold
# search_settings2.fp_tanimoto_threshold = 0.0

for query, inchi_key, name, spectrum_fp, spectrum_fp_count in zip(df_query['spectrum'], df_query['inchi_key'],
                                                                  df_query['name'], df_query['spectrum_fp'],
                                                                  df_query['spectrum_fp_count']):
    # search db consists only of true hits
    if args.only_trues:
        match_df = df[(df["inchi_key"] == inchi_key) & (df["name"] == name)]
        # if also asking for a bunch of false hits, concat them
        if args.num_false != 0:
            # screen out exact matches
            df = df[df["inchi_key"] != inchi_key]
            if args.nonrandom:
                # add in false hits starting at 100th row
                match_df = pd.concat([match_df, df.iloc[pd.np.r_[100:100+args.num_false]]], ignore_index=True)
            else:
                match_df = pd.concat([match_df, df.sample(n=args.num_false)], ignore_index=True)
        spectra = match_df['spectrum'].values
        spectra_fp = match_df['spectrum_fp'].values
        inchi_keys = match_df['inchi_key'].values

    query.inchi_key = inchi_key  # sometimes the spectra is missing or has the incorrect inchikey
    # query.id = index  #  sometimes the ids don't match.  Not sure what this comment means!
    results = query.search_spectra(settings=search_settings, spectrum_fp=spectrum_fp,
                                   spectrum_fp_count=spectrum_fp_count, spectra=spectra,
                                   inchi_keys=inchi_keys, spectra_fp=spectra_fp, results=results)
    # hits2 = query.search_spectra(match_df, settings=search_settings2, spectrum_fp=row['spectrum_fp'],
    #                              spectrum_fp_count=row['spectrum_fp_count'])

# convert results into a dataframe
mux = pd.MultiIndex.from_arrays([results["query_id"], results["hit_id"]], names=["query_id", "hit_id"])
del results["query_id"]
del results["hit_id"]
result_df = pd.DataFrame(results, index=mux)

print("Calculate tanimoto similarities")
# create tanimoto similarity
result_df["ecfp4_similarity"] = 0.0
result_df["ecfp4_similarity_boost"] = 0.0
# make sure the indices are sorted
result_df = result_df.sort_index()
df = df.sort_index()
df_query = df_query.sort_index()
for t in result_df.itertuples():
    fp1 = df_query.at[t.Index[0], args.fingerprint]
    fp2 = df.at[t.Index[1], args.fingerprint]
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    result_df.at[t.Index, "ecfp4_similarity"] = similarity
    # if the inchi or name matches, boost the score
    if df_query.at[t.Index[0], "inchi_key"] == df.at[t.Index[1], "inchi_key"]:
        similarity += 0.01
    if df_query.at[t.Index[0], "name"] == df.at[t.Index[1], "name"]:
        similarity += 0.01
    result_df.at[t.Index, "ecfp4_similarity_boost"] = similarity

# todo: it turns out that the inchi_key field in the spectrum can be different from the one in the dataframe.  may have been fixed
# workaround
# result_df.loc[result_df["ecfp4_similarity_boost"] == 1.02, "identical"] = 1

result_df.sort_values(by=["query_id", "composite_score"], inplace=True)
# rank hits by score within group
if not args.min_size:
    result_df["rank_score"] = result_df.groupby("query_id")["composite_score"].rank("dense", ascending=True)
    result_df["rank_cosine"] = result_df.groupby("query_id")["cosine_score"].rank("dense", ascending=False)
    result_df.to_csv(f"{args.output}.csv")
    trues = result_df.query("identical == 1")
    falses = result_df.query("identical != 1")
    min_false_score = falses['composite_score'].min()
    max_false_cosine = falses['cosine_score'].max()
    print(f"min false score={min_false_score}, max false cosine={max_false_cosine}")
    print(f"num trues below score threshold = {trues[trues['composite_score'] <= min_false_score].shape[0]}")
    print(f"num trues above cosine threshold = {trues[trues['cosine_score'] >= max_false_cosine].shape[0]}")
    trues.to_csv(f"{args.output}_true.csv")
    falses.to_csv(f"{args.output}_false.csv")
else:
    del result_df['query_name']
    del result_df['query_formula']
    del result_df['query_inchi_key']
    del result_df['hit_name']
    del result_df['hit_formula']
    del result_df['hit_inchi_key']

result_df.to_pickle(f"{args.output}.pkl")
result_df.to_parquet(f"{args.output}.parquet")

if args.output_queries != "":
    df_query.to_pickle(f"{args.output_queries}.pkl")

# modify so that mode for trues plus small number of falses
# set match_df to have these
# change identity match to match inchi key AND name
# find highest scoring false, both for rank and
# count number of trues above these thesholds, print out

# ideas to try:
# adjust match ratio
# different match ratios for match/unmatch
# noise filter (just straight threshold)
# min m/z to allow for manual cutoff
# auto gain control to rescale peaks, using average of peaks 2-3.

import argparse
from masskit.utils.general import discounted_cumulative_gain
import sqlalchemy
from masskit.data_specs.base_library import supported_library_types
from masskit.data_specs.spectral_library import *
import numpy as np
import pandas as pd
import logging

"""
compare the results of a search against a gold standard structure comparison search
for example, a spectral similarity search vs structure comparison, or,
predicted structural similarity versus structure comparison

we assume the files for comparison are structure similarly:
all have a integer multiindex of (query_id, hit_id)
all have an identical field (todo: is this important.  what if they differ?)
all have a score field, whose name can be configured.  The score sorting direction can be inverted.

"""
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

parser.add_argument('--golden_search_results',
                    help="input pkl file containing golden (usually structure) search results",
                    default="")
parser.add_argument('--golden_score', help="score field for the golden results, e.g. ecfp4_similarity_boost",
                    default="ecfp4_similarity_boost")
parser.add_argument('--golden_score_ascending', help="is the gold score ascending?",
                    default=False, action='store_true')
parser.add_argument('--worst_golden_score',
                    help="the golden score used for comparison search results not found in the golden search results",
                    default=0.0, type=float)

parser.add_argument('--comparison_search_results', help="input pkl file containing structure search results",
                    default="")
parser.add_argument('--comparison_score', help="score field for the comparison results",
                    default="predictions")
parser.add_argument('--comparison_score_ascending', help="is the comparison score ascending?",
                    default=False, action='store_true')

parser.add_argument('--search_library', help="search library",
                    default="")
parser.add_argument('--query_library', help="query library",
                    default="")
parser.add_argument('--filter', help="pandas query used to filter both the library and queries, like nce == 35.0",
                    default="")

parser.add_argument('--output', help="output prefix for data files", default="test")

parser.add_argument('--tanimoto_threshold',
                    help="the minimum cosine threshold for search results (but not mssearch results. 0=all)",
                    default=0.3, type=float)  # about 0.3 for ecfp4, 0.45 for ecfp2
parser.add_argument('--max_identical', help="The allowable maximum rank of an identical match (0=no limit)",
                    default=0, type=int)
parser.add_argument('--remove_identical', help="remove identical matches before analysis",
                    default=False, action='store_true')
parser.add_argument(
    "--lib_type",
    choices=[x.__name__ for x in supported_library_types],
    default='TandemMolLib',
    help="what library format to read?",
)

args = parser.parse_args()

comparison_score = args.comparison_score
comparison_score_rank_column = comparison_score + "_rank"

golden_score = args.golden_score
golden_score_rank_column = golden_score + "_rank"

# load in comparison search results
df_comparison = pd.read_parquet(args.comparison_search_results, engine='pyarrow')

# load in golden search results
df_golden = pd.read_parquet(args.golden_search_results, engine='pyarrow')
if args.tanimoto_threshold:
    df_golden = df_golden[df_golden[golden_score] >= args.tanimoto_threshold]

# load in search library
engine = sqlalchemy.create_engine(f"sqlite:///{args.search_library}")
df_search_library = LibraryAccessor.read_sql(engine, lib_type=globals()[args.lib_type])
if args.filter:
    df_search_library = df_search_library.query(args.filter)
# force the indices to int32
df_search_library.index = df_search_library.index.astype("int32")
# useful map from nistno to chemical name
search_nistno2name = dict(zip(df_search_library.index.to_list(), df_search_library.name))

# load in query library
engine = sqlalchemy.create_engine(f"sqlite:///{args.query_library}")
df_query_library = LibraryAccessor.read_sql(engine, lib_type=globals()[args.lib_type])
if args.filter:
    df_query_library = df_query_library.query(args.filter)
# force the indices to int32
df_query_library.index = df_query_library.index.astype("int32")
# useful map from nistno to chemical name
query_nistno2name = dict(zip(df_query_library.index.to_list(), df_query_library.name))


def deskew(df, df_query_library_in, df_search_library_in):
    """
    fix version skew by forcing search results to match query and search libraries
    :param df: data frame to fix
    :param df_query_library_in: query library
    :param df_search_library_in: search library
    :return: deskewed data frame
    """
    # make sure queries are in the query library dataframe (sometimes the query library has version skew)
    query_ok = df.index.isin(df_query_library_in.index, level=0)
    # make sure hits are in the search library (sometimes the search library has version skew)
    hit_ok = df.index.isin(df_search_library_in.index, level=1)
    return df[query_ok & hit_ok]


# deal with version skew
df_golden = deskew(df_golden, df_query_library, df_search_library)
df_comparison = deskew(df_comparison, df_query_library, df_search_library)

# eliminate results that are identical structures
if args.remove_identical:
    df_golden = df_golden.query('identical == False')
    df_comparison = df_comparison.query('identical == False')

# subset queries to test set
test_ids = df_query_library.query("set == 'test'").index.values

# subset search results to test queries only
df_golden = df_golden[df_golden.index.isin(test_ids, level=0)]
df_comparison = df_comparison[df_comparison.index.isin(test_ids, level=0)]

# add in the names of the queries and hits for readability
df_golden['query_name'] = \
    [query_nistno2name.get(x, None) for x in df_golden.index.get_level_values(0).to_list()]
df_golden['hit_name'] = \
    [search_nistno2name.get(x, None) for x in df_golden.index.get_level_values(1).to_list()]

# add rankings
df_golden[golden_score_rank_column] = df_golden.groupby(
    level=["query_id"])[golden_score].rank(method="first", ascending=args.golden_score_ascending)

df_comparison[comparison_score_rank_column] = df_comparison.groupby(
    level=["query_id"])[comparison_score].rank(method="first", ascending=args.comparison_score_ascending)

# golden score discounted cumulative gain
golden_dcg = []  # of the top 10
golden_dcg_3 = []  # of the top 3

# predicted tanimoto discounted cumulative gain
comparison_dcg = []
comparison_dcg_3 = []
comparison_rank_identical = []
comparison_identical_over_one = 0  # identical hits over 1.0
comparison_over_one = np.count_nonzero(df_comparison[comparison_score] > 1.0)  # all hits > 1.0

intersect_top_3 = []  # number of top 3 golden hits in comparison top 3
intersect_top_10 = []  # number of top 10 golden hits in comparison top 10

# iterate through queries
for query in test_ids:
    # create dataframe hits matching the query
    df_golden_hits = df_golden.query("query_id == @query")
    df_comparison_hits = df_comparison.query("query_id == @query")

    df_intersection = pd.merge(df_golden_hits, df_comparison_hits, how='outer', left_index=True, right_index=True,
                               suffixes=('_golden', ''))

    # for the top 10 comparison hits, how many are in the top 10 golden hits?
    df_top_10 = df_intersection[(df_intersection[golden_score_rank_column] <= 10) &
                                (df_intersection[comparison_score_rank_column] <= 10)]
    intersect_top_10.append(len(df_top_10.index))
    # for the top 3 comparison hits, how many are in the top 10 golden hits?
    df_top_3 = df_intersection[(df_intersection[golden_score_rank_column] <= 3) &
                               (df_intersection[comparison_score_rank_column] <= 3)]
    intersect_top_3.append(len(df_top_3.index))

    # compute dcg, dcg_3 for comparison hits.  check to see if all of the hits have the golden score
    # note that by default, na's are sorted to the end, irrespective of the value of ascending
    df_intersection = df_intersection.sort_values(by=comparison_score, ascending=args.comparison_score_ascending)
    relevance_comparison = df_intersection[golden_score].values[0:10]
    # replace na's (hits not found by the golden search) with a poor score
    relevance_comparison = np.nan_to_num(relevance_comparison, nan=args.worst_golden_score)

    assert args.golden_score_ascending is False, 'discounted_cumulative_gain must be modified for ascending scores'
    comparison_dcg.append(discounted_cumulative_gain(relevance_comparison))
    comparison_dcg_3.append(discounted_cumulative_gain(relevance_comparison[0:3]))
    print(f"predicted comparison score discounted cumulative gain = {comparison_dcg[-1]}")

    # compute highest identity hit for comparison score
    identical = list(np.where(df_intersection["identical"] == 1)[0] + 1)
    if not identical:
        print(f"unable to find identical match for query {query}")
    else:
        if not args.max_identical or identical[0] <= args.max_identical:
            comparison_rank_identical.append(identical[0])
            print(f"comparison score identical at row {identical[0]}")
            if df_intersection.iloc[identical[0] - 1][comparison_score] > 1.0:
                comparison_identical_over_one += 1

    # compute dcg for golden score so this can be used in comparisons.
    df_golden_hits = df_golden_hits.sort_values(by=golden_score, ascending=args.golden_score_ascending)
    relevance_golden = df_golden_hits[golden_score].values[0:10]
    golden_dcg.append(discounted_cumulative_gain(relevance_golden))
    relevance_golden = df_golden_hits[golden_score].values[0:3]
    golden_dcg_3.append(discounted_cumulative_gain(relevance_golden))

print(f"comparison score column name is {comparison_score}")
print(f"average comparison_dcg top 10 = {np.mean(comparison_dcg)}, std dev = {np.std(comparison_dcg)}")
print(f"average comparison_dcg_3 top 3 = {np.mean(comparison_dcg_3)}, std dev = {np.std(comparison_dcg_3)}")
print(f"average golden_dcg top 10 = {np.mean(golden_dcg)}, std dev = {np.std(golden_dcg)}")
print(f"average golden_dcg_3 top 3 = {np.mean(golden_dcg_3)}, std dev = {np.std(golden_dcg_3)}")
if not args.remove_identical:
    print(
        f"average comparison score identical rank = {np.mean(comparison_rank_identical)},"
        f" median={np.median(comparison_rank_identical)}, total top matches={len(comparison_rank_identical)}")
    print(f"number of comparison identical hits with comparison scores above 1.0 = {comparison_identical_over_one}")
print(f"number of comparison scores above 1.0 = {comparison_over_one}")
print(f"number of test queries = {len(test_ids)}")
print(f"average number of top 3 golden hits in top 3 comparison score hits = {np.mean(intersect_top_3)},"
      f" std dev = {np.std(intersect_top_3)}")
print(f"average number of top 10 golden hits in top 10 comparison score hits = {np.mean(intersect_top_10)},"
      f" std dev = {np.std(intersect_top_10)}")

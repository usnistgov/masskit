import numpy as np
from rdkit import DataStructs
from masskit.config import TandemMLConfig


"""
common utility functions
"""


def discounted_cumulative_gain(relevance_array):
    return np.sum((np.power(2.0, relevance_array) - 1.0) / np.log2(np.arange(2, len(relevance_array) + 2)))


def tandem_options(parser):
    """
    command line options for tandem spectra
    :param parser: argument parser
    """
    # parser.add_argument('--collision_energy', help="filter library by collision energy, e.g. '10'", default="")
    # parser.add_argument('--ion_mode', help="filter library by positive or negative, e.g. 'P'", default="")
    # parser.add_argument('--instrument_type', help="filter library by dissociation used, e.g. 'HCD'", default="")
    parser.add_argument('--filter', help="expression to filter spectral libraries, e.g. ion_mode == 'P' and "
                        "nce == 35.0 and instrument_type == 'HCD'", default="")
    parser.add_argument('--filter_queries', help="filter queries also", default=False, action='store_true')
    parser.add_argument('--energy_channel', help="create channel in the spectra that encodes the energy",
                        default=False, action='store_true')
    parser.add_argument('--mz_channel', help="create channel containing mz", default=False, action='store_true')
    parser.add_argument('--precursor_minus_mz_channel', help="create channel containing precursor-mz", default=False,
                        action='store_true')


def tandem_options2config(args, config: TandemMLConfig):
    """
    use args to set up config for tandem spectra machine learning
    :param args: command line arguments
    :param config: configuration
    """
    config.energy_channel = args.energy_channel
    config.mz_channel = args.mz_channel
    config.precursor_minus_mz_channel = args.precursor_minus_mz_channel
    config.input_shape = config.get_input_shape()


def calculate_similarity_scores(query, hit, do_inchi_only):
    """
    calculate whether a match between structures is an exact match and if it should be boosted
    :param query: namedtuple containing query row info
    :param hit:  namedtuple containing hit row info
    :param do_inchi_only:  should we match only by inchi_key or also match by name
    :return: boolean whether the match was exact, the boost to the score
    """
    exact_match = (query.inchi_key == hit.inchi_key)
    if not do_inchi_only:
        exact_match = exact_match and (query.name == hit.name)
    score_boost = (0.02 if exact_match else 0.0)
    return exact_match, score_boost


def clean_up_df(df):
    """
    clean up dataframe to slim it down.  fields unused in searching
    :param df: the dataframe to slim down
    """
    del df['collision_gas']
    del df['column']
    del df['estimated_ri']
    del df['estimated_ri_error']
    del df['experimental_ri']
    del df['experimental_ri_data']
    del df['experimental_ri_error']
    del df['instrument']
    del df['ionization']
    del df['isomeric_smiles']
    del df['precursor_type']
    del df['sample_inlet']
    del df['smiles']
    del df['spectrum_type']
    del df['stdnp']
    del df['stdnp_error']
    del df['stdpolar']
    del df['stdpolar_data']
    del df['stdpolar_error']
    del df['synonyms']
    del df['mol']


def search_tanimoto(query_fp, query_fp_count, library_fps, library_fp_counts, tanimoto_threshold=0.3):
    """
    search an array of fingerprints subject to a speed up heuristic.
    Note: seems slower than the bulk search, at least for thresholds < 0.65
    :param query_fp: query fingerprint
    :param query_fp_count: number of bits set in the query fingerprint
    :param library_fps: library fingerprints
    :param library_fp_counts: number of bits set in each library fingerprint
    :param tanimoto_threshold: the minimum tanimoto score
    :return: a numpy array of tanimoto values for each library entry
    """
    if tanimoto_threshold > 0.0:
        tanimoto_array = np.zeros(len(library_fps))
        tanimoto_bound = (tanimoto_threshold + 1)/tanimoto_threshold
        for i in range(len(library_fps)):
            min_count = float(min(query_fp_count, library_fp_counts[i]))
            if (query_fp_count + library_fp_counts[i]) / min_count < tanimoto_bound:
                tanimoto_array[i] = DataStructs.TanimotoSimilarity(query_fp, library_fps[i])
    else:
        tanimoto_array = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, library_fps))
    return tanimoto_array

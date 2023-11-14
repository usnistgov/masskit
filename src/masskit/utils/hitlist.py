import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from . import general as mkgeneral


class Score(ABC):
    """
    base class for scoring a hitlist

    :param hit_table_map: a TableMap for the hits
    :type hit_table_map: TableMap
    :param query_table_map: a TableMap for the queries.  uses hit_table_map if set to None
    :type query_table_map: TableMap
    :param score_name: the name of the score
    :type score_name: str
    """

    def __init__(self, hit_table_map=None, query_table_map=None, score_name=None):
        if score_name is not None:
            self.score_name = score_name
        self.hit_table_map = hit_table_map
        if query_table_map is None:
            self.query_table_map = hit_table_map
        else:
            self.query_table_map = query_table_map

    def __call__(self, *args, **kwargs):
        return self.score(*args, **kwargs)

    @abstractmethod
    def score(self, hitlist):
        """
        do the scoring on the hitlist

        :param hitlist: the hitlist to score
        :type hitlist: Hitlist
        """
        raise NotImplementedError


class CosineScore(Score):
    # inits on hitlist, returns hitlist with a particular score
    def __init__(self, hit_table_map, query_table_map=None, score_name=None, scale=1.0):
        """
        cosine score calculator

        :param hit_table_map: the table map for hits
        :param query_table_map: the table map for queries
        :param score_name: the name of the score, default is "cosine_score"
        :param scale: the maximum value of the cosine score
        """
        if score_name is None:
            score_name = "cosine_score"  # set the default
        super().__init__(hit_table_map, query_table_map, score_name)
        self.scale = scale

    def score(self, hitlist):
        hitlist = hitlist.to_pandas()
        hitlist[self.score_name] = None
        query_id_old = None
        query_spectrum = None
        query_ids = hitlist.index.get_level_values(0)
        hit_ids = hitlist.index.get_level_values(1)
        cosine_scores = np.empty((len(hitlist.index),))
        for j in range(len(hitlist.index)):
            query_id = query_ids[j]
            if query_id == -1:
                continue
            if query_id != query_id_old:
                query_id_old = query_id
                # retrieve query spectrum
                query_spectrum = self.query_table_map.getitem_by_id(query_id)['spectrum']
            hit_id = hit_ids[j]
            # retrieve hit spectrum
            hit_spectrum = self.hit_table_map.getitem_by_id(hit_id)['spectrum']
            # calculate score
            cosine_scores[j] = query_spectrum.cosine_score(hit_spectrum, scale=self.scale)
        hitlist[self.score_name] = cosine_scores


class PeptideIdentityScore(Score):
    def __init__(self, score_name=None):
        """
        identity score for peptide hitlist

        :param hit_table_map: the table map for hits
        :param query_table_map: the table map for queries
        :param score_name: the name of the score, default is "tanimoto"
        """
        if score_name is None:
            score_name = "identity"  # set the default
        super().__init__(score_name=score_name)
        
    def score(self, hitlist):
        hitlist = hitlist.to_pandas()
        peptide = hitlist['peptide'].to_numpy()
        identity = np.zeros(peptide.shape, dtype=np.bool_)
        mod_names = hitlist['mod_names'].to_numpy()
        mod_positions = hitlist['mod_positions'].to_numpy()
        charge = hitlist['charge'].to_numpy()
        spectrum = hitlist['spectrum'].to_numpy()
        
        for i in range(len(hitlist.index)):
            if spectrum[i].mod_names is not None:
                query_mods = set(zip(spectrum[i].mod_names, spectrum[i].mod_positions))
            else:
                query_mods = set()
            if mod_names[i] is not None:
                hit_mods = set(zip(mod_names[i], mod_positions[i]))
            else:
                hit_mods = set()
            if query_mods == hit_mods and spectrum[i].charge == charge[i] and peptide[i] == spectrum[i].peptide:
                identity[i] = True
        
        hitlist[self.score_name] = identity
        


class TanimotoScore(Score):
    # inits on hitlist, returns hitlist with a particular score
    def __init__(self, hit_table_map, query_table_map=None, score_name=None, scale=1.0,
                 hit_fingerprint_column=None, query_fingerprint_column=None):
        """
        Tanimoto score calculator

        :param hit_table_map: the table map for hits
        :param query_table_map: the table map for queries
        :param score_name: the name of the score, default is "tanimoto"
        :param scale: the maximum value of the cosine score
        """
        if score_name is None:
            score_name = "tanimoto"  # set the default
        super().__init__(hit_table_map, query_table_map, score_name)
        self.scale = scale
        if hit_fingerprint_column is None:
            self.hit_fingerprint_column = 'ecfp4'
        else:
            self.hit_fingerprint_column = hit_fingerprint_column
        if query_fingerprint_column is None:
            self.query_fingerprint_column = 'ecfp4'
        else:
            self.query_fingerprint_column = query_fingerprint_column

    def score(self, hitlist):
        hitlist = hitlist.to_pandas()
        hitlist[self.score_name] = None
        query_id_old = None
        query_fingerprint = None
        query_ids = hitlist.index.get_level_values(0)
        hit_ids = hitlist.index.get_level_values(1)
        tanimotos = np.zeros((len(hitlist.index),))
        for j in range(len(hitlist.index)):
            query_id = query_ids[j]
            if query_id == -1:
                continue

            if query_id != query_id_old:
                query_id_old = query_id
                query_row = self.query_table_map.getrow_by_id(query_id)
                # retrieve query fingerprint
                query_fingerprint = self.query_table_map.to_arrow().slice(query_row, 1)[self.query_fingerprint_column].to_numpy()[0]
            hit_id = hit_ids[j]
            try:
                hit_row = self.hit_table_map.getrow_by_id(hit_id)
            except IndexError:
                # not all hits are in the store as the store is filtered and the index is not
                continue
            # retrieve hit fingerprint
            hit_fingerprint = self.hit_table_map.to_arrow().slice(hit_row, 1)[self.hit_fingerprint_column].to_numpy()[0]
            # calculate score
            numerator = np.unpackbits(np.bitwise_and(query_fingerprint, hit_fingerprint).view('uint8')).sum()
            denominator = np.unpackbits(np.bitwise_or(query_fingerprint, hit_fingerprint).view('uint8')).sum()
            if denominator:
                tanimotos[j] = numerator / denominator
        hitlist[self.score_name] = tanimotos


class Hitlist(ABC):
    """
    base class for a list of hits from a search

    :param hitlist: the hitlist
    :type hitlist: pandas.DataFrame
    """

    def __init__(self, hitlist=None):
        super().__init__()
        self._hitlist = hitlist

    @property
    def hitlist(self):
        """
        get the hitlist as a pandas dataframe

        :return: the hitlist
        :rtype: pandas.DataFrame
        """
        return self._hitlist

    @hitlist.setter
    def hitlist(self, value):
        """
        set the hitlist

        :param value: the hitlist
        :type value: pandas.DataFrame
        """
        self._hitlist = value

    def get_query_ids(self):
        """
        return list of unique query ids

        :return: query ids
        :rtype: np.int64
        """
        return self.hitlist.index.get_level_values(0).unique()

    def to_pandas(self):
        """
        get hitlist as a pandas dataframe
        """
        return self.hitlist

    def sort(self, score=None, ascending=False):
        """
        sort the hitlist per query

        :param score: name of the score.  default is cosine_score
        :param ascending: should the score be ascending
        """
        if score is None:
            score = 'cosine_score'
        self.hitlist.sort_values(by=["query_id", score], inplace=True, ascending=ascending)

    def save(self, file):
        self.hitlist.to_pickle(file)

    def load(self, file):
        self.hitlist = pd.read_pickle(file)
        return self


class HitlistCompare(ABC):
    """
    base class for comparing two hitlists.

    :param comparison_score: the column name of the score to compare to ground truth, 'cosine_score' default
    :param truth_score: the column name of the ground truth score, default same as comparison score
    :param comparison_score_ascending: is the comparison score ascending?
    :param truth_score_ascending: is the ground truth score ascending?
    :param comparison_score_rank: what is the column name of the rank of the comparison score, default appends
     _rank to comparison_score
    :param truth_score_rank: what is the column name of the rank of the truth score, default appends _rank to
     truth_score
    """

    def __init__(self, comparison_score=None, truth_score=None, comparison_score_ascending=False,
                 truth_score_ascending=False, comparison_score_rank=None, truth_score_rank=None):
        """
        initialize comparator

        """
        super().__init__()
        if comparison_score is None:
            self.comparison_score = "cosine_score"
        else:
            self.comparison_score = comparison_score
        if truth_score is None:
            self.truth_score = self.comparison_score
        else:
            self.truth_score = truth_score
        if truth_score_rank is None:
            self.truth_score_rank = f'{self.truth_score}_truth_rank'
        else:
            self.truth_score_rank = truth_score_rank
        if comparison_score_rank is None:
            self.comparison_score_rank = f'{self.comparison_score}_rank'
        else:
            self.comparison_score_rank = comparison_score_rank
        self.comparison_score_ascending = comparison_score_ascending
        self.truth_score_ascending = truth_score_ascending

    def __call__(self, *args, **kwargs):
        return self.compare(*args, **kwargs)

    @abstractmethod
    def compare(self, compare_hitlist, truth_hitlist=None, recall_values=None, rank_method=None):
        """
        compare hitlist to a ground truth hitlist.

        :param compare_hitlist: Hitlist to compare to
        :param truth_hitlist: Hitlist that serves as ground truth
        :param recall_values: lengths of the hitlist subsets used to calculate recall. [1, 3, 10] by default
        :param rank_method: what method to use to do ranking? "min" is default
        """

        raise NotImplementedError


class CompareRecallDCG(HitlistCompare):
    """
    compare two hitlists by ranking the scores and computing the recall and discounted cumulative gain
    at several subsets of the hit list
    with different lengths.  The recall values consist of the number of hits in each subset in the compare hitlist
    that have a rank equal to or better than the ground truth hitlist.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recall_values = [x for x in range(1, 11)]

    def compare(self, compare_hitlist, truth_hitlist=None, recall_values_in=None, rank_method=None):
        """
        compare hitlist to a ground truth hitlist.  Rank according to scores, then compute recall.
        the ranking applied is min ranking, where a group of hits that are tied are given a rank as if there had been
        one member of the group.

        :param compare_hitlist: Hitlist to compare to
        :param truth_hitlist: Hitlist that serves as ground truth
        :param recall_values_in: lengths of the hitlist subsets used to calculate recall. [1, 3, 10] by default
        :param rank_method: what method to use to do ranking? "min" is default
        """

        if recall_values_in is not None:
            self.recall_values = recall_values_in
        if rank_method is None:
            rank_method = "min"

        # add rankings.  "min" ranking means that if there is a tie, the numerically lower rank of the group is
        # assigned to all members.  We use min as for recall the highest ranking
        truth_hitlist.to_pandas()[self.truth_score_rank] = truth_hitlist.to_pandas().groupby(
            level=["query_id"])[self.truth_score].rank(method=rank_method, ascending=self.comparison_score_ascending)

        compare_hitlist.to_pandas()[self.comparison_score_rank] = compare_hitlist.to_pandas().groupby(
            level=["query_id"])[self.comparison_score].rank(method=rank_method, ascending=self.truth_score_ascending)

        truth_queries = truth_hitlist.to_pandas().index.get_level_values(0).unique()
        compare_queries = compare_hitlist.to_pandas().index.get_level_values(0).unique()
        query_ids = np.intersect1d(truth_queries, compare_queries)
        if len(query_ids) != len(compare_queries):
            print(f"some comparison queries (total num {len(compare_queries)}) not in the ground truth hitlist (total num {len(truth_queries)}). "
                             f"Number of intersecting queries {len(query_ids)}). Missing truth ids = {set(truth_queries) - set(compare_queries)}")

        columns = {}
        for recall_value in self.recall_values:
            columns[('truth_dcg', recall_value)] = []
            columns[('comparison_dcg', recall_value)] = []
            columns[('recall', recall_value)] = []
        columns['truth_max_score'] = []
        columns['comparison_max_score'] = []

        for query in query_ids:
            # create dataframe hits matching the query
            truth_hits = truth_hitlist.to_pandas().query(f"query_id == {query}")
            compare_hits = compare_hitlist.to_pandas().query(f"query_id == {query}")

            columns['truth_max_score'].append(truth_hits[self.truth_score].max())
            columns['comparison_max_score'].append(compare_hits[self.comparison_score].max())

            df_intersection = pd.merge(truth_hits, compare_hits, how='outer', left_index=True,
                                       right_index=True,
                                       suffixes=('_truth', ''))

            # for the top 10 comparison hits, how many are in the top 10 golden hits?
            for recall_value in self.recall_values:
                value = len(df_intersection[(df_intersection[self.truth_score_rank] <= recall_value) &
                                            (df_intersection[self.comparison_score_rank] <= recall_value)])
                # deal with the case where there are ties in both the truth score and comparison score
                if value > recall_value:
                    value = recall_value
                columns[('recall', recall_value)].append(value)

                # compute dcg.  check to see if all the hits have the golden score
                # note that by default, na's are sorted to the end, irrespective of the value of ascending
                if self.truth_score_ascending is False and self.comparison_score_ascending is False:
                    df_intersection = df_intersection.sort_values(by=self.comparison_score,
                                                                  ascending=self.comparison_score_ascending)
                    relevance_comparison = df_intersection[self.truth_score].values[0:recall_value]
                    # replace na's (hits not found by the golden search) with 0, which will not add to the dcg
                    relevance_comparison = np.nan_to_num(relevance_comparison)
                    columns[('comparison_dcg', recall_value)].append(mkgeneral.discounted_cumulative_gain(relevance_comparison))

                    # compute dcg for golden score so this can be used in comparisons.
                    truth_hits = truth_hits.sort_values(by=self.truth_score, ascending=self.truth_score_ascending)
                    relevance_golden = truth_hits[self.truth_score].values[0:recall_value]
                    columns[('truth_dcg', recall_value)].append(mkgeneral.discounted_cumulative_gain(relevance_golden))
                else:
                    logging.warning("Unable to compute dcg with ascending scores")
            pass

        return pd.DataFrame(columns, index=query_ids)


class IdentityRecall(HitlistCompare):
    """
    examing a hitlist with an identity colum and compute the recall 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recall_values = [x for x in range(1, 6)]

    def compare(self, compare_hitlist, recall_values_in=None, rank_method=None, identity_column=None):
        """
        compute recall of compare hitlist 
        
        :param compare_hitlist: Hitlist that serves as ground truth
        :param recall_values_in: lengths of the hitlist subsets used to calculate recall. [1, 3, 10] by default
        :param rank_method: what method to use to do ranking? "min" is default
        :param identity_column: column name of the identity column
        """

        if recall_values_in is not None:
            self.recall_values = recall_values_in
        if rank_method is None:
            rank_method = "min"
        if identity_column is None:
            identity_column = 'identity'

        # add rankings.  "min" ranking means that if there is a tie, the numerically lower rank of the group is
        # assigned to all members.  We use min as for recall the highest ranking
        hitlist = compare_hitlist.hitlist
        hitlist[self.comparison_score_rank] = hitlist.groupby(level=["query_id"])[self.comparison_score].rank(method=rank_method, ascending=self.comparison_score_ascending)

        query_ids = hitlist.index.get_level_values(0).unique()

        columns = {}
        for recall_value in self.recall_values:
            columns[('recall', recall_value)] = []

        for query in query_ids:
            # create dataframe hits matching the query
            hits = hitlist.query(f"query_id == {query}")

            # for the hits, are there any identical hits?
            for recall_value in self.recall_values:
                value = len(hits[(hits[self.comparison_score_rank] <= recall_value) & hits[identity_column]])
                # deal with the case where there are ties in both the truth score and comparison score
                if value > recall_value:
                    value = recall_value
                columns[('recall', recall_value)].append(value)

        return pd.DataFrame(columns, index=query_ids)

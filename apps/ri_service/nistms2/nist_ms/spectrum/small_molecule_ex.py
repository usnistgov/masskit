from .small_molecule import *

"""
Extensions to the spectrum base classes
"""


class UnitMassSpectrumScoreDevelopment(UnitMassSpectrum):
    """
    class for containing the unit mass spectra used for score development
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def score_rank_ks(self, spectrum, index1, index2):
        """
        see if the intensity distribution is the same.  not a very powerful test and doesn't
        work in all instances
        :param spectrum:
        :param index1:
        :param index2:
        :return:
        """
        # make sure the index arrays are populated
        if len(index1) < 1 or len(index2) != len(index1):
            return 0.0
        # start with the minimum position at the first index
        min_pos = 0
        # scan through the rest of the index array
        for i in range(1, len(index2)):
            # if it points to a lower value, change min_pos to point to that position
            if spectrum.products.intensity[index2[i]] < spectrum.products.intensity[index2[min_pos]]:
                min_pos = i
        # threshold_library_intensities = \
        #     spectrum.products.intensity[spectrum.products.intensity >= spectrum.products.intensity[index2[min_pos]]]
        # now, take the intensity of the query peak matching the min intensity library peak by using the same
        # min_pos
        # threshold_query_intensities = \
        #     self.products.intensity[self.products.intensity >= self.products.intensity[index1[min_pos]]]
        threshold_library_intensities = np.take(spectrum.products.intensity, index2)
        threshold_query_intensities = np.take(self.products.intensity, index1)
        d, p = sts.ks_2samp(threshold_library_intensities, threshold_query_intensities)
        logging.debug(f"ks p={p}, ks D={d}")
        return 1.0 - p

    @staticmethod
    def unmatched_score(spectrum, index2):
        """
        first attempt at unmatched score using rank probabilities of intensity
        didn't seem to work, but kept here as it is generally useful for computing rank intensity
        probabilities
        :param spectrum: the spectrum whose intensities are to be ranked
        :param index2: the match index
        :return: p value
        """
        n = spectrum.products.mz.shape[0]
        m = n - len(index2)
        p = 1.0
        if m != 0:
            average = (m * (n + 1)) / 2.0
            stddev = math.sqrt(average * (n - m) / 6.0)
            if stddev != 0.0:
                sum_intensity = 0
                for i, rank in enumerate(spectrum.products.rank):
                    if i not in index2:
                        sum_intensity += rank
                p = 0.5 * (1.0 + spc.erf((sum_intensity - average) / (stddev * math.sqrt(2.0))))
            logging.debug(f"average = {average}, stddev={stddev}, unmatched prob = {p}")
        return p

    def total_score(self, spectrum, new_hist, min_match=3, match_ratio=0.9, cosine_threshold=100):
        """
        try to match two spectra and calculate the probability of a match
        :param spectrum: the spectrum to match against
        :param new_hist: histogram of mz match probabilities
        :param min_match: minimum number of peaks to match
        :param match_ratio: the max/min difference in intensity ratio of two peaks needed to declare a match
        :param cosine_threshold: the minimum allowable cosine_threshold to compute scores
        :return: the p value, match prob, unmatch prob, matched rank prob, cosine score, cosine score on filtered
        spectra, number of matches, success in calculating.
        filtering = cut peaks (a) with intensity below 10 and
           (b) below min of 50 Da or max of min mz of either spectra.
        """
        # intersect the spectra
        #        common_mz, index1, index2 = np.intersect1d(self.products.mz, spectrum.products.mz, assume_unique=True,
        #                                                   return_indices=True)
        common_mz, index1, index2 = self.products.intersect(spectrum.products)
        matches = len(common_mz)
        if matches >= min_match:
            cosine_score = self.products.cosine_score(spectrum.products, index1, index2)
            if cosine_score < cosine_threshold:
                return 1.0, 1.0, 1.0, 1.0, cosine_score, cosine_score, matches, False
            max_min_mz = min([max([np.min(spectrum.products.mz), np.min(self.products.mz)]), 50])
            # create filtered versions of the spectra
            self.filtered = self.product_class()
            spectrum.filtered = self.product_class()
            self.filtered = self.products.copy(min_mz=max_min_mz, min_intensity=10)
            spectrum.filtered = spectrum.products.copy(min_mz=max_min_mz, min_intensity=10)
            common_mz, index1, index2 = self.filtered.intersect(spectrum.filtered)
            # note: the rank score seems to work better without the intensity ratio filter
            if len(common_mz) >= min_match:
                matched_rank_prob = self.filtered.score_rank(spectrum.filtered, index1, index2)
                cosine_score_filtered = self.filtered.cosine_score(spectrum.filtered, index1, index2)
                # redo the match using intensity ratio filter
                common_mz, index1, index2 = self.filtered.match(spectrum.filtered, common_mz, index1, index2,
                                                                match_ratio)
                matches = len(common_mz)
                if matches >= min_match:
                    # calculate the cosine score on the filtered peaks
                    match_prob = self.filtered.score_match(common_mz, new_hist, min_match)
                    unmatched_prob = spectrum.filtered.unmatched_score2(index2)
                    #                stat, p_val = sts.combine_pvalues([match_prob * matched_rank_prob, unmatched_prob])
                    return matched_rank_prob * (1000 - cosine_score) ** 4, match_prob, unmatched_prob, \
                        matched_rank_prob, cosine_score, cosine_score_filtered, matches, True
                else:
                    return 1.0, 1.0, 1.0, 1.0, cosine_score, cosine_score, matches, False
            else:
                return 1.0, 1.0, 1.0, 1.0, cosine_score, cosine_score, 0, False
        else:
            return 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, matches, False

    def search_spectra(self, df, new_hist, results, cosine_threshold=200, jitter=False, jitter_val=0):
        """
        search a standard dataframe of spectra
        :param df: the dataframe to search
        :param new_hist: the histogram of mz
        :param results: search results
        :param cosine_threshold: the minimum cosine score to place in results
        :param jitter: add a random jitter to all peak mz
        :param jitter_val: how much jitter to add.  0 = random
        """
        for index, row in df.iterrows():
            subject = row["spectrum"]

            identical = (self.inchi_key == row["inchi_key"]) & (self.name == row["name"])
            if jitter and not identical:
                # add a random shift to mz values
                if jitter_val == 0:
                    delta = randint(4, 14)
                else:
                    delta = jitter_val
                subject.precursor.jitter = delta
                subject.products.jitter = delta
            score, match_prob, unmatched_prob, matched_rank_prob, cosine_score, cosine_score_filtered, matches, \
                success = self.total_score(subject, new_hist, cosine_threshold=cosine_threshold)
            if success:
                e_val = score * df.shape[0]
                # changed key from f"{self.id}_{index}" to tuple
                results[(self.id, index)] = [self.name, self.formula, self.inchi_key, row["name"], row["formula"],
                                             row["inchi_key"], e_val,
                                             score, match_prob, unmatched_prob, matched_rank_prob, cosine_score,
                                             cosine_score_filtered, matches,
                                             1 if identical else 0]
            #                logging.debug(f"hit={results[index]}")
            # reset the jitter
            subject.precursor.jitter = 0
            subject.products.jitter = 0
        return



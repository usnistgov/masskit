import numpy as np

from massspec_ml.pytorch.samplers import BaseSampler


class LengthSampler(BaseSampler):
    """
    sampler based on length of a peptide.  borrowed from alphafold 2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def probability(self):
        """
        method to compute the probability of sampling a particular record

        :return: numpy array with the probability of sampling, from [0,1]
        """
        return np.minimum(
            np.maximum(self.dataset.get_column('peptide_len'),
                       self.dataset.config.ml.sampler.min_length) * self.dataset.config.ml.sampler.scale,
            self.dataset.config.ml.sampler.max_length) / self.dataset.config.ml.sampler.max_length

class LengthSampler2(BaseSampler):
    """
    sampler based on length of a peptide.  borrowed from alphafold 2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def probability(self):
        """
        method to compute the probability of sampling a particular record

        :return: numpy array with the probability of sampling, from [0,1]
        """
        a = self.dataset.config.ml.sampler.a
        b = self.dataset.config.ml.sampler.b
        c = self.dataset.config.ml.sampler.c
        d = self.dataset.config.ml.sampler.d
        mipr = self.dataset.config.ml.sampler.min_prob
        mapr = self.dataset.config.ml.sampler.max_prob
        return np.minimum(
            np.maximum(a - b*c**(np.vectorize(len)(self.dataset.get_column('peptide'))-d),
                       mipr), mapr)

class LengthSampler3(BaseSampler):
    """
    sampler based on length of a peptide.  borrowed from alphafold 2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def probability(self):
        """
        method to compute the probability of sampling a particular record

        :return: numpy array with the probability of sampling, from [0,1]
        """
        maxlen = self.dataset.config.ml.sampler.max_length
        minlen = self.dataset.config.ml.sampler.min_length
        mipr = self.dataset.config.ml.sampler.min_prob
        a = self.dataset.config.ml.sampler.a
        b = self.dataset.config.ml.sampler.b
        c = self.dataset.config.ml.sampler.c
        d = self.dataset.config.ml.sampler.d
        alpha = ((maxlen-minlen)/mipr)*np.log(c)
        peptides = np.minimum(np.maximum(np.vectorize(len)(self.dataset.get_column('peptide')), minlen), maxlen)
        return np.log((a - peptides)/b)/alpha + d

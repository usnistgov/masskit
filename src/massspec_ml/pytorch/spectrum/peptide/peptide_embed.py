import torch
from massspec.peptide.encoding import Uniprot21, mod_masses
import massspec_ml.pytorch.spectrum.peptide.peptide_constants as constants
import logging
import numpy as np
from massspec_ml.pytorch.spectrum.spectrum_embed import Embed1D

"""
embedding classes
- each embedding can be specified in the configuration
- each embedding has a corresponding *_embed function and *_channels function
- *_embed function returns the embedding
- *_channels function returns the number of channels in the function
- the channels function returns the total number of channels in all embeddings
- the embed function returns the embedding for a row of data
"""


class EmbedPeptide(Embed1D):
    """
    peptide 1D embedding
    """

    def __init__(self, config):
        super().__init__(config)
        self.mods = self.config.ml.embedding.peptide.mods
        # conversion objects
        self.u21 = Uniprot21()
        # create mod reverse mapping
        self.mod2index = {}
        for i, mod in enumerate(self.mods):
            # convert the modification names in self.mods into standard integer ids and relate to int embedding
            self.mod2index[mod_masses.dictionary.index(mod)] = i + 1  # addition of 1 allows for no mod

    def get_mod_list(self, row):
        """
        given a data row, return a numpy array of mods

        :param row: data row
        :return: mods as numpy array
        """
        mod_list = np.zeros(self.max_len)
        if row["mod_names"] is not None and row["mod_positions"] is not None:
            for name, position in zip(row["mod_names"], row["mod_positions"]):
                if position >= len(mod_list):
                    continue
                # get the int embedding
                mod_list[position] = self.mod2index.get(name, 0)
        return mod_list

    def mods_embed(self, row):
        """
        embed the peptide modifications as a one hot tensor

        :param row: data record
        :return: one hot tensor
        """
        mod_list = self.get_mod_list(row)
        return Embed1D.list2one_hot(mod_list, self.mods_channels())

    def mods_channels(self):
        """
        number of channels for modifications

        :return: number of channels
        """
        # addition of 1 allows for no mods
        return len(self.mods) + 1

    def mods_singleton_embed(self, row):
        """
        embed the modifications as a float tensor ranging from 0 to the number of mods

        :param row: data record
        :return: float tensor
        """
        mod_list = self.get_mod_list(row)
        return torch.unsqueeze(torch.FloatTensor(mod_list), -1)

    @staticmethod
    def mods_singleton_channels():
        """
        the number of mods_singleton channels

        :return: the number of mod channels
        """
        return 1

    def peptide_embed(self, row):
        """
        embed the peptide sequence as a one hot tensor

        :param row: data record
        :return: one hot tensor
        """
        peptide = self.u21.encode(row["peptide"], count=self.max_len)
        return Embed1D.list2one_hot(peptide, self.peptide_channels())

    @staticmethod
    def peptide_channels():
        """
        number of channels for peptide embedding

        :return: number of channels
        """
        return constants.PEPTIDE_CLASSES

    def peptide_singleton_embed(self, row):
        """
        embed the peptide sequence as a float tensor ranging from 0 to the number of amino acids

        :param row: data record
        :return: one hot tensor
        """
        peptide = self.u21.encode(row["peptide"], count=self.max_len)
        return torch.unsqueeze(torch.FloatTensor(peptide), -1)

    @staticmethod
    def peptide_singleton_channels():
        """
        number of channels for peptide singleton embedding

        :return: number of channels
        """
        return 1

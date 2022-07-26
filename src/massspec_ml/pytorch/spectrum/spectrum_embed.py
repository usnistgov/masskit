import bisect
import logging
import numpy as np
import torch
from massspec_ml.pytorch.embed import Embed


class Embed1D(Embed):
    """
    generic 1d embedding
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_len = self.config.ml.embedding.max_len
        self.min_charge = self.config.ml.embedding.min_charge
        self.max_charge = self.config.ml.embedding.max_charge
        # get nce bins by name
        self.nce_bins = self.config.ml.embedding.nce_bins
        self.nce_lo = min(self.nce_bins)
        self.nce_hi = max(self.nce_bins)
        self.ev_bins = self.config.ml.embedding.ev_range
        self.ev_lo = min(self.config.ml.embedding.ev_range)
        self.ev_hi = max(self.config.ml.embedding.ev_range)

    def nce_embed(self, row):
        """
        embed the nce as a one hot tensor

        :param row: data record
        :return: one hot tensor
        """
        # note that bisect returns values from 0 to the length of the array
        if row['nce'] is not None:
            nce_bin = bisect.bisect_left(self.nce_bins[1:-2], row["nce"])
        else:
            logging.error(f'record {row["id"]} with missing nce information')
            nce_bin = 0
        nce_list = np.repeat(nce_bin, self.max_len)
        return Embed1D.list2one_hot(nce_list, self.nce_channels())

    def nce_channels(self):
        """
        the number of nce channels

        :return: the number of nce channels
        """
        return len(self.nce_bins) - 1

    def nce_singleton_embed(self, row):
        """
        embed the nce as a single float value from 0 to 1

        :param row: data record
        :return: FloatTensor
        """
        if row['nce'] is not None:
            value = float(row["nce"] - self.nce_lo) / (self.nce_hi - self.nce_lo)
        else:
            logging.error(f'record {row["id"]} with missing nce information')
            value = 0.0
        return torch.unsqueeze(torch.FloatTensor(np.repeat(value, self.max_len)), -1)

    @staticmethod
    def nce_singleton_channels():
        """
        the number of nce channels

        :return: the number of nce_float channels
        """
        return 1

    def ev_embed(self, row):
        """
        embed the ev as a one hot tensor

        :param row: data record
        :return: one hot tensor
        """
        # note that bisect returns values from 0 to the length of the array
        if row['ev'] is not None:
            ev_bin = bisect.bisect_left(self.ev_bins[1:-2], row["ev"])
        else:
            logging.error(f'record {row["id"]} with missing ev information')
            ev_bin = 0
        ev_list = np.repeat(ev_bin, self.max_len)
        return Embed1D.list2one_hot(ev_list, self.ev_channels())

    def ev_channels(self):
        """
        the number of ev channels

        :return: the number of ev channels
        """
        return len(self.ev_bins) - 1

    def ev_singleton_embed(self, row):
        """
        embed the ev as a single float value from 0 to 1

        :param row: data record
        :return: FloatTensor
        """
        if row['ev'] is not None:
            value = float(row["ev"] - self.nce_lo) / (self.nce_hi - self.nce_lo)
        else:
            logging.error(f'record {row["id"]} with missing ev information')
            value = 0.0
        return torch.unsqueeze(torch.FloatTensor(np.repeat(value, self.max_len)), -1)

    @staticmethod
    def ev_singleton_channels():
        """
        the number of ev channels

        :return: the number of ev_float channels
        """
        return 1

    def charge_embed(self, row):
        """
        embed the charge as a one hot tensor

        :param row: data record
        :return: one hot tensor
        """
        # clip the charge values
        charge = (
            max(self.min_charge, min(self.max_charge, row["charge"])) - self.min_charge
        )
        charge_list = np.repeat(charge, self.max_len)
        return Embed1D.list2one_hot(charge_list, self.charge_channels())

    def charge_channels(self):
        """
        the number of charge channels.
        no charge, which should be rare, is encoded as an empty vector

        :return: the number of charge channels
        """
        return self.max_charge - self.min_charge + 1

    def charge_singleton_embed(self, row):
        """
        embed the charge as a float tensor ranging from 0 to 1

        :param row: data record
        :return: float tensor
        """
        charge = max(self.min_charge, min(self.max_charge, row["charge"]))
        if self.min_charge == self.max_charge:
            charge = 1.0
        else:
            charge = float(charge - self.min_charge) / (
                self.max_charge - self.min_charge
            )
        return torch.unsqueeze(torch.FloatTensor(np.repeat(charge, self.max_len)), -1)

    @staticmethod
    def charge_singleton_channels():
        """
        the number of charge_float channels

        :return: the number of charge channels
        """
        return 1
from abc import ABC, abstractmethod
import logging

import torch

class BasicEmbed(ABC):
    """
    base embedding class

    """
    def __init__(self, config):
        self.config = config

    def embed(self, row):
        """
        call the requested embedding functions as listed in config.ml.embedding.embeddings

        :param row: the data row
        :return: the concatenated one hot tensor of the embeddings
        """
        return row

    @property
    def channels(self):
        """
        return the number of channels in the encoding

        :return: the number of channels
        """
        return None

    @staticmethod
    def list2one_hot(list_in, num_classes):
        """
        convert a list of integers into a one hot tensor

        :param list_in: the list of integers
        :param num_classes: the number of classes
        :return: the one hot tensor
        """
        return torch.nn.functional.one_hot(
            torch.LongTensor(list_in), num_classes=num_classes
        ).float()


class Embed(BasicEmbed):
    """
    generic embedding

    each embedding has a member function ending in _embed to create the embedding
    and another member function ending in _channels that gives the number of channels in the embedding
    both functions take a dict called "row"
    """
    def __init__(self, config):
        super().__init__(config)

    def embed(self, row):
        """
        call the requested embedding functions as listed in config.ml.embedding.embeddings

        :param row: the data row
        :return: the concatenated one hot tensor of the embeddings
        """
        try:
            embeddings = [
                getattr(self, func + "_embed")(row)
                for func in self.config.ml.embedding.embeddings
            ]
        except KeyError as e:
            logging.error(f"not able to find embedding: {e}")
            raise
        one_hot = torch.cat(embeddings, dim=-1)
        if self.config.ml.embedding.channel_first:
            one_hot = torch.transpose(one_hot, -1, -2)
        return one_hot

    @property
    def channels(self):
        """
        return the number of channels in the encoding

        :return: the number of channels
        """
        try:
            num_channels = sum(
                [
                    getattr(self, func + "_channels")()
                    for func in self.config.ml.embedding.embeddings
                ]
            )
        except KeyError as e:
            logging.error(f"not able to find embedding: {e}")
            raise
        return num_channels


class EmbedXor(Embed):
    """
    embedding for toy xor network
    """

    def xor_embed(self, row):
        """
        embed the nce as a single float value from 0 to 1

        :param row: data record
        :return: FloatTensor
        """
        return torch.FloatTensor(row['input'])

    @staticmethod
    def xor_channels():
        """
        the number of nce channels

        :return: the number of nce_float channels
        """
        return 2
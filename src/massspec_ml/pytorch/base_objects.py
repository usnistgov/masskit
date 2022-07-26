from abc import ABC, abstractmethod
from collections import namedtuple

from torch import Tensor
from torch.nn import Module

ModelInput = namedtuple('ModelInput', ('x', 'y', 'index'))
ModelOutput = namedtuple('ModelOutput', ('y_prime', 'score', 'var'), defaults=(None, None, None))


class BaseLoss(Module, ABC):
    """
    abstract base class for losses
    loss is implemented as a pytorch module
    """

    def __init__(self, config=None, set=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    @abstractmethod
    def forward(self, output, batch, params=None) -> Tensor:
        """
        calculate the loss

        :param output: output dictionary from the model
        :param batch: batch data from the dataloader
        :param params: optional dictionary of parameters, such as epoch type
        :return: loss tensor
        """
        pass
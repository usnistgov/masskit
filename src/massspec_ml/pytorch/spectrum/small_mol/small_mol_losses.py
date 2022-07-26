from massspec_ml.pytorch.spectrum.spectrum_base_objects import BaseSpectrumLoss
from massspec_ml.pytorch.base_objects import BaseLoss
import torch
from torch import Tensor
import torch.nn.functional as functional

"""
Various losses

- the "output" from the model is a dictionary
  - output.y_prime contains a batch of predicted spectra
- "batch" is the input to the model
  - batch.y is a batch of experimental spectra corresponding to the predicted spectra
- each batch of spectra is a float 32 tensor of shape (batch, channel, mz_bins)
  - by convention, channel 0 are intensities, which are not necessarily scaled
  - channel 1 are standard deviations of the corresponding intensities
"""


class SearchLoss(BaseLoss):
    """
    loss based on search hitlist
    """
    __constants__ = ['threshold', 'epsilon']
    
    def __init__(self, threshold=0.2, epsilon=1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.epsilon = epsilon

    def forward(self, output, batch, params=None) -> Tensor:
        """
        Computes hitlist loss
        assumes:
        - batch.y < 0 means hitlist row is invalid
        - batch.y >= 1.0 means identical match
        - batch.y and output.y_prime is structured (batch, ..., hitlistrow)

        :param output: ModelOutput
        :param batch: ModelInput
        :param params: optional parameters, defaults to None
        :return: loss
        """
        # is_valid = is_label_valid(labels)
        is_valid = batch.y >= 0.0
        # threshold after finding valid values
        y_prime = torch.clamp(output.y_prime, min=self.threshold)
        y = torch.clamp(batch.y, min=self.threshold)
        # labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
        y_prime = is_valid * y_prime
        # labels_2 = tf.identity(labels)
        # logits_2 = tf.compat.v1.where(is_valid, logits, tf.zeros_like(logits))
        y = is_valid * y
        # count the number of valid hitlist positions and add an epsilong
        denominator = is_valid.sum(dim=-1) + self.epsilon
        denominator = denominator.repeat_interleave(y.shape[-1]).view(y.shape)
        # mse = tf.keras.losses.MSE(labels_2, logits_2)
        # mse over last dimension
        # to do: don't include invalids
        mse = torch.square(y_prime - y) / denominator
        # weights = tf.to_float(weights)
        # weights = tf.range(tf.shape(mse)[-1])
        weights = torch.arange(y.shape[-1], dtype=torch.float32, device=y.device).repeat(y.shape[0],1)
        # don't underweight identical hits, set them to 0
        is_not_identical = batch.y < 1.0
        weights = is_not_identical * weights
        # weights = 1.0/tf.math.log(weights + 2.0)
        weights = 1.0 / torch.log(weights + 2.0)
        # return_val = mse * weights
        return_val = mse * weights
        # return_val = backend.sum(return_val)
        return_val = return_val.sum()
        return return_val

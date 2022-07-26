from abc import abstractmethod
import torch

from massspec_ml.pytorch.base_objects import BaseLoss
from massspec_ml.pytorch.spectrum.peptide.peptide_constants import EPSILON
from torch import Tensor
from massspec.utils.general import class_for_name


# input named tuple for model.  A named tuple is used instead of a dict as some code requires a constant
# container. 'x' is the input Tensor, 'y' is are the experimental spectra, 'index' is the index into the database
# for the row corresponding to 'x' and 'y'

# output named tuple for model. A named tuple is used instead of a dict as some code requires a constant
# container. 'y_prime' is the predicted spectrum. 'score' is an optional score created in the model,
# like kl divergence. var is a measure of variance


class SpectrumModule(torch.nn.Module):
    """
    base class for a spectrum module.  contains the configuration object.
    """
    def __init__(self, config):
        super(SpectrumModule, self).__init__()
        self.config = config

    @property
    def bins(self):
        """
        calculate number of bins in the output spectrum
        """
        return int(self.config.ms.max_mz / self.config.ms.bin_size)


class SpectrumModel(SpectrumModule):
    """
    base class for spectral prediction models
    - the "output" from the model is a dictionary
      - output['y_prime'] contains a batch of predicted spectra
    - "batch" is the input to the model
      - batch['y'] is a batch of experimental spectra corresponding to the predicted spectra
      - each batch of spectra is a float 32 tensor of shape (batch, channel, mz_bins)
    - by convention, channel 0 are intensities, which are not necessarily scaled
    - channel 1 are standard deviations of the corresponding intensities
    """

    def __init__(self, config):
        super(SpectrumModel, self).__init__(config)
        self.embedding = class_for_name(self.config.paths.modules.embeddings,
                                        self.config.ml.embedding.embedding_type)(self.config)

    @property
    def channels(self):
        """
        calculate number of channels in the input
        """
        return self.embedding.channels


class BaseSpectrumLoss(BaseLoss):
    """
    abstract base class for spectrum losses
    assumes spectra have dimensions (batch, channel, mz_array)
    """
    def __init__(self, intensity_channel=0, variance_channel=1, epsilon=EPSILON,
                 *args, **kwargs) -> None:
        """
        init BaseSpectrumLoss

        :param intensity_channel: which channel to operate on
        :param variance_channel: the channel in the predicted spectra that has the predicted variance per peak
        :param epsilon: small value, used in division, etc.
        """
        super(BaseSpectrumLoss, self).__init__(*args, **kwargs)
        self.intensity_channel = intensity_channel
        self.variance_channel = variance_channel
        self.epsilon = epsilon

    def extract_spectra(self, output, batch) -> (Tensor, Tensor):
        predicted_spectrum = output.y_prime[:, self.intensity_channel:self.intensity_channel + 1, :]
        true_spectrum = batch.y
        if self.config.ml.loss.sqrt_intensity:
            predicted_spectrum[predicted_spectrum < 0.0] = 0.0
            predicted_spectrum = predicted_spectrum.sqrt()
            true_spectrum[true_spectrum < 0.0] = 0.0
            true_spectrum = true_spectrum.sqrt()
        return predicted_spectrum, true_spectrum

    def extract_variance(self, input_tensor: Tensor) -> Tensor:
        return input_tensor[:, self.variance_channel:self.variance_channel + 1, :]

    @abstractmethod
    def forward(self, output, batch, params=None) -> Tensor:
        """
        calculate the loss

        :param output: output dictionary from the model, type ModelOutput
        :param batch: batch data from the dataloader, type ModelInput
        :param params: optional dictionary of parameters, such as epoch type
        :return: loss tensor
        """
        pass


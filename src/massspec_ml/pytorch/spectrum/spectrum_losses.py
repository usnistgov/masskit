from massspec_ml.pytorch.spectrum.spectrum_base_objects import BaseSpectrumLoss
from massspec_ml.pytorch.base_objects import BaseLoss
import torch
from torch import Tensor
import torch.nn.functional as functional

"""
Various losses for predicting spectra

- the "output" from the model is a dictionary
  - output.y_prime contains a batch of predicted spectra
- "batch" is the input to the model
  - batch.y is a batch of experimental spectra corresponding to the predicted spectra
- each batch of spectra is a float 32 tensor of shape (batch, channel, mz_bins)
  - by convention, channel 0 are intensities, which are not necessarily scaled
  - channel 1 are standard deviations of the corresponding intensities
"""


class SpectrumMSELoss(BaseSpectrumLoss):
    """
    mean square error of intensity channel
    """

    def __init__(self, *args, **kwargs) -> None:
        super(SpectrumMSELoss, self).__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        predicted_spectrum, true_spectrum = self.extract_spectra(output, batch)
        return functional.mse_loss(predicted_spectrum, true_spectrum)


class SpectrumMSEKLLoss(BaseSpectrumLoss):
    """
    mean square error of intensity channel plus KL divergence
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        predicted_spectrum, true_spectrum = self.extract_spectra(output, batch)
        return functional.mse_loss(predicted_spectrum, true_spectrum) + \
            output.score / self.config.input[params['loop']].num


class SpectrumCosineLoss(BaseSpectrumLoss):
    """
    cosine similarity of intensity channel
    """

    def __init__(self, *args, **kwargs) -> None:
        super(SpectrumCosineLoss, self).__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        predicted_spectrum, true_spectrum = self.extract_spectra(output, batch)
        return -functional.cosine_similarity(predicted_spectrum, true_spectrum, dim=-1).mean()


class SpectrumCosineKLLoss(BaseSpectrumLoss):
    """
    cosine similarity of intensity channel and KL divergence
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        predicted_spectrum, true_spectrum = self.extract_spectra(output, batch)
        return -functional.cosine_similarity(predicted_spectrum, true_spectrum, dim=-1).mean() + \
            output.score / self.config.input[params['loop']].num


class SpectrumLogCosineKLLoss(BaseSpectrumLoss):
    """
    log of cosine similarity of intensity channel and KL divergence
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        predicted_spectrum, true_spectrum = self.extract_spectra(output, batch)
        return (1.0 - functional.cosine_similarity(predicted_spectrum, true_spectrum, dim=-1).mean()).log() + \
            output.score / self.config.input[params['loop']].num


class SpectrumNormalNLL(BaseSpectrumLoss):
    """
    negative log likelihood loss for a normal distribution for a spectral model that emits predictions
    and variance of that prediction
    omits constants in log likelihood
    """

    def __init__(self, *args, **kwargs) -> None:
        super(SpectrumNormalNLL, self).__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        predicted_spectrum, true_spectrum = self.extract_spectra(output, batch)
        diff = batch.y - output.y_prime
        variance = self.extract_variance(output.y_prime)
        if torch.isnan(variance).any():
            raise FloatingPointError("variance contains a NaN")
        return torch.mean(variance.log()) + torch.mean(diff.square() / (variance + self.epsilon))


"""
base losses
"""


class MSELoss(BaseLoss):
    """
    mean square error
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        return_val = functional.mse_loss(output.y_prime, batch.y)
        return return_val


class MSEKLLoss(BaseLoss):
    """
    mean square error plus kl divergence
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, output, batch, params=None) -> Tensor:
        return_val = functional.mse_loss(output.y_prime, batch.y) + \
                     output.score / self.config.input[params['loop']].num
        return return_val

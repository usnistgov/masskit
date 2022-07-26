from massspec_ml.pytorch.spectrum.spectrum_losses import *
from torchmetrics import Metric
import torch


class BaseMetric(Metric):
    """
    base class for metrics
    """
    def __init__(self, device='cuda', config=None, *args, **kwargs):
        """
        initialize the metric

        :param metric_device: which device to put the metric tensors on
        :param config: configuration
        """
        super().__init__(*args, **kwargs)
        self.config = config
        self.metric_device = device

    def update(self, output, batch):
        """
        update during batch

        :param output: standard ModelOutput from model
        :param batch: standard ModelInput batch information
        :return:

        Note: in the current version of torchmetrics, update() is called *twice* on each step, once to aggregate the
        current step to the accumulators and second to call compute() on the values for the current step (the value
        of the accumulators is stashed and restored during this last process).  This is
        only done when compute_on_step is true.  In future versions of torchmetrics, this behavior will become
        optional: https://github.com/PyTorchLightning/metrics/issues/344.  2021-09-07
        """
        pass

    def compute(self):
        pass

    @staticmethod
    def extract_spectra(output, batch):
        """
        Given the input and output to a model, extract the spectra

        :param output: model output
        :param batch: model input
        :return: predicted spectra and true spectra as Tensors
        """
        predicted_spectrum = output.y_prime[:, 0:1, :]
        true_spectrum = batch.y
        return predicted_spectrum, true_spectrum


class BaseLossMetric(BaseMetric):
    """
    base class for metrics
    """
    def __init__(self, loss_class=None, device='cuda', negate=False, config=None, *args, **kwargs):
        """
        initialize the metric

        :param loss_class: the loss class to use for the metric.
        :param metric_device: which device to put the metric tensors on
        :param config: configuration
        :param negate: should the metric be negated
        """
        super().__init__(device=device, config=config, *args, **kwargs)
        if negate:
            self.negate = -1.0
        else:
            self.negate = 1.0
        self.add_state("metric_sum", default=torch.tensor(0.0, dtype=torch.float64, device=self.metric_device),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, device=self.metric_device), dist_reduce_fx="sum")
        self.loss = loss_class(config=config)

    def update(self, output, batch):
        """
        update the metric for each step.  This is automatically called by forward() and all the arguments to
        forward are given to update.

        :param output: standard ModelOutput from model
        :param batch: standard ModelInput batch information
        """
        self.metric_sum += self.negate * self.loss(output, batch)
        self.total += 1

    def compute(self):
        return self.metric_sum / self.total


class SpectrumMSEMetric(BaseLossMetric):
    def __init__(self, config=None, *args, **kwargs):
        super().__init__(loss_class=SpectrumMSELoss, config=config, *args, **kwargs)


class SpectrumCosineMetric(BaseLossMetric):
    def __init__(self, config=None, *args, **kwargs):
        super().__init__(loss_class=SpectrumCosineLoss, config=config, negate=True, *args, **kwargs)


class SpectrumNormalNLLMetric(BaseLossMetric):
    def __init__(self, config=None, *args, **kwargs):
        super().__init__(loss_class=SpectrumNormalNLL, config=config, *args, **kwargs)


class MSEMetric(BaseLossMetric):
    """
    standard mean squared error
    """
    def __init__(self, config=None, *args, **kwargs):
        super().__init__(loss_class=MSELoss, config=config, *args, **kwargs)


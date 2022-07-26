from torch import nn

from massspec_ml.pytorch.base_objects import ModelOutput
from massspec_ml.pytorch.spectrum.spectrum_base_objects import *
# from bayesian_torch.layers import Conv1dFlipout

"""
demo of a simple pytorch lightning model for predicting spectra
plus a simple pytorch lightning data module
"""


class DenseSpectrumNet(SpectrumModel):
    """
    simple n layer dense network
    """

    def __init__(self, config):
        super().__init__(config)

        self.layer1 = nn.Sequential(
            nn.Linear(
                self.channels * self.config.ml.embedding.max_len,
                self.bins,
            ),
            nn.BatchNorm1d(self.bins),
            nn.ReLU(),
        )
        self.middle_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(self.bins, self.bins),
                    nn.BatchNorm1d(self.bins),
                    nn.ReLU(),
                )
                for _ in range(self.config.ml.model.DenseSpectrumNet.num_middle_layers)
            ]
        )
        self.fc = nn.Linear(self.bins, self.bins)

    def forward(self, x):
        inp = x[0]
        out = torch.flatten(inp, start_dim=1)
        out = self.layer1(out)
        out = self.middle_layers(out)
        out = self.fc(out)
        out = torch.unsqueeze(out, -2)
        return ModelOutput(out)


class XorModel(SpectrumModel):

    def __init__(self, config):
        super(XorModel, self).__init__(config)
        if self.config.ml.bayesian_network.bayes:
            pass
            # self.fc1 = Conv1dFlipout(1, 4, 2)
            # self.fc2 = Conv1dFlipout(4, 1, 1)
        else:
            # self.fc1 = nn.Linear(2, 4)
            # self.fc2 = nn.Linear(4, 1)
            self.fc1 = nn.Conv1d(1, 4, 2)
            self.fc2 = nn.Conv1d(4, 1, 1)

    def forward(self, x):
        x = x[0]
        x = torch.unsqueeze(x, 1)  # added to make convolution work
        if self.config.ml.bayesian_network.bayes:
            x, score = self.fc1(x)
            x = torch.relu(x)
            x, score2 = self.fc2(x)
            score += score2
        else:
            score = 1.0
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
        x = torch.squeeze(x, -1)  # added to make convolution work
        return ModelOutput(x, score)

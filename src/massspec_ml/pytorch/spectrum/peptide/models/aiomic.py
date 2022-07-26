# from bayesian_torch.layers import Conv1dFlipout
from torch import nn
import numpy as np
from math import prod

from massspec_ml.pytorch.base_objects import ModelOutput
from massspec_ml.pytorch.spectrum.spectrum_base_objects import *

"""
models for peptide spectrum prediction
"""


class ResBlock(SpectrumModule):
    """
    residual block
    """

    def __init__(self, config, inch, outch, ks, pad):  # use config
        """
        initialize residual module

        :param config: configuration
        :param inch: number of input channels
        :param outch: number of output channels
        """
        super().__init__(config)
        bayes = config.ml.bayesian_network.bayes
        self.conv1 = nn.Conv1d(inch, outch, ks, 1, padding=pad, bias=False)
        self.norm1 = nn.BatchNorm1d(outch)
        self.act = torch.relu
        self.conv2 = nn.Conv1d(outch, outch, ks, 1, padding=pad, bias=False)
        self.norm2 = nn.BatchNorm1d(outch)
        self.shortcut = (
            nn.Conv1d(inch, outch, 1, 1, 0) if inch != outch else nn.Identity()  # could be bayes
        )

    def forward(self, inp):
        """
        forward pass for residual module

        :param inp: input tensor
        :return: the block
        """
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.act(out)
        return self.act(torch.add(self.shortcut(inp), self.norm2(self.conv2(out))))


class CompoundConv(SpectrumModule):
    """
    compound convolution
    """

    def __init__(self, config, inch, outch, rang):  # use config
        """
        initialize compound convolution module

        :param config: configurations
        :param inch: input channels
        :param outch: output channels
        :param rang: list of filter sizes
        """
        super().__init__(config)
        bayes = config.ml.bayesian_network.bayes
        # length = (rang[1] - rang[0]) // rang[2]
        # first = [(m if m == 0 else int(np.ceil(m / 2))) for m in range(length)]
        # second = [
        #     (m + 1 if m == 0 else int(np.ceil((m + 1) / 2))) for m in range(length)
        # ]
        # converted to ModuleList to make zip work in forward()
        # self.pads = nn.ModuleList(
        #     [nn.ConstantPad1d((first[m], second[m]), 0) for m in range(length)]
        # )
        # Parameters are not found without nn.ModuleList
        self.convs = nn.ModuleList(
                [nn.Conv1d(inch, outch, m, 1, padding=m//2) for m in rang]
            )

    def forward(self, inp):
        """
        forward pass of compound convolution module

        :param inp: input tensor
        :return: output tensor
        """
        # return torch.cat([m(n(inp)) for m, n in zip(self.convs, self.pads)], dim=1)
        return torch.cat([m(inp) for m in self.convs], dim=1)


class AIomicsModel(SpectrumModel):
    def __init__(self, config):
        super().__init__(config)
        outdim = self.bins
        dropout = self.config.ml.model.AIomicsModel.dropout
        filtstart = self.config.ml.model.AIomicsModel.filtstart
        outin = self.config.ml.model.AIomicsModel.outin
        branches = self.config.ml.model.AIomicsModel.branches
        filtmid = self.config.ml.model.AIomicsModel.filtmid
        kss = self.config.ml.model.AIomicsModel.kss
        filtend = self.config.ml.model.AIomicsModel.filtend
        rang = self.config.ml.model.AIomicsModel.CompoundConv.rang
        bayes = self.config.ml.bayesian_network.bayes
        
        self.aa = self.channels
        self.upsample = nn.Upsample(scale_factor=2)
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()

        # First block - different sized convolutions
        self.first = CompoundConv(self.config, self.aa, filtstart, rang)
        inch = len(rang)*filtstart

        # Middle blocks - all with residual connections
        self.outer, self.inner = outin
        filts = np.linspace(filtmid[0], filtmid[1], self.outer, dtype='int')

        self.branches = nn.ModuleList(nn.ModuleList() for m in range(branches))
        for m in range(self.outer):
            """
            Complicated to figure inch and outch
            - Tensorflow makes this much easier
            class Infer(Module):
              def __init__(self, cls, *args, **kwargs):
                super(Infer, self).__init__()
                self.shape_dim = kwargs.pop('shape_dim', 1)
                self.cls = cls
                self.args = args
                self.kwargs = kwargs
            
                self.module = None
            
              def forward(self, x):
                if self.module is None:
                  try:
                    self.module = self.cls(x.shape[self.shape_dim], *self.args, **self.kwargs)
                  except IndexError as e:
                    raise ShapeInferenceError(f"Improper shape dim ({self.shape_dim}) selected for {self.cls} with input of shape {x.shape}")
                return self.module(x)
            """
            outch = filts[m]//branches
            branch = [[] for n in range(branches)]
            for n in range(self.inner):
                for o in range(branches):
                    ks = kss[0] + o*kss[1]
                    pad = (ks-1)//2
                    branch[o].append(ResBlock(self.config, inch, outch, ks, pad))
                inch = outch
            for o in range(branches):
                self.branches[o].append(nn.Sequential(*branch[o]))
            inch = branches*outch
        
        # Second-to-last block - without residual connection
        filtend = filtmid[-1] if filtend==0 else filtend
        if bayes:
            pass
            # self.second_to_last = Conv1dFlipout(inch, filtend, 1, stride=1, padding=1, bias=False)
                                                           # prior_mean=0.0, prior_variance=0.05,
                                                           # posterior_mu_init=0.0, posterior_rho_init=-7.0)
        else:
            self.second_to_last = nn.Conv1d(inch, filtend, 1, 1, 0, bias=False) # third number must be 1 in orig
        last = []
        last.append(nn.BatchNorm1d(filtend))
        last.append(nn.ReLU(True))
        last.append(self.drop)

        # Last block - project to outdim
        last.append(nn.Conv1d(filtend, outdim, 1, 1))
        last.append(nn.Sigmoid())
        self.last = nn.Sequential(*last)

        # Initialize weight
        #std = self.config.ml.model.AIomicsModel.std
        #for weight in self.parameters():
        #    if len(weight.shape)>1:
        #        #nn.init.normal_(weight, 0.0, std)
        #        #nn.init.kaiming_normal_(weight, nonlinearity='relu')
        #        nn.init.xavier_normal_(weight, gain=1.0)
        #nn.init.zeros_(self.last[-2].bias)
        
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, inp):
        #out = torch.nn.functional.one_hot(inp, self.aa).float()
        #out = torch.einsum("ijk -> ikj", out)
        score = 0.0
        inp = inp[0]  # refer to input elements by index, not name.  otherwise tensorboard logger will not work
        out = self.first(inp)
        #out = self.drop(out)
        for m in range(self.outer):
            out = self.upsample(out) if m>0 else out
            outs = []
            for n in range(len(self.branches)):
                outs.append(self.branches[n][m](out))
            out = torch.cat(outs, dim=1)
            #out = self.drop(out)
        if self.config.ml.bayesian_network.bayes:
            out, score = self.second_to_last(out)
        else:
            out = self.second_to_last(out)
        out = self.last(out)
        return ModelOutput(out.mean(dim=-1).view(out.shape[0], 1, out.shape[1]), score)


class SqEx(SpectrumModule):
    def __init__(self, config, inch):
        super().__init__(config)
        d = config.ml.model.PredFull.SqEx.d
        self.squeeze = lambda x: x.sum(dim=tuple(range(2, len(x.size()))))
        self.divisor = lambda x: prod(x.shape[2:])
        self.exc1 = nn.Linear(inch, inch // d)
        self.exc2 = nn.Linear(inch // d, inch)
        self.excite = lambda x: torch.sigmoid(self.exc2(torch.relu(self.exc1(x))))

    def forward(self, inp):
        out = self.squeeze(inp) / self.divisor(inp)
        return self.excite(out).view(*tuple(out.shape), 1) * inp


class pfblock(SpectrumModule):
    def __init__(self, config, inch, outch, ks, sqez=True, act=torch.relu):
        super().__init__(config)
        #ks = config.ml.model.PredFull.pfblock.ks
        self.conv = nn.Conv1d(inch, outch, ks, 1, 1)
        self.norm = nn.BatchNorm1d(outch)
        self.sqex = SqEx(config, outch) if sqez else nn.Identity()
        self.act = act
        self.shortcut = (
            nn.Conv1d(inch, outch, 1, 1, 0) if inch != outch else nn.Identity()
        )

    def forward(self, inp):
        out = self.sqex(self.norm(self.conv(inp)))
        return self.act(self.shortcut(inp) + out)

class PFCompoundConv(nn.Module):
    def __init__(self, inch, outch, rang=(2,10,1), norm=False):
        super(PFCompoundConv, self).__init__()
        length = (rang[1] - rang[0] + 1) // rang[2]
        first = [(m if m==0 else int(np.ceil(m/2))) for m in range(length)]
        second = [(m+1 if m==0 else int(np.ceil((m+1)/2))) for m in range(length)]
        self.pads = [nn.ConstantPad1d((first[m], second[m]), 0) for m in range(length)]
        self.convs = nn.ModuleList([nn.Conv1d(inch, outch, m, 1, 0) for m in range(*rang)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(outch) if norm else nn.Identity() for m in range(*rang)])
    def forward(self, inp):
        return torch.cat([l(m(n(inp))) for l,m,n in zip(self.norms, self.convs, self.pads)], dim=1)

class PredFull(SpectrumModel):
    def __init__(self, config):
        super().__init__(config)
        seq_len = self.config.ml.embedding.max_len
        outdim = self.bins
        inch = self.channels
        #rang = self.config.ml.model.PredFull.CompoundConv.rang
        
        #self.pos = torch.arange(0,seq_len, 1, dtype=torch.float32).view(1,1,-1)
        #self.pos = self.pos/self.pos.max();self.pos = 2*(self.pos-self.pos.mean())
        self.first = PFCompoundConv(inch, 64, rang=(2,10,1), norm=True)
        self.block0 = nn.Sequential(nn.Conv1d(inch, 512, 1, 1), nn.BatchNorm1d(512))
        self.block1 = nn.Sequential(*[pfblock(config, 512, 512, 3) for m in range(8)])
        self.block2 = nn.Sequential(
            *[pfblock(config, 512, 512, 1, sqez=False, act=nn.ReLU()) for _ in range(3)]
        )
        self.final = nn.Sequential(
            nn.Conv1d(512, outdim, 1, 1, 0), nn.Sigmoid(), nn.AvgPool1d(seq_len, 1, 0)
        )

    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, inp):
        # out = torch.nn.functional.one_hot(inp, 22).transpose(1, 2).float()
        inp = inp[0]
        #inp = torch.cat([inp, self.pos.repeat([inp.shape[0],1,1])], dim=1)
        #print(inp.shape)
        out = self.first(inp) + self.block0(inp)
        out = self.block1(out)
        out = self.final(out)
        out = torch.einsum(
            "ijk -> ikj", out
        )  # force transpose of channel and spectrum dimensions
        return ModelOutput(out, None, None)

class AIomicsModelUnc(SpectrumModel):
    def __init__(self, config):
        super().__init__(config)
        outdim = self.bins
        dropout = self.config.ml.model.AIomicsModelUnc.dropout
        filtstart = self.config.ml.model.AIomicsModelUnc.filtstart
        outin = self.config.ml.model.AIomicsModelUnc.outin
        branches = self.config.ml.model.AIomicsModelUnc.branches
        filtmid = self.config.ml.model.AIomicsModelUnc.filtmid
        kss = self.config.ml.model.AIomicsModelUnc.kss
        filtend = self.config.ml.model.AIomicsModelUnc.filtend
        rang = self.config.ml.model.AIomicsModelUnc.CompoundConv.rang
        bayes = self.config.ml.bayesian_network.bayes
        self.floor = self.config.ml.model.AIomicsModelUnc.floor
        
        self.aa = self.channels
        self.upsample = nn.Upsample(scale_factor=2)
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()

        # First block - different sized convolutions
        self.first = CompoundConv(self.config, self.aa, filtstart, rang)
        inch = len(rang)*filtstart

        # Middle blocks - all with residual connections
        self.outer, self.inner = outin
        filts = np.linspace(filtmid[0], filtmid[1], self.outer, dtype='int')

        self.branches = nn.ModuleList(nn.ModuleList() for m in range(branches))
        for m in range(self.outer):
            """
            Complicated to figure inch and outch
            - Tensorflow makes this much easier
            class Infer(Module):
              def __init__(self, cls, *args, **kwargs):
                super(Infer, self).__init__()
                self.shape_dim = kwargs.pop('shape_dim', 1)
                self.cls = cls
                self.args = args
                self.kwargs = kwargs
            
                self.module = None
            
              def forward(self, x):
                if self.module is None:
                  try:
                    self.module = self.cls(x.shape[self.shape_dim], *self.args, **self.kwargs)
                  except IndexError as e:
                    raise ShapeInferenceError(f"Improper shape dim ({self.shape_dim}) selected for {self.cls} with input of shape {x.shape}")
                return self.module(x)
            """
            outch = filts[m]//branches
            branch = [[] for n in range(branches)]
            for n in range(self.inner):
                for o in range(branches):
                    ks = kss[0] + o*kss[1]
                    pad = (ks-1)//2
                    branch[o].append(ResBlock(self.config, inch, outch, ks, pad))
                inch = outch
            for o in range(branches):
                self.branches[o].append(nn.Sequential(*branch[o]))
            inch = branches*outch
        
        # Second-to-last block - without residual connection
        filtend = filtmid[-1] if filtend==0 else filtend
        if bayes:
            self.second_to_last = Conv1dFlipout(inch, filtend, 1, stride=1, padding=1, bias=False)
        else:
            self.second_to_last = nn.Conv1d(inch, filtend, 1, 1, padding=0, bias=False)
        self.Var = nn.Linear(filtend*(2**(self.outer-1))*self.config.ml.embedding.max_len, 1)
        last = []
        last.append(nn.BatchNorm1d(filtend))
        last.append(nn.ReLU(True))
        last.append(self.drop)

        # Last block - project to outdim
        last.append(nn.Conv1d(filtend, outdim, 1, 1))
        last.append(nn.Sigmoid())
        self.last = nn.Sequential(*last)

    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, inp):
        #out = torch.nn.functional.one_hot(inp, self.aa).float()
        #out = torch.einsum("ijk -> ikj", out)
        score = None
        inp = inp[0]  # refer to input elements by index, not name.  otherwise tensorboard logger will not work
        out = self.first(inp)
        out = self.drop(out)
        for m in range(self.outer):
            out = self.upsample(out) if m>0 else out
            outs = []
            for n in range(len(self.branches)):
                outs.append(self.branches[n][m](out))
            out = torch.cat(outs, dim=1)
            out = self.drop(out)
        if self.config.ml.bayesian_network.bayes:
            out, score = self.second_to_last(out)
        else:
            out = self.second_to_last(out)
        var = torch.abs(self.Var(torch.flatten(out, 1, -1))) + self.floor
        out = self.last(out)
        # ModelOutput2 is code I added to base_objects.py
        return ModelOutput(out.mean(dim=-1).view(out.shape[0], 1, out.shape[1]), score, var.view(out.shape[0], 1, 1))

def test_modifier(model, config):
    """
    test version of a modifing function used to modify a model during transfer learning

    :param model: the model
    :param config: configuration
    :return: modified model
    """
    return model

import torch
from torch import nn
import torch.nn.functional as F
from massspec_ml.pytorch.base_objects import ModelOutput
from massspec_ml.pytorch.spectrum.spectrum_base_objects import *

"""
demo of a simple pytorch lightning model for predicting spectra
plus a simple pytorch lightning data module
"""


class SimpleNet(SpectrumModel):
    """
    simple n layer dense network
    """

    def __init__(self, config):
        super().__init__(config)
        fp_size = self.config.ml.model.SimpleNet.fp_size

        self.layer1 = nn.Sequential(
            nn.Linear(
                self.bins,
                fp_size,
            ),
            nn.BatchNorm1d(fp_size),
            nn.ReLU(),
        )
        self.middle_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(fp_size, fp_size),
                    nn.BatchNorm1d(fp_size),
                    nn.ReLU(),
                )
                for _ in range(self.config.ml.model.SimpleNet.num_middle_layers)
            ]
        )
        self.fc = nn.Linear(fp_size, fp_size)

    def forward(self, x):
        # input should be (batch, 20000)
        inp = x[0]
        out = self.layer1(inp)
        out = self.middle_layers(out)
        out = self.fc(out)
        out = nn.functional.normalize(out, dim=-1)
        # output should be (batch, self.config.ml.model.SimpleNet.fp_size)
        return ModelOutput(out)


class AIMSNet(SpectrumModel):
    """
    generates fingerprint for AIMS hybrid search
    """

    def __init__(self, config):
        super().__init__(config)
        fp_size = self.config.ml.model.AIMSNet.fp_size
        # pool_size = int(1/self.config.ms.bin_size)

#        self.rate = torch.nn.Parameter(torch.full((self.bins,), 0.5))
        self.fc_in = nn.Linear(self.bins, fp_size)
        self.conv = torch.nn.Conv1d(1, 1, 9, stride=1, padding='same')
        # self.maxpool = torch.nn.MaxPool1d(pool_size, stride=int(1/self.config.ms.bin_size), padding=pool_size//2)
        self.fc = nn.Linear(fp_size, fp_size)

    def forward(self, x):
        x = x[0]
        
        # query = query * self.rate_0
#        x = self.rate * x
        x = self.fc_in(x)
        # conv = Conv1D(1, int(25/self.config.bin_size + 1), padding='same', activation=None, name="correlation_conv")
        x = torch.unsqueeze(x, dim=1)
        x = self.conv(x)
        # query = MaxPool1D(pool_size=int(1/self.config.bin_size)+1, strides=int(1/self.config.bin_size), padding='same', name='query_maxpool')(query)
        # x = self.maxpool(x)
        x = self.fc(x)
        x = torch.squeeze(x)

        # query = Flatten(name='flatten_query')(query)
        x = nn.functional.normalize(x, dim=-1)
        return ModelOutput(x)


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)
    

class LocalLinear(nn.Module):
    def __init__(self,in_features,local_features,kernel_size,padding=0,stride=1,bias=True):
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        fold_num = (in_features+2*padding-self.kernel_size)//self.stride+1
        weight = torch.empty((fold_num,kernel_size,local_features))
        # torch.nn.init.normal_ or torch.randn(fold_num,kernel_size,local_features)
        torch.nn.init.xavier_uniform_(weight)
        if bias:
            bias_tensor = torch.empty((fold_num,local_features))
            torch.nn.init.xavier_uniform_(weight)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias_tensor) if bias else None

    def forward(self, x:torch.Tensor):
        x = F.pad(x,[self.padding]*2,value=0)
        x = x.unfold(-1,size=self.kernel_size,step=self.stride)
        x = torch.matmul(x.unsqueeze(2),self.weight).squeeze(2)+self.bias
        return x


class ResNetBaseline(SpectrumModel):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    self.config.ml.model.ResNetBaseline.mid_channels:
        The 3 residual blocks will have as output channels:
        [self.config.ml.model.ResNetBaseline.mid_channels, self.config.ml.model.ResNetBaseline.mid_channels * 2, self.config.ml.model.ResNetBaseline.mid_channels * 2]
    self.config.ml.model.ResNetBaseline.fp_size:
        The number of output classes
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.mid_channels = self.config.ml.model.ResNetBaseline.mid_channels
        self.fp_size = self.config.ml.model.ResNetBaseline.fp_size
        self.mid_size = self.config.ml.model.ResNetBaseline.get('mid_size', self.fp_size)
        
        self.fc_in = nn.Linear(self.bins, self.mid_size)
        
        if self.config.ml.model.ResNetBaseline.get('more_blocks', False):
            self.layers = nn.Sequential(*[
                ResNetBlock(in_channels=1, out_channels=self.mid_channels),
                ResNetBlock(in_channels=self.mid_channels, out_channels=self.mid_channels * 2),
                ResNetBlock(in_channels=self.mid_channels * 2, out_channels=self.mid_channels),
                ResNetBlock(in_channels=self.mid_channels, out_channels=self.mid_channels),
                ResNetBlock(in_channels=self.mid_channels, out_channels=1),
            ])
        else: 
            self.layers = nn.Sequential(*[
                ResNetBlock(in_channels=1, out_channels=self.mid_channels),
                ResNetBlock(in_channels=self.mid_channels, out_channels=self.mid_channels),
                ResNetBlock(in_channels=self.mid_channels, out_channels=1),
            ])
        self.final = nn.Linear(self.mid_size, self.fp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = x[0]
        x = torch.unsqueeze(x, dim=1)
        x = self.fc_in(x)
        # x = x.view((x.shape[0], self.mid_channels, -1))
        x = self.layers(x)
        # x = torch.squeeze(x)  # only works with mid_channels=1, otherwise change to flatten all but first dimension
        x = torch.flatten(x, start_dim=1)
        x = self.final(x)
        x = nn.functional.normalize(x, dim=-1)
        return ModelOutput(x)


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


   
class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNetBaseline_new(SpectrumModel):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, config):
        super().__init__(config)
        
        in_channels = self.config.ml.model.ResNetBaseline.in_channels
        base_filters = self.config.ml.model.ResNetBaseline.base_filters
        n_classes = self.config.ml.model.ResNetBaseline.n_classes
        self.verbose = self.config.ml.model.ResNetBaseline.verbose
        self.n_block = self.config.ml.model.ResNetBaseline.n_block
        self.kernel_size = self.config.ml.model.ResNetBaseline.kernel_size
        self.stride = self.config.ml.model.ResNetBaseline.stride
        self.groups = self.config.ml.model.ResNetBaseline.groups
        self.use_bn = self.config.ml.model.ResNetBaseline.use_bn
        self.use_do = self.config.ml.model.ResNetBaseline.use_do

        self.downsample_gap = self.config.ml.model.ResNetBaseline.downsample_gap # 2 for base model
        self.increasefilter_gap = self.config.ml.model.ResNetBaseline.increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = x[0]
        out = torch.unsqueeze(dim=1)
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        
        return ModelOutput(out)
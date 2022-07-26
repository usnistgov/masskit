from massspec_ml.pytorch.base_objects import ModelOutput
from massspec_ml.pytorch.spectrum.spectrum_base_objects import *

class AIomicsRNN(SpectrumModel):
    def __init__(self, config):
        super().__init__(config)
        outdim = self.bins
        max_len = self.config.ml.embedding.max_len
        embed_dim = self.config.ml.model.AIomicsRNN.embed_dim
        hidden_size = self.config.ml.model.AIomicsRNN.hidden_size
        layers = self.config.ml.model.AIomicsRNN.layers
        bidir = self.config.ml.model.AIomicsRNN.bidir
        
        #self.embed = torch.nn.Embedding(max_len, embed_dim)
        self.embed = torch.nn.Linear(max_len, embed_dim)
        self.GRU = torch.nn.GRU(embed_dim, hidden_size, layers, bidirectional=bidir)
        bidir = 2 if bidir==True else 1
        self.dense = torch.nn.Linear(bidir*hidden_size, outdim)
        self.pool = torch.nn.AvgPool1d(max_len, 1)
    
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
     
    def forward(self, x):
        out = self.embed(x[0])
        out, hidden = self.GRU(out)
        out = self.dense(out)
        out = torch.sigmoid(out)
        return ModelOutput(out.mean(dim=1).view(out.shape[0], 1, out.shape[-1]))

import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# info: https://zhuanlan.zhihu.com/p/40920384

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True) # Or use nn.LSTM()?

    def forward(self, src, hidden=None):
        embedded = self.embed(src) # src is the sequence of word idx of an input sentence.
        outputs, hidden = self.gru(embedded, hidden) # outputs refer to all the output paramters from each of time step,
                                                    # hidden is the last hidden state. shape=(sent_length, batch_size, embedding_size)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden
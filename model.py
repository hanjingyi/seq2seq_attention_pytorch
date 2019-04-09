import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# info: https://zhuanlan.zhihu.com/p/40920384
# info: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# info: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb

class EncoderAtt(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(EncoderAtt, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers=n_layers

        self.embed = nn.Embedding(input_size, embed_size) #input_size is the input vocab size
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True) # GRU: https://pytorch.org/docs/0.3.1/nn.html#gru

    def forward(self, src, hidden=None):
        embedded = self.embed(src) # src is a batch of word sequence idx?
        outputs, hidden = self.gru(embedded, hidden)
        # INPUT:
               # embedded(seq_len, batch, input_size)
               # hidden(num_layers * num_directions, batch, hidden_size)
               # The input can also be a packed variable length seq using torch.nn.utils.rnn.pack_padded_sequence()
        # OUTPUT:
               # outputs(seq_len, batch, hidden_size * num_directions)
               # hidden(num_layers * num_directions, batch, hidden_size)


        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



# class DecoderAtt(nn.Module):
#     def __init__(self, embed_size, hidden_size, output_size,n_layers=1, dropout=0.2):
#         super(DecoderAtt, self).__init__()
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#
#         self.embed = nn.Embedding(output_size, embed_size) # output vocab size
#         self.dropout = nn.Dropout(dropout, inplace=True)
#         self.attention = Attention(hidden_size)
#         self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
#                           n_layers, dropout=dropout)
#         self.out = nn.Linear(hidden_size * 2, output_size)
#
#     def forward(self, input, last_hidden, encoder_outputs):
#         # Get the embedding of the current input word (last output word)
#         embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
#         embedded = self.dropout(embedded)
#         # Calculate attention weights and apply to encoder outputs
#         attn_weights = self.attention(last_hidden[-1], encoder_outputs)
#         context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
#         context = context.transpose(0, 1)  # (1,B,N)
#         # Combine embedded input word and attended context, run through RNN
#         rnn_input = torch.cat([embedded, context], 2)
#         output, hidden = self.gru(rnn_input, last_hidden)
#         output = output.squeeze(0)  # (1,B,N) -> (B,N)
#         context = context.squeeze(0)
#         output = self.out(torch.cat([output, context], 1))
#         output = F.log_softmax(output, dim=1)
#         return output, hidden, attn_weights










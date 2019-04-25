import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# info: https://zhuanlan.zhihu.com/p/40920384
# info: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# info: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb
# info: https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
# info: (*) https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/seq2seq.py

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 variable_lengths=False,n_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.input_size=input_size
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.variable_lengths = variable_lengths

        self.embedding=nn.Embedding(input_size,embed_size)
        self.gru=nn.GRU(embed_size,hidden_size,n_layers,dropout=dropout)

    def forward(self, src_batch,hidden=None,input_lengths=None):
        embedded=self.embedding(src_batch)
        embedded=self.input_dropout(embedded)
        if self.variable_lengths:
            embedded=nn.utils.rnn.pack_padded_sequence(embedded,input_lengths, batch_first=True)
        output,hidden=self.gru(embedded, hidden)
        if self.variable_lengths:
            output, _=nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size,n_layers=1, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(embed_size, hidden_size,n_la     )
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, decode_function=F.log_softmax,device=None):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.device=device
        self.decode_function=decode_function

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal."
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers."

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

        '''
            Resets parameter data pointer so that they can use faster code paths.

            Right now, this works only if the module is on the GPU and cuDNN is enabled.
            Otherwise, it's a no-op.
            
        '''

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src=[src_sent_len, batch_size]
        # tgt=[tgt_sent_len, batch_size]
        # teacher_forcing_ratio is the probability to use teacher forcing.

        batch_size=tgt.shape[1]
        max_len=tgt.shape[0]
        tgt_vocab_size=self.decoder.output_size

        # tensor to store decoder outputs
        outputs=Variable(torch.zeros(max_len, batch_size,tgt_vocab_size)) # with gpu, add ".to(self.device)", but how about with cpu.

        # last hidden state of the encoder is used as the initial hidden state of the decoder.
        hidden, output=self.encoder(src)








# class EncoderAtt(nn.Module):
#     def __init__(self, input_size, embed_size, hidden_size,
#                  n_layers=1, dropout=0.5):
#         super(EncoderAtt, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.embed_size = embed_size
#         self.n_layers=n_layers
#
#         self.embed = nn.Embedding(input_size, embed_size) #input_size is the input vocab size
#         self.gru = nn.GRU(embed_size, hidden_size, n_layers,
#                           dropout=dropout, bidirectional=True) # GRU: https://pytorch.org/docs/0.3.1/nn.html#gru
#
#     def forward(self, src, hidden=None):
#         embedded = self.embed(src) # src is a batch of word sequence idx?
#         outputs, hidden = self.gru(embedded, hidden)
#         # nn.GRU INPUT:
#                # embedded(seq_len, batch, input_size)
#                # hidden(num_layers * num_directions, batch, hidden_size)
#                # The input can also be a packed variable length seq using torch.nn.utils.rnn.pack_padded_sequence()
#         # nn.GRU OUTPUT:
#                # outputs(seq_len, batch, hidden_size * num_directions)
#                # hidden(num_layers * num_directions, batch, hidden_size)
#
#
#         # sum bidirectional outputs
#         outputs = (outputs[:, :, :self.hidden_size] +
#                    outputs[:, :, self.hidden_size:])
#         return outputs, hidden


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










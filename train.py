import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import os
import random
import math
import time

from functions import *

# Get input_size and output(dimensionality of the one-hot vectors that will be input to the encoder/decoder.)
# The Preprocessing steps will deliver temporal outputs for saving memory space.
removeEmptyParallel('en-bg.en','en-bg.bg')
removeDuplicateParallel('en-bg.en.noEmpty.sl','en-bg.en.noEmpty.sl')
os.remove('en-bg.en.noEmpty.sl')
os.remove('en-bg.bg.noEmpty.tl')
src_vocab=text2vocab('en-bg.en.noEmpty.sl.noDuplicate.sl',200)
tgt_vocab=text2vocab('en-bg.en.noEmpty.sl.noDuplicate.tl',200)
os.remove('en-bg.en.noEmpty.sl.noDuplicate.sl')
os.remove('en-bg.en.noEmpty.sl.noDuplicate.tl')

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

# Prepare train, val and test sets



# Initialize weights in PyTorch by creating a function

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer=optim.Adam(model.parameters())

# PAD_IDX=TRG.vocab.stoi['<pad>']
criterion=nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def train(model, iterator, optimizer, criterion, clip ):
    model.train()
    epoch_loss=0
    for i, batch in enumerate(iterator):
        src=batch.src
        trg=batch.trg

        optimizer.zero_grad()
        output=model(src, trg)

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss=criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)




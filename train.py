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

input_size=len(src_vocab)
output_size=len(tgt_vocab)
enc_emb_dim=256
dec_emb_dim=256
enc_hid_dim=512
dec_hid_dim=512
enc_dropout=0.5
dec_dropout=0.5

en-bg.en.noEmpty.sl.noDuplicate.sl.vocab

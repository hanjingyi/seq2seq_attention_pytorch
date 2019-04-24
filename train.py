import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import time

input_size=len(src.vocab)
output_size=len(trg.vocab)
enc_emb_dim=256
dec_emb_dim=256
enc_hid_dim=512
dec_hid_dim=512
enc_dropout=0.5
dec_dropout=0.5





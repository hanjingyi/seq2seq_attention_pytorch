import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import spacy


spacy_en=spacy.load('en_core_web_sm')
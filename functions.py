import collections
from colorama import Fore
from colorama import Style
import random
import hashlib
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.nn.init as init

#
# def segment_counts(input):
#         lines=len([line for line in input])
#         return lines

def truecase(str):
    return str[0].lower()+str[1:]

def uppercase2normal(sent):
    '''
    This function convert a sentence to lowercase if the whole sentence is in uppercase,
    else, convert the leading character to lowercase (truecase).
    '''
    if sent.isupper():
        return sent.lower()
    else:
        return truecase(sent)



# def diff_src_tgt_warn(src,tgt):
#     if segment_counts(src)!=segment_counts(tgt):
#         print(f" {Fore.RED}WARNING: {src} and {tgt} have different segment counts.{Style.RESET_ALL}")
#     else:
#         pass


def removeEmptyParallel(src,tgt):
    with open(src,'r', encoding='utf8') as s,\
         open(tgt,'r',encoding='utf8') as t,\
         open(f"{src}.noEmpty.sl",'w', encoding='utf8') as os,\
         open(f"{tgt}.noEmpty.tl",'w', encoding='utf8') as ot:

        for line_s, line_t in zip(s,t):
            if line_s.strip() and line_t.strip():
                os.write(line_s)
                ot.write(line_t)

def hash_sent(sent):
    return hashlib.md5(sent.strip().encode('utf8')).hexdigest()


def removeDuplicateParallel(src,tgt):
    hash_set_src=set()
    hash_set_tgt=set()
    with open(src, 'r', encoding='utf8') as s, \
         open(tgt, 'r', encoding='utf8') as t, \
         open(f"{src}.noDuplicate.sl", 'w', encoding='utf8') as os, \
         open(f"{tgt}.noDuplicate.tl", 'w', encoding='utf8') as ot:
        for line_s, line_t in zip(s,t):
            hash_s=hash_sent(line_s)
            hash_t=hash_sent(line_t)
            if hash_s not in hash_set_src and hash_t not in hash_set_tgt:
                os.write(line_s)
                ot.write(line_t)
                hash_set_src.add(hash_s)
                hash_set_tgt.add(hash_t)


def text2vocab(input, vocab_size):
    '''

    input: a list of preprocessed input file.
    vocab_size: define the size of the vocabulary.
    :return: a vocabulary output file.

    Convert input corpus to a top ranked n vocab list including <s>, </s> and UNK.
    For languages that need segmentation (e.g. Zh,Ja and De etc.), please preprocess the corpus before passing in.
    '''


    tmp = []
    final = []  # Add idx of <s>, </s>, <pad> and UNK.
    with open(input,'r',encoding='utf8') as f:
        for line in f:
            line=uppercase2normal(line)
            line=word_tokenize(line)
            line=' '.join(line)
            for w in line.strip().split():
                tmp.append(w)
        w_count=collections.Counter(tmp).most_common()[:vocab_size]
        for t in w_count:
            final.append(t[0])
        final.insert(0, '</s>')
        final.insert(0, '<s>')
        final.insert(0, '<unk>')
        final.insert(0, '<pad>')
    return final


def word2idx(vocab):
    '''
    Return a dictionary {word:idx}.
    '''
    with open(vocab,'r',encoding='utf8') as f:
        word2idx={line.strip(): idx for idx,line in enumerate(f)}
    return word2idx


def sent2idx(vocab,sent):
    dict=word2idx(vocab)
    s=[dict['<s>']]
    for w in sent.strip().split():
        if w not in dict.keys():
            s.append(dict['<unk>'])
        else:
            s.append(dict[w])
    s.append(dict['</s>'])
    return s

def splitTrainTuneTest(src,tgt,sum_size,tune_size,test_size):
    '''
    sum_size: original corpus size (before splitting).
    tune_size: tune set size.
    test_size: test set size.
    '''
    with open(src,'r',encoding='utf8') as s,\
         open(tgt,'r', encoding='utf8') as t,\
         open('train.src','w',encoding='utf8') as train_s,\
         open('train.tgt','w',encoding='utf8') as train_t,\
         open('tune.src','w',encoding='utf8') as tune_s,\
         open('tune.tgt','w',encoding='utf8') as tune_t,\
         open('test.src','w',encoding='utf8') as test_s,\
         open('test.tgt','w',encoding='utf8') as test_t:
        rand = random.sample(range(sum_size), sum_size)
        # print(rand)
        rand_tune = [i for i in rand[:tune_size]]
        print(rand_tune)
        rand_test = [i for i in rand[tune_size:tune_size+test_size]]
        print(rand_test)
        for i, (line_s, line_t) in enumerate(zip(s,t)):
            if i in rand_tune:
                tune_s.write(line_s)
                tune_t.write(line_t)
            elif i in rand_test:
                test_s.write(line_s)
                test_t.write(line_t)
            else:
                train_s.write(line_s)
                train_t.write(line_t)


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    Info: https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
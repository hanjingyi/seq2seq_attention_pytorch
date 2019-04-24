
from nltk.tokenize import word_tokenize
from colorama import Fore
from colorama import Style

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def truecase(str):
    return str[0].lower()+str[1:]

def uppercase2normal(input):
    '''
    This function convert a sentence to lowercase if the whole sentence is in uppercase,
    else, convert the leading character to lowercase (truecase).
    '''
    l=[]
    for line in input:
        if line.isupper():
            l.append(line.lower())
        else:
            l.append(truecase(line))
    return l


def segment_counts(input):
    # with open(input,'r',encoding='utf8') as f:
        lines=len([line for line in input])
        return lines

def diff_src_tgt_warn(src,tgt):
    if segment_counts(src)!=segment_counts(tgt):
        print(f" {Fore.RED}WARNING: {src} and {tgt} have different segment counts.{Style.RESET_ALL}")
    else:
        pass

def removeEmptyDuplicates(src,tgt):
    '''
   Remove duplicated src or tgt segments.
   Remove empty lines.
    '''

    s=[line_s for idx, (line_s,line_t) in enumerate(set(zip(src,tgt))) if line_s.strip() and line_t.strip()]
    t=[line_t for idx, (line_s,line_t) in enumerate(set(zip(src,tgt))) if line_s.strip() and line_t.strip()]
    return s,t

def tokenizer(input):
    return [word_tokenize(line) for line in input]


def loadParallelCorpus(src,tgt):
    with open(src,'r',encoding='utf8') as sf, open(tgt,'r',encoding='utf8') as tf:
        sl=[line.strip() for line in sf]
        tl=[line.strip() for line in tf]
        diff_src_tgt_warn(sl,tl)
        sl,tl=removeEmptyDuplicates(sl,tl)
        sl=uppercase2normal(sl)
        tl=uppercase2normal(tl)
        print(sl[:3])
        print(tl[:3])
    return sl,tl

def text2vocab(input, output, vocab_size):
    '''

    :param input: a list of preprocessed input file.
    :param output: an output file
    :param vocab_size: define the size of the vocabulary.
    :return: a vocabulary output file.

    Convert input corpus to a top ranked n vocab list including <s>, </s> and UNK.
    For languages that need segmentation (e.g. Zh,Ja and De etc.), please preprocess the corpus before passing in.
    '''


    tmp = []
    final = []  # Add idx of <s>, </s> and UNK.
    with open(output, 'w', encoding='utf8') as o:
        input=tokenizer(input)
        for line in input:
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
        o.write('\n'.join(final))

s_tmp,t_tmp=loadParallelCorpus('PC.en-it.en','PC.en-it.it')
s_tmp,t_tmp=loadParallelCorpus('en-bg.bg','en-bg.en')


def vocab2embedding(vocab,embedding_size=5):
    '''
    Return word embeddings for each word from given vocaulary file.
    the tensor will the shape[vocabulary_size+3, embedding_size]
    '''
    with open(vocab,'r',encoding='utf8') as f:
        l=[line.strip() for line in f]
        word2idx={i:w for i,w in enumerate(l)}
        embeds=nn.Embedding(len(dict),embedding_size) #shape=(vocab_size, vector_dimen)
        # lookup_tensor=torch.tensor([word2idx[w] for w in word2idx.keys()],dtype=torch.long)
        # hello_embed=embeds(lookup_tensor)
        # print(hello_embed)
        for w in word2idx.keys():
            lookup_tensor=torch.tensor([word2idx[w]], dtype=torch.long)
            hello_embed=embeds(lookup_tensor)
            print(hello_embed)
    # return word2idx

embed=nn.Embedding(10,5)
embed(torch.LongTensor([3]))


sub='fl'
str=["flower","flow","flight"]


# for some in str:
import collections
import string
import random

for i in range(len(str)):
    set(str[0]).intersection(str[1])



def removeParallelSubsample(src,tgt,sample_src, sample_tgt):
    with open(src,'r',encoding='utf8') as s, open(tgt,'r', encoding='utf8') as t, \
        open(f"{src}.rest_subsample.sl", 'w',encoding='utf8' ) as os, open(f"{tgt}.rest_subsample.tl", 'w',encoding='utf8') as ot, \
        open(sample_src,'r', encoding='utf8') as samp_s, open(sample_tgt,'r',encoding='utf8') as samp_t:
        sf=[line for line in samp_s]
        tf=[line for line in samp_t]
        for idx, (line_s,line_t) in enumerate(zip(s,t)):
            if line_s not in sf:
                os.write(line_s)
                ot.write(line_t)


def removeParallelTag(src,tgt):
    with open(src, 'r', encoding='utf8') as s, open(tgt, 'r', encoding='utf8') as t, \
            open(f"{src}.garb_tag.sl", 'w', encoding='utf8') as garb_s, open(f"{tgt}.garb_tag.tl", 'w', encoding='utf8') as garb_t, \
            open(f"{src}.no_tag.sl", 'w', encoding='utf8') as os, open(f"{tgt}.no_tag.tl", 'w',encoding='utf8') as ot:
        for idx, (line_s,line_t) in enumerate(zip(s,t)):
            if "<cf" in line_s and "/cf>" in line_s:
                garb_s.write(line_s)
                garb_t.write(line_t)
            else:
                os.write(line_s)
                ot.write(line_t)


def removeParallelDuplicates(src,tgt):
    hash_set_s=set()
    hash_set_t=set()
    with open(src, 'r', encoding='utf8') as s, open(tgt, 'r', encoding='utf8') as t, \
        open(f"{src}.no_duplicates.sl", 'w', encoding='utf8') as os, open(f"{tgt}.no_duplicates.tl", 'w', encoding='utf8') as ot:
        for idx,(line_s, line_t) in enumerate(zip(s,t)):
            hash_s=hash_sent(line_s)
            hash_t=hash_sent(line_t)
            if hash_s not in hash_set_s and hash_t not in hash_set_t:
                os.write(line_s)
                ot.write(line_t)
                hash_set_s.add(hash_s)
                hash_set_t.add(hash_t)


s=["flower","flow","flight","fofight"]


def findintersection(s1,s2):
    return [c for c in s1 if c in s2]

def subindex(s1,s2):
    for i in range(len(s1)):
        if s1[:i] in s2:
            print(s2.index(s1[:1]))


t=[]
for i in range(len(s)-1):
    t=[c for c in s[0] if c in s[i]]




    if t:
        l.append(list(t))

l2=[' '.join(w) for w in l]




if len(l)==len(s)-1:

    print('yes')
else:
    print('empty')




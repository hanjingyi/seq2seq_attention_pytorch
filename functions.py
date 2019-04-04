import tensorflow as tf
import tensorboard

import collections
import spacy
from torchtext
from colorama import Fore
from colorama import Style


def tokenizer(input):
    return [word_tokenize(line) for line in input]

def segment_counts(input):
    with open(input,'r',encoding='utf8') as f:
        lines=len([line for line in f])
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


def loadParallelCorpus(src,tgt):
    with open(src,'r',encoding='utf8') as sf, open(tgt,'r',encoding='utf8') as tf:
        sl=[line.strip() for line in sf]
        tl=[line.strip() for line in tf]
        diff_src_tgt_warn(sl,tl)
        sl,tl=removeEmptyDuplicates(sl,tl)
        # s_file=[word_tokenize(line) for line in sl]
        # t_file=[word_tokenize(line) for line in tl]
    return s_file,t_file

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
        for line in input:
            for w in line.strip().split():
                tmp.append(w)
        w_count=collections.Counter(tmp).most_common()[:vocab_size]
        for t in w_count:
            final.append(t[0])
        final.insert(0, '</s>')
        final.insert(0, '<s>')
        final.insert(0, '<unk>')
        o.write('\n'.join(final))


def vocab2embedding(vocab,embedding_size):
    '''
    Return word embeddings for each word from given vocaulary file.
    the tensor will the shape[vocabulary_size+3, embedding_size]
    '''
    with open(vocab,'r',encoding='utf8') as f:
        word2idx={idx:line.strip for idx,line.strip() in enumerate(f)}
    return word2idx


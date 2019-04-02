import tensorflow as tf
import tensorboard

import collections
import spacy
from torchtext
from colorama import Fore
from colorama import Style


def removeEmptyDuplicates(src,tgt):
    '''
   Remove duplicated src or tgt segments.
   Remove empty lines.
    '''

    s=[line_s for idx, (line_s,line_t) in enumerate(set(zip(src,tgt))) if line_s and line_t]
    t=[line_t for idx, (line_s,line_t) in enumerate(set(zip(src,tgt))) if line_s and line_t]
    return s,t

def segment_counts(input):
    with open(input,'r',encoding='utf8') as f:
        lines=len([line for line in f])
        return lines

def diff_src_tgt_warn(src,tgt):
    if segment_counts(src)!=segment_counts(tgt):
        print(f" {Fore.RED}WARNING: {src} and {tgt} have different segment counts.{Style.RESET_ALL}")
    else:
        pass

def text2vocab(input, output, vocab_size):
    '''
    Convert input corpus to a top ranked n vocab list including <s>, </s> and UNK.
    For languages that need segmentation (e.g. Zh,Ja and De etc.), please preprocess the corpus before passing in.
    '''
    tmp = []
    final = []  # Add idx of <s>, </s> and UNK.
    with open(input, 'r', encoding='utf8') as f, open(output, 'w', encoding='utf8') as o:
        for line in f:
            for w in line.strip().split():
                tmp.append(w)
        w_count=collections.Counter(tmp).most_common()[:vocab_size]
        for t in w_count:
            final.append(t[0])
        final.insert(0, '</s>')
        final.insert(0, '<s>')
        final.insert(0, '<unk>')
        o.write('\n'.join(final))


def vocab2embedding(input_vocab,embedding_size):
    '''
    Return word embeddings for each word from given vocaulary file.
    the tensor will the shape[vocabulary_size+3, embedding_size]
    '''

    with open(input_vocab,'r',encoding='utf8') as f:
        vocab=[w for w in f]

    word_embedding=tf.get_variable("word_embeddings",
                                   [len(vocab), embedding_size])
    word_ids=[idx for idx,w in enumerate(vocab)]
    embedded_word_ids=tf.nn.embedding_lookup(word_embedding,word_ids)
    return embedded_word_ids
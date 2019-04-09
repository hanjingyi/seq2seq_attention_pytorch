import collections
from colorama import Fore
from colorama import Style


def tokenizer(input):
    return [word_tokenize(line) for line in input]

def segment_counts(input):
        lines=len([line for line in input])
        return lines

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
        sl=uppercase2normal(sl)
        tl=uppercase2normal(tl)
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
    final = []  # Add idx of <s>, </s>, <pad> and UNK.
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


def word2idx(vocab):
    '''
    Return a dictionary {idx:word}.
    '''
    with open(vocab,'r',encoding='utf8') as f:
        word2idx={line.strip(): idx for idx,line in enumerate(f)}
    return word2idx

s='this is a test .'

def sent2idx(vocab,sent):
    s=[]
    dict=word2idx(vocab)
    for w in sent.strip().split():
        if w not in dict.keys():
            s.append(dict['<unk>'])
        else:
            s.append(dict[w])
    return s


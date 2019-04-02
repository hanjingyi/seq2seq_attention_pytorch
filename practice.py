import spacy
from nltk.tokenize import word_tokenize

s=['this is a test.',
   'this is a test.',
   '',
   'this is another test.',
   '11223']

t=['esta es una prueba.',
   'esta es una prueba.',
   'hello word',
   'esta tambien es una prueba.',
   '',
   'hello word']



def removeEmptyDuplicates(src,tgt):
    '''
   Remove duplicated src or tgt segments.
   Remove empty lines.
    '''

    s=[line_s for idx, (line_s,line_t) in enumerate(set(zip(src,tgt))) if line_s and line_t]
    t=[line_t for idx, (line_s,line_t) in enumerate(set(zip(src,tgt))) if line_s and line_t]
    return s,t

src,tgt=removeEmptyDuplicates(s,t)

def loadParallelCorpus(src,tgt):
    with open(src,'r',encoding='utf8') as s, open(tgt,'r',encoding='utf8') as t:
        sl,tl=removeEmptyDuplicates(s,t)
        print(sl[:10])
        print(tl[:10])
        s_file=[word_tokenize(line) for line in sl]
        t_file=[word_tokenize(line) for line in tl]
    return s_file,t_file

s_tmp,t_tmp=loadParallelCorpus('en-bg.bg','en-bg.en')
s_tmp,t_tmp=loadParallelCorpus('PC.en-it.en','PC.en-it.it')



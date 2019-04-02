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



def removeDuplicates(src,tgt):
    '''
   Remove duplicated src or target segments.
   Remove empty lines.
    '''

    s=[line_s for idx, (line_s,line_t) in enumerate(set(zip(src,tgt))) if line_s and line_t]
    t=[line_t for idx, (line_s,line_t) in enumerate(set(zip(src,tgt))) if line_s and line_t]
    return s,t

src,tgt=removeDuplicates(s,t)



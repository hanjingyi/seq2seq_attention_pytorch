from functions import *
import numpy as np


np_data=np.arange(6).reshape((2,3))
torch_data=torch.from_numpy(np_data)




# Preprocess parallel data

src_vocab=text2vocab('en-bg.en.txt',1000)
trg_vocab=text2vocab('en-bg.bg.txt',1000)







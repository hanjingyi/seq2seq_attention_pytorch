##########################################################
# Numpy VS. Array
#########################################################
import torch
import numpy as np

# Convert numpy data to torch data:

np_data=np.arange(6).reshape((2,3))
torch_data=torch.from_numpy(np_data)

# Convert torch data to numpy data:

tensor2array=torch_data.numpy()


data=[-1,-2,3,4]
# Convert to 32bit float number:
tensor=torch.FloatTensor(data)
# Convert to absolut numebers:
torch.abs(tensor) # convert to tensor before using torch functions.
np.abs(data)

# Math computations are very similar to numpy, e.g.
np.mean(data)
torch.mean(tensor)

data_np=np.array(data)
data_np=data_np.reshape(2,2)
tensor=torch.FloatTensor(data_np)

    # Matrices computation: example：https://www.mathsisfun.com/algebra/matrix-introduction.html
    # np.malmul() is equal to torch.mm()
    # but data.dot() is not equal to tensor.dot()
print(np.matmul(data_np,data_np))
print(torch.mm(tensor,tensor))

#####################################################
# Torch Variable
#####################################################
import torch
from torch.autograd import Variable

tensor=torch.FloatTensor()
data_1=[[1,3,5,7],[9,2,4,3]]
data_2=[[0,4,5],[2,3,7],[3,5,9],[1,6,11]]
data_1_torch=torch.FloatTensor(data_1)
data_2_torch=torch.FloatTensor(data_2)
mulmax_result=torch.mm(data_1_torch,data_2_torch)

# create variable with given tensor.
variable=Variable(data_1_torch, requires_grad=True)

tensor_out=torch.mean(data_1_torch*data_1_torch)
variable_out=torch.mean(variable*variable)

# 误差反向传递
variable_out.backward()
variable.grad

# Check variable 里面的值
variable.data

# variable convert to numpy
variable.data.numpy()



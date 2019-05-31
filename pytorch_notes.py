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

#############################
# Activation function
# Plot with matplot
##############################

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# fake data
x=torch.linspace(-5,5,200)
x=Variable(x)
y_relu=F.relu(x).numpy()
y_sigmoid=torch.sigmoid(x).numpy()
y_tanh=torch.tanh(x).numpy()
y_softplus=F.softplus(x).numpy()

plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np,y_relu,c='blue',label='relu')
plt.ylim(-1,5)
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np,y_sigmoid,c='blue',label='sigmoid')
plt.ylim((-0.2,1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np,y_tanh,c='blue',label='tanh')
plt.ylim((-1.2,1.2))
plt.legend(loc='best')

#############################
#  Implementation - regression
############################

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Create random x and its y.
x=torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
y=x.pow(2) + 0.2*torch.rand(x.size()) # .pow() -> 二次方
# Convert matrix to variable, since network only allows variables for further computation.
x,y=Variable(x),Variable(y)
# To plot the data
# plt.scatter(x.numpy(),y.numpy(),c='r')
# plt.show()

# Implement neural network structure

class Net(torch.nn.Module):
    def __init__(self, n_feature,n_hidden,n_output):
        super(Net, self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x


network=Net(1,10,1)

# Visualization
plt.ion() # something about plotting
plt.show()

optimizer=torch.optim.SGD(network.parameters(), lr=0.5)
loss_func=torch.nn.MSELoss() # Mean Square Error is useful for regression problem.

for t in range(200):
    prediction=network(x)
    loss=loss_func(prediction, y) # always maintain the order of first prediction then true y label.

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

###################################
# Implementation - Classification
##################################

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Create data

n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)  # class0 x data (tensor), shape=(100,2)
y0=torch.zeros(100) #class0 y data (tensor), shape=(100,1)
x1=torch.normal(-2*n_data,1) #class1 x data (tensor), shape=(100,2)
y1=torch.ones(100) # class1 y data (tensor), shape=(100,1)
x=torch.cat((x0,x1),0).type(torch.FloatTensor) # FloatTensor=32-bit floating
# torch.cat((x0,x1),0) why 0,
y=torch.cat((y0,y1),).type(torch.LongTensor) # LongTensor=64-bit integer

x, y=Variable(x), Variable(y)

# Plot real data

plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=y.data.numpy(),s=100,lw=0,cmap='RdYlGn')
plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden=torch.nn.Linear(n_feature, n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

network=Net(2,10,2) # 2 dimentional features, 10 hidden states and 2 binary prediction output)
print(network)

optimizer=torch.optim.SGD(network.parameters(),lr=0.02)
loss_func=torch.nn.CrossEntropyLoss()

# Visualization
plt.ion() # something about plotting
plt.show()


for t in range(200):
    out=network(x) # the output of network is still a sequence of numbers, to compute
                   # final prediction we should use softmax (F.softmax(out)).
    loss=loss_func(out,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 2 ==0:
        plt.cla()
        prediction=torch.max(F.softmax(out),1)[1]
        pred_y=prediction.data.numpy().squeeze()
        target_y=y.data.numpy()
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=pred_y, s=100, lw=0,cmap='RdYlGn')
        accuracy=sum(pred_y==target_y) /200
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()


####################################
# Simple implementation methods without define class by yourself
###################################

# implemention of previous classification network using torch.nn.Sequential()
network_2= torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2),
)

print(network)
print(network_2)

################################
# Save and restore a model
###############################
def save(x,y):
    network_2=torch.nn.Sequential(
        torch.nn.Linear(2,10),
        torch.nn.ReLU(), # Note: When construct network by torch.nn.Sequential(),
                        # instead of using F.relu, use torch.nn.ReLU
        torch.nn.Linear(10,2)
    )

    optimizer=torch.optim.SGD(network_2.parameters(), lr=0.02)
    loss_func=torch.nn.CrossEntropyLoss()

    plt.ion()
    plt.show()

    for t in range(200):
        out=network_2(x)
        loss=loss_func(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 2 == 0:
            plt.cla()
            prediction = torch.max(F.softmax(out), 1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
            accuracy = sum(pred_y == target_y) / 200
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

    plt.ion()
    plt.show()

    # torch.save(network_2.state_dict(), 'tmp_network_2_param.pkl') #
    torch.save(network_2, 'tmp_network_2.pkl')   # Save entire network.

def restore_entire_network():
    network_2_restore= torch.load('tmp_network_2.pkl')

def restore_network_param(model_name):
    network_2_restore_param=torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.ReLU,
        torch.nn.Linear(10, 2)
    )

    network_2_restore_param.load_state_dict(torch.load('tmp_network_2_param.pkl'))
    prediction=network_2(x)


###################################
# Training with batches
###################################

import torch
import torch.utils.data as Data

BATCH_SIZE=5
x=torch.linspace(1,10,10)
y=torch.linspace(10,1,10)









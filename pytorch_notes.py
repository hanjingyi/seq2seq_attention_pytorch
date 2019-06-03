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
# Training with batches using DataLoader.
###################################

import torch
import torch.utils.data as Data
torch.manual_seed(7) # Make it reproducible.

BATCH_SIZE=5
x=torch.linspace(1,10,10)
y=torch.linspace(10,1,10)

# number of epoch -> Repeat the training (with entire data set) how many times.
# number of batch -> Each training step using how many data samples.

torch_dataset=Data.TensorDataset(x, y) #  先转换成 torch 能识别的 Dataset

loader=Data.DataLoader(
    dataset=torch_dataset,  # Must be torch TensorDataset format.
    batch_size=BATCH_SIZE, # Mini batch size.
    shuffle=True, # Important to shuffle dataset for each training epoch.
    num_workers=2 #  多线程来读数据
)

for epoch in range(3): # Training the whole dataset three times.
    for step, (batch_x, batch_y) in enumerate(loader):
        print('Epoch:' , epoch, '| Step:', step, '|batch x:', batch_x.numpy(), '|batch_y:', batch_y.numpy())


##############################
# Speed up training by SGD / Momentum / AdaGrad / RMSProp / Adam
# Adam combines mos of the advantages of previous methods, so it is one of the most
# effective and commonly used one.
##############################

import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt

# hyper parameters
LR=0.01
BATCH_SIZE=32
EPOCH=12

# Fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)   # hidden layer
        self.predict = torch.nn.Linear(20, 1)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


# Define different networks for the comparison

net_SGD=Net()
net_Momentum=Net()
net_RMSprop=Net()
net_Adam=Net()
nets=[net_SGD, net_Momentum, net_RMSprop, net_Adam]

# Define different optimizers for the comparison

opt_SGD=torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum=torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
opt_RMSprop=torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam=torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
opts=[opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]


# Define loss function and record loss of different networks during the training.
loss_func=torch.nn.MSELoss()
loss_histories=[[],[],[],[]]

for epoch in range(EPOCH):
    print('Epoch:', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x=Variable(batch_x)
        b_y=Variable(batch_y)

        for net, opt, loss_history in zip(nets, opts, loss_histories):
            output=net(b_x)
            print(output)
            loss=loss_func(output, b_y)
            opt.zero_grad() # Clean gradients for next train
            loss.backward() # backpropagation, compute gradients
            opt.step() # Apply gradients
            print(loss.data.numpy())
            loss_history.append(loss.data.numpy()) # loss recoder



# Plot

labels=['SGD','Momentum','RMSprop','Adam']
for i, loss_history in enumerate(loss_histories):
    plt.plot(loss_history, label=labels[i])

plt.legend(loc='best')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.ylim((0,0.2))
plt.show()






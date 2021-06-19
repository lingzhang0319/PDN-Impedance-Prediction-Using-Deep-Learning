# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:45:55 2020

@author: lingzhang0319
"""

from utils import folder_init
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter
import os
import numpy as np
import warnings
import random
from collections import deque

def wrap_np(state, device):
    return torch.from_numpy(state[np.newaxis, :]).float().unsqueeze(0).to(device)


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        x1 = torch.from_numpy(np.load(os.path.join(NEW_DATA_PATH, str(ID)+'.npz'))['x1'])
        x2 = torch.from_numpy(np.load(os.path.join(NEW_DATA_PATH, str(ID)+'.npz'))['x2'])
        y = torch.from_numpy(np.load(os.path.join(NEW_DATA_PATH, str(ID)+'.npz'))['y'][0:132])
        
        x1 = x1.permute(2, 0, 1)
        
        # normalize input data
        # x1[0,:,:] = (x1[0,:,:] - 1)/1
        # x1[1:,:,:] = (x1[1:,:,:] - 5)/5
        # x2[0:9] = (x2[0:9] - 1)/1
        # x2[9:] = (x2[9:] - 5)/5

        return x1, x2, y
    

# this is one way to define a network
class Net(nn.Module):
    def __init__(self, n2d, n1d):
        super(Net, self).__init__()
        self.n2d = n2d
        self.batch_size = batch_size
        self.linear1 = nn.Linear(n1d, n2d*n2d)   # hidden layer
        self.convs = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(16*n2d*n2d, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 132)
        )

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        x3 = self.linear1(x2)      
        x3 = torch.reshape(x3, (N, 1, H, W))
        x3 = torch.cat((x1, x3), 1)
        y = self.convs(x3)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y

warnings.filterwarnings("ignore")
folder_init()

dataname = 'test3_no_normalization'
NEW_DATA_PATH = 'supervised_data'
SUMMARY_PATH = "source/summary/" + dataname
MODEL_PATH = 'trained_model/'

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

writer = SummaryWriter(SUMMARY_PATH)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

batch_size = 128

# Parameters
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 0}
max_epochs = 100

#complete_list = list(range(700000, 700200))
complete_list = list(range(0, 950000))
#random.shuffle(complete_list)
num_test_cases = 10000
# Datasets
#test_id  = complete_list[:num_test_cases]
#training_id = complete_list[num_test_cases:]
training_id  = complete_list[:len(complete_list)-num_test_cases]
test_id = complete_list[len(complete_list)-num_test_cases:]


# Generators
training_set = Dataset(training_id)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(test_id)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

net = Net(n2d=16, n1d=17)     # define the network
# print(net)  # net architecture
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

n = 0

lowest_loss = 100
train_loss_deque = deque(maxlen=10)  # last 10 losses
test_loss_deque = deque(maxlen=10)  # last 10 losses

# Loop over epochs
for epoch in range(max_epochs):
    print('\n')
    print('Epoch ' + str(epoch))
    # Training
    for x1s, x2s, ys in training_generator:
        n += 1
        # Transfer to GPU
        x1s = x1s.float().to(device)
        x2s = x2s.float().to(device)
        ys = ys.float().to(device)
        net.to(device)
        prediction = net(x1s, x2s)     # input x and predict based on x
        
        train_loss = torch.sqrt(loss_func(prediction, ys))     # must be (1. nn output, 2. target)
        train_loss_deque.append(train_loss.detach().cpu().numpy())

        optimizer.zero_grad()   # clear gradients for next train
        train_loss.backward()   # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        
        if n % 10 == 0:
            print('Epoch ' + str(epoch) + ', training loss is ' + str(train_loss.detach().cpu().numpy()))
            # Validation
        if n % 50 == 0:
            with torch.set_grad_enabled(False):
                for x1s, x2s, ys in validation_generator:
                    # Transfer to GPU
                    x1s = x1s.float().to(device)
                    x2s = x2s.float().to(device)
                    ys = ys.float().to(device)
                    net.to(device)
                    net.eval()
                    prediction = net(x1s, x2s)     # input x and predict based on x
                    test_loss = torch.sqrt(loss_func(prediction, ys))     # must be (1. nn output, 2. target)
                    test_loss_deque.append(test_loss.detach().cpu().numpy())
                    
                    print('Epoch ' + str(epoch) + ', test loss is ' + str(test_loss.detach().cpu().numpy()))
                    
                    if np.mean(test_loss_deque) < lowest_loss:
                        torch.save(net.state_dict(), os.path.join(MODEL_PATH + dataname + '_best_model'))
                        lowest_loss = np.mean(test_loss_deque)
                        print('Best model saved !')
                    break
                    
                    break
        
        if n % 100 == 0:
            writer.add_scalar("Training loss", np.mean(train_loss_deque), n)
            writer.add_scalar("Test loss", np.mean(test_loss_deque), n)



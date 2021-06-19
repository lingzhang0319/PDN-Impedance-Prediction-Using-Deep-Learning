# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:42:45 2020

@author: lingzhang0319
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:16:48 2020

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
import time
import matplotlib.pyplot as plt
from pdn_class import PDN
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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

        return x1, x2, y
    
    
# this is one way to define a network
class Net(nn.Module):
    def __init__(self, n2d, n1d):
        super(Net, self).__init__()
        self.n2d = n2d
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


brd = PDN()

dataname = 'test3_no_normalization'
NEW_DATA_PATH = 'supervised_data/'
SUMMARY_PATH = "source/summary/" + dataname
MODEL_PATH = 'trained_model/'

net = Net(n2d=16, n1d=17) 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")
net.to(device)

net.load_state_dict(torch.load(os.path.join(MODEL_PATH, dataname +'_best_model'), map_location=torch.device('cpu')))
net.eval()

# ID = random.randint(0, 950000-10000)
ID = random.randint(950000-10000, 950000)   # randomly select a case from the testing data
# ID = 941762

x1 = torch.from_numpy(np.load(os.path.join(NEW_DATA_PATH, str(ID)+'.npz'))['x1']).float().unsqueeze(0).permute(0,3,1,2).to(device)
x2 = torch.from_numpy(np.load(os.path.join(NEW_DATA_PATH, str(ID)+'.npz'))['x2']).float().unsqueeze(0).to(device)
y = torch.from_numpy(np.load(os.path.join(NEW_DATA_PATH, str(ID)+'.npz'))['y'][0:132]).float().unsqueeze(0).to(device)

sxy = np.load(os.path.join(NEW_DATA_PATH, str(ID)+'.npz'))['sxy']


t1 = time.time()
y_predict = net(x1, x2)
print(time.time()-t1)

plt.loglog(brd.freq.f[0:132]/1e6, np.power(10, y.squeeze().detach().cpu().numpy()/20))
plt.loglog(brd.freq.f[0:132]/1e6, np.power(10, y_predict.squeeze().detach().cpu().numpy()/20),'r--')
plt.grid(which='both')
plt.xlabel('Frequency(MHz)')
plt.ylabel('Impedance(Ohm)')
plt.legend(['Ground truth','Predicted by DNN'])
plt.title('Case #' + str(ID))
plt.show()

x1 = x1.squeeze(0).detach().cpu().numpy()
x2 = x2.squeeze(0).detach().cpu().numpy()
y = y.detach().cpu().numpy()

brd_shape_ic = x1[0,:,:]
num_decaps = np.count_nonzero(x1[1:,:,:])
print('Number of decaps is ' + str(num_decaps))
num_stackup = np.count_nonzero(x2[9:])
print('Number of layers is ' + str(num_stackup+1))


# =============================================================================
# Plot board shape, decap placement, and board stackup
# =============================================================================

ic_x_indx, ic_y_indx = np.where(x1[0,:] == 2)
top_x_indx, top_y_indx = np.nonzero(x1[1,:])
bot_x_indx, bot_y_indx = np.nonzero(x1[2,:])

ic_pwr_x_mm = (ic_x_indx + 0.5)/16*200 - 1
ic_pwr_y_mm = (ic_y_indx + 0.5)/16*200
ic_gnd_x_mm = (ic_x_indx + 0.5)/16*200 + 1
ic_gnd_y_mm = (ic_y_indx + 0.5)/16*200

top_pwr_x_mm = (top_x_indx + 0.5)/16*200 - 1
top_pwr_y_mm = (top_y_indx + 0.5)/16*200
top_gnd_x_mm = (top_x_indx + 0.5)/16*200 + 1
top_gnd_y_mm = (top_y_indx + 0.5)/16*200

bot_pwr_x_mm = (bot_x_indx + 0.5)/16*200 - 1
bot_pwr_y_mm = (bot_y_indx + 0.5)/16*200
bot_gnd_x_mm = (bot_x_indx + 0.5)/16*200 + 1
bot_gnd_y_mm = (bot_y_indx + 0.5)/16*200

fig, ax = plt.subplots()
ax.plot(sxy[:,1]*1e3, 200 - sxy[:,0]*1e3)
ax.plot(ic_y_indx * 200 / 16, 200 - (ic_x_indx+0.5) * 200 / 16, 'ro')
ax.plot((top_y_indx)*200/16, 200 - (top_x_indx+0.5)*200/16, 'b*')
ax.plot((bot_y_indx)*200/16, 200 - (bot_x_indx+0.5)*200/16, 'g+')

for i in range(0, top_x_indx.shape[0]):
    ax.annotate(str(int(x1[1,top_x_indx[i],top_y_indx[i]])), xy=(((top_y_indx[i]+0.2)*200/16, 200 - (top_x_indx[i]+0.5)*200/16)), 
                    color='red')
for i in range(0, bot_x_indx.shape[0]):
    ax.annotate(str(int(x1[2,bot_x_indx[i],bot_y_indx[i]])), xy=(((bot_y_indx[i]+0.2)*200/16, 200 - (bot_x_indx[i]+0.5)*200/16)), 
                    color='red')

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.legend(['PCB', 'IC', 'Top decaps', 'Bottom decaps'])
plt.show()


'''Plot stackup'''
if np.where(x2==0)[0].shape[0] != 0:
    num_layers = np.where(x2 == 0)[0][0]
else:
    num_layers = 9
stackup = x2[0:num_layers] - 1
die_t = x2[9:9+num_layers-1]

die_t_reverse = list(reversed(list(die_t)))
stackup_reverse = list(reversed(list(stackup)))

die_t_string = list(map(str, reversed(list(die_t))))
df = pd.DataFrame([reversed(list(die_t))], columns=die_t_string, index=['Stackup'])
ax = df.plot(kind='bar', stacked=True, rot=360, legend='reverse')
ax.xaxis.set_label_text("")
ax.yaxis.set_label_text("Thickness (mm)")


for i in range(0, len(stackup_reverse)):
    if i == 0:
        ax.annotate('GND',(-0.2, 0))
    elif int(stackup_reverse[i]) == 1:
        ax.annotate('PWR',(-0.2, np.sum(die_t[-i:])-0.01))
    elif int(stackup_reverse[i]) == 0:
        ax.annotate('GND',(-0.2, np.sum(die_t[-i:]) - 0.01))

plt.show()

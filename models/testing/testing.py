#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


# Model Definition

# In[2]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(573, 2048)
        self.hidden1 = nn.Linear(2048, 1024)
        self.hidden2 = nn.Linear(1024, 512)
        self.hidden3 = nn.Linear(512, 256)
        self.hidden4 = nn.Linear(256, 128)
        self.hidden5 = nn.Linear(128, 64)
        self.hidden6 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = self.output(x)
        return x


# In[3]:


useCUDA = True
batchSize = 1


# In[4]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not useCUDA:
	device = torch.device('cpu')


# In[5]:


modelPath  = "../trained_models/1680241798"

net = Net()
net.load_state_dict(torch.load(modelPath))
net = net.to(device)


# In[ ]:





# In[ ]:


net.eval()

# Test the network

test_data = torch.load('../dataset/testSet.pt')

criterion = nn.MSELoss()

testLoader = DataLoader(test_data, batch_size=batchSize, shuffle=True)


runningLoss = 0
for data in testLoader:
	X, y = data
	y = y.unsqueeze(1)
	X, y = X.to(device), y.to(device)
	output = net(X)
	loss = criterion(output, y)
	runningLoss += loss.item()

print("Test Loss: ", runningLoss/(len(testLoader)*batchSize))


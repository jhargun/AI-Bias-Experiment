#!/usr/bin/env python
# coding: utf-8

# In[19]:


import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time
import pandas as pd


# Model Definition

# In[28]:


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


# Variables Definintions

# In[21]:


t = int(time.time())
useCUDA = True
dataPath = "../large_field_preprocessed_data.csv"
epochs = 50
batchSize = 32
modelPath = f"../trained_models/{t}"


# Device Check

# In[22]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not useCUDA:
	device = torch.device('cpu')


# Model and Dataset Creation

# In[29]:


net = Net()
net = net.to(device)


trainSet = torch.load("../dataset/trainSet.pt")
trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)


# Training

# In[30]:


optimizer = optim.Adam(net.parameters(), lr =1e-5)

criterion = nn.MSELoss(reduction='mean')

print("Epochs Started")

for epoch in range(epochs):
	running_loss = 0.0
	for i, data in enumerate(trainLoader):
		X, y = data
		y = y.unsqueeze(1)
		X = X.to(device)
		y = y.to(device)

		net.zero_grad()

		output = net(X)

		loss = criterion(output, y)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		if i % 1000 == 999:    # print every 1000 mini-batches
			print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
			running_loss = 0.0

torch.save(net.state_dict(), modelPath)
torch.cuda.empty_cache()


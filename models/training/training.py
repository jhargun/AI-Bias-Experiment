#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time
import pandas as pd
import os


# Variables Definintions

# In[ ]:


t = int(time.time())
inputSize = 573
useCUDA = True
epochs = 1000
batchSize = 8192
lr = 1e-4

name = "AllFields"

dir = os.getcwd()
modelFolder = f"{dir}/../trained_models/"
if not os.path.exists(modelFolder):
	os.makedirs(modelFolder)

modelPath = f"{modelFolder}{t}-{batchSize}-{inputSize}-{name}.pt"


# Device Check

# In[ ]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not useCUDA:
	device = torch.device('cpu')


# Model Definition

# In[ ]:


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.input = nn.Linear(inputSize, 2048)
#         self.hidden1 = nn.Linear(2048, 1024)
#         self.hidden2 = nn.Linear(1024, 512)
#         self.hidden3 = nn.Linear(512, 256)
#         self.hidden4 = nn.Linear(256, 128)
#         self.hidden5 = nn.Linear(128, 64)
#         self.hidden6 = nn.Linear(64, 32)
#         self.output = nn.Linear(32, 1)
        
#     def forward(self, x):
#         x = F.leaky_relu(self.input(x))
#         x = F.leaky_relu(self.hidden1(x))
#         x = F.leaky_relu(self.hidden2(x))
#         x = F.leaky_relu(self.hidden3(x))
#         x = F.leaky_relu(self.hidden4(x))
#         x = F.leaky_relu(self.hidden5(x))
#         x = F.leaky_relu(self.hidden6(x))
#         x = self.output(x)
#         return x
    
# net = Net()
net = nn.Sequential(nn.Linear(inputSize, 8192),
                    nn.LeakyReLU(),
                    nn.Linear(8192, 4096),
                    nn.LeakyReLU(),
                    nn.Linear(4096, 2048),
                    nn.LeakyReLU(),
                    nn.Linear(2048, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64,32),
                    nn.LeakyReLU(),
                    nn.Linear(32,1))

net = net.to(device)


# Model and Dataset Creation

# In[ ]:


trainSetX = torch.load("../dataset/trainSetX.pt")
trainSetY = torch.load("../dataset/trainSetY.pt")
# trainSetX = trainSetX.to(device)
# trainSetY = trainSetY.to(device)
print(trainSetX.shape)
print(trainSetY.shape)


# In[ ]:


testSetX = torch.load("../dataset/testSetX.pt")
testSetY = torch.load("../dataset/testSetY.pt")
# testSetX = testSetX.to(device)
# testSetY = testSetY.to(device)
print(testSetX.shape)
print(testSetY.shape)


# In[ ]:


trainSet = TensorDataset(trainSetX, trainSetY)
trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)


# In[ ]:


testSet = TensorDataset(testSetX, testSetY)
testLoader = DataLoader(testSet, batch_size=int(len(testSet)/10), shuffle=True)


# Training

# In[ ]:


optimizer = optim.Adam(net.parameters(), lr = lr)

trainCriterion = nn.L1Loss(reduction='mean')
testCriterion = nn.L1Loss(reduction='mean')

print("Epochs Started")

bestLoss = float('inf')

for epoch in range(epochs):
	net.train()
	running_loss = 0.0
	torch.cuda.empty_cache()
	for i, data in enumerate(trainLoader):
		X, y = data
		y = y.unsqueeze(1)
		X = X.to(device)
		y = y.to(device)
  
		net.zero_grad()
		output = net(X)
		loss = trainCriterion(output, y)
		# print(loss)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		# if i % 1000 == 999:    # print every 1000 mini-batches
	print(f'Epoch {epoch+1}, Train loss: {running_loss/len(trainLoader):.3f}')
	running_loss = 0.0
	
	
	torch.cuda.empty_cache()
	net.eval()
	with torch.no_grad():
		test_running_loss = 0.0
		for i, data in enumerate(testLoader):
			torch.cuda.empty_cache()
			X, y = data
			y = y.unsqueeze(1)
			X = X.to(device)
			y = y.to(device)
   
			output = net(X)
			testLoss = testCriterion(output, y)
			test_running_loss += testLoss.item()
		print(f'Epoch {epoch+1}, Test loss: {test_running_loss/len(testLoader):.3f}')
		if test_running_loss/len(testLoader) < bestLoss:
			bestLoss = test_running_loss/len(testLoader)
			torch.save(net.state_dict(), modelPath)
			print("Model Saved")

torch.cuda.empty_cache()


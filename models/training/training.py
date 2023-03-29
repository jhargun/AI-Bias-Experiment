import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time
from training.model import Net
import sys


def run_train(modelPath, xPath, yPath, epochs=20, prevModelPath = None, useCUDA = False, batchSize = 10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not useCUDA:
        device = torch.device('cpu')
    print("Loading numpy")
    train_x = np.load(xPath)
    train_y = np.load(yPath)
    print(train_x.shape)
    print(train_y.shape)

    print("Loading Device")
    
    tensor_x = torch.Tensor(train_x)
    tensor_y = torch.Tensor(train_y)
    tensor_y = tensor_y.type(torch.LongTensor)

    net = Net()
    tensor_x = tensor_x.to(device)
    tensor_y = tensor_y.to(device)
    net.to(device)


    trainSet = TensorDataset(tensor_x, tensor_y)
    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)

    print("Device loaded")

    if(prevModelPath != None):
        net.load_state_dict(torch.load(prevModelPath))

    optimizer = optim.Adam(net.parameters(), lr =1e-4)

    criterion = nn.MSELoss()

    print("Epochs Started")

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainLoader):
            X, y = data
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
    del tensor_x
    del tensor_y
    torch.cuda.empty_cache()
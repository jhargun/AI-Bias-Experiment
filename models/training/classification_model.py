import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, output, target, multiplier=1):
		return torch.mean(torch.abs(output - target))



class ClassificationNet(nn.Module):
    def __init__(self, inputSize: int):
        super().__init__()
        
        # Change input size
        self.input = nn.Linear(inputSize, 1024)
        self.hidden1 = nn.Linear(1024, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.hidden4 = nn.Linear(128, 64)
        self.hidden5 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 11)
        
    def forward(self, x):
        print('hi',x.shape)
        x = F.leaky_relu(self.input(x))
        x = F.leaky_relu(self.hidden1(x))
        x = F.leaky_relu(self.hidden2(x))
        x = F.leaky_relu(self.hidden3(x))
        x = F.leaky_relu(self.hidden4(x))
        x = F.leaky_relu(self.hidden5(x))
        x = self.output(x)
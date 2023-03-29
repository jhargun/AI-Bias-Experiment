import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, output, target, multiplier=1):
		return torch.mean(torch.abs(output - target))



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Change input size
        self.input = nn.Linear(784, 128)
        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.output(x)
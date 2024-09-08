import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleDNN, self).__init__()
        self.layer1 = nn.Linear(input_shape, 200)
        self.layer2 = nn.Linear(200, 100)
        self.layer3 = nn.Linear(100, 50)
        self.layer4 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
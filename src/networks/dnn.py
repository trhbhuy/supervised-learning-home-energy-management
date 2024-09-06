import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def _activation(self, x):
        return F.relu(x)

    def forward(self, x):
        x = self._activation(self.layer1(x))
        x = self._activation(self.layer2(x))
        x = self._activation(self.layer3(x))
        return self.layer4(x)

# class DNN(nn.Module):
#     def __init__(self, n_features, n_target):
#         super(DNN, self).__init__()
#         self.layer1 = nn.Linear(n_features, 200)
#         self.layer2 = nn.Linear(200, 100)
#         self.layer3 = nn.Linear(100, 50)
#         self.layer4 = nn.Linear(50, n_target)

#     def forward(self, x):
#         x = torch.relu(self.layer1(x))
#         x = torch.relu(self.layer2(x))
#         x = torch.relu(self.layer3(x))
#         return self.layer4(x)

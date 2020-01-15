import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size):

        super(MultiLayerPerceptron, self).__init__()

        ######

        # 4.3 YOUR CODE HERE


        self.fc1 = nn.Linear(input_size, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)

        self.fc2 = nn.Linear(64, 1)
        ######

    def forward(self, features):

        pass
        ######

        # 4.3 YOUR CODE HERE
        x = F.relu(self.fc1(features.float()))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = torch.sigmoid(self.fc2(x))
        return x
        ######

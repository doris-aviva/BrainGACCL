import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(torch.nn.Module):
    def __init__(self, dim_c, num_classes, drop):
        super(Classifier, self).__init__()

        dim = 64

        self.fc1 = nn.Linear(dim, dim_c)
        self.fc2 = nn.Linear(dim_c, num_classes)
        self.drop = drop

    def forward(self, x, device):
        x = x.to(device)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


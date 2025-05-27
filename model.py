import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from GINlayer import WGIN


class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes, drop, dim=64, num_layers=2):
        super(Net, self).__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(self.num_layers):

            if i:
                nn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            else:
                nn = nn.Sequential(nn.Linear(num_features, dim), nn.ReLU(), nn.Linear(dim, dim))
            conv = WGIN(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

        self.fc = nn.Linear(dim * 2, dim)#  * 2
        self.fc1 = nn.Linear(dim, dim)#  * 2
        self.fc2 = nn.Linear(dim, num_classes)

        self.drop = drop

    def get_embeddings(self, x, edge_index, edge_weight, batch, device):
        xs = []
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = self.bns[i](x)
            xs.append(x)

        x_pool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(x_pool, 1)
        return x

    def forward(self, x, edge_index, edge_weight, batch, device):
        x = self.get_embeddings(x, edge_index, edge_weight, batch, device)
        x = F.relu(self.fc(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

    def forward_cl(self, x, edge_index, edge_weight, batch, device):
        x = self.get_embeddings(x, edge_index, edge_weight, batch, device)
        x = F.relu(self.fc(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.fc1(x))
        return x

    def get_node_embeddings(self, x, edge_index, edge_weight, batch, device):
        xs = []
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = self.bns[i](x)
            xs.append(x)
        return x


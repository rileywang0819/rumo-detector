import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from tools.earlystopping import EarlyStopping
from tqdm import tqdm
from tools.evalutate import *
from process.load_data import *
from process.rand5fold import *


class Network(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device, edges=None):
        super(Network, self).__init__()
        self.conv1 = GCNConv(in_feats, 2 * hid_feats)
        self.conv2 = GCNConv(2 * hid_feats, out_feats)
        self.device = device
        self.fc = nn.Linear(out_feats, 4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = scatter_mean(x, data.batch, dim=0)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


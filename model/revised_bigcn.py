import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from torch.tensor import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm
from tools.earlystopping import EarlyStopping
from tools.evalutate import *
from process.load_data import *
from process.rand5fold import *
from model.revised_gcn import *


class TDGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super(TDGCN, self).__init__()
        self.device = device
        self.setup_layers(in_feats, hid_feats, out_feats)
        self.conv1 = GCNConv(in_feats, hid_feats)
        # root feature enhancement
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def setup_layers(self, in_feats, hid_feats, out_feats):
        self.pos_base_aggregator = SignedLayerBase(in_feats, hid_feats)
        self.neg_base_aggregator = SignedLayerBase(in_feats, hid_feats)
        self.pos_deep_aggregator = SignedLayerDeep(2 * hid_feats, out_feats)
        self.neg_deep_aggregator = SignedLayerDeep(2 * hid_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        pos_link, neg_link = data.TD_pos_link, data.TD_neg_link
        pos_link = torch.from_numpy(np.array(Tensor.cpu(pos_link), dtype=np.int64).T).type(torch.long).to(self.device)
        neg_link = torch.from_numpy(np.array(Tensor.cpu(neg_link), dtype=np.int64).T).type(torch.long).to(self.device)
        h_pos, h_neg = [], []
        h_pos.append(torch.tanh(self.pos_base_aggregator(x, pos_link)))
        h_neg.append(torch.tanh(self.neg_base_aggregator(x, neg_link)))
        concat_h = torch.cat((h_pos[0], h_neg[0]), 1)
        h_pos.append(torch.tanh(self.pos_deep_aggregator(concat_h, pos_link, edge_index)))
        h_neg.append(torch.tanh(self.neg_deep_aggregator(concat_h, neg_link, edge_index)))
        x = torch.cat((h_pos[1], h_neg[1]), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x


class BUGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super(TDGCN, self).__init__()
        self.device = device
        self.setup_layers(in_feats, hid_feats, out_feats)

    def setup_layers(self, in_feats, hid_feats, out_feats):
        self.pos_base_aggregator = SignedLayerBase(in_feats, hid_feats)
        self.neg_base_aggregator = SignedLayerBase(in_feats, hid_feats)
        self.pos_deep_aggregator = SignedLayerDeep(hid_feats, out_feats)
        self.neg_deep_aggregator = SignedLayerDeep(hid_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        pos_link, neg_link = data.BU_pos_link, data.BU_neg_link
        pos_link = torch.from_numpy(np.array(Tensor.cpu(pos_link), dtype=np.int64).T).type(torch.long).to(self.device)
        neg_link = torch.from_numpy(np.array(Tensor.cpu(neg_link), dtype=np.int64).T).type(torch.long).to(self.device)
        h_pos, h_neg = [], []
        h_pos.append(torch.tanh(self.pos_base_aggregator(x, pos_link)))
        h_neg.append(torch.tanh(self.neg_base_aggregator(x, neg_link)))
        concat_h = torch.cat((h_pos[0], h_neg[0]), 1)
        h_pos.append(torch.tanh(self.pos_deep_aggregator(concat_h, pos_link, edge_index)))
        h_neg.append(torch.tanh(self.neg_deep_aggregator(concat_h, neg_link, edge_index)))
        x = torch.cat((h_pos[1], h_neg[1]), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class Network(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super(Network, self).__init__()
        self.SignedBUGCN = TDGCN(in_feats, hid_feats, out_feats, device)
        self.SignedTDGCN = TDGCN(in_feats, hid_feats, out_feats, device)
        self.fc = nn.Linear((out_feats + hid_feats) * 2, 4)

    def forward(self, data):
        TD_x = self.SignedTDGCN(data)
        BU_x = self.SignedBUGCN(data)
        x = torch.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


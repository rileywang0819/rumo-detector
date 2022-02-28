import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm
from tools.earlystopping import EarlyStopping
from tools.evalutate import *
from process.load_data import *
from process.rand5fold import *


class TDGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super(TDGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        # root feature enhancement
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.device = device

    def forward(self, data):
        # x: 节点的特征矩阵， shaper=[节点个数, 节点特征数]
        # edge_index: 图的邻接矩阵(COO格式), shape=[2, 边的数量], type=torch.long
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_index = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            # data.batch标记了data内每一条数据和每一个batch的归属关系
            index = torch.eq(data.batch, num_batch)
            # boolean indexing，某个batch的root_extend全部赋值为x1中根节点的值
            root_extend[index] = x1[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class BUGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super(BUGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)
        self.device = device

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_index = data.rootindex
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[root_index[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x


class Network(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device, edges=None):
        super(Network, self).__init__()
        self.BUGCN = BUGCN(in_feats, hid_feats, out_feats, device)
        self.TDGCN = TDGCN(in_feats, hid_feats, out_feats, device)
        # fully connected layer, output 4 classes
        self.fc = nn.Linear((out_feats + hid_feats) * 2, 4)

    def forward(self, data):
        TD_x = self.TDGCN(data)
        BU_x = self.BUGCN(data)
        x = torch.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        # print("suspend")
        return x


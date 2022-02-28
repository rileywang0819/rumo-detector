import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GCNConv
from torch.tensor import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_mean, scatter_add


# =========================
#      layer classes
# =========================

class SignedLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SignedLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.conv1 = GCNConv(in_feats, out_feats)
        self.conv2 = GCNConv(in_feats, out_feats)
        self.weight = Parameter(torch.Tensor(2 * in_feats, out_feats))
        self.bias = Parameter(torch.Tensor(out_feats))


class SignedLayerBase(SignedLayer):
    def forward(self, x, edge_index):
        row, col = edge_index
        out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        out = torch.cat((torch.add(out, x), x), 1)
        out = torch.matmul(out, self.weight)
        out += self.bias
        # out = F.normalize(out, p=2, dim=-1)
        out += self.conv1(x, edge_index)
        return out

class SignedLayerDeep(SignedLayer):
    def forward(self, x, edge_index_signed, edge_index):
        out = self.conv2(x, edge_index_signed)
        temp = self.conv2(x, edge_index)

        return out


class Network(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, device):
        super(Network, self).__init__()
        self.device = device
        self.setup_layers(in_feats, hid_feats, out_feats)
        self.fc = nn.Linear((out_feats + hid_feats), 4)

    def setup_layers(self, in_feats, hid_feats, out_feats):
        self.pos_base_aggregator = SignedLayerBase(in_feats, hid_feats)
        self.neg_base_aggregator = SignedLayerBase(in_feats, hid_feats)
        self.pos_deep_aggregator = SignedLayerDeep(2 * hid_feats, out_feats)
        self.neg_deep_aggregator = SignedLayerDeep(2 * hid_feats, out_feats)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        pos_link, neg_link = data.pos_link, data.neg_link
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
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

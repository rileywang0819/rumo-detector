import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data


def read_graph(edge_path):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers.
    :param args: Arguments object.
    :return edges: Edges dictionary.
    """
    edges = {
        "negative_edges": [],
        "positive_edges": []
    }

    for line in open(edge_path):
        line = line.rstrip()
        reply_id, replied_id = int(line.split('\t')[1]), int(line.split('\t')[2])
        # print(line.split('\t')[0])
        link_type = int(line.split('\t')[3])

        if link_type == -1:
            edges["negative_edges"].append([int(reply_id), int(replied_id)])
        else:
            edges["positive_edges"].append([int(reply_id), int(replied_id)])

    edges["edge_count"] = len(edges["negative_edges"]) + len(edges["positive_edges"])
    edges["node_count"] = len(set([edge[0] for edge in edges["positive_edges"]]
                                  + [edge[1] for edge in edges["positive_edges"]]
                                  + [edge[0] for edge in edges["negative_edges"]]
                                  + [edge[1] for edge in edges["negative_edges"]]
                                  ))
    return edges


# ==============================
#   non-directed dataset
# ==============================

class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, droprate=0, data_path=None):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))


# ==============================
#   linked-graph dataset
# ==============================

class LinkedGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=10000, droprate=0, data_path=None, link_path=None):
        # 过滤到过低或过高的树
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path  # <Phemegraph>
        self.link_path = link_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']

        # store link information
        link_path = os.path.join(self.link_path, id + ".txt")
        link_data = read_graph(link_path)

        new_edgeindex = edgeindex

        return Data(
            x=torch.tensor(data['x'], dtype=torch.float32),
            edge_index=torch.LongTensor(new_edgeindex),
            pos_link=torch.LongTensor(link_data['positive_edges']),
            neg_link=torch.LongTensor(link_data['negative_edges']),
            y=torch.LongTensor([int(data['y'])]),
            root=torch.LongTensor(data['root']),
            rootindex=torch.LongTensor([int(data['rootindex'])])
        )


# ==============================
#   bi-directed dataset
# ==============================

class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, tddroprate=0, budroprate=0, data_path=None):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow, bucol]
        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex), BU_edge_index=torch.LongTensor(bunew_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))

# TODO
class BiLinkedGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, tddroprate=0, budroprate=0, data_path=None, link_path=None):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.link_path = link_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']

        # read directed graph
        link_path = os.path.join(self.link_path, id + ".txt")
        link_data = read_graph(link_path)

        reversed_link_data = {'positive_edges': [], 'negative_edges': []}
        for link in link_data['positive_edges']:
            reversed_link_data['positive_edges'].append([link[1], link[0]])
        for link in link_data['negative_edges']:
            reversed_link_data['negative_edges'].append([link[1], link[0]])

        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow, bucol]

        return Data(
            x=torch.tensor(data['x'], dtype=torch.float32),
            edge_index=torch.LongTensor(new_edgeindex),
            BU_edge_index=torch.LongTensor(bunew_edgeindex),
            BU_pos_link=torch.LongTensor(link_data['positive_edges']),
            BU_neg_link=torch.LongTensor(link_data['positive_edges']),
            TD_pos_link=torch.LongTensor(reversed_link_data['positive_edges']),
            TD_neg_link=torch.LongTensor(reversed_link_data['negative_edges']),
            y=torch.LongTensor([int(data['y'])]),
            root=torch.LongTensor(data['root']),
            rootindex=torch.LongTensor([int(data['rootindex'])]))

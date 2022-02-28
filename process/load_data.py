import os


from process.dataset import *


# =======================
#        load tree
# =======================

# cwd = os.path.abspath(os.path.join(os.getcwd(), ".."))
cwd = os.getcwd()
# print(cwd)


def loadTree(dataname):

    if 'Pheme' in dataname:
        treePath = os.path.join(cwd, "data\\" + "Pheme\\PhemeText\\" + "data.TD_RvNN.vol_5000.txt")
        # print(treePath)
    elif 'Twitter' in dataname:
        treePath = os.path.join(cwd, "data\\" + "Twitter\\" + dataname + "\\data.TD_RvNN.vol_5000.txt")
        # print(treePath)
    print("loading twitter tree data......")
    treeDic = {}

    for line in open(treePath):
        line = line.rstrip()
        """
        eid: root id
        indexP: parent tweet id of the current tweet
        indexC: current tweet id 
        max_degree: parent number
        maxL: max text length of the tree
        Vec: word dictionary {word_id: count_times}
        """
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]

        if not treeDic.__contains__(eid):
            treeDic[eid] = {}

        """
        treeDic = {
            tree1_id: {tweet1_id: {属性}, tweet2_id: {属性}, ...}
            tree2_id: ...
        }
        """
        treeDic[eid][indexC] = {
            'parent': indexP,
            'max_degee': max_degree,
            'maxL': maxL,
            'word_vec': Vec
        }
    print(f"\t#trees: {len(treeDic)}")

    return treeDic


# =======================
#        load data
# =======================

# for gcn model
def loadData(dataname, treeDic, fold_x_train, fold_x_test, droprate):
    if 'Twitter' in dataname:
        data_path = os.path.join(cwd, "data\\" + "Twitter\\" + dataname + "graph")
        # print(data_path)
    elif dataname == 'Pheme':
        data_path = os.path.join(cwd, "data\\" + "Pheme\\" + dataname + "graph")
        # print(data_path)
    print("loading train set......")
    traindata_lst = GraphDataset(fold_x_train, treeDic, droprate=droprate, data_path=data_path)
    print("num of train:", len(traindata_lst))
    print("loading test set......")
    testdata_lst = GraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("num of test:", len(testdata_lst))
    return traindata_lst, testdata_lst



def loadData2(dataname, treeDic, fold_x_train, fold_x_test, droprate):
    if 'Twitter' in dataname:
        raise Exception("Twitter dataset cannot be used for stance-guided model")
    elif dataname == 'Pheme':
        data_path = os.path.join(cwd, "data\\" + "Pheme\\" + dataname + "graph")
        link_path = os.path.join(cwd, "data\\" + "Pheme\\" + dataname + "link")
        # print(data_path)
    print("loading train set......")
    traindata_lst = LinkedGraphDataset(fold_x_train, treeDic, droprate=droprate, data_path=data_path, link_path=link_path)
    print("num of train:", len(traindata_lst))
    print("loading test set......")
    testdata_lst = LinkedGraphDataset(fold_x_test, treeDic, data_path=data_path, link_path=link_path)
    print("num of test:", len(testdata_lst))
    return traindata_lst, testdata_lst


def loadBiData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate, BUdroprate):
    if 'Twitter' in dataname:
        data_path = os.path.join(cwd, "data\\" + "Twitter\\" + dataname + "graph")
        # print(data_path)
    elif dataname == 'Pheme':
        data_path = os.path.join(cwd, "data\\" + "Pheme\\" + dataname + "graph")
        # print(data_path)
    print("loading train set......")
    traindata_lst = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("num of train:", len(traindata_lst))
    print("loading test set......")
    testdata_lst = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("num of test:", len(testdata_lst))
    return traindata_lst, testdata_lst


def loadBiData2(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate, BUdroprate):
    if 'Twitter' in dataname:
        raise Exception("Twitter dataset cannot be used for stance-guided model")
    elif dataname == 'Pheme':
        data_path = os.path.join(cwd, "data\\" + "Pheme\\" + dataname + "graph")
        link_path = os.path.join(cwd, "data\\" + "Pheme\\" + dataname + "link")
    print("loading train set......")
    traindata_lst = BiLinkedGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate,
                                   data_path=data_path, link_path=link_path)
    print("num of train:", len(traindata_lst))
    print("loading test set......")
    testdata_lst = BiLinkedGraphDataset(fold_x_test, treeDic, data_path=data_path, link_path=link_path)
    print("num of test:", len(testdata_lst))
    return traindata_lst, testdata_lst



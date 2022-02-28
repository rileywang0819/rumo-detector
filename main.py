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


if len(sys.argv) < 4:
    print("You should enter 1) dataset_name; 2) iteration_times; 3) model_name")
    sys.exit()
if sys.argv[3] == 'gcn':
    from model.gcn import Network
elif sys.argv[3] == 'bigcn':
    from model.bigcn import Network
elif sys.argv[3] == 'revised_gcn':
    from model.revised_gcn import Network
elif sys.argv[3] == 'revised_bigcn':
    from model.revised_bigcn import Network
else:
    print("Invalid Input. Check your input again.")
    sys.exit()


# def train_model(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr,
#               weight_decay, patient, n_epochs, batchsize, dataname, iter):

def train_model(treeDic, x_test, x_train, lr, weight_decay, patience,
                n_epochs, batchsize, dataname, iter, modelname, droprate, device, edge_path=None):
    model = Network(in_feats=5000, hid_feats=64, out_feats=64, device=device).to(device)
    if modelname == 'bigcn':
        BU_params = list(map(id, model.BUGCN.conv1.parameters()))
        BU_params += list(map(id, model.BUGCN.conv2.parameters()))
        # len(BU_params) = 4, 记录BU-GCN模型的参数所在的内存地址
        # 过滤出除BU-GCN以外的所有网络模型参数并保存
        base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
        param = [
            {'params': base_params},
            {'params': model.BUGCN.conv1.parameters(), 'lr': lr / 5},
            {'params': model.BUGCN.conv2.parameters(), 'lr': lr / 5}
        ]
    elif modelname == 'gcn' or 'revised_gcn' or 'revised_bigcn':
        param = model.parameters()
        # gcn_params = list(map(id, model.conv1.parameters()))
        # gcn_params += list(map(id, model.conv2.parameters()))
        # base_params = filter(lambda p: id(p) not in gcn_params, model.parameters())
        # param = [
        #     {'params': base_params},
        #     {'params': model.conv1.parameters(), 'lr': lr / 5},
        #     {'params': model.conv2.parameters(), 'lr': lr / 5}
        # ]
    optimizer = torch.optim.Adam(param, lr=lr, weight_decay=weight_decay)

    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):
        if modelname == 'bigcn':
            # 加载双向的图数据
            train_data_list, test_data_list = loadBiData(
                dataname, treeDic, x_train, x_test, TDdroprate=droprate, BUdroprate=droprate
            )
        elif modelname == 'gcn':
            train_data_list, test_data_list = loadData(
                dataname, treeDic, x_train, x_test, droprate=droprate
            )
        elif modelname == 'revised_gcn':
            train_data_list, test_data_list = loadData2(
                dataname, treeDic, x_train, x_test, droprate=droprate
            )
        elif modelname == 'revised_bigcn':
            train_data_list, test_data_list = loadBiData2(
                dataname, treeDic, x_train, x_test, TDdroprate=droprate, BUdroprate=droprate
            )
        # change "num_workers" if you need concurrency
        train_loader = DataLoader(train_data_list, batch_size=batchsize, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_data_list, batch_size=batchsize, shuffle=False, num_workers=0)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqmd_train_loader = tqdm(train_loader)      # process visualization

        for batch_data in tqmd_train_loader:
            batch_data.to(device)
            out_labels = model(batch_data)  # y_hat
            loss = F.nll_loss(out_labels, batch_data.y)
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            # 统计预测正确的个数
            correct = pred.eq(batch_data.y).sum().item()
            train_acc = correct / len(batch_data.y)
            avg_acc.append(train_acc)
            print("\tIter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(
                iter, epoch, batch_idx, loss.item(), train_acc
            ))
            batch_idx += 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        # record validation result temporarily
        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1 = [], [], [], [], []
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2 = [], [], [], []
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3 = [], [], [], []
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], []

        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for batch_data in tqdm_test_loader:
            batch_data.to(device)
            val_out = model(batch_data)
            val_loss = F.nll_loss(val_out, batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(batch_data.y).sum().item()
            val_acc = correct / len(batch_data.y)

            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, \
                    Acc4, Prec4, Recll4, F4 = evaluation4class(val_pred, batch_data.y)
            temp_val_Acc_all.append(Acc_all)
            temp_val_Acc1.append(Acc1)
            temp_val_Prec1.append(Prec1)
            temp_val_Recll1.append(Recll1)
            temp_val_F1.append(F1)
            temp_val_Acc2.append(Acc2)
            temp_val_Prec2.append(Prec2)
            temp_val_Recll2.append(Recll2)
            temp_val_F2.append(F2)
            temp_val_Acc3.append(Acc3)
            temp_val_Prec3.append(Prec3)
            temp_val_Recll3.append(Recll3)
            temp_val_F3.append(F3)
            temp_val_Acc4.append(Acc4)
            temp_val_Prec4.append(Prec4)
            temp_val_Recll4.append(Recll4)
            temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)

        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(
            epoch, np.mean(temp_val_losses), np.mean(temp_val_accs))
        )

        # show result after each epoch
        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
        torch.cuda.empty_cache()

        # early stopping
        early_stopping(
            np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1),
            np.mean(temp_val_F2), np.mean(temp_val_F3), np.mean(temp_val_F4), model,
            modelname, dataname
        )
        accs = np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break

    # return train_losses, val_losses, train_accs, val_accs, accs, F1, F2, F3, F4
    return accs, F1, F2, F3, F4


if __name__ == '__main__':
    lr = 0.0005     # learning rate
    weight_decay = 1e-4     # avoid over-fitting
    patience = 10   # early stop: max 10 epochs without improvement
    n_epochs = 10   # 10, 50, 100, 200
    batchsize = 64  # 16, 32, 64, 128, 256
    # TDdroprate = 0.2
    # BUdroprate = 0.2
    droprate = 0.2  # 0, 0.2
    datasetname = sys.argv[1]   # 'Pheme', 'Twitter15', 'Twitter16'
    iterations = int(sys.argv[2])   # 100
    modelname = sys.argv[3]
    model = "GCN"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []

    for iter in range(iterations):

        fold0_x_test, fold0_x_train, \
        fold1_x_test, fold1_x_train, \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test, fold4_x_train = load5foldData(datasetname)

        treeDic = loadTree(datasetname)

        # train and test
        accs0, F1_0, F2_0, F3_0, F4_0 = train_model(
            treeDic, fold0_x_test, fold0_x_train,
            lr, weight_decay, patience, n_epochs,
            batchsize, datasetname, iterations, modelname, droprate, device
        )
        accs1, F1_1, F2_1, F3_1, F4_1 = train_model(
            treeDic, fold0_x_test, fold0_x_train,
            lr, weight_decay, patience, n_epochs,
            batchsize, datasetname, iterations, modelname, droprate, device
        )
        accs2, F1_2, F2_2, F3_2, F4_2 = train_model(
            treeDic, fold0_x_test, fold0_x_train,
            lr, weight_decay, patience, n_epochs,
            batchsize, datasetname, iterations, modelname, droprate, device
        )
        accs3, F1_3, F2_3, F3_3, F4_3 = train_model(
            treeDic, fold0_x_test, fold0_x_train,
            lr, weight_decay, patience, n_epochs,
            batchsize, datasetname, iterations, modelname, droprate, device
        )
        accs4, F1_4, F2_4, F3_4, F4_4 = train_model(
            treeDic, fold0_x_test, fold0_x_train,
            lr, weight_decay, patience, n_epochs,
            batchsize, datasetname, iterations, modelname, droprate, device
        )

        test_accs.append((accs0 + accs1 + accs2 + accs3 + accs4) / 5)
        NR_F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
        FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
        TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
        UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)

    print(f"Total_Test_Accuracy: {sum(test_accs) / iterations}"
          f"\tNR F1: {sum(NR_F1) / iterations}"
          f"\tFR F1: {sum(FR_F1) / iterations}"
          f"\tTR F1: {sum(TR_F1) / iterations}"
          f"\tUR F1: {sum(UR_F1) / iterations}")

    print("\n\n===================== ALL DONE =====================")

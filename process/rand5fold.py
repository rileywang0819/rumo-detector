import random
import os

from random import shuffle


# cwd = os.path.abspath(os.path.join(os.getcwd(), ".."))
cwd = os.getcwd()
# print(cwd)


def load5foldData(obj):

    if 'Pheme' in obj:
        labelPath = os.path.join(cwd, "data\\" + "Pheme\\PhemeText\\" + obj + "_label_All.txt")
        # print(labelPath)
    elif 'Twitter' in obj:
        labelPath = os.path.join(cwd, "data\\" + "Twitter\\" + obj + "\\" + obj + "_label_All.txt")
        # print(labelPath)

    labelset_nonR = ['news', 'non-rumor']
    labelset_f = ['false']
    labelset_t = ['true']
    labelset_u = ['unverified']
    print("loading tree label......")

    NR, F, T, U = [], [], [], []    # 用于记录4个label分别对应的root_ID
    l1 = l2 = l3 = l4 = 0   # 记录4个label分别对应的root的个数
    labelDic = {}   # key: root_id ---- value: label

    for line in open(labelPath):    # for each line
        line = line.rstrip()
        # label: label of root tweet; eid: root_id of the tree
        label, eid = line.split('\t')[0], line.split('\t')[2]
        labelDic[eid] = label.lower()
        if label in labelset_nonR:    # non-rumor or news
            NR.append(eid)
            l1 += 1
        elif label in labelset_f:    # false
            F.append(eid)
            l2 += 1
        elif label in labelset_t:    # true
            T.append(eid)
            l3 += 1
        elif label in labelset_u:    # unverified
            U.append(eid)
            l4 += 1

    print("\t#source tweets: " + str(len(labelDic)))
    print("\t#non rumors: " + str(l1))
    print("\t#false rumors:" + str(l2))
    print("\t#true rumors:" + str(l3))
    print("\t#unverified rumors:" + str(l4))
    random.shuffle(NR)
    random.shuffle(F)
    random.shuffle(T)
    random.shuffle(U)

    """5-fold cross validation"""
    fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
    fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
    len1 = int(l1 * 0.2)
    len2 = int(l2 * 0.2)
    len3 = int(l3 * 0.2)
    len4 = int(l4 * 0.2)

    """80% for train, 20% for test"""
    fold0_x_test.extend(NR[0:len1])
    fold0_x_test.extend(F[0:len2])
    fold0_x_test.extend(T[0:len3])
    fold0_x_test.extend(U[0:len4])
    fold0_x_train.extend(NR[len1:])
    fold0_x_train.extend(F[len2:])
    fold0_x_train.extend(T[len3:])
    fold0_x_train.extend(U[len4:])

    fold1_x_train.extend(NR[0:len1])
    fold1_x_train.extend(NR[len1 * 2:])
    fold1_x_train.extend(F[0:len2])
    fold1_x_train.extend(F[len2 * 2:])
    fold1_x_train.extend(T[0:len3])
    fold1_x_train.extend(T[len3 * 2:])
    fold1_x_train.extend(U[0:len4])
    fold1_x_train.extend(U[len4 * 2:])
    fold1_x_test.extend(NR[len1:len1 * 2])
    fold1_x_test.extend(F[len2:len2 * 2])
    fold1_x_test.extend(T[len3:len3 * 2])
    fold1_x_test.extend(U[len4:len4 * 2])

    fold2_x_train.extend(NR[0:len1 * 2])
    fold2_x_train.extend(NR[len1 * 3:])
    fold2_x_train.extend(F[0:len2 * 2])
    fold2_x_train.extend(F[len2 * 3:])
    fold2_x_train.extend(T[0:len3 * 2])
    fold2_x_train.extend(T[len3 * 3:])
    fold2_x_train.extend(U[0:len4 * 2])
    fold2_x_train.extend(U[len4 * 3:])
    fold2_x_test.extend(NR[len1 * 2:len1 * 3])
    fold2_x_test.extend(F[len2 * 2:len2 * 3])
    fold2_x_test.extend(T[len3 * 2:len3 * 3])
    fold2_x_test.extend(U[len4 * 2:len4 * 3])

    fold3_x_train.extend(NR[0:len1 * 3])
    fold3_x_train.extend(NR[len1 * 4:])
    fold3_x_train.extend(F[0:len2 * 3])
    fold3_x_train.extend(F[len2 * 4:])
    fold3_x_train.extend(T[0:len3 * 3])
    fold3_x_train.extend(T[len3 * 4:])
    fold3_x_train.extend(U[0:len4 * 3])
    fold3_x_train.extend(U[len4 * 4:])
    fold3_x_test.extend(NR[len1 * 3:len1 * 4])
    fold3_x_test.extend(F[len2 * 3:len2 * 4])
    fold3_x_test.extend(T[len3 * 3:len3 * 4])
    fold3_x_test.extend(U[len4 * 3:len4 * 4])

    fold4_x_train.extend(NR[0:len1 * 4])
    fold4_x_train.extend(NR[len1 * 5:])
    fold4_x_train.extend(F[0:len2 * 4])
    fold4_x_train.extend(F[len2 * 5:])
    fold4_x_train.extend(T[0:len3 * 4])
    fold4_x_train.extend(T[len3 * 5:])
    fold4_x_train.extend(U[0:len4 * 4])
    fold4_x_train.extend(U[len4 * 5:])
    fold4_x_test.extend(NR[len1 * 4:len1 * 5])
    fold4_x_test.extend(F[len2 * 4:len2 * 5])
    fold4_x_test.extend(T[len3 * 4:len3 * 5])
    fold4_x_test.extend(U[len4 * 4:len4 * 5])

    # element: root_id of trees
    fold0_test = list(fold0_x_test)
    shuffle(fold0_test)
    fold0_train = list(fold0_x_train)
    shuffle(fold0_train)
    fold1_test = list(fold1_x_test)
    shuffle(fold1_test)
    fold1_train = list(fold1_x_train)
    shuffle(fold1_train)
    fold2_test = list(fold2_x_test)
    shuffle(fold2_test)
    fold2_train = list(fold2_x_train)
    shuffle(fold2_train)
    fold3_test = list(fold3_x_test)
    shuffle(fold3_test)
    fold3_train = list(fold3_x_train)
    shuffle(fold3_train)
    fold4_test = list(fold4_x_test)
    shuffle(fold4_test)
    fold4_train = list(fold4_x_train)
    shuffle(fold4_train)

    return list(fold0_test), list(fold0_train), \
           list(fold1_test), list(fold1_train), \
           list(fold2_test), list(fold2_train), \
           list(fold3_test), list(fold3_train), \
           list(fold4_test), list(fold4_train)

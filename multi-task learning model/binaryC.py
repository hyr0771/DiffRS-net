
import time
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  TensorDataset, DataLoader

import os
import copy
import math

from scipy.special import softmax
import scipy.stats as ssxx
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,recall_score,precision_score,precision_recall_curve,f1_score,auc

import matplotlib.pyplot as plt
import seaborn as sns;
# sns.set_theme(color_codes=True)

random_seed = 12

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

torch.cuda.set_device("cuda:0")
os.environ['CUDA_VISIBLE_DEVICES']='1'

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0001, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >self.patience:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

mrna = pd.read_csv('D:/jupyter notebook/ThreeOmics/data/mRNA.txt', sep='\t')
mir = pd.read_csv('D:/jupyter notebook/ThreeOmics/data/miRNA.txt', sep='\t')
meth = pd.read_csv('D:/jupyter notebook/ThreeOmics/data/Meth.txt', sep='\t')
clincal = pd.read_csv('D:/jupyter notebook/ThreeOmics/data/label.txt', sep='\t')

mrna.index = mrna['gene_name']
del mrna['gene_name']

mir.index = mir['gene_name']
del mir['gene_name']

meth.index = meth['gene_name']
del meth['gene_name']

mrna = mrna.T
mir = mir.T
meth = meth.T

mrna_feaure_num = 500
mir_feaure_num = 200
meth_feaure_num = 500

mrna = mrna.iloc[:, :mrna_feaure_num]
mir = mir.iloc[:, :mir_feaure_num]
meth = meth.iloc[:, :meth_feaure_num]

clincal.index = clincal['id']
del clincal['id']

label = clincal['label']

mrna.insert(0, 'label', label)
mir.insert(0, 'label', label)
meth.insert(0, 'label', label)

Basal_mrna = mrna[mrna['label'].values == 0]
Her2_mrna = mrna[mrna['label'].values == 1]
LumA_mrna = mrna[mrna['label'].values == 2]
LumB_mrna = mrna[mrna['label'].values == 3]

Basal_mir = mir[mir['label'].values == 0]
Her2_mir = mir[mir['label'].values == 1]
LumA_mir = mir[mir['label'].values == 2]
LumB_mir = mir[mir['label'].values == 3]

Basal_meth = meth[meth['label'].values == 0]
Her2_meth = meth[meth['label'].values == 1]
LumA_meth = meth[meth['label'].values == 2]
LumB_meth = meth[meth['label'].values == 3]


def connect(A,B):
    data = pd.concat([A,B],axis=0)
    return data.sample(frac=1.0,random_state=1)

def change_label(data):
    origin_label = data['label'].values
    xx = np.unique(origin_label)
    new_label = []
    for i in origin_label:
        if i < xx.mean():
            new_label.append(0)
        else:
            new_label.append(1)
    data['label'] = np.array(new_label)
    return data


Basal_Her2_mrna = connect(Basal_mrna, Her2_mrna)
Basal_LumA_mrna = connect(Basal_mrna, LumA_mrna)
Basal_LumB_mrna = connect(Basal_mrna, LumB_mrna)

Her2_LumA_mrna = connect(Her2_mrna, LumA_mrna)
Her2_LumB_mrna = connect(Her2_mrna, LumB_mrna)

Lum_AB_mrna = connect(LumA_mrna, LumB_mrna)

Basal_Her2_mir = connect(Basal_mir, Her2_mir)
Basal_LumA_mir = connect(Basal_mir, LumA_mir)
Basal_LumB_mir = connect(Basal_mir, LumB_mir)

Her2_LumA_mir = connect(Her2_mir, LumA_mir)
Her2_LumB_mir = connect(Her2_mir, LumB_mir)

Lum_AB_mir = connect(LumA_mir, LumB_mir)

Basal_Her2_meth = connect(Basal_meth, Her2_meth)
Basal_LumA_meth = connect(Basal_meth, LumA_meth)
Basal_LumB_meth = connect(Basal_meth, LumB_meth)

Her2_LumA_meth = connect(Her2_meth, LumA_meth)
Her2_LumB_meth = connect(Her2_meth, LumB_meth)

Lum_AB_meth = connect(LumA_meth, LumB_meth)



Basal_Her2_mrna = change_label(Basal_Her2_mrna)
Basal_Her2_mir = connect(Basal_mir, Her2_mir)
Basal_Her2_meth = connect(Basal_meth, Her2_meth)


Basal_LumA_mrna = change_label(Basal_LumA_mrna)
Basal_LumA_mir = change_label(Basal_LumA_mir)
Basal_LumA_meth = change_label(Basal_LumA_meth)

Basal_LumB_mrna = change_label(Basal_LumB_mrna)
Basal_LumB_mir = change_label(Basal_LumB_mir)
Basal_LumB_meth = change_label(Basal_LumB_meth)

Her2_LumA_mrna = change_label(Her2_LumA_mrna)
Her2_LumA_mir = change_label(Her2_LumA_mir)
Her2_LumA_meth = change_label(Her2_LumA_meth)

Her2_LumB_mrna = change_label(Her2_LumB_mrna)
Her2_LumB_mir = change_label(Her2_LumB_mir)
Her2_LumB_meth = change_label(Her2_LumB_meth)

Lum_AB_mrna = change_label(Lum_AB_mrna)
Lum_AB_mir = change_label(Lum_AB_mir)
Lum_AB_meth = change_label(Lum_AB_meth)

import torch
import torch.nn as nn


class mtlAttention(nn.Module):
    def __init__(self, In_Nodes1, In_Nodes2, In_Nodes3, Modules):
        super(mtlAttention, self).__init__()
        self.Modules = Modules
        self.sigmoid = nn.Sigmoid()

        self.task1_FC1_x = nn.Linear(In_Nodes1, Modules, bias=False)
        self.task1_FC1_y = nn.Linear(In_Nodes1, Modules, bias=False)

        self.task2_FC1_x = nn.Linear(In_Nodes2, Modules, bias=False)
        self.task2_FC1_y = nn.Linear(In_Nodes2, Modules, bias=False)

        self.task3_FC1_x = nn.Linear(In_Nodes3, Modules, bias=False)
        self.task3_FC1_y = nn.Linear(In_Nodes3, Modules, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.task1_FC2 = nn.Sequential(nn.Linear(Modules * 2, 64), nn.ReLU())
        self.task2_FC2 = nn.Sequential(nn.Linear(Modules * 2, 64), nn.ReLU())
        self.task3_FC2 = nn.Sequential(nn.Linear(Modules * 2, 64), nn.ReLU())

        self.task1_FC3 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.task2_FC3 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.task3_FC3 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())

        self.task1_FC4 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.task2_FC4 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.task3_FC4 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())

        self.task1_FC5 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
        self.task2_FC5 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
        self.task3_FC5 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward_one(self, xg, xm, xl):
        xg_x = self.task1_FC1_x(xg)
        xm_x = self.task2_FC1_x(xm)
        xl_x = self.task3_FC1_x(xl)

        xg_y = self.task1_FC1_y(xg)
        xm_y = self.task2_FC1_y(xm)
        xl_y = self.task3_FC1_y(xl)

        # Concatenate transformed inputs for attention mechanism
        xg = torch.cat([xg_x.reshape(-1, 1, self.Modules), xg_y.reshape(-1, 1, self.Modules)], dim=1)
        xm = torch.cat([xm_x.reshape(-1, 1, self.Modules), xm_y.reshape(-1, 1, self.Modules)], dim=1)
        xl = torch.cat([xl_x.reshape(-1, 1, self.Modules), xl_y.reshape(-1, 1, self.Modules)], dim=1)

        # Normalize for attention mechanism
        norm_g = torch.norm(xg, dim=1, keepdim=True)
        xg = xg.div(norm_g)

        norm_m = torch.norm(xm, dim=1, keepdim=True)
        xm = xm.div(norm_m)

        norm_l = torch.norm(xl, dim=1, keepdim=True)
        xl = xl.div(norm_l)

        # Calculate the attention scores for G-M, G-L, and M-L
        energy_gm = torch.bmm(xg.permute(0, 2, 1), xm)
        energy_gl = torch.bmm(xg.permute(0, 2, 1), xl)
        energy_ml = torch.bmm(xm.permute(0, 2, 1), xl)

        # Apply softmax to get the attention weights
        attention_gm = self.softmax(energy_gm)
        attention_gl = self.softmax(energy_gl)
        attention_ml = self.softmax(energy_ml)

        # Apply the attention weights to the corresponding inputs
        xg_value = torch.bmm(xg, attention_gm)
        xm_value_gm = torch.bmm(xm, attention_gm.permute(0, 2, 1))

        xg_value_gl = torch.bmm(xg, attention_gl)
        xl_value_gl = torch.bmm(xl, attention_gl.permute(0, 2, 1))

        xm_value_ml = torch.bmm(xm, attention_ml)
        xl_value_ml = torch.bmm(xl, attention_ml.permute(0, 2, 1))

        # Combine the values for each input
        xg_combined = xg_value + xg_value_gl
        xm_combined = xm_value_gm + xm_value_ml
        xl_combined = xl_value_gl + xl_value_ml

        xg = xg_combined
        xm = xm_combined
        xl = xl_combined

        # Concatenate attention results
        xg = xg.view(-1, self.Modules * 2)
        xm = xm.view(-1, self.Modules * 2)
        xl = xl.view(-1, self.Modules * 2)

        xg = self.task1_FC2(xg)
        xm = self.task2_FC2(xm)
        xl = self.task3_FC2(xl)

        xg = self.task1_FC3(xg)
        xm = self.task2_FC3(xm)
        xl = self.task3_FC3(xl)

        xg = self.task1_FC4(xg)
        xm = self.task2_FC4(xm)
        xl = self.task3_FC4(xl)

        xg = self.task1_FC5(xg)
        xm = self.task2_FC5(xm)
        xl = self.task3_FC5(xl)

        return xg, xm, xl


def model(omics1, omics2, omics3, learningRate, weightDecay):
    xg_data = omics1.iloc[:, 1:].values
    xm_data = omics2.iloc[:, 1:].values
    xl_data = omics3.iloc[:, 1:].values

    label = omics1['label']
    random_state = random.randint(1, 1000)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    j = 0
    for train_index, test_index in skf.split(xg_data, label):
        Xg_train, Xg_test = xg_data[train_index, :], xg_data[test_index, :]
        Xm_train, Xm_test = xm_data[train_index, :], xm_data[test_index, :]
        Xl_train, Xl_test = xl_data[train_index, :], xl_data[test_index, :]

        yg_train, yg_test = label[train_index], label[test_index]
        j = j + 1
        if j == 1:  # CV1 test
            break
    # set hyperparmater
    earlyStoppingPatience = 100

    learningRate = learningRate


    weightDecay = weightDecay

    num_epochs = 500000

    # change form
    y_train = np.array(yg_train).flatten().astype(int)
    y_test = np.array(yg_test).flatten().astype(int)


    Xg = torch.tensor(Xg_train, dtype=torch.float32).cuda()
    Xm = torch.tensor(Xm_train, dtype=torch.float32).cuda()
    Xl = torch.tensor(Xl_train, dtype=torch.float32).cuda()

    Xg_test = torch.tensor(Xg_test, dtype=torch.float32).cuda()
    Xm_test = torch.tensor(Xm_test, dtype=torch.float32).cuda()
    Xl_test = torch.tensor(Xl_test, dtype=torch.float32).cuda()

    y = torch.tensor(y_train, dtype=torch.float32).cuda()


    ds = TensorDataset(Xg, Xm, Xl, y)
    loader = DataLoader(ds, batch_size=y_train.shape[0], shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    In_Nodes1 = Xg_train.shape[1]
    In_Nodes2 = Xm_train.shape[1]
    In_Nodes3 = Xl_train.shape[1]

    #     print('In_Nodes1', In_Nodes1)
    #     print('In_Nodes2', In_Nodes2)

    net = mtlAttention(In_Nodes1, In_Nodes2, In_Nodes3, 64)
    net = net.to(device)
    early_stopping = EarlyStopping(
        patience=earlyStoppingPatience, verbose=False)
    optimizer = optim.Adam(
        net.parameters(), lr=learningRate, weight_decay=weightDecay)
    loss_fn = nn.BCELoss()

    start = time.time()
    for epoch in (range(num_epochs)):
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        for i, data in enumerate(loader, 0):
            xg, xm, xl, y = data
            output1, output2, output3 = net.forward_one(xg, xm, xl)
            output1 = output1.squeeze()
            output2 = output2.squeeze()
            output3 = output3.squeeze()

            net.train()
            optimizer.zero_grad()
            loss = (loss_fn(output1, y) + loss_fn(output2, y) + loss_fn(output3, y)) / 3
            loss.backward(retain_graph=True)
            optimizer.step()

            running_loss1 += loss_fn(output1, y.view(-1)).item()
            running_loss2 += loss_fn(output2, y.view(-1)).item()
            running_loss3 += loss_fn(output3, y.view(-1)).item()

        early_stopping(running_loss1 + running_loss2 + running_loss3, net)
        if early_stopping.early_stop:
            # #             print("Early stopping")
            #             print("--------------------------------------------------------------------------------------------------")
            break

    # test
    test1, test2, test3 = net.forward_one(
        Xg_test.clone().detach(), Xm_test.clone().detach(), Xl_test.clone().detach())
    test1 = test1.cpu().detach().numpy()
    test2 = test2.cpu().detach().numpy()
    test3 = test3.cpu().detach().numpy()

    final = (test1 + test2 + test3) / 3
    ACC = accuracy_score(list(y_test), np.where(final > 0.5, 1, 0))
    F1 = f1_score(list(y_test), np.where(final > 0.5, 1, 0))
    AUC = roc_auc_score(y_test.reshape(-1), final)

    #     ACC_task1 = accuracy_score(list(y_test), np.where(test1 > 0.5, 1, 0))
    #     ACC_task2 = accuracy_score(list(y_test), np.where(test2 > 0.5, 1, 0))
    #     F1_task1 = f1_score(list(y_test), np.where(test1 > 0.5, 1, 0))
    #     F1_task2 = f1_score(list(y_test), np.where(test2 > 0.5, 1, 0))
    #     AUC_task1 = roc_auc_score(y_test.reshape(-1), test1)
    #     AUC_task2 = roc_auc_score(y_test.reshape(-1), test2)

    return ACC, F1, AUC

ACC = []
F1 = []
AUC = []

for i in range(0, 30):
    acc, f1, auc = model(Basal_Her2_mrna, Basal_Her2_mir, Basal_Her2_meth, 0.0001, 0.001)
    ACC.append(acc)
    F1.append(f1)
    AUC.append(auc)

print("*******Basal_Her2********")
print("*******ACC****************************************")
print("mean ACC:", np.array(ACC).mean())
print('acc list:\n', ACC)

print("*******F1*****************************************")
print("mean F1:", np.array(F1).mean())
print('F1 list:\n', F1)

print("*******AUC****************************************")
print("mean AUC:", np.array(AUC).mean())
print('AUC list:\n', AUC)

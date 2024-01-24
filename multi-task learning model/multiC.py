import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os
import copy
import math

from scipy.special import softmax
import scipy.stats as ss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, recall_score, precision_score, \
    precision_recall_curve, f1_score, auc

import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.model_selection import KFold, StratifiedKFold
# sns.set_theme(color_codes=True)
import warnings
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
warnings.filterwarnings("ignore")

random_seed = 0

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

torch.cuda.set_device("cuda:0")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0001, path='checkpoint.pt', trace_func=print):
        """
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
            if self.counter > self.patience:
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
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class mtlAttention(nn.Module):
    def __init__(self, In_Nodes1, In_Nodes2, In_Nodes3, Modules, dropout_rate):
        super(mtlAttention, self).__init__()
        self.Modules = Modules
        self.dropout_rate = dropout_rate

        # Task 1 layers
        self.task1_FC1_x = nn.Linear(In_Nodes1, Modules, bias=False)
        self.task1_FC1_y = nn.Linear(In_Nodes1, Modules, bias=False)
        self.task1_FC2 = nn.Sequential(nn.Linear(Modules * 2, 32), nn.ReLU())
        self.task1_FC3 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.task1_FC4 = nn.Sequential(nn.Linear(16, 4), nn.Softmax(dim=-1))

        # Task 2 layers
        self.task2_FC1_x = nn.Linear(In_Nodes2, Modules, bias=False)
        self.task2_FC1_y = nn.Linear(In_Nodes2, Modules, bias=False)
        self.task2_FC2 = nn.Sequential(nn.Linear(Modules * 2, 32), nn.ReLU())
        self.task2_FC3 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.task2_FC4 = nn.Sequential(nn.Linear(16, 4), nn.Softmax(dim=-1))

        # Task 3 layers
        self.task3_FC1_x = nn.Linear(In_Nodes3, Modules, bias=False)
        self.task3_FC1_y = nn.Linear(In_Nodes3, Modules, bias=False)
        self.task3_FC2 = nn.Sequential(nn.Linear(Modules * 2, 32), nn.ReLU())
        self.task3_FC3 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.task3_FC4 = nn.Sequential(nn.Linear(16, 4), nn.Softmax(dim=-1))

        # Common layers
        self.softmax = nn.Softmax(dim=-1)
        nn.Dropout(dropout_rate)

    def forward_one(self, xg, xm, xl):
        # Task 1 forward pass
        xg_x = self.task1_FC1_x(xg)
        xg_y = self.task1_FC1_y(xg)

        # Task 2 forward pass
        xm_x = self.task2_FC1_x(xm)
        xm_y = self.task2_FC1_y(xm)

        # Task 3 forward pass
        xl_x = self.task3_FC1_x(xl)
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

        # Sequential layers for each task
        xg = self.task1_FC2(xg)
        xg = self.task1_FC3(xg)
        xg = self.task1_FC4(xg)

        xm = self.task2_FC2(xm)
        xm = self.task2_FC3(xm)
        xm = self.task2_FC4(xm)

        xl = self.task3_FC2(xl)
        xl = self.task3_FC3(xl)
        xl = self.task3_FC4(xl)

        return xg, xm, xl



mrna = pd.read_csv('D:/jupyter notebook/ThreeOmics/data/mRNA.txt',sep='\t')
mir = pd.read_csv('D:/jupyter notebook/ThreeOmics/data/miRNA.txt',sep='\t')
meth = pd.read_csv('D:/jupyter notebook/ThreeOmics/data/Meth.txt',sep='\t')
clincal = pd.read_csv('D:/jupyter notebook/ThreeOmics/data/label.txt',sep='\t')


mrna.index = mrna['gene_name']
del mrna['gene_name']
mir.index = mir['gene_name']
del mir['gene_name']
meth.index = meth['gene_name']
del meth['gene_name']


mrna_feature_num = 500
mir_feature_num = 100
meth_feature_num = 500


mrna = mrna.iloc[:mrna_feature_num, :]
mir = mir.iloc[:mir_feature_num, :]
meth = meth.iloc[:meth_feature_num, :]

mrna = mrna.T
mir = mir.T
meth = meth.T
y = clincal['label'].values

mrna = mrna.values
mir = mir.values
meth = meth.values


mrna = min_max_scaler.fit_transform(mrna)
mir = min_max_scaler.fit_transform(mir)
meth = min_max_scaler.fit_transform(meth)

label = y
n_SKFold = KFold(n_splits=5, shuffle=True,random_state = 913 )

j = 0
for train_index, test_index in n_SKFold.split(mrna):
    Xg_train, Xg_test = mrna[train_index, :], mrna[test_index, :]
    Xm_train, Xm_test = mir[train_index, :], mir[test_index, :]
    Xl_train, Xl_test = meth[train_index, :], meth[test_index, :]
    yg_train, yg_test = label[train_index], label[test_index]
    j = j + 1
    if j == 1:  
        break

Xg_train = Xg_train.astype(float)
Xg_test = Xg_test.astype(float)
Xm_train = Xm_train.astype(float)
Xm_test = Xm_test.astype(float)
Xl_train = Xl_train.astype(float)
Xl_test = Xl_test.astype(float)

train_losses, test_losses = [], []
start = time.time()

earlyStoppingPatience = 300
# learningRate = 5e-6
learningRate = 0.00001
# weightDecay = 0.001
weightDecay = 0.001
num_epochs = 300000
# l1_lambda = 0.001

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

Xg_test = torch.tensor(Xg_test, dtype=torch.float32).cuda()
Xm_test = torch.tensor(Xm_test, dtype=torch.float32).cuda()
Xl_test = torch.tensor(Xl_test, dtype=torch.float32).cuda()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
In_Nodes1 = Xg_train.shape[1]
In_Nodes2 = Xm_train.shape[1]
In_Nodes3 = Xl_train.shape[1]

# mtlAttention(In_Nodes1,In_Nodes2, # of module)
net = mtlAttention(In_Nodes1, In_Nodes2, In_Nodes3, 64, 0.1)
net = net.to(device)
early_stopping = EarlyStopping(patience=earlyStoppingPatience, verbose=False)
optimizer = optim.Adam(net.parameters(), lr=learningRate, weight_decay=weightDecay)
loss_fn = nn.CrossEntropyLoss()

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
        optimizer.zero_grad()  # 将梯度清0
        y = y.to(torch.int64)

        loss = (loss_fn(output1, y) + loss_fn(output2, y) + loss_fn(output3, y)) / 3
        #         l1_regularization = torch.norm(torch.nn.utils.parameters_to_vector(net.parameters()), 1)
        #         loss += l1_lambda * l1_regularization

        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss1 += loss_fn(output1, y.view(-1)).item()
        running_loss2 += loss_fn(output2, y.view(-1)).item()
        running_loss3 += loss_fn(output3, y.view(-1)).item()

    early_stopping(running_loss1 + running_loss2 + running_loss3, net)
    if early_stopping.early_stop:
        print("Early stopping")
        print("--------------------------------------------------------------------------------------------------")
        break

    if (epoch + 1) % 200 == 0 or epoch == 0:
        print(
            'Epoch [{}/{}], Loss: {:.4f}, Cross_entry_loss_task1; {:.4f}, Cross_entry_loss_task2; {:.4f},Cross_entry_loss_task3; {:.4f}'.format(
                epoch + 1, num_epochs,
                running_loss1 + running_loss2 + running_loss3,
                running_loss1,
                running_loss2,
                running_loss3))

### Test
test1, test2, test3 = net.forward_one(Xg_test.clone().detach(), Xm_test.clone().detach(), Xl_test.clone().detach())
test1 = test1.cpu().detach().numpy()
test2 = test2.cpu().detach().numpy()
test3 = test3.cpu().detach().numpy()
Test_all = (test1 + test2 + test3) / 3

pre1 = np.argmax(test1, axis=1)
pre2 = np.argmax(test2, axis=1)
pre3 = np.argmax(test3, axis=1)
pre_all = np.argmax(Test_all, axis=1)


print("预测值:", np.array(pre_all))

print("------------------------------------")
print("准确率:", accuracy_score(y_test, pre_all))
print("F1:", f1_score(y_test, pre_all, average='macro'))

print("time :", time.time() - start)

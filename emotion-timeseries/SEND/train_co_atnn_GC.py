from __future__ import print_function
import torch
from co_attn_GC_model import MovieNet
from utils import MovieDataset, adjust_learning_rate, eval_ccc
from torch.utils.data import DataLoader
import time
import math
import warnings
import pickle
import numpy as np
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")
from scipy.stats.stats import pearsonr
from block import fusions
from clstm import cLSTM, train_model_gista, train_model_adam, cLSTMSparse

args = {}

## Network Arguments
args['F_length'] = 32
args['A_length'] = 88
args['T_length'] = 300
args['dropout_prob'] = 0.5
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['train_flag'] = True
args['model_path'] = 'trained_models/rnned_finetune_traf_5_3.tar'
args['optimizer'] = 'adam'
args['embed_dim'] = 128
args['h_dim'] = 512
args['n_layers'] = 1
args['attn_len'] = 3
num_epochs = 10
batch_size = 1
lr=1e-4


with open('../data_dicts/train.pickle', 'rb') as handle:
    train_raw = pickle.load(handle)
with open('../data_dicts/val.pickle', 'rb') as handle:
    val_raw = pickle.load(handle)
with open('../data_dicts/test.pickle', 'rb') as handle:
    test_raw = pickle.load(handle)

for key in list(test_raw.keys()):
    is_any_nan = []
    seq_len = test_raw[key][0]
    f = test_raw[key][2]
    is_any_nan.append(True in np.isnan(f))
    a = test_raw[key][1]
    is_any_nan.append(True in np.isnan(a))
    t = test_raw[key][3]
    is_any_nan.append(True in np.isnan(t))
    if True in is_any_nan:
        print(key, is_any_nan)

    ## Initialize data loaders
trSet = MovieDataset(train_raw)
valSet = MovieDataset(val_raw)
testSet= MovieDataset(test_raw)
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=8)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=8)
testLoader = DataLoader(testSet,batch_size=batch_size,shuffle=True,num_workers=8)
# Initialize network
net = MovieNet(args)
print(net)

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(pytorch_total_params)
if args['use_cuda']:
    net = net.cuda()

## Initialize optimizer
optimizer = torch.optim.RMSprop(net.parameters(), lr=lr) if args['optimizer']== 'rmsprop' else torch.optim.Adam(net.parameters(),lr=lr, weight_decay=0.9)
crossEnt = torch.nn.BCELoss()
mse = torch.nn.MSELoss(reduction='sum')
## Variables holding train and validation loss values:
train_loss = []
val_loss = []
ccc = []
prev_val_loss = math.inf

for epoch_num in range(num_epochs):
#    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    adjust_learning_rate(optimizer, epoch_num, lr)

## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train()
    # Variables to track training performance:
    avg_tr_loss = 0
    CCC = 0
    for i, data in enumerate(trDataloader):
        st_time = time.time()
        train, labels, F, A, T = data
        if args['use_cuda']:
            train = torch.nn.Parameter(train).cuda()
            F = torch.nn.Parameter(F).cuda()
            A = torch.nn.Parameter(A).cuda()
            T = torch.nn.Parameter(T).cuda()
            labels = torch.nn.Parameter(labels).cuda()
        train.requires_grad_()
        F.requires_grad_()
        A.requires_grad_()
        T.requires_grad_()
        labels.requires_grad_()

        # Forward pass
        emot_score, input_clstm, shared_encoder  = net(train, F, A, T, labels)
        train_model_gista(shared_encoder, input_clstm, lam=6.6, lam_ridge=1e-4, lr=0.001, max_iter=50, check_every=1000, truncation=64)
        # GC_est = shared_encoder.GC().cpu().data.numpy()
        # print('pred: ', emot_score, 'labels: ', labels)
        emot_score = emot_score.squeeze(dim=0)
        # if torch.max(emot_score) != 0:
        #     emot_score = ((emot_score - torch.min(emot_score))/(torch.max(emot_score) - torch.min(emot_score)))*100
        labels = labels.squeeze(dim=0)
        labels = (labels - torch.min(labels))/(torch.max(labels) - torch.min(labels))

        # l = maskedMSE(emot_score, labels)
        l = mse(emot_score, labels)

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward(retain_graph=True)
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        # Track average train loss and average train time:
        batch_time = time.time()-st_time
        avg_tr_loss += l.item()

        # Calculate CCC Value
        CCC += eval_ccc(emot_score, labels, CCC)
        # print(CCC)


    print("Epoch no:",epoch_num+1, "| Avg train loss:",format(avg_tr_loss/len(trSet),'0.4f'), "| Training CCC Value:", format(CCC/len(trSet)) )

    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________



## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    # net.eval()
    avg_val_loss = 0
    CCC = 0

    for i, data in enumerate(valDataloader):
        st_time = time.time()
        val, labels, F, A, T = data
        if args['use_cuda']:
            val = torch.nn.Parameter(val, requires_grad=False).cuda()
            F = torch.nn.Parameter(F, requires_grad=False).cuda()
            A = torch.nn.Parameter(A, requires_grad=False).cuda()
            T = torch.nn.Parameter(T, requires_grad=False).cuda()            
            labels = torch.nn.Parameter(labels, requires_grad=False).cuda()

        # Forward pass
        emot_score, embed, shared_encoder  = net(val, F, A, T, labels)
        emot_score = emot_score.squeeze(dim=1)
        # if torch.max(emot_score) != 0:
        #     emot_score = ((emot_score - torch.min(emot_score)) / (torch.max(emot_score) - torch.min(emot_score))) * 100
        labels = (labels - torch.min(labels))/(torch.max(labels) - torch.min(labels))
        labels = labels.squeeze(dim=0)
        l = mse(emot_score, labels)
        avg_val_loss += l.item()

        # Compute CCC
        CCC += eval_ccc(emot_score, labels, CCC)

    ccc.append(CCC/len(valSet))
        # Print validation loss and update display variables
    print('Validation loss :',format(avg_val_loss/len(valSet),'0.4f'), "| Validation CCC Value:", format(CCC/len(valSet)))
    print("===================================================================================== \n" )


    #__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# ccc_smooth = signal.savgol_filter(l2_norm_bo, 97, 3)
# lr_ave = signal.savgol_filter(l2_norm_lr, 97, 3)

plt.plot(ccc)
plt.show()
torch.save(net.state_dict(), args['model_path'])



## Test:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# net.eval()
avg_test_loss = 0
CCC = 0

for i, data in enumerate(testLoader):
    st_time = time.time()
    test, labels, F, A, T = data
    if args['use_cuda']:
        test = torch.nn.Parameter(test, requires_grad=False).cuda()
        F = torch.nn.Parameter(F, requires_grad=False).cuda()
        A = torch.nn.Parameter(A, requires_grad=False).cuda()
        T = torch.nn.Parameter(T, requires_grad=False).cuda()            
        labels = torch.nn.Parameter(labels, requires_grad=False).cuda()

    # Forward pass
    emot_score, embed, shared_encoder  = net(test, F, A, T, labels)
    emot_score = emot_score.squeeze(dim=1)
    # if torch.max(emot_score) != 0:
    #     emot_score = ((emot_score - torch.min(emot_score)) / (torch.max(emot_score) - torch.min(emot_score))) * 100
    labels = (labels - torch.min(labels))/(torch.max(labels) - torch.min(labels))
    labels = labels.squeeze(dim=0)
    l = mse(emot_score, labels)
    avg_test_loss += l.item()

    # Compute CCC
    CCC += eval_ccc(emot_score, labels, CCC)

    ccc.append(CCC/len(testSet))
    # Print test loss and update display variables
print('Test loss :',format(avg_test_loss/len(testSet),'0.4f'), "| Test CCC Value:", format(CCC/len(testSet)))
print("===================================================================================== \n" )

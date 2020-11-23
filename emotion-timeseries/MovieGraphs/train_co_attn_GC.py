from __future__ import print_function
import torch
from model_co_atnn_GC import MovieNet
from utils_co_attn import MovieGraphDataset, adjust_learning_rate, AverageMeter, accuracy_multihots
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
import torch.nn.functional as F
from clstm import cLSTM, train_model_gista, train_model_adam, cLSTMSparse

args = {}

# MG DIMS
args['Face_length'] = 204
args['Va_length'] = 41
args['trans_length'] = 300
args['scene_length'] = 300
args['desc_length'] = 300
args['sit_length'] = 300
args['out_layer'] = 27
args['dropout_p rob'] = 0.5
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
args['attn_len'] = 15
num_epochs = 10
batch_size = 1
lr = 1e-2

with open('./data_dicts/train.pickle', 'rb') as handle:
    u = pickle._Unpickler(handle)
    u.encoding = 'latin1'
    train_raw = u.load()

with open('./data_dicts/val.pickle', 'rb') as handle:
    v = pickle._Unpickler(handle)
    v.encoding = 'latin1'
    val_raw = v.load()
with open('./data_dicts/test.pickle', 'rb') as handle:
    v = pickle._Unpickler(handle)
    v.encoding = 'latin1'
    test_raw = v.load()

## Initialize data loaders
trSet = MovieGraphDataset(train_raw)
valSet = MovieGraphDataset(val_raw)
testSet = MovieGraphDataset(test_raw)
trDataloader = DataLoader(trSet, batch_size=batch_size, shuffle=True, num_workers=8)
valDataloader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=8)
testDataLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=8)
# Initialize network
net = MovieNet(args)
print(net)
if args['use_cuda']:
    net = net.cuda()

## Initialize optimizer
optimizer = torch.optim.RMSprop(net.parameters(), lr=lr) if args['optimizer']== 'rmsprop' else torch.optim.Adam(net.parameters(),lr=lr, weight_decay=0.9)
crossEnt = torch.nn.BCELoss()
mse = torch.nn.MSELoss(reduction='sum')
train_loss = []
val_loss = []
prev_val_loss = math.inf

for epoch_num in range(num_epochs):
    adjust_learning_rate(optimizer, epoch_num, lr)
    ## Train
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.train()
    epoch_loss = 0
    acc = 0
    for i, data in enumerate(trDataloader):
        st_time = time.time()
        train, labels, F, Va, emb_desc, emb_sit, emb_sce, emb_trans = data
        if args['use_cuda']:
            train = torch.nn.Parameter(train).cuda()
            labels = torch.nn.Parameter(labels).cuda()
            F = torch.nn.Parameter(F).cuda()
            Va = torch.nn.Parameter(Va).cuda()
            emb_desc = torch.nn.Parameter(emb_desc).cuda()
            emb_sit = torch.nn.Parameter(emb_sit).cuda()
            emb_sce = torch.nn.Parameter(emb_sce).cuda()
            emb_trans = torch.nn.Parameter(emb_trans).cuda()
            
        train.requires_grad_()
        labels.requires_grad_()
        F.requires_grad_()
        Va.requires_grad_()
        emb_desc.requires_grad_()
        emb_sit.requires_grad_()
        emb_sce.requires_grad_()
        emb_trans.requires_grad_()

        # Forward pass
        emot_score, input_clstm, shared_encoder = net(train, F, Va, emb_desc, emb_sit, emb_sce, emb_trans, labels)
        train_model_gista(shared_encoder, input_clstm, lam=6.6, lam_ridge=1e-4, lr=0.001, max_iter=10, check_every=1000, truncation=64)
        # GC_est = shared_encoder.GC().cpu().data.numpy()
        
        emot_score = emot_score.squeeze(dim=0)
        labels = labels.squeeze(dim=0)

        log_softmax = torch.nn.LogSoftmax(dim=1)
        log_softmax_output = log_softmax(emot_score)
        loss = - torch.sum(log_softmax_output * labels) / emot_score.shape[0]
        losses.update(loss.item(), train.size(0))
        prec1 = accuracy_multihots(emot_score, labels, topk=(1, 3))
        top1.update(prec1[0], train.size(0))
        acc += prec1[0]
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
    print("Epoch no:",epoch_num+1, "| Avg train loss:", format(epoch_loss/len(trSet),'0.4f'), "| Training Accuracy Value:", format(acc/len(trSet)) )

    ## Validate
    net.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    epoch_loss = 0
    acc = 0
    for i, data in enumerate(valDataloader):
        st_time = time.time()
        val, labels, F, Va, emb_desc, emb_sit, emb_sce, emb_trans = data
        if args['use_cuda']:
            val = torch.nn.Parameter(val, requires_grad=False).cuda()
            labels = torch.nn.Parameter(labels, requires_grad=False).cuda()
            F = torch.nn.Parameter(F, requires_grad=False).cuda()
            Va = torch.nn.Parameter(Va, requires_grad=False).cuda()
            emb_desc = torch.nn.Parameter(emb_desc, requires_grad=False).cuda()
            emb_sit = torch.nn.Parameter(emb_sit, requires_grad=False).cuda()
            emb_sce = torch.nn.Parameter(emb_sce, requires_grad=False).cuda()
            emb_trans = torch.nn.Parameter(emb_trans, requires_grad=False).cuda()
            
        # Forward pass
        emot_score, input_clstm, shared_encoder  = net(val, F, Va, emb_desc, emb_sit, emb_sce, emb_trans, labels)
        emot_score = emot_score.squeeze(dim=0)
        labels = labels.squeeze(dim=0)

        log_softmax = torch.nn.LogSoftmax(dim=1)
        log_softmax_output = log_softmax(emot_score)
        loss = - torch.sum(log_softmax_output * labels) / emot_score.shape[0]
        losses.update(loss.item(), val.size(0))
        prec1 = accuracy_multihots(emot_score, labels, topk=(1, 3))
        top1.update(prec1[0], val.size(0))
        epoch_loss += loss
        acc += prec1[0]

    print("Epoch no:",epoch_num+1, "| Avg validation loss:", format(epoch_loss/len(valSet),'0.4f'), "| Validation Accuracy Value:", format(acc/len(valSet)))
    print('---------------------------------------')

testDataLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=8)
## Test
net.eval()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
avg_test_loss = 0
acc = 0
for i, data in enumerate(testDataLoader):
    st_time = time.time()
    test, labels, F, Va, emb_desc, emb_sit, emb_sce, emb_trans = data
    if args['use_cuda']:
        test = torch.nn.Parameter(test, requires_grad=False).cuda()
        labels = torch.nn.Parameter(labels, requires_grad=False).cuda()
        F = torch.nn.Parameter(F, requires_grad=False).cuda()
        Va = torch.nn.Parameter(Va, requires_grad=False).cuda()
        emb_desc = torch.nn.Parameter(emb_desc, requires_grad=False).cuda()
        emb_sit = torch.nn.Parameter(emb_sit, requires_grad=False).cuda()
        emb_sce = torch.nn.Parameter(emb_sce, requires_grad=False).cuda()
        emb_trans = torch.nn.Parameter(emb_trans, requires_grad=False).cuda()
        
    # Forward pass
    emot_score, input_clstm, shared_encoder  = net(test, F, Va, emb_desc, emb_sit, emb_sce, emb_trans, labels)
    emot_score = emot_score.squeeze(dim=0)
    labels = labels.squeeze(dim=0)

    log_softmax = torch.nn.LogSoftmax(dim=1)
    log_softmax_output = log_softmax(emot_score)
    loss = - torch.sum(log_softmax_output * labels) / emot_score.shape[0]
    losses.update(loss.item(), val.size(0))
    prec1 = accuracy_multihots(emot_score, labels, topk=(1, 3))
    top1.update(prec1[0], val.size(0))
    avg_test_loss += loss
    acc += prec1[0]

print('Test loss :',format(avg_test_loss/len(testSet),'0.4f'), "| Test Accuracy Value:", format(acc/len(testSet)))
print("===================================================================================== \n" )

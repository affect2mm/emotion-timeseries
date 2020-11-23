from __future__ import print_function
import torch
from model_co_attn_GC import MovieNet
from utils_co_attn_GC import adjust_learning_rate, MediaEvalDataset, prsn, save_ckp, load_ckp
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
from scipy.stats.mstats import pearsonr
from clstm import cLSTM, train_model_gista, train_model_adam, cLSTMSparse

args = {}

best_model_path="./best_model"
checkpoint_path="./checkpoints"
valid_loss_min=np.Inf
## Network Arguments

args['Face_length'] = 204
args['Va_length'] = 317
args['audio_length'] = 1583
args['scene_length'] = 4096
args['out_layer'] = 2

args['dropout_prob'] = 0.5
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['train_flag'] = True
args['model_path'] = 'trained_models/media_eval_model.tar'
args['optimizer'] = 'adam'
args['embed_dim'] = 512
args['h_dim'] = 512
args['n_layers'] = 1
args['attn_len'] = 6
num_epochs = 20
batch_size = 1
lr=1e-4
GC_est=None

with open('./data_dicts/train.pickle', 'rb') as handle:
    u = pickle._Unpickler(handle)
    u.encoding = 'latin1'
    train_raw = u.load()


with open('./data_dicts/val.pickle', 'rb') as handle:
    v = pickle._Unpickler(handle)
    v.encoding = 'latin1'
    # train_raw = pickle.load(handle)
    val_raw = v.load()

with open('./data_dicts/test.pickle', 'rb') as handle:
    v = pickle._Unpickler(handle)
    v.encoding = 'latin1'
    # train_raw = pickle.load(handle)
    test_raw = v.load()

trSet = MediaEvalDataset(train_raw)
valSet = MediaEvalDataset(val_raw)
testSet = MediaEvalDataset(test_raw)
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=8)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=8)
testDataloader = DataLoader(testSet,batch_size=batch_size,shuffle=True,num_workers=8)

# Initialize network
net = MovieNet(args)
if args['use_cuda']:
    net = net.cuda()

## Initialize optimizer
optimizer = torch.optim.RMSprop(net.parameters(), lr=lr) if args['optimizer']== 'rmsprop' else torch.optim.Adam(net.parameters(),lr=lr, weight_decay=0.9)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
crossEnt = torch.nn.BCELoss()
mse = torch.nn.MSELoss(reduction='sum')


for epoch_num in range(num_epochs):
#    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    adjust_learning_rate(optimizer, epoch_num, lr)

## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train()
    # Variables to track training performance:
    avg_tr_loss = 0
    for i, data in enumerate(trDataloader):
        st_time = time.time()
        train, labels, F, Va, scene, audio = data
        labels1 = labels[0]
        labels2= labels[1]
        if args['use_cuda']:
            train = torch.nn.Parameter(train).cuda()
            labels1 = torch.nn.Parameter(labels1).cuda()
            labels2 = torch.nn.Parameter(labels2).cuda()
            F = torch.nn.Parameter(F).cuda()
            Va = torch.nn.Parameter(Va).cuda()
            scene = torch.nn.Parameter(scene).cuda()
            audio = torch.nn.Parameter(audio).cuda()

        train.requires_grad_()
        labels1.requires_grad_()
        labels2.requires_grad_()
        F.requires_grad_()
        Va.requires_grad_()
        scene.requires_grad_()
        audio.requires_grad_()
        

        # Forward pass
        emot_score, input_clstm, shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6  = net(train, F, Va, scene, audio, labels)
        train_model_gista(shared_encoder, input_clstm, lam=0.5, lam_ridge=1e-4, lr=0.001, max_iter=1, check_every=1000, truncation=64)
        GC_est = shared_encoder.GC().cpu().data.numpy()
        
        emot_score = emot_score.squeeze(dim=0)
        labels1 = labels1.T
        labels2 = labels2.T

        emot_score = (2*(emot_score - torch.min(emot_score))/(torch.max(emot_score) - torch.min(emot_score))) -1
        l = mse(emot_score[:,0].unsqueeze(dim=1), labels1) + mse(emot_score[:,1].unsqueeze(dim=1), labels2)

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        # scheduler.step()
        avg_tr_loss += l.item()
        

    # print(GC_est)
    print("Epoch no:",epoch_num+1, "| Avg train loss:",format(avg_tr_loss/len(trSet),'0.4f') )

    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________



## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.eval()
    valmse = 0
    aromse = 0
    valpcc = 0
    aropcc = 0
    for i, data in enumerate(testDataloader):
        st_time = time.time()
        val, labels,  F, Va, scene, audio  = data
        labels1 = labels[0]
        labels2 = labels[1]
        if args['use_cuda']:
            val = torch.nn.Parameter(val).cuda()
            labels1 = torch.nn.Parameter(labels1).cuda()
            labels2 = torch.nn.Parameter(labels2).cuda()
            F = torch.nn.Parameter(F).cuda()
            Va = torch.nn.Parameter(Va).cuda()
            scene = torch.nn.Parameter(scene).cuda()
            audio = torch.nn.Parameter(audio).cuda()

        # Forward pass
        emot_score, input_clstm, shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6  = net(val, F, Va, scene, audio, labels)


        emot_score = emot_score.squeeze(dim=0)
        labels1 = labels1.T
        labels2 = labels2.T

        emot_score = (2*(emot_score - torch.min(emot_score))/(torch.max(emot_score) - torch.min(emot_score))) -1
        # labels1 = (2*(labels1 - torch.min(labels1)))/(torch.max(labels1) - torch.min(labels1)) -1
        # labels2 = (2*(labels2 - torch.min(labels2)))/(torch.max(labels2) - torch.min(labels2)) -1
        valmse += mse(emot_score[:, 0].unsqueeze(dim=1), labels1)/labels1.shape[0]
        aromse += mse(emot_score[:, 1].unsqueeze(dim=1), labels2)/labels2.shape[0]

        valpcc += pearsonr(emot_score[:, 0].unsqueeze(dim=1).cpu().detach().numpy(), labels1.cpu().detach().numpy())[0]
        aropcc += pearsonr(emot_score[:, 1].unsqueeze(dim=1).cpu().detach().numpy(), labels2.cpu().detach().numpy())[0]
    epoch_valmse = valmse/len(valSet)
    epoch_aromse = aromse/len(valSet)
    epoch_valpcc = valpcc / len(valSet)
    epoch_aropcc = aropcc / len(valSet)
    val_loss=epoch_valmse
    # val_loss=(epoch_aromse+epoch_valmse)/2
    print("Epoch Valence MSE:", epoch_valmse.item() , "Epoch Arousal MSE:", epoch_aromse.item(),"\nEpoch Valence PCC:", epoch_valpcc.item() ,
          "Epoch Arousal PCC:", epoch_aropcc.item(),"\n","==========================")

    checkpoint = {
        'epoch': epoch_num + 1,
        'valid_loss_min_valence': epoch_valmse,
        'valid_loss_min_arousal': epoch_aromse,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
        
    # save checkpoint
    save_ckp(checkpoint, False, checkpoint_path+"/train_co_attn_GC_current_checkpoint.pt", best_model_path+"/train_co_attn_GC_best_model.pt")
    
    ## TODO: save the model if validation loss has decreased
    if val_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,val_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path+"/train_co_attn_GC_current_checkpoint.pt", best_model_path+"/train_co_attn_GC_best_model.pt")
        valid_loss_min = val_loss
    

net=MovieNet(args)
net, optimizer, start_epoch, valid_loss_min_valence, valid_loss_min_arousal = load_ckp(best_model_path+"/train_co_attn_GC_best_model.pt", net, optimizer)



net.eval()
testmse = 0
aromse = 0
testpcc = 0
aropcc = 0
for i, data in enumerate(testDataloader):
    st_time = time.time()
    test, labels, F, Va, scene, audio = data
    labels1 = labels[0]
    labels2 = labels[1]
    if args['use_cuda']:
        test = torch.nn.Parameter(test).cuda()
        labels1 = torch.nn.Parameter(labels1).cuda()
        labels2 = torch.nn.Parameter(labels2).cuda()
        F = torch.nn.Parameter(F).cuda()
        Va = torch.nn.Parameter(Va).cuda()
        scene = torch.nn.Parameter(scene).cuda()
        audio = torch.nn.Parameter(audio).cuda()


    # Forward pass
    emot_score, input_clstm, shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6  = net(test, F, Va, scene, audio, labels)
    print(att_1, att_2, att_3, att_4, att_5, att_6)

    emot_score = emot_score.squeeze(dim=0)
    labels1 = labels1.T
    labels2 = labels2.T

    emot_score = (2*(emot_score - torch.min(emot_score))/(torch.max(emot_score) - torch.min(emot_score))) -1
    print(emot_score)
    # labels1 = (2*(labels1 - torch.min(labels1)))/(torch.max(labels1) - torch.min(labels1)) -1
    # labels2 = (2*(labels2 - torch.min(labels2)))/(torch.max(labels2) - torch.min(labels2)) -1
    testmse += mse(emot_score[:, 0].unsqueeze(dim=1), labels1)/labels1.shape[0]
    aromse += mse(emot_score[:, 1].unsqueeze(dim=1), labels2)/labels2.shape[0]

    testpcc += pearsonr(emot_score[:, 0].unsqueeze(dim=1).cpu().detach().numpy(), labels1.cpu().detach().numpy())[0]
    aropcc += pearsonr(emot_score[:, 1].unsqueeze(dim=1).cpu().detach().numpy(), labels2.cpu().detach().numpy())[0]

test_testmse = testmse/len(testSet)
test_aromse = aromse/len(testSet)
test_testpcc = testpcc / len(testSet)
test_aropcc = aropcc / len(testSet)

print("Test Valence MSE:", test_testmse.item() , "Test Arousal MSE:", test_aromse.item(),"\Test Valence PCC:", test_testpcc.item() ,
        "Test Arousal PCC:", test_aropcc.item(),"\n","==========================")

import csv
with open("GC_LIRIS.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(GC_est)

from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
from keras.utils import to_categorical
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
from block import fusions
import shutil

#___________________________________________________________________________________________________________________________

# New Dataloader for MovieClass

class MediaEvalDataset(Dataset):


    def __init__(self, data):
        self.data = data
        self.movie_idx = list(self.data.keys()) # ['tt03045', 'tt0840830' ...] etc
        self.num_samples = len(list(self.data.keys())) # 51 movies ideally
        self.new_data = {}
        for movie in self.movie_idx:
            num_clips = list(self.data[movie].keys())
            self.new_data[movie] = []
            self.new_data[movie].append(len(num_clips))
            self.new_data[movie].append( np.array([self.data[movie][clip]['face'] for clip in num_clips]) )
            self.new_data[movie].append( np.array([self.data[movie][clip]['va'] for clip in num_clips]) )
            self.new_data[movie].append( np.array([self.data[movie][clip]['scene'] for clip in num_clips]) )
            self.new_data[movie].append( np.array([self.data[movie][clip]['audio'] for clip in num_clips]) )
            self.new_data[movie].append( np.array([self.data[movie][clip]['valence'] for clip in num_clips]) )
            self.new_data[movie].append( np.array([self.data[movie][clip]['arousal'] for clip in num_clips]) )



    def __len__(self):
        return self.num_samples



    def __getitem__(self, idx):
        idx = self.movie_idx[idx]
        F = self.new_data[idx][1]
        Va = self.new_data[idx][2]
        scene = self.new_data[idx][3]
        audio = self.new_data[idx][4]
        y = [self.new_data[idx][5][:10], self.new_data[idx][6][:10]]

        combined = np.hstack([F, Va, scene, audio])
        return combined[:10], y, F[:10], Va[:10], scene[:10], audio[:10]


#________________________________________________________________________________________________________________________________________


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch == 100:
        lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def prsn(emot_score, labels):
    """Computes concordance correlation coefficient."""

    labels_mu = torch.mean(labels)
    emot_mu = torch.mean(emot_score)
    vx = emot_score - emot_mu
    vy = labels - labels_mu
    # prsn_corr = torch.mean(vx * vy)
    prsn_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    return prsn_corr

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min_valence = checkpoint['valid_loss_min_valence']
    valid_loss_min_arousal = checkpoint['valid_loss_min_arousal']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min_valence.item(), valid_loss_min_arousal.item()

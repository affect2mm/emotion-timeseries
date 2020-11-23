from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
from block import fusions

#__________________________________________________________________________________________________________________________

# New Dataloader for MovieClass



### Dataset class for the NGSIM datase
class MovieDataset(Dataset):


    def __init__(self, data):
        self.data = data
        self.movie_idx = list(self.data.keys())
        self.num_samples = len(list(self.data.keys()))
        pass


    def __len__(self):
        return self.num_samples



    def __getitem__(self, idx):
        idx = self.movie_idx[idx]
        F = self.data[idx][2]
        A = self.data[idx][1]
        T = self.data[idx][3]
        y = self.data[idx][4]
        
        combined = np.hstack([F, A, T])
        #shape: timestamps*sum of dim_modality

        # Convert to torch tensors
        F = torch.Tensor(F)
        A = torch.Tensor(A)
        T = torch.Tensor(T)
        # y = torch.Tensor(y)

        # Instantiate fusion classes
        FA = fusions.Block([F.shape[1], A.shape[1]], T.shape[1])
        FAT = fusions.Block([T.shape[1], T.shape[1]], F.shape[1]+A.shape[1]+T.shape[1])

        # compute fusions
        temp_output_FA = FA([F, A])
        final_FAT = FAT([temp_output_FA, T])

        # return final_FAT.cpu().detach().numpy(), y
        return combined, y, F, A, T


#________________________________________________________________________________________________________________________________________


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch == 100:
        lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def eval_ccc(emot_score, labels, CCC):
    """Computes concordance correlation coefficient."""

    emot_mu = torch.mean(emot_score)
    emot_sigma = torch.std(emot_score)
    labels_mu = torch.mean(labels)
    labels_sigma = torch.std(labels)
    vx = emot_score - emot_mu
    vy = labels - labels_mu
    prsn_corr = torch.mean(vx * vy)
    CCC = (2 * prsn_corr) / (emot_sigma ** 2 + labels_sigma ** 2 + (emot_mu - labels_mu) ** 2)
    return CCC
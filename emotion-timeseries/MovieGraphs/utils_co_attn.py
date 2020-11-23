from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
from keras.utils import to_categorical
import numpy as np
import torch
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
from block import fusions
twenty_six_labels = {'Affection': ['loving', 'friendly'], 'Anger': ['anger', 'furious', 'resentful', 'outraged', 'vengeful'],
'Annoyance': ['annoy', 'frustrated', 'irritated', 'agitated', 'bitter', 'insensitive', 'exasperated', 'displeased'],
'Anticipation':	['optimistic', 'hopeful', 'imaginative', 'eager'],
'Aversion':	['disgusted', 'horrified', 'hateful'],
'Confidence':	['confident', 'proud', 'stubborn', 'defiant', 'independent', 'convincing'],
'Disapproval':	['disapproving', 'hostile', 'unfriendly', 'mean', 'disrespectful', 'mocking', 'condescending', 'cunning', 'manipulative', 'nasty', 'deceitful', 'conceited', 'sleazy', 'greedy', 'rebellious', 'petty'],
'Disconnection':	['indifferent', 'bored', 'distracted', 'distant', 'uninterested', 'self-centered', 'lonely', 'cynical', 'restrained', 'unimpressed', 'dismissive']                                        ,
'Disquietment':	['worried', 'nervous', 'tense', 'anxious','afraid', 'alarmed', 'suspicious', 'uncomfortable', 'hesitant', 'reluctant', 'insecure', 'stressed', 'unsatisfied', 'solemn', 'submissive']                 ,
'Doubt/Conf':	['confused', 'skeptical', 'indecisive']                                                                                                                                         ,
'Embarrassment':	['embarrassed', 'ashamed', 'humiliated']                                                                                                                                     ,
'Engagement':	['curious', 'serious', 'intrigued', 'persistent', 'interested', 'attentive', 'fascinated']                                                                                                ,
'Esteem':	['respectful', 'grateful']                                                                                                                                                           ,
'Excitement':	['excited', 'enthusiastic', 'energetic', 'playful', 'impatient', 'panicky', 'impulsive', 'hasty']                                                                                             ,
'Fatigue':	['tired', 'sleepy', 'drowsy']                                                                                                                                                            ,
'Fear':	['scared', 'fearful', 'timid', 'terrified']                                                                                                                                                     ,
'Happiness':	['cheerful', 'delighted', 'happy', 'amused', 'laughing', 'thrilled', 'smiling', 'pleased', 'overwhelmed', 'ecstatic', 'exuberant']                                                                     ,
'Pain':	['pain']                                                                                                                                                                                    ,
'Peace':	['content', 'relieved', 'relaxed', 'calm', 'quiet', 'satisfied', 'reserved', 'carefree']                                                                                                ,
'Pleasure':	['funny', 'attracted', 'aroused', 'hedonistic', 'pleasant', 'flattered', 'entertaining', 'mesmerized']    ,
'Sadness':	['sad', 'melancholy', 'upset', 'disappointed', 'discouraged', 'grumpy', 'crying', 'regretful', 'grief-stricken', 'depressed', 'heartbroken', 'remorseful', 'hopeless', 'pensive', 'miserable']                         ,
'Sensitivity':	['apologetic', 'nostalgic']                                                                                                                                                               ,
'Suffering':	['offended', 'hurt', 'insulted', 'ignorant', 'disturbed', 'abusive', 'offensive'],
'Surprise':	['surprise', 'surprised', 'shocked', 'amazed', 'startled', 'astonished', 'speechless', 'disbelieving', 'incredulous'],
'Sympathy':	['kind', 'compassionate', 'supportive', 'sympathetic', 'encouraging', 'thoughtful', 'understanding', 'generous', 'concerned', 'dependable', 'caring', 'forgiving', 'reassuring', 'gentle'],
'Yearning':	['jealous', 'determined', 'aggressive', 'desperate', 'focused', 'dedicated', 'diligent'] ,
'None': ['None']}

class MovieGraphDataset(Dataset):
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
            self.new_data[movie].append( np.array([self.data[movie][clip]['embed_description'] for clip in num_clips]) )
            self.new_data[movie].append( np.array([self.data[movie][clip]['embed_situation'] for clip in num_clips]) )
            self.new_data[movie].append( np.array([self.data[movie][clip]['embed_scene'] for clip in num_clips]) )
            self.new_data[movie].append( np.array([self.data[movie][clip]['embed_transcript'] for clip in num_clips]) )
            self.new_data[movie].append( np.array([self.data[movie][clip]['emotions'] for clip in num_clips]) )
            for f in range(len(num_clips)):
                emot_labels = self.new_data[movie][7][f]
                if len(emot_labels) == 0:
                    emot_labels.append('None')
                labels = list(twenty_six_labels.keys())
                integer_mapping = {x: i for i, x in enumerate(labels)}
                vec = [integer_mapping[word] for word in labels]
                encoded = to_categorical(vec)
                emot_encoding = []
                for emot in emot_labels:
                    emot_encoding.append(list(encoded[integer_mapping[emot]]))
                emot_labels = [sum(x) for x in zip(*emot_encoding)]
                self.new_data[movie][7][f] = emot_labels
            self.new_data[movie][7] = np.array(list(self.new_data[movie][7]))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = self.movie_idx[idx]
        F = self.new_data[idx][1]
        Va = self.new_data[idx][2]
        emb_desc = self.new_data[idx][3]
        emb_sit = self.new_data[idx][4]
        emb_sce = self.new_data[idx][5]
        emb_trans = self.new_data[idx][6]
        y = self.new_data[idx][7]

        combined = np.hstack([F, Va, emb_desc, emb_sit, emb_sce, emb_trans])

        F = torch.Tensor(F)
        Va = torch.Tensor(Va)
        emb_desc = torch.Tensor(emb_desc)
        emb_sit = torch.Tensor(emb_sit)
        emb_sce = torch.Tensor(emb_sce)
        emb_trans = torch.Tensor(emb_trans)
        # Instantiate fusion classes
        fusion1 = fusions.Block([F.shape[1], Va.shape[1]], emb_desc.shape[1])
        fusion2 = fusions.Block([emb_desc.shape[1], emb_desc.shape[1]], F.shape[1] + Va.shape[1] + emb_desc.shape[1])

        fusion3 = fusions.Block([emb_sit.shape[1], emb_sce.shape[1]], emb_trans.shape[1])
        fusion4 = fusions.Block([emb_trans.shape[1], emb_trans.shape[1]], emb_sit.shape[1] + emb_sce.shape[1] + emb_trans.shape[1])

        # compute fusions
        temp_output_fusion1 = fusion1([F, Va])
        first_three= fusion2([temp_output_fusion1, emb_desc])
        temp_output_fusion2 = fusion3([emb_sit, emb_sce])
        second_three = fusion4([temp_output_fusion2, emb_trans])

        fusion5 = fusions.Block([first_three.shape[1], second_three.shape[1]], first_three.shape[1]+second_three.shape[1])
        final_fused = fusion5([first_three, second_three])
        return combined, y, F, Va, emb_desc, emb_sit, emb_sce, emb_trans


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch == 100:
        lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def accuracy_multihots(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # maxk = max(topk)
        # batch_size = target.size(0)
        batch_size = 1

        _, pred = output.topk(1, 1, True, True)
        target_value = torch.gather(target, 1, pred)
        # target_inds_one = (target != 0).nonzero()

        correct_k = (target_value > 0).float().sum(0, keepdim=False).sum(0, keepdim=True)
        correct_k /= target.shape[0]
        res = (correct_k.mul_(100.0))
        return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

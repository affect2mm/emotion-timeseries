from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from clstm import cLSTM, train_model_gista, train_model_adam, cLSTMSparse

def pad_shift(x, shift, padv=0.0):
    """Shift 3D tensor forwards in time with padding."""
    if shift > 0:
        padding = torch.ones(x.size(0), shift, x.size(2)).to(x.device) * padv
        return torch.cat((padding, x[:, :-shift, :]), dim=1)
    elif shift < 0:
        padding = torch.ones(x.size(0), -shift, x.size(2)).to(x.device) * padv
        return torch.cat((x[:, -shift:, :], padding), dim=1)
    else:
        return x

def convolve(x, attn):
    """Convolve 3D tensor (x) with local attention weights (attn)."""
    stacked = torch.stack([pad_shift(x, i) for
                           i in range(attn.shape[2])], dim=-1)
    return torch.sum(attn.unsqueeze(2) * stacked, dim=-1)

class MovieNet(nn.Module):
    """Multimodal encoder-decoder LSTM model.
    modalities -- list of names of each input modality
    dims -- list of dimensions for input modalities
    embed_dim -- dimensions of embedding for feature-level fusion
    h_dim -- dimensions of LSTM hidden state
    n_layers -- number of LSTM layers
    attn_len -- length of local attention window
    """

    def __init__(self, args, device='cuda:0'):
        # device=torch.device('cuda:0')
        super(MovieNet, self).__init__()
        self.face_len = args['F_length']
        self.audio_len = args['A_length']
        self.text_len = args['T_length']
        self.total_mod_len = self.face_len + self.audio_len +self.text_len
        # self.total_mod_len = 3*(self.text_len)
        self.embed_dim = args['embed_dim']
        self.h_dim = args['h_dim']
        self.n_layers = args['n_layers']
        self.attn_len = args['attn_len']


         # linear for word-guided visual attention
        self.text_linear = nn.Linear(self.text_len, self.h_dim, bias=True)
        self.face_linear = nn.Linear(self.face_len, self.h_dim, bias=True)
        self.audio_linear = nn.Linear(self.audio_len, self.h_dim, bias=True)
        self.att_linear = nn.Linear(self.h_dim * 2, 1)
        self.unimodal_text = nn.Linear(self.h_dim,1)
        self.unimodal_face = nn.Linear(self.h_dim,1)
        self.unimodal_audio = nn.Linear(self.h_dim,1)
        
        # Create raw-to-embed FC+Dropout layer for each modality
        # self.embed = nn.Sequential(nn.Dropout(0.5),
                                    #   nn.Linear( self.total_mod_len, self.embed_dim),
                                    #   nn.ReLU())
        # self.add_module('embed_{}'.format(m), self.embed[m])
        # Layer that computes attention from embeddings
        # self.attn = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
        #                           nn.ReLU(),
        #                           nn.Linear(self.embed_dim, self.attn_len),
        #                           nn.Softmax(dim=1))
        # Encoder computes hidden states from embeddings for each modality
        # self.encoder = nn.LSTM(self.embed_dim, self.h_dim, self.n_layers, batch_first=True)
        self.shared_encoder = cLSTM(3,self.h_dim, batch_first=True).cuda(device=device) 
        # self.encoder = cLSTMSparse(self.embed_dim,self.h_dim, batch_first=True)
        self.enc_h0 = nn.Parameter(torch.zeros(self.n_layers, 1, self.h_dim))
        self.enc_c0 = nn.Parameter(torch.zeros(self.n_layers, 1, self.h_dim))
        # Decodes targets and LSTM hidden states
        self.decoder = nn.LSTM(1 + self.attn_len, self.h_dim, self.n_layers, batch_first=True)
        # self.decoder = cLSTMSparse(1 + self.embed_dim, self.sparsity, self.h_dim, batch_first=True)
        self.dec_h0 = nn.Parameter(torch.zeros(self.n_layers, 1, self.h_dim))
        self.dec_c0 = nn.Parameter(torch.zeros(self.n_layers, 1, self.h_dim))
        # Final MLP output network
        self.out = nn.Sequential(nn.Linear(self.h_dim, 4),
                                 nn.ReLU(),
                                 nn.Linear(4, 1))
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, x, face_features, audio_features, text_features, target=None, tgt_init=0.0):

        # Get batch dim
        x = x.float()
        batch_size, seq_len = x.shape[0], x.shape[1]
        # Set initial hidden and cell states for encoder
        h0 = self.enc_h0.repeat(1, batch_size, 1)
        c0 = self.enc_c0.repeat(1, batch_size, 1)

        text_features_rep = self.text_linear(text_features)
        face_features_rep = self.face_linear(face_features)
        audio_features_rep = self.audio_linear(audio_features)

        concat_features = torch.cat([text_features_rep, face_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_1 = self.att_linear(concat_features).squeeze(-1)
        att_1 = torch.softmax(att_1, dim=-1)

        concat_features = torch.cat([face_features_rep, audio_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_2 = self.att_linear(concat_features).squeeze(-1)
        att_2 = torch.softmax(att_2, dim=-1)

        concat_features = torch.cat([text_features_rep, audio_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_3 = self.att_linear(concat_features).squeeze(-1)
        att_3 = torch.softmax(att_3, dim=-1)
        
        unimodal_text_input= torch.tanh(text_features_rep)
        unimodal_text_input = self.unimodal_text(unimodal_text_input).squeeze(-1)
        unimodal_text_input = torch.softmax(unimodal_text_input, dim=-1)

        unimodal_face_input= torch.tanh(face_features_rep)
        unimodal_face_input = self.unimodal_face(unimodal_face_input).squeeze(-1)
        unimodal_face_input = torch.softmax(unimodal_face_input, dim=-1)

        unimodal_audio_input= torch.tanh(audio_features_rep)
        unimodal_audio_input = self.unimodal_audio(unimodal_audio_input).squeeze(-1)
        unimodal_audio_input = torch.softmax(unimodal_audio_input, dim=-1)

        enc_input_unimodal_cat=torch.cat([unimodal_text_input, unimodal_face_input, unimodal_audio_input], dim=-1)
        enc_input_unimodal_cat = enc_input_unimodal_cat.reshape(batch_size, seq_len, self.attn_len)
        attn=torch.cat([att_1, att_2, att_3], dim=-1)
        attn = attn.reshape(batch_size, seq_len, self.attn_len)


        enc_out, _ = self.shared_encoder(enc_input_unimodal_cat)
        context = convolve(enc_out, attn)
        
        # Set initial hidden and cell states for decoder
        h0 = self.dec_h0.repeat(1, batch_size, 1)
        c0 = self.dec_c0.repeat(1, batch_size, 1)
        if target is not None:
            target = target.float()
            # Concatenate targets from previous timesteps to context
            dec_in = torch.cat([pad_shift(target, 1, tgt_init), context], 2)
            dec_out, _ = self.decoder(dec_in, (h0, c0))
            # Undo the packing
            # dec_out, _ = pad_packed_sequen,ce(dec_out, batch_first=True)
            # Flatten temporal dimension
            dec_out = dec_out.reshape(-1, self.h_dim)
            # Compute predictions from decoder outputs
            predicted = self.out(dec_out).view(batch_size, seq_len, 1)
        else:
            # Use earlier predictions to predict next time-steps
            predicted = []
            p = torch.ones(batch_size, 1).to(self.device) * tgt_init
            h, c = h0, c0
            for t in range(seq_len):
                # Concatenate prediction from previous timestep to context
                i = torch.cat([p, context[:, t, :]], dim=1).unsqueeze(1)
                # Get next decoder LSTM state and output
                o, (h, c) = self.decoder(i, (h, c))
                # Computer prediction from output state
                p = self.out(o.view(-1, self.h_dim))
                predicted.append(p.unsqueeze(1))
            predicted = torch.cat(predicted, dim=1)
        # Mask target entries that exceed sequence lengths
        # predicted = predicted * mask.float()
        return predicted, enc_input_unimodal_cat, self.shared_encoder

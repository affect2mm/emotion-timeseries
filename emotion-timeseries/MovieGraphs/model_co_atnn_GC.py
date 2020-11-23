from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import warnings
warnings.filterwarnings('ignore')
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

    def __init__(self, args, device=torch.device('cuda:0')):
        super(MovieNet, self).__init__()
        self.face_len = args['Face_length']
        self.audio_len = args['Va_length']
        self.text_len = args['trans_length']
        self.scene_length = args['scene_length']
        self.desc_length = args['desc_length']
        self.sit_length = args['sit_length']
        self.out_layer = args['out_layer']
        self.total_mod_len = self.face_len + self.audio_len +self.text_len + self.desc_length + self.sit_length + self.scene_length
        # self.total_mod_len = 3*(self.text_len)
        self.embed_dim = args['embed_dim']
        self.h_dim = args['h_dim']
        self.n_layers = args['n_layers']
        self.attn_len = args['attn_len']

        self.text_linear = nn.Linear(self.text_len, self.h_dim, bias=True)
        self.face_linear = nn.Linear(self.face_len, self.h_dim, bias=True)
        self.audio_linear = nn.Linear(self.audio_len, self.h_dim, bias=True)
        self.scene_linear = nn.Linear(self.scene_length, self.h_dim, bias=True)
        self.desc_linear = nn.Linear(self.desc_length, self.h_dim, bias=True)
        self.sit_linear = nn.Linear(self.sit_length, self.h_dim, bias=True)
        self.att_linear = nn.Linear(self.h_dim * 2, 1)

        self.unimodal_text = nn.Linear(self.h_dim,1)
        self.unimodal_face = nn.Linear(self.h_dim,1)
        self.unimodal_audio = nn.Linear(self.h_dim,1)
        self.unimodal_scene = nn.Linear(self.h_dim,1)
        self.unimodal_desc = nn.Linear(self.h_dim,1)
        self.unimodal_sit = nn.Linear(self.h_dim,1)
        
        

        # Create raw-to-embed FC+Dropout layer for each modality
        # self.embed = nn.Sequential(nn.Dropout(0.1),
        #                               nn.Linear( self.total_mod_len, self.embed_dim),
        #                               nn.ReLU())
        # # self.add_module('embed_{}'.format(m), self.embed[m])
        # # Layer that computes attention from embeddings
        # self.attn = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
        #                           nn.ReLU(),
        #                           nn.Linear(self.embed_dim, self.attn_len),
        #                           nn.Softmax(dim=1))
        # Encoder computes hidden states from embeddings for each modality
        self.shared_encoder = cLSTM(6,self.h_dim, batch_first=True).cuda(device=device) 
        # self.encoder = nn.LSTM(self.embed_dim, self.h_dim, self.n_layers, batch_first=True)
        self.enc_h0 = nn.Parameter(torch.zeros(self.n_layers, 1, self.h_dim))
        self.enc_c0 = nn.Parameter(torch.zeros(self.n_layers, 1, self.h_dim))
        # Decodes targets and LSTM hidden states
        # self.decoder = nn.LSTM(1 + self.h_dim, self.h_dim, self.n_layers, batch_first=True)
        self.decoder = nn.LSTM(6, self.h_dim, self.n_layers, batch_first=True)
        self.dec_h0 = nn.Parameter(torch.zeros(self.n_layers, 1, self.h_dim))
        self.dec_c0 = nn.Parameter(torch.zeros(self.n_layers, 1, self.h_dim))
        # Final MLP output network
        self.out = nn.Sequential(nn.Dropout(0.1),nn.Linear(self.h_dim, 512),
                                 nn.ReLU(),
                                #  nn.Linear(self.h_dim, 512),
                                #  nn.ReLU(),
                                #  nn.Linear(self.h_dim, 512),
                                #  nn.ReLU(),
                                 nn.Linear(512, self.out_layer))
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, x, face_features, audio_features, desc_features, sit_features , scene_features, text_features , target=None, tgt_init=0.0):
        # Get batch dim

        # print(face_features.shape, audio_features.shape, desc_features.shape, sit_features.shape, scene_features.shape, text_features.shape)
        # exit()
        # #  F, Va, emb_desc, emb_sit, emb_sce, emb_trans
        x = x.float()
        face_features=face_features.float()
        audio_features=audio_features.float()
        text_features=text_features.float()
        batch_size, seq_len = x.shape[0], x.shape[1]
        # Set initial hidden and cell states for encoder
        h0 = self.enc_h0.repeat(1, batch_size, 1)
        c0 = self.enc_c0.repeat(1, batch_size, 1)
        # Convert raw features into equal-dimensional embeddings
        # embed = self.embed(x)
        # Compute attention weights


        text_features_rep = self.text_linear(text_features)        
        face_features_rep = self.face_linear(face_features)
        audio_features_rep = self.audio_linear(audio_features)
        scene_features_rep = self.scene_linear(scene_features)
        desc_features_rep = self.desc_linear(desc_features)
        sit_features_rep = self.sit_linear(sit_features)


        # attention multiplexers

        # pairwise text
        concat_features = torch.cat([text_features_rep, audio_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_1 = self.att_linear(concat_features).squeeze(-1)
        att_1 = torch.softmax(att_1, dim=-1)

        concat_features = torch.cat([text_features_rep, face_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_2 = self.att_linear(concat_features).squeeze(-1)
        att_2 = torch.softmax(att_2, dim=-1)

        concat_features = torch.cat([text_features_rep, scene_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_3 = self.att_linear(concat_features).squeeze(-1)
        att_3 = torch.softmax(att_3, dim=-1)

        concat_features = torch.cat([text_features_rep, desc_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_4 = self.att_linear(concat_features).squeeze(-1)
        att_4 = torch.softmax(att_4, dim=-1)

        concat_features = torch.cat([text_features_rep, sit_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_5 = self.att_linear(concat_features).squeeze(-1)
        att_5 = torch.softmax(att_5, dim=-1)
                
        concat_features = torch.cat([face_features_rep, audio_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_6 = self.att_linear(concat_features).squeeze(-1)
        att_6 = torch.softmax(att_6, dim=-1)
        
        concat_features = torch.cat([face_features_rep, scene_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_7 = self.att_linear(concat_features).squeeze(-1)
        att_7 = torch.softmax(att_7, dim=-1)
        
        concat_features = torch.cat([face_features_rep, desc_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_8 = self.att_linear(concat_features).squeeze(-1)
        att_8 = torch.softmax(att_8, dim=-1)

        concat_features = torch.cat([face_features_rep, sit_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_9 = self.att_linear(concat_features).squeeze(-1)
        att_9 = torch.softmax(att_9, dim=-1)
        
        # pairwise audio
        concat_features = torch.cat([audio_features_rep, scene_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_10 = self.att_linear(concat_features).squeeze(-1)
        att_10 = torch.softmax(att_10, dim=-1)        
        
        concat_features = torch.cat([audio_features_rep, desc_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_11 = self.att_linear(concat_features).squeeze(-1)
        att_11 = torch.softmax(att_11, dim=-1)        
        
        concat_features = torch.cat([audio_features_rep, sit_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_12 = self.att_linear(concat_features).squeeze(-1)
        att_12 = torch.softmax(att_12, dim=-1)        
        
        # pairwise scene
        concat_features = torch.cat([scene_features_rep, desc_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_13 = self.att_linear(concat_features).squeeze(-1)
        att_13 = torch.softmax(att_13, dim=-1)        
        
        concat_features = torch.cat([scene_features_rep, sit_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_14 = self.att_linear(concat_features).squeeze(-1)
        att_14 = torch.softmax(att_14, dim=-1)        
        
        # pairwise desc 
        concat_features = torch.cat([desc_features_rep, sit_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)
        att_15 = self.att_linear(concat_features).squeeze(-1)
        att_15 = torch.softmax(att_15, dim=-1)        

        unimodal_text_input= torch.tanh(text_features_rep)
        unimodal_text_input = self.unimodal_text(unimodal_text_input).squeeze(-1)
        unimodal_text_input = torch.softmax(unimodal_text_input, dim=-1)

        unimodal_face_input= torch.tanh(face_features_rep)
        unimodal_face_input = self.unimodal_face(unimodal_face_input).squeeze(-1)
        unimodal_face_input = torch.softmax(unimodal_face_input, dim=-1)

        unimodal_audio_input= torch.tanh(audio_features_rep)
        unimodal_audio_input = self.unimodal_audio(unimodal_audio_input).squeeze(-1)
        unimodal_audio_input = torch.softmax(unimodal_audio_input, dim=-1)

        unimodal_scene_input= torch.tanh(scene_features_rep)
        unimodal_scene_input = self.unimodal_text(unimodal_scene_input).squeeze(-1)
        unimodal_scene_input = torch.softmax(unimodal_scene_input, dim=-1)

        unimodal_desc_input= torch.tanh(desc_features_rep)
        unimodal_desc_input = self.unimodal_face(unimodal_desc_input).squeeze(-1)
        unimodal_desc_input = torch.softmax(unimodal_desc_input, dim=-1)

        unimodal_sit_input= torch.tanh(sit_features_rep)
        unimodal_sit_input = self.unimodal_audio(unimodal_sit_input).squeeze(-1)
        unimodal_sit_input = torch.softmax(unimodal_sit_input, dim=-1)

        enc_input_unimodal_cat=torch.cat([unimodal_text_input, unimodal_face_input, unimodal_audio_input, unimodal_scene_input, unimodal_desc_input, unimodal_sit_input], dim=-1)
        enc_input_unimodal_cat = enc_input_unimodal_cat.reshape(batch_size, seq_len, 6)
        attn=torch.cat([att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10, att_11, att_12, att_13, att_14, att_15], dim=-1)
        attn = attn.reshape(batch_size, seq_len, self.attn_len)


        enc_out, _ = self.shared_encoder(enc_input_unimodal_cat)
        context = convolve(enc_out, attn)
        

        # Set initial hidden and cell states for decoder
        h0 = self.dec_h0.repeat(1, batch_size, 1)
        c0 = self.dec_c0.repeat(1, batch_size, 1)
        if target is not None:
            target = target.float()
            # Concatenate targets from previous timesteps to context
            dec_in = context
            # Pack the input to mask padded entries
            # dec_in = pack_padded_sequence(dec_in, lengths, batch_first=True)
            # Forward propagate decoder LSTM
            dec_out, _ = self.decoder(dec_in, (h0, c0))
            # Undo the packing
            # dec_out, _ = pad_packed_sequence(dec_out, batch_first=True)
            # Flatten temporal dimension
            dec_out = dec_out.reshape(-1, self.h_dim)
            # Compute predictions from decoder outputs
            predicted = self.out(dec_out).view(batch_size, seq_len, self.out_layer)
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

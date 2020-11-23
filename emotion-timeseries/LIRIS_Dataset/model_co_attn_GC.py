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
    """Multimodal encoder-decoder LSTM model.
    modalities -- list of names of each input modality
    dims -- list of dimensions for input modalities
    embed_dim -- dimensions of embedding for feature-level fusion
    h_dim -- dimensions of LSTM hidden state
    n_layers -- number of LSTM layers
    attn_len -- length of local attention window
    """

    def __init__(self, args, device=torch.device('cuda:0')):
        super(MovieNet, self).__init__()
        self.face_len = args['Face_length']
        self.va_len = args['Va_length']
        self.audio_len = args['audio_length']
        self.scene_length = args['scene_length']
        self.out_layer = args['out_layer']
        self.total_mod_len = self.face_len + self.va_len +self.audio_len + self.scene_length
        # self.total_mod_len = 3*(self.text_len)
        self.embed_dim = args['embed_dim']
        self.h_dim = args['h_dim']
        self.n_layers = args['n_layers']
        self.attn_len = args['attn_len']
        self.dropout=0.5

        self.face_linear = nn.Linear(self.face_len, self.h_dim, bias=True)
        self.va_linear = nn.Linear(self.va_len, self.h_dim, bias=True)
        self.audio_linear = nn.Linear(self.audio_len, self.h_dim, bias=True)
        self.scene_linear = nn.Linear(self.scene_length, self.h_dim, bias=True)
        self.att_linear1 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)
        self.att_linear2 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)
        self.att_linear3 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)
        self.att_linear4 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)
        self.att_linear5 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)
        self.att_linear6 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)
        
        self.unimodal_face = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU())
        self.unimodal_va = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU())
        self.unimodal_audio = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU())
        self.unimodal_scene = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim,1)
        

        self.shared_encoder = cLSTM(4,self.h_dim, batch_first=True).cuda(device=device) 
        self.enc_h0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))
        self.enc_c0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))
        # Decodes targets and LSTM hidden states
        # self.decoder = nn.LSTM(1 + self.h_dim, self.h_dim, self.n_layers, batch_first=True)
        self.decoder = nn.LSTM(6, self.h_dim, self.n_layers, batch_first=True)
        self.dec_h0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))
        self.dec_c0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))
        # Final MLP output network
        self.out = nn.Sequential(nn.Linear(self.h_dim, 2),#128
                                #  nn.LeakyReLU(),
                                #  nn.Linear(512, 8),
                                 nn.LeakyReLU(),
                                 nn.Linear(2, self.out_layer))
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, x, face_features, va_features, scene_features, audio_features, target=None, tgt_init=0.0):
        # Get batch dim
        x = x.float()
        
        face_features=face_features.float()
        va_features=va_features.float()
        audio_features=audio_features.float()
        scene_features=scene_features.float()

        batch_size, seq_len = x.shape[0], x.shape[1]
        # Set initial hidden and cell states for encoder
        h0 = self.enc_h0.repeat(1, batch_size, 1)
        c0 = self.enc_c0.repeat(1, batch_size, 1)
        # Convert raw features into equal-dimensional embeddings
        # embed = self.embed(x)

        face_features_rep = self.face_linear(face_features)
        va_features_rep = self.va_linear(va_features)
        audio_features_rep = self.audio_linear(audio_features)
        scene_features_rep = self.scene_linear(scene_features)

        concat_features = torch.cat([face_features_rep, va_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_1 = self.att_linear1(concat_features).squeeze(-1)
        att_1 = torch.softmax(att_1, dim=-1)

        concat_features = torch.cat([face_features_rep, audio_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_2 = self.att_linear2(concat_features).squeeze(-1)
        att_2 = torch.softmax(att_2, dim=-1)

        concat_features = torch.cat([face_features_rep, scene_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_3 = self.att_linear3(concat_features).squeeze(-1)
        att_3 = torch.softmax(att_3, dim=-1)

        concat_features = torch.cat([va_features_rep, audio_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_4 = self.att_linear4(concat_features).squeeze(-1)
        att_4 = torch.softmax(att_4, dim=-1)

        concat_features = torch.cat([va_features_rep, scene_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_5 = self.att_linear5(concat_features).squeeze(-1)
        att_5 = torch.softmax(att_5, dim=-1)

        concat_features = torch.cat([audio_features_rep, scene_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_6 = self.att_linear6(concat_features).squeeze(-1)
        att_6 = torch.softmax(att_6, dim=-1)


        unimodal_va_input= va_features_rep
        unimodal_va_input = self.unimodal_va(unimodal_va_input).squeeze(-1)
        unimodal_va_input = torch.softmax(unimodal_va_input, dim=-1)

        unimodal_face_input= face_features_rep
        unimodal_face_input = self.unimodal_face(unimodal_face_input).squeeze(-1)
        unimodal_face_input = torch.softmax(unimodal_face_input, dim=-1)

        unimodal_audio_input= audio_features_rep
        unimodal_audio_input = self.unimodal_audio(unimodal_audio_input).squeeze(-1)
        unimodal_audio_input = torch.softmax(unimodal_audio_input, dim=-1)

        unimodal_scene_input= scene_features_rep
        unimodal_scene_input = self.unimodal_scene(unimodal_scene_input).squeeze(-1)
        unimodal_scene_input = torch.softmax(unimodal_scene_input, dim=-1)
        # print(unimodal_face_input.shape, unimodal_va_input.shape, unimodal_audio_input.shape, unimodal_scene_input.shape)

        enc_input_unimodal_cat=torch.cat([unimodal_face_input, unimodal_va_input, unimodal_audio_input, unimodal_scene_input], dim=-1)
        # print(enc_input_unimodal_cat.shape, batch_size, seq_len)
        enc_input_unimodal_cat = enc_input_unimodal_cat.reshape(batch_size, seq_len, 4)
        attn=torch.cat([att_1, att_2, att_3, att_4, att_5, att_6], dim=-1)
        attn = attn.reshape(batch_size, seq_len, self.attn_len)

       
        enc_out, _ = self.shared_encoder(enc_input_unimodal_cat)
        # Undo the packing
        # enc_out, _ = pad_packed_sequence(enc_out, batch_first=True)
        # Convolve output with attention weights
        # i.e. out[t] = a[t,0]*in[t] + ... + a[t,win_len-1]*in[t-(win_len-1)]
        context = convolve(enc_out, attn)

        # Set initial hidden and cell states for decoder
        h0 = self.dec_h0.repeat(1, batch_size, 1)
        c0 = self.dec_c0.repeat(1, batch_size, 1)
        if target is not None:
            # print(target[0].shape)
            # exit()
            target_0 = target[0].float().reshape(batch_size, seq_len, 1)
            target_0 = torch.nn.Parameter(target_0).cuda()
            target_1 = target[1].float().reshape(batch_size, seq_len, 1)
            target_1 = torch.nn.Parameter(target_1).cuda()
            # print(pad_shift(target, 1, tgt_init), context.shape)
            # Concatenate targets from previous timesteps to context
            dec_in = torch.cat([pad_shift(target_0, 1, tgt_init),pad_shift(target_1, 1, tgt_init), context], 2)
            dec_out, _ = self.decoder(dec_in, (h0, c0))
            # Undo the packing
            dec_out = dec_out.reshape(-1, self.h_dim)


           
            # dec_in = context           
            # dec_out, _ = self.decoder(dec_in, (h0, c0))            
            # dec_out = dec_out.reshape(-1, self.h_dim)
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
        return predicted, enc_input_unimodal_cat, self.shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6

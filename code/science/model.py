from __future__ import division

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import relu

class FeatureTransformer(nn.Module):
    '''
    take an n x d_in matrix and transform it into a n x d_out matrix
    where the n x d_in matrix is the n examples each with d_in dimensions
    '''

    def __init__(self, d_in, d_out):
        super(FeatureTransformer, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        # takes in (*, d_in) tensors and outputs (*, d_out) tensors
        self.linear = nn.Linear(d_in, d_out)
        # self.relu = nn.ReLU()

    def forward(self, input):
        return relu(self.linear(input))


class ChainEncoder(nn.Module):
    '''
    encodes N chains at the same time
    assumes that each of the chains are of the same length
    '''

    def __init__(self, v_feature_lengths, e_feature_lengths, out_length, pooling):
        super(ChainEncoder, self).__init__()
        self.out_length = feature_enc_length = out_length
        num_layers = 1
        self.rnn_type = 'LSTM'
        self.pooling = pooling
        self.v_feature_lengths = v_feature_lengths
        self.e_feature_lengths = e_feature_lengths

        self.v_feature_encoders = nn.ModuleList()
        self.e_feature_encoders = nn.ModuleList()
        for d_in in self.v_feature_lengths:
            self.v_feature_encoders.append(
                FeatureTransformer(d_in, feature_enc_length))
        for d_in in self.e_feature_lengths:
            self.e_feature_encoders.append(
                FeatureTransformer(d_in, feature_enc_length))

        # RNN famlity layer: input (seq_len, batch_size, d_in), output (seq_len, batch_size, d_out * D) where D=2 for bidirectional, D=1 otherwise
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=feature_enc_length,
                              hidden_size=out_length, num_layers=num_layers)
        elif self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size=feature_enc_length,
                                hidden_size=out_length, num_layers=num_layers)

    def forward(self, input):
        '''
        input is a list of v_features, and e_features
        v_features is a list of num_vertices tuples
        each tuple is an N x d_in Variable, in which N is the batch size, and d_in is the feature length
        e_features is structured similarly
        '''
        v_features, e_features = input
        # v_features.shape == (num_vertices, batch_size, variable feature_len)
        # e_features.shape == (num_edges, batch_size, variable feature_len)

        v_encodes = []
        for i in range(len(v_features)):  # 4 vertices 
            v_enc = None
            for j in range(len(v_features[i])):  # feature in each vertex
                curr_encoder = self.v_feature_encoders[j]
                if v_enc is None:
                    v_enc = curr_encoder(v_features[i][j])
                else:
                    v_enc += curr_encoder(v_features[i][j])
            v_enc = v_enc / len(v_features[i])  # each feature encode is of shape (batch_size, out_length)
            v_encodes.append(v_enc)

        e_encodes = []
        for i in range(len(e_features)):  # 3 edges
            e_enc = None
            for j in range(len(e_features[i])):
                curr_encoder = self.e_feature_encoders[j]
                if e_enc is None:
                    e_enc = curr_encoder(e_features[i][j])
                else:
                    e_enc += curr_encoder(e_features[i][j])
            e_enc = e_enc / len(e_features[i])
            e_encodes.append(e_enc)

        combined_encs = [0] * (len(v_encodes)+len(e_encodes))
        # interleave vertices and edges
        combined_encs[::2] = v_encodes
        combined_encs[1::2] = e_encodes
        combined_encs = torch.stack(combined_encs, dim=0).detach().clone()
        # combined_encs has shape (#V+#E) x batch_size x out_length

        if self.rnn_type == 'RNN':
            output, hidden = self.rnn(combined_encs)
        elif self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.lstm(combined_encs)
        if self.pooling == 'last':
            return output[-1]
        else:
            return torch.mean(output, dim=0)


class Predictor(nn.Module):
    '''
    takes two feature vectors and produces a prediction
    '''

    def __init__(self, feature_len):
        super(Predictor, self).__init__()
        self.linear = nn.Linear(feature_len, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)  # will use NLLLoss in learn.py

    def forward(self, vec1, vec2):
        # combined = torch.cat((vec1, vec2), dim=1)
        a = self.linear(vec1)
        b = self.linear(vec2)
        combined = torch.cat((a, b), dim=1)
        return self.logsoftmax(combined)

class JointModel(nn.Module):
    '''
    Combine ChainEncoder and Predictor together
    '''

    def __init__(self,  v_feature_lengths, e_feature_lengths, out_length, pooling):
        super(JointModel, self).__init__()
        self.encoder = ChainEncoder(v_feature_lengths, e_feature_lengths, out_length, pooling)
        self.predictor = Predictor(out_length)

    def forward(self, input1, input2):
        vec1 = self.encoder(input1)
        vec2 = self.encoder(input2)
        return self.predictor(vec1, vec2)

"""
resnet for time series data, pytorch version
Shenda Hong, Nov 2019
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
    
class Network(nn.Module):
    """
    Input:
        X: (n_samples, n_length, n_channel)
        Y: (n_samples)
        
    Output:
        out: (n_samples, n_classes), logits
        
    Pararmetes:
        n_classes: number of classes in classification task
        d_model: alias of n_channel
        d_emb: expand 76 to a higher dim
        nhead: number of head in transformer
        dim_feedforward: dimension of hidden
        dropout: dropout rate
    """

    def __init__(self, n_classes, d_model, d_emb, nhead, dim_feedforward, dropout, verbose=False, **kwargs):
        super(Network, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.d_emb = d_emb
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.n_classes = n_classes
        self.verbose = verbose
        
        self.expand_layer = nn.Conv1d(in_channels=self.d_model, 
                                      out_channels=self.d_emb, 
                                      kernel_size=2)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_emb, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.dense = nn.Linear(self.d_emb, self.n_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = x
        if self.verbose:
            print('input (n_samples, n_length, n_channel)', out.shape)
            
        out = out.permute(0, 2, 1)
        if self.verbose:
            print('transpose (n_samples, n_channel, n_length)', out.shape)
        out = self.expand_layer(out)
        if self.verbose:
            print('expand_layer', out.shape)
            
        out = out.permute(2, 0, 1)
        if self.verbose:
            print('transpose (n_length, n_samples, n_channel)', out.shape)
        out = self.transformer_encoder(out)
        if self.verbose:
            print('transformer_encoder', out.shape)

        out = out.mean(0)
        if self.verbose:
            print('global pooling', out.shape)

        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        
        return out    
    
    def say_name(self):
        """
        not finished
        """
        return "{}".format('transformer')    
    
    
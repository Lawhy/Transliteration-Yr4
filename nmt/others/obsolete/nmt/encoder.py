# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NMTEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, encoder_layers=1, dropout=0.1, encoder_type='biGRU'):
        """
        Args:
            num_embeddings (int): number of embeddings is the size of source vocabulary
            embedding_size (int): size of the embedding vectors
            rnn_hidden_size (int): size of the RNN hidden state vectors 
        """
        super(NMTEncoder, self).__init__()
    
        # nn.Embedding will do the auto-padding and the dimension is decreased, of course not one-hot!
        self.source_embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=0)
        self.num_layers = encoder_layers
        self.num_directions = 2
        self.hidden_size = rnn_hidden_size
        
        # no need for drop-out if the layer number is 1
        if self.num_layers == 1:
            dropout = 0
            
        self.dropout = nn.Dropout(dropout)
        
        # right now only encoder_layer = 1 works fine... as such, drop_out needs to be 0
        if encoder_type == 'biGRU':
            self.birnn = nn.GRU(embedding_size, rnn_hidden_size, num_layers=encoder_layers, 
                                dropout=dropout, bidirectional=True, batch_first=True)
        elif encoder_type == 'biLSTM':
            self.birnn = nn.LSTM(embedding_size, rnn_hidden_size, num_layers=encoder_layers, 
                                 dropout=dropout, bidirectional=True, batch_first=True)
        else:
            raise ValueError('Unsupported encoder type!')
    
    def forward(self, src_batch, src_lengths):
        """The forward pass of the model
        
        Args:
            x_source (torch.Tensor): the input data tensor.
                x_source.shape is (batch, seq_size)
            x_lengths (torch.Tensor): a vector of lengths for each item in the batch
        Returns:
            a tuple: x_unpacked (torch.Tensor), x_birnn_h_n (torch.Tensor)
                x_unpacked.shape = (batch, seq_size, rnn_hidden_size * 2)
                x_birnn_h_n.shape = (batch, rnn_hidden_size * 2)
        """
        # (batch_size, longest-sequence-size-in-this-batch) -- requires sorting
        src_embedded = self.dropout(self.source_embedding(src_batch))
        # create PackedSequence; x_packed.data.shape=(number_items, embedding_size)
        # 这里的 number_items 就是有多少个非零的embedding, 详见notebooks/encoder_understanding
        src_packed = pack_padded_sequence(src_embedded, src_lengths.detach().cpu().numpy(), 
                                          batch_first=True)
        
        # x_birnn_h_n.shape = (num_rnn (if bidirectional then 2), batch_size, feature_size) 
        # x_birnn_out contains the last layer outputs, x_birnn_h_n contains all the h_n in each layer
        birnn_out, birnn_h_n  = self.birnn(src_packed)
        
        # take the last layer hidden states
        birnn_h_n = birnn_h_n.view(self.num_layers, self.num_directions, birnn_h_n.size(1), self.hidden_size)[-1]
        # permute to (batch_size, num_rnn, feature_size)
        birnn_h_n = birnn_h_n.permute(1, 0, 2)
        
        # flatten features; reshape to (batch_size, num_rnn * feature_size)
        birnn_h_n = birnn_h_n.contiguous().view(birnn_h_n.size(0), -1)
        
        birnn_out_unpacked, _ = pad_packed_sequence(birnn_out, batch_first=True)
        
        # birnn_out_unpacked: [batch_size, src_length, encoder_hidden_size * 2]
        # birnn_h_n: [batch_size, encoder_hidden_size * 2]
        return birnn_out_unpacked, birnn_h_n


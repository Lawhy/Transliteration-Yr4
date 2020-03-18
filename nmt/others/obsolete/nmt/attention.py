# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

# ### Attention In Details

# Parameters:
# 1. Encoder outputs: $h_1, h_2, ..., h_n$, where n is the source length. ($keys$ or $values$)
# 2. Previous decoder hidden state: $\bar{h}_{t-1}$. ($query$)
#
# Computation (for decoder state at time t):
# 1. $\textbf{score}_t = \tanh(W[query * n, keys])$; (Linear mapping, query repeated n times)
# 2. $\textbf{a}_t = \textbf{softmax}(score)$; (gives probability distribution of which key to look at)
# 3. $\textbf{c}_t = \sum_{k=1}^n a_{t, k} h_k$; (context vector, the weighted sum of encoder outputs)
# 4. $\tilde{h}_t = f([h_t, c_t])$
#

class Attention(nn.Module):
    
    def __init__(self, encoder_hidden_size, decoder_hiddeb_size):
        super().__init__()
        
        self.hidden_map = nn.Linear((encoder_hidden_size * 2) + decoder_hidden_size, decoder_hidden_size)
        self.v = nn.Parameter(torch.rand(decoder_hidden_size))
        
    def forward(self, query_vector, encoder_outputs):
        
        # query_vector: [batch_size, decoder_hidden_size], the decoder hidden state from the last time step
        # encoder_outputs: [batch size, src sent len, encoder_hidden_size * 2]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # repeat encoder hidden state src_len times
        # recall: torch.unsqueeze insert extra dimension
        # query_vector: [batch_size, src_len, decoder_hidden_size]
        query_vector = query_vector.unsqueeze(1).repeat(1, src_len, 1)
        
        # concat: [batch_size, src_len, (encoder_hidden_size * 2) + decoder_hidden_size] 
        # score: [batch size, src_len, decoder_hidden_size]
        score = torch.tanh(self.hidden_map(torch.cat((query_vector, encoder_outputs), dim = 2))) 
          
        # score = [batch size, decoder_hidden_size, src_len]
        score = score.permute(0, 2, 1)
        
        # v = [batch_size, 1, decoder_hidden_size]  
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        
        # torch.bmm(v, energy) [batch_size, 1, decoder_hidden_size] @ [batch_size, decoder_hidden_size, src_len]
        # (1, decoder_hidden_size) * (decoder_hidden_size, src_len) -> (1, src_len)
        # 相当于把所有的 hidden information 变成句子长度的信息, 然后 apply softmax
        # attention= [batch_size, src_len]
        attention = torch.bmm(v, score).squeeze(1)
        
        return F.softmax(attention, dim=1)


# ------------------------------------------------------------------------------------
# #### Below are attention code from the book "Natural Language Processing with Pytorch"

def verbose_attention(encoder_state_vectors, query_vector):
    """A descriptive version of the neural attention mechanism 
    
    Args:
        encoder_state_vectors (torch.Tensor): 3dim tensor from bi-GRU in encoder
        query_vector (torch.Tensor): hidden state in decoder GRU
    Returns:
        
    """
    batch_size, num_vectors, vector_size = encoder_state_vectors.size()
    vector_scores = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size), 
                              dim=2)
    vector_probabilities = F.softmax(vector_scores, dim=1)
    weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, num_vectors, 1)
    context_vectors = torch.sum(weighted_vectors, dim=1)
    return context_vectors, vector_probabilities, vector_scores

def terse_attention(encoder_state_vectors, query_vector):
    """A shorter and more optimized version of the neural attention mechanism
    
    Args:
        encoder_state_vectors (torch.Tensor): 3dim tensor from bi-GRU in encoder
        query_vector (torch.Tensor): hidden state
    """
    vector_scores = torch.matmul(encoder_state_vectors, query_vector.unsqueeze(dim=2)).squeeze()
    vector_probabilities = F.softmax(vector_scores, dim=-1)
    context_vectors = torch.matmul(encoder_state_vectors.transpose(-2, -1), 
                                   vector_probabilities.unsqueeze(dim=2)).squeeze()
    return context_vectors, vector_probabilities

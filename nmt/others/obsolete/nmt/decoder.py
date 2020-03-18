import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from nmt.attention import verbose_attention, terse_attention

class NMTDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, bos_index, initial_state_size,
                 decoder_layers=1, sampling_temperature=3, dropout=0.1):
        """
        Args:
            num_embeddings (int): number of embeddings is also the number of 
                unique words in target vocabulary 
            embedding_size (int): the embedding vector size
            rnn_hidden_size (int): size of the hidden rnn state
            bos_index(int): begin-of-sequence index
            sample_probability (float): the schedule sampling parameter
                probabilty of using model's predictions at each decoder step
        """
        super(NMTDecoder, self).__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self.target_embedding = nn.Embedding(num_embeddings=num_embeddings, 
                                             embedding_dim=embedding_size, 
                                             padding_idx=0)
        self.gru_cell = nn.GRUCell(embedding_size + rnn_hidden_size, 
                                   rnn_hidden_size)
        if decoder_layers > 1:
            # the last layer is specialised for attention
            self.rnn = nn.GRU(embedding_size + rnn_hidden_size, rnn_hidden_size, num_layers=decoder_layers - 1, 
                                 dropout=dropout, batch_first=True)
            
        # map the last encoder state to a new decoder state (encoder_hidden_size * 2, decoder_hidden_size)    
        self.hidden_map = nn.Linear(initial_state_size, rnn_hidden_size)
        
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_embeddings)
        self.bos_index = bos_index
        self.sampling_temperature = sampling_temperature
        self.dropout = nn.Dropout(dropout)  
        
        self.num_layers = decoder_layers 
        
    def _init_indices(self, batch_size):
        """ return the BEGIN-OF-SEQUENCE index vector """
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_index
    
    def _init_context_vectors(self, batch_size):
        """ return a zeros vector for initializing the context """
        return torch.zeros(batch_size, self._rnn_hidden_size)
    
    def _init_decoder(self, initial_hidden_state, target_sequence=None, sample_probability=0.0):
        
        if target_sequence is None:
            sample_probability = 1.0
        else:
            # We are making an assumption there: The batch is on first
            # The input is (Batch, Seq)
            # We want to iterate over sequence so we permute it to (S, B)
            target_sequence = target_sequence.permute(1, 0)
            output_sequence_size = target_sequence.size(0)
            
        # use the provided encoder hidden state as the initial hidden state
        # with a tanh activation function
        h_0 = torch.tanh(self.hidden_map(initial_hidden_state))
        
        return h_0, target_sequence, output_sequence_size, sample_probability
    
    
    def is_using_sample(self, sample_probability):
        return np.random.random() < sample_probability
    
    
    def pred_y_t_index(self, prediction_vector):
        score_for_y_t_index = self.classifier(self.dropout(prediction_vector))
        p_y_t_index = F.softmax(score_for_y_t_index * self.sampling_temperature, dim=1)
        # _, y_t_index = torch.max(p_y_t_index, 1)
        y_t_index = torch.multinomial(p_y_t_index, 1).squeeze()
        return y_t_index
    
    
    def forward_step(self, encoder_state, y_t_index, h_t, context_vectors):

        # Step 1: Embed word and concat with previous context
        y_input_vector = self.target_embedding(y_t_index)
        rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)

        # Step 2: Make a GRU step, getting a new hidden vector
        h_t = self.gru_cell(rnn_input, h_t)
        self._cached_ht.append(h_t.cpu().detach().numpy())

        # Step 3: Use the current hidden to attend to the encoder state
        context_vectors, p_attn, _ = verbose_attention(encoder_state_vectors=encoder_state, 
                                                       query_vector=h_t)

        # auxillary: cache the attention probabilities for visualization
        self._cached_p_attn.append(p_attn.cpu().detach().numpy())

        # Step 4: Use the current hidden and context vectors to make a prediction to the next word
        prediction_vector = torch.cat((context_vectors, h_t), dim=1)
        score_for_y_t_index = self.classifier(self.dropout(prediction_vector))

        # auxillary: collect the prediction scores
        return h_t, context_vectors, prediction_vector, score_for_y_t_index
        

            
    def forward(self, encoder_state, initial_hidden_state, target_sequence, sample_probability=0.0):
        """The forward pass of the model
        
        Args:
            encoder_state (torch.Tensor): the output of the NMTEncoder
            initial_hidden_state (torch.Tensor): The last hidden state in the  NMTEncoder
            target_sequence (torch.Tensor): the target text data tensor
            sample_probability (float): the schedule sampling parameter
                probabilty of using model's predictions at each decoder step
        Returns:
            output_vectors (torch.Tensor): prediction vectors at each output step
        """
        h_t, target_sequence, output_sequence_size, sample_probability = \
                                        self._init_decoder(initial_hidden_state, target_sequence, sample_probability)
        
        batch_size = encoder_state.size(0)
        # initialize context vectors to zeros
        context_vectors = self._init_context_vectors(batch_size)
        # initialize first y_t word as BOS
        y_t_index = self._init_indices(batch_size)
        
        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)

        output_vectors = []
        # all for visualisation
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()
        
#         if self.num_layers > 1:
#             last_layer_hidden_states, _ = self.rnn(self.target_embedding(target_sequence), h_t.unsqueeze(1))
#             h_t = last_layer_hidden_states[-1]
#             print(h_t.shape)
         
        for i in range(output_sequence_size):
            # Schedule sampling is whe
            use_sample = self.is_using_sample(sample_probability)
            if not use_sample:
                y_t_index = target_sequence[i]

            h_t, context_vectors, prediction_vector, score_for_y_t_index = self.forward_step(encoder_state, \
                                                                                        y_t_index, \
                                                                                        h_t,
                                                                                        context_vectors)

            if use_sample:
                y_t_index = self.pred_y_t_index(prediction_vector)

            # auxillary: collect the prediction scores
            output_vectors.append(score_for_y_t_index)

        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)
        
        return output_vectors

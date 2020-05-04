# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, nchar, nhid, nlayers, max_input_len, dropout=0.5, batch_first=False):
        super(LSTMModel, self).__init__()
        self.nchar = nchar
        self.nhid = nhid
        self.nlayers = nlayers
        self.max_input_len = max_input_len
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(nchar, nhid, nlayers, dropout=dropout, batch_first=batch_first)
        self.decoder = nn.Linear(nhid, nchar)

        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, lengths):
        """
        Args:
            input : one hot encoding representation of a character
            hidden : hidden state of previous timestep 
        """
        packed_input = pack_padded_sequence(input, lengths, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_input, hidden)
        output, _ = pad_packed_sequence(
            packed_output,
            padding_value=0, 
            total_length=self.max_input_len
        )
        output = self.drop(output)
        decoded = self.decoder(output)
        return F.log_softmax(decoded, dim=2), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                    weight.new_zeros(self.nlayers, batch_size, self.nhid))
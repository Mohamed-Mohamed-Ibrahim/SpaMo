import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TemporalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(TemporalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, lengths):
        """
        x: [Batch, Time, Channels]
        lengths: [Batch] Tensor of actual sequence lengths
        """
        lengths_cpu = lengths.to('cpu')
        
        packed_x = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        
        packed_out, _ = self.lstm(packed_x)
        
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        return self.layer_norm(out)
import torch
import torch.nn as nn

class TemporalConv(nn.Module):
    """
    1D Temporal Convolutional Network.
    """
    def __init__(self, input_size, hidden_size):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.kernel_size = ['K5', 'P2', 'K5', 'P2']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=True))
            elif ks[0] == 'K':
                kernel_size = int(ks[1])
                padding = (kernel_size - 1) // 2
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=kernel_size, stride=1, padding=padding)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
                
        self.temporal_conv = nn.Sequential(*modules)

    def update_lgt(self, lgt):
        feat_len = lgt.clone() if isinstance(lgt, torch.Tensor) else torch.tensor(lgt)
        
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.ceil(feat_len / int(ks[1]))
            else:
                pass
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }
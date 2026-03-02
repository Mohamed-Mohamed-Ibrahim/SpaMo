import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

class FeatureAugmenter(nn.Module):
    def __init__(
        self, 
        feature_dim: int = 1024,         # <--- Add your feature dimension here (e.g., 1024 for I3D)
        aug_prob: float = 0.5,          
        frame_dropout_prob: float = 0.1, 
        span_mask_prob: float = 0.1,     
        channel_drop_prob: float = 0.05, 
        max_span_length: int = 10        
    ):
        super().__init__()
        self.aug_prob = aug_prob
        self.frame_dropout_prob = frame_dropout_prob
        self.span_mask_prob = span_mask_prob
        self.channel_drop_prob = channel_drop_prob
        self.max_span_length = max_span_length
        
        # Create a learnable token to replace 0.0
        # Initialize it randomly with a small variance
        self.mask_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

    def forward(
        self, 
        features: torch.Tensor, 
        lengths: Union[List[int], torch.Tensor]
    ) -> torch.Tensor:
        
        # Bypass if evaluating or probability fails
        if not self.training or torch.rand(1).item() > self.aug_prob:
            return features
            
        B, T, D = features.shape
        device = features.device
        aug = features.clone()

        # -------------------------
        # (1) FRAME DROPOUT (Properly Scaled)
        # -------------------------
        if self.frame_dropout_prob > 0:
            frame_mask = torch.bernoulli(
                torch.full((B, T, 1), 1 - self.frame_dropout_prob, device=device)
            )
            # Scale the remaining frames to maintain energy
            aug = (aug * frame_mask) / (1.0 - self.frame_dropout_prob)

        # -------------------------
        # (2) CHANNEL DROPOUT (Properly Scaled)
        # -------------------------
        if self.channel_drop_prob > 0:
            channel_mask = torch.bernoulli(
                torch.full((B, 1, D), 1 - self.channel_drop_prob, device=device)
            )
            aug = (aug * channel_mask) / (1.0 - self.channel_drop_prob)

        # -------------------------
        # (3) SPAN MASKING (Using Learnable Token)
        # -------------------------
        if self.span_mask_prob > 0:
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device)
                
            for b in range(B):
                L = int(lengths[b])
                if L <= 1: 
                    continue
                    
                target_mask = int(L * self.span_mask_prob)
                masked = 0
                attempts = 0
                
                while masked < target_mask and attempts < 20:
                    attempts += 1
                    span_len = int(torch.randint(1, self.max_span_length + 1, (1,)))
                    if span_len >= L: 
                        break
                    
                    start = int(torch.randint(0, L - span_len, (1,)))
                    
                    # INSTEAD OF 0.0, WE PLUG IN THE LEARNABLE TOKEN
                    # It automatically broadcasts to the shape of the span
                    aug[b, start:start+span_len] = self.mask_token
                    
                    masked += span_len

        return aug

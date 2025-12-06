import torch
import torch.nn as nn
from typing import List, Union

class FeatureAugmenter(nn.Module):
    def __init__(
        self, 
        aug_prob: float = 0.5,          # Chance of applying augmentation to a batch
        frame_dropout_prob: float = 0.1, # Drop individual frames
        span_mask_prob: float = 0.1,     # Mask continuous time spans
        channel_drop_prob: float = 0.05, # Drop feature dimensions
        max_span_length: int = 10        # Maximum contiguous masked span length
    ):
        super().__init__()
        self.aug_prob = aug_prob
        self.frame_dropout_prob = frame_dropout_prob
        self.span_mask_prob = span_mask_prob
        self.channel_drop_prob = channel_drop_prob
        self.max_span_length = max_span_length

    def forward(
        self, 
        features: torch.Tensor, 
        lengths: Union[List[int], torch.Tensor]
    ) -> torch.Tensor:

        if not self.training or torch.rand(1).item() > self.aug_prob:
            return features

        B, T, D = features.shape
        device = features.device

        aug = features.clone()

        # -------------------------
        # (1) FRAME DROPOUT
        # -------------------------
        if self.frame_dropout_prob > 0:
            frame_mask = torch.bernoulli(
                torch.full((B, T, 1), 1 - self.frame_dropout_prob, device=device)
            )
            aug = aug * frame_mask

        # -------------------------
        # (2) CHANNEL DROPOUT
        # -------------------------
        if self.channel_drop_prob > 0:
            channel_mask = torch.bernoulli(
                torch.full((B, 1, D), 1 - self.channel_drop_prob, device=device)
            )
            aug = aug * channel_mask

        # -------------------------
        # (3) SPAN MASKING
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
                    aug[b, start:start+span_len] = 0.0
                    masked += span_len

        return aug

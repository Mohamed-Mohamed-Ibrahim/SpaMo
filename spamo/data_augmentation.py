import torch
import torch.nn as nn
from typing import List, Union


class FeatureAugmenter(nn.Module):
    """
    Applies Gaussian noise augmentation to visual features during training.
    
    This augmenter is designed for transformer hidden state features (CLIP ViT, VideoMAE)
    which are not L2-normalized and typically have values in range [-5, 5] with std ~1-2.
    """
    
    def __init__(self, noise_std: float = 0.1, noise_prob: float = 0.5):
        """
        Initialize the feature augmenter.
        
        Args:
            noise_std: Standard deviation of Gaussian noise to add to features.
                      Recommended range: 0.05 to 0.2 for transformer hidden states.
            noise_prob: Probability of applying augmentation to a batch during training.
        """
        super().__init__()
        self.noise_std = noise_std
        self.noise_prob = noise_prob
    
    def forward(
        self, 
        features: torch.Tensor, 
        lengths: Union[List[int], torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply Gaussian noise augmentation to features.
        
        Args:
            features: Input features of shape [batch_size, seq_len, feature_dim].
            lengths: Actual sequence lengths before padding, either as a list of integers
                    or a 1D tensor. Used to mask padding positions.
        
        Returns:
            Augmented features with same shape as input.
        """
        if not self.training or torch.rand(1).item() > self.noise_prob:
            return features
        
        noise = torch.randn_like(features) * self.noise_std
        
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=features.device)
        
        batch_size, seq_len, _ = features.shape
        mask = torch.zeros(batch_size, seq_len, 1, device=features.device)
        
        for i, length in enumerate(lengths):
            mask[i, :length, :] = 1.0
        
        noise = noise * mask
        augmented_features = features + noise
        
        return augmented_features
import torch
import torch.nn as nn
from typing import List, Union


class FeatureAugmenter(nn.Module):
    def __init__(
        self, 
        noise_ratio: float = 0.1,  # Noise as percentage of feature std
        noise_prob: float = 0.5,
        use_adaptive: bool = True
    ):
        """
        Initialize the feature augmenter.
        
        Args:
            noise_ratio: Noise level as a ratio of feature standard deviation.
                        e.g., 0.1 means noise_std = 0.1 * feature_std
            noise_prob: Probability of applying augmentation to a batch.
            use_adaptive: If True, compute noise_std from data variance.
                         If False, use noise_ratio as fixed noise_std.
        """
        super().__init__()
        self.noise_ratio = noise_ratio
        self.noise_prob = noise_prob
        self.use_adaptive = use_adaptive
    
    def forward(
        self, 
        features: torch.Tensor, 
        lengths: Union[List[int], torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply Gaussian noise augmentation to features.
        
        Args:
            features: Input features of shape [batch_size, seq_len, feature_dim].
            lengths: Actual sequence lengths before padding.
        
        Returns:
            Augmented features with same shape as input.
        """
        if not self.training or torch.rand(1).item() > self.noise_prob:
            return features
        
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=features.device)
        
        batch_size, seq_len, _ = features.shape
        
        # Create mask for valid positions
        mask = torch.zeros(batch_size, seq_len, 1, device=features.device)
        for i, length in enumerate(lengths):
            mask[i, :length, :] = 1.0
        
        # Compute adaptive noise std if enabled
        if self.use_adaptive:
            # Compute std only on valid (non-padded) positions
            valid_features = features * mask
            num_valid = mask.sum()
            
            feature_std = torch.sqrt(
                (valid_features ** 2).sum() / num_valid
            )
            noise_std = self.noise_ratio * feature_std
        else:
            noise_std = self.noise_ratio
        
        # Generate and apply noise
        noise = torch.randn_like(features) * noise_std
        noise = noise * mask
        augmented_features = features + noise
        
        return augmented_features
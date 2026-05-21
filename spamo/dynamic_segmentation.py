import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DynamicSegmenter(nn.Module):
    """
    Computes motion-based importance scores for frames to guide adaptive masking.
    Enhanced with multi-layer networks and attention mechanisms for powerful feature scoring.
    """
    def __init__(self, hidden_dim=512, motion_threshold=0.5, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = motion_threshold
        self.num_layers = num_layers
        
        # Multi-layer motion detection network
        layers = []
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim // 2
            out_channels = hidden_dim // 2
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.GELU())
        
        self.motion_detector = nn.Sequential(*layers)
        
        # Output projection to single score
        self.score_projection = nn.Sequential(
            nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 4, 1, kernel_size=1)
        )
        
        # Temporal attention for context-aware scoring
        self.attention_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.attention_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Adaptive threshold learning
        self.threshold_param = nn.Parameter(torch.tensor([motion_threshold]))
        self.norm_layer = nn.LayerNorm(hidden_dim)
    
    def forward(self, features, lengths):
        """
        Args:
            features: (B, T, C) tensor
            lengths: list of int, actual lengths per batch
        Returns:
            importance_scores: (B, T) tensor, higher values indicate more important frames
        """
        B, T, C = features.shape
        
        # Normalize features for better gradient flow
        normalized_features = self.norm_layer(features)
        
        # Self-attention to capture temporal context
        attn_output, _ = self.attention(normalized_features, normalized_features, normalized_features)
        
        # Combine with original features
        enhanced_features = features + 0.3 * attn_output
        
        # Multi-layer motion detection
        motion_scores = self.motion_detector(enhanced_features.permute(0, 2, 1))
        
        # Project to single score
        motion_scores = self.score_projection(motion_scores).squeeze(1)  # (B, T)
        
        # Apply sigmoid with learned offset for adaptive thresholding
        importance_scores = torch.sigmoid(motion_scores + self.threshold_param)
        
        return importance_scores

    def segment(self, features, importance_scores, lengths):
        """
        Convert frame-level features into adaptive temporal segments with gentle downsampling.
        Preserves temporal information by keeping full resolution or gentle 2x downsampling.
        FIX: Uses quantile-based threshold for stability.
        """
        batch_size, max_len, hidden_dim = features.shape
        segmented_sequences = []
        segmented_lengths = []

        for b in range(batch_size):
            length = lengths[b]
            seq = features[b, :length]
            scores = importance_scores[b, :length]
            
            # FIX 4: Use quantile instead of mean-std for stable threshold
            adaptive_threshold = scores.quantile(0.3)
            
            boundaries = [0]
            for t in range(1, length):
                # Smooth boundary detection with hysteresis
                if scores[t] < adaptive_threshold and scores[t - 1] >= adaptive_threshold:
                    boundaries.append(t)
            boundaries.append(length)

            segments = []
            
            for start, end in zip(boundaries, boundaries[1:]):
                if end <= start:
                    continue
                
                segment_seq = seq[start:end]
                
                # FIX 3: Don't collapse aggressively - gentle downsampling instead of 1 vector per segment
                # Keep frames or downsample by 2 if segment is long
                if segment_seq.shape[0] > 10:
                    # Gentle downsampling: every 2nd frame
                    segment_seq = segment_seq[::2]
                
                segments.append(segment_seq)

            # If no segments found, use full sequence or downsample
            if len(segments) == 0:
                if seq.shape[0] > 10:
                    segments = [seq[::2]]
                else:
                    segments = [seq]

            segmented_sequences.append(torch.cat(segments, dim=0))
            segmented_lengths.append(segmented_sequences[-1].shape[0])

        max_seg_len = max(segmented_lengths)
        output = features.new_zeros((batch_size, max_seg_len, hidden_dim))
        for b, seg in enumerate(segmented_sequences):
            output[b, :seg.shape[0]] = seg

        return output, segmented_lengths

class AdaptiveMasker(nn.Module):
    """
    Applies adaptive masking to variable-length segments based on frame importance.
    Uses smooth masking (noise injection) instead of hard zeroing for better robustness.

    The optional `seed` parameter enables deterministic masking behavior for
    reproducible training and evaluation.
    """
    def __init__(self, mask_prob=0.15, min_mask_len=5, max_mask_len=20, mask_type='noise', seed=None):
        super().__init__()
        self.mask_prob = mask_prob
        self.min_len = min_mask_len
        self.max_len = max_mask_len
        self.mask_type = mask_type  # 'zero', 'noise', or 'smooth'
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else None
        self.generator = None
    
    def forward(self, features, importance_scores, lengths, training=True):
        """
        Args:
            features: (B, T, C) tensor
            importance_scores: (B, T) tensor from DynamicSegmenter
            lengths: list of int, actual lengths per batch
            training: bool, only apply masking during training
        Returns:
            masked_features: (B, T, C) tensor with adaptive masking applied
        """
        if not training or (self.rng.random() if self.rng is not None else random.random()) > self.mask_prob:
            return features
        
        masked = features.clone()
        B, T, C = features.shape
        
        for b in range(B):
            seq_len = lengths[b]
            scores = importance_scores[b, :seq_len]
            
            if seq_len <= self.min_len:
                continue
            
            # Compute adaptive threshold
            score_mean = scores.mean()
            score_std = scores.std() + 1e-8
            threshold = score_mean - 0.3 * score_std
            
            # Find low-importance regions
            low_imp_mask = scores < threshold
            low_imp_indices = low_imp_mask.nonzero(as_tuple=True)[0]
            
            if len(low_imp_indices) == 0:
                continue
            
            # Create contiguous mask regions
            mask_starts = []
            current_start = low_imp_indices[0].item()
            
            for i in range(1, len(low_imp_indices)):
                if low_imp_indices[i].item() - low_imp_indices[i-1].item() > 1:
                    mask_starts.append((current_start, low_imp_indices[i-1].item() + 1))
                    current_start = low_imp_indices[i].item()
            mask_starts.append((current_start, low_imp_indices[-1].item() + 1))
            
            # Randomly select one region to mask
            if mask_starts:
                if self.rng is not None:
                    start, end = self.rng.choice(mask_starts)
                else:
                    start, end = random.choice(mask_starts)
                mask_len = min(end - start, self.max_len)
                mask_start = max(0, start)
                mask_end = min(seq_len, mask_start + mask_len)
                
                if self.mask_type == 'noise':
                    # Add Gaussian noise instead of zeroing
                    shape = masked[b, mask_start:mask_end].shape
                    if self.seed is not None:
                        # Ensure generator matches the feature device
                        if self.generator is None or self.generator.device != masked.device:
                            self.generator = torch.Generator(device=masked.device)
                            self.generator.manual_seed(self.seed)
                        noise = torch.randn(shape, device=masked.device, dtype=masked.dtype, generator=self.generator) * 0.1
                    else:
                        noise = torch.randn(shape, device=masked.device, dtype=masked.dtype) * 0.1
                    masked[b, mask_start:mask_end] = masked[b, mask_start:mask_end] + noise
                elif self.mask_type == 'smooth':
                    # Smooth masking with fade-in/fade-out
                    fade_len = min(3, (mask_end - mask_start) // 4)
                    for i in range(mask_start, mask_end):
                        if i - mask_start < fade_len:
                            alpha = (i - mask_start) / fade_len
                        elif i - mask_start >= mask_end - mask_start - fade_len:
                            alpha = (mask_end - i) / fade_len
                        else:
                            alpha = 0.3
                        masked[b, i] = masked[b, i] * (1 - alpha)
                else:  # 'zero'
                    masked[b, mask_start:mask_end] = 0
        
        return masked
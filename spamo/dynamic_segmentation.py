import torch
import torch.nn as nn
import random

class DynamicSegmenter(nn.Module):
    """
    Computes motion-based importance scores for frames to guide adaptive masking.
    Inspired by dynamic clip partitioning in AVRET, but adapted for full-sequence processing.
    """
    def __init__(self, hidden_dim=512, motion_threshold=0.5):
        super().__init__()
        self.motion_detector = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        self.threshold = motion_threshold  # Not used in this simplified version, but kept for future extension
    
    def forward(self, features, lengths):
        """
        Args:
            features: (B, T, C) tensor
            lengths: list of int, actual lengths per batch
        Returns:
            importance_scores: (B, T) tensor, higher values indicate more important frames
        """
        motion_scores = self.motion_detector(features.permute(0, 2, 1)).squeeze(1)  # (B, T)
        importance_scores = torch.sigmoid(motion_scores)  # Normalize to [0,1]
        return importance_scores

    def segment(self, features, importance_scores, lengths):
        """
        Convert frame-level features into adaptive temporal segments.

        Each segment is mean-pooled between boundaries derived from importance scores,
        which produces a shorter, adaptive sequence that preserves high-motion units.
        """
        batch_size, max_len, hidden_dim = features.shape
        segmented_sequences = []
        segmented_lengths = []

        for b in range(batch_size):
            length = lengths[b]
            seq = features[b, :length]
            scores = importance_scores[b, :length]
            boundaries = [0]

            for t in range(1, length):
                if scores[t] < self.threshold and scores[t - 1] >= self.threshold:
                    boundaries.append(t)
            boundaries.append(length)

            segments = []
            for start, end in zip(boundaries, boundaries[1:]):
                if end <= start:
                    continue
                segment = seq[start:end].mean(dim=0, keepdim=True)
                segments.append(segment)

            if len(segments) == 0:
                segments = [seq.mean(dim=0, keepdim=True)]

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
    Masks low-importance regions with variable lengths, forcing robust inference.
    """
    def __init__(self, mask_prob=0.15, min_mask_len=5, max_mask_len=20):
        super().__init__()
        self.mask_prob = mask_prob
        self.min_len = min_mask_len
        self.max_len = max_mask_len
    
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
        if not training:
            return features
        
        masked = features.clone()
        for b in range(features.shape[0]):
            seq_len = lengths[b]
            seq = features[b, :seq_len]
            scores = importance_scores[b, :seq_len]
            
            # Decide whether to mask this sequence
            if random.random() < self.mask_prob and seq_len > self.min_len:
                # Identify low-importance regions (below median)
                low_imp_indices = (scores < scores.median()).nonzero(as_tuple=True)[0]
                if len(low_imp_indices) > 0:
                    # Randomly select a start point in low-importance areas
                    start = random.choice(low_imp_indices.tolist())
                    # Random mask length within bounds
                    max_possible = seq_len - start
                    length = random.randint(self.min_len, min(self.max_len, max_possible))
                    # Apply mask (set to zero)
                    masked[b, start:start+length] = 0
        return masked

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalSignCLLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 0.07,
        temporal_window: int = 5,
    ):
        super().__init__()
        self.temperature = temperature
        self.temporal_window = temporal_window
    
    def forward(
        self,
        visual_embeddings: torch.Tensor,
        visual_masks: torch.Tensor,
    ) -> torch.Tensor:
  
        batch_size, seq_len, embed_dim = visual_embeddings.shape
        device = visual_embeddings.device
        
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(visual_embeddings, p=2, dim=2)  # [B, T, D]
        
        total_loss = 0.0
        batch_count = 0
        
        # Process each sequence in batch
        for b in range(batch_size):
            # Get valid sequence length (accounting for padding)
            valid_len = int(visual_masks[b].sum().item())
            
            # Skip if sequence too short
            if valid_len < 2:
                continue
            
            # Extract and compute loss for this sequence
            seq_embeddings = embeddings[b, :valid_len, :]  # [T, D]
            seq_loss = self._compute_sequence_loss(seq_embeddings)
            
            if seq_loss is not None:
                total_loss += seq_loss
                batch_count += 1
        
        # Return averaged loss
        if batch_count == 0:
            # Return zero tensor that requires grad for stability
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss / batch_count
    
    def _compute_sequence_loss(self, embeddings: torch.Tensor) -> Optional[torch.Tensor]:
   
        seq_len = embeddings.shape[0]
        
        # Compute similarity matrix: [T, T]
        # Using normalized embeddings, this is cosine similarity
        sim_matrix = torch.mm(embeddings, embeddings.t())
        
        # Scale by temperature
        sim_matrix = sim_matrix / self.temperature
        
        # Mask out diagonal (self-similarity)
        sim_matrix.fill_diagonal_(-1e9)
        
        # Compute NT-Xent loss
        loss = 0.0
        valid_frames = 0
        
        for i in range(seq_len):
            # Find positive pair indices (within temporal window)
            pos_start = max(0, i - self.temporal_window)
            pos_end = min(seq_len, i + self.temporal_window + 1)
            
            # Collect indices (excluding self)
            pos_indices = list(range(pos_start, pos_end))
            if i in pos_indices:
                pos_indices.remove(i)
            
            # Skip if no positive pairs
            if len(pos_indices) == 0:
                continue
            
            # Get logits for this anchor
            logits_i = sim_matrix[i]  # [T]
            pos_logits = logits_i[pos_indices]  # [num_pos]
            
            # Numerically stable NT-Xent loss using log-sum-exp trick
            # Find maximum for stability
            max_logit = torch.max(logits_i).detach()
            
            # Compute exp with stability
            logits_stable = logits_i - max_logit
            pos_logits_stable = pos_logits - max_logit
            
            # Sum of positive exponentials
            pos_exp_sum = torch.exp(pos_logits_stable).sum()
            
            # Sum of all exponentials  
            all_exp_sum = torch.exp(logits_stable).sum()
            
            # NT-Xent loss = -log(mean(exp(pos)) / mean(exp(all)))
            #              = -log(sum(exp(pos)) / sum(exp(all)))
            frame_loss = -torch.log(pos_exp_sum / (all_exp_sum + 1e-8))
            
            # Only add if finite (skip NaN/Inf)
            if torch.isfinite(frame_loss):
                loss += frame_loss
                valid_frames += 1
        
        # Return None if no valid frames processed
        if valid_frames == 0:
            return None
        
        # Return averaged loss
        return loss / valid_frames


class SignCLLoss(nn.Module):
 
    
    def __init__(
        self,
        temperature: float = 0.07,
        temporal_window: int = 5,
        use_cosine_similarity: bool = True,
    ):
        super().__init__()
        self.temporal_impl = TemporalSignCLLoss(temperature, temporal_window)
        self.use_cosine_similarity = use_cosine_similarity
    
    def forward(
        self,
        visual_embeddings: torch.Tensor,
        visual_masks: torch.Tensor,
        video_ids: Optional[list] = None,
    ) -> torch.Tensor:
  
        return self.temporal_impl(visual_embeddings, visual_masks)

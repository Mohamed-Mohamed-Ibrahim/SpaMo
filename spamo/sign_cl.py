
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

        # Similarity matrix: [T, T] (cosine because embeddings are normalized)
        logits = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Build positive mask: True where |i-j| <= temporal_window and i != j
        idx = torch.arange(seq_len, device=embeddings.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        pos_mask = (dist <= self.temporal_window)
        pos_mask.fill_diagonal_(False)

        # For numerical stability compute per-row log-sum-exp
        # Work in float32 for stable exp/log when needed
        logits_f = logits.float()

        row_max, _ = torch.max(logits_f, dim=1, keepdim=True)
        logits_stable = logits_f - row_max
        exp_logits = torch.exp(logits_stable)

        # Denominator: sum over all (including positives, excluding diag since exp(-inf)~0)
        denom = exp_logits.sum(dim=1)  # [T]

        # Numerator: sum over positive positions for each anchor
        numer = (exp_logits * pos_mask.float()).sum(dim=1)  # [T]

        # Valid anchors are those that have at least one positive
        valid = numer > 0
        if valid.sum() == 0:
            return None

        # Per-anchor loss
        loss_per_anchor = -torch.log(numer[valid] / (denom[valid] + 1e-8))

        # Return averaged loss (cast back to embeddings dtype)
        return loss_per_anchor.mean().to(embeddings.dtype)


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


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SignCLLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 0.07,
        temporal_window: int = 5,
        use_cosine_similarity: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.temporal_window = temporal_window
        self.use_cosine_similarity = use_cosine_similarity
    
    def forward(
        self,
        visual_embeddings: torch.Tensor,
        visual_masks: torch.Tensor,
        video_ids: Optional[list] = None,
    ) -> torch.Tensor:
  
        batch_size, seq_len, embedding_dim = visual_embeddings.shape
        
        # Flatten batch and sequence dimensions: [batch_size * seq_len, embedding_dim]
        flat_embeddings = visual_embeddings.reshape(-1, embedding_dim)
        flat_masks = visual_masks.reshape(-1)
        
        # Create index mapping: (batch_idx, seq_idx) -> flat_idx
        batch_indices, seq_indices = torch.where(visual_masks == 1)
        
        if len(batch_indices) < 2:
            # Not enough valid frames for contrastive learning
            return torch.tensor(0.0, device=visual_embeddings.device, requires_grad=True)
        
        # Normalize embeddings for cosine similarity
        if self.use_cosine_similarity:
            flat_embeddings = F.normalize(flat_embeddings, p=2, dim=1)
        
        # Compute similarity matrix: [num_valid_frames, num_valid_frames]
        valid_indices = torch.where(flat_masks == 1)[0]
        valid_embeddings = flat_embeddings[valid_indices]
        
        similarity_matrix = torch.mm(valid_embeddings, valid_embeddings.t())  # [N, N]
        similarity_matrix = similarity_matrix / self.temperature
        
        # Construct positive pair mask based on temporal neighborhoods
        pos_mask = self._get_positive_pairs_mask(
            batch_indices, seq_indices, batch_size, seq_len
        )
        
        # Diagonal is always identity, not a positive pair
        pos_mask.fill_diagonal_(False)
        
        # NT-Xent loss computation
        # Remove self-similarity (diagonal)
        similarity_matrix.fill_diagonal_(-9e15)
        
        # For each anchor, compute loss
        pos_logits = similarity_matrix[pos_mask].reshape(len(valid_indices), -1)
        neg_logits = similarity_matrix
        
        # Gather negative logits (all except positive pairs)
        neg_mask = ~pos_mask
        neg_logits_gathered = []
        
        for i in range(len(valid_indices)):
            neg_idx = torch.where(neg_mask[i])[0]
            if len(neg_idx) > 0:
                neg_logits_gathered.append(neg_logits[i, neg_idx])
        
        # Compute contrastive loss using InfoNCE
        loss = self._compute_infonce_loss(
            similarity_matrix, pos_mask
        )
        
        return loss
    
    def _get_positive_pairs_mask(
        self,
        batch_indices: torch.Tensor,
        seq_indices: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
 
        num_valid = len(batch_indices)
        pos_mask = torch.zeros((num_valid, num_valid), dtype=torch.bool, device=batch_indices.device)
        
        for i in range(num_valid):
            for j in range(i + 1, num_valid):
                # Same video (batch)
                if batch_indices[i] == batch_indices[j]:
                    # Within temporal window
                    seq_dist = abs(seq_indices[i].item() - seq_indices[j].item())
                    if seq_dist <= self.temporal_window and seq_dist > 0:
                        pos_mask[i, j] = True
                        pos_mask[j, i] = True
        
        return pos_mask
    
    def _compute_infonce_loss(
        self,
        similarity_matrix: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
 
        num_frames = similarity_matrix.shape[0]
        
        if pos_mask.sum() == 0:
            # No positive pairs found
            return torch.tensor(0.0, device=similarity_matrix.device, requires_grad=True)
        
        loss = 0.0
        valid_count = 0
        
        for i in range(num_frames):
            pos_indices = torch.where(pos_mask[i])[0]
            
            if len(pos_indices) == 0:
                # This frame has no positive pairs, skip it
                continue
            
            # Positive logits for this frame
            pos_logits = similarity_matrix[i, pos_indices]
            
            # Negative logits (all other frames)
            neg_logits = similarity_matrix[i]
            
            # Compute loss for this anchor
            # log_softmax over all samples including positives
            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.zeros(len(pos_logits), dtype=torch.long, device=similarity_matrix.device)
            
            frame_loss = F.cross_entropy(
                logits.unsqueeze(0),
                labels.unsqueeze(0),
                reduction='none'
            ).sum()
            
            loss += frame_loss
            valid_count += 1
        
        if valid_count == 0:
            return torch.tensor(0.0, device=similarity_matrix.device, requires_grad=True)
        
        return loss / valid_count


class TemporalSignCLLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 0.07,
        temporal_window: int = 5,
        hard_neg_ratio: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.temporal_window = temporal_window
        self.hard_neg_ratio = hard_neg_ratio
    
    def forward(
        self,
        visual_embeddings: torch.Tensor,
        visual_masks: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len, embed_dim = visual_embeddings.shape
        device = visual_embeddings.device
        
        # Normalize embeddings
        embeddings = F.normalize(visual_embeddings, p=2, dim=2)  # [B, T, D]
        
        loss = 0.0
        count = 0
        
        for b in range(batch_size):
            # Get valid frame indices for this batch
            valid_len = int(visual_masks[b].sum().item())
            if valid_len < 2:
                continue
            
            batch_embeddings = embeddings[b, :valid_len, :]  # [T, D]
            
            # Compute similarity matrix for this batch
            sim_matrix = torch.mm(batch_embeddings, batch_embeddings.t())  # [T, T]
            sim_matrix = sim_matrix / self.temperature
            
            # Create positive pair mask (temporal neighbors)
            pos_mask = torch.zeros((valid_len, valid_len), dtype=torch.bool, device=device)
            for i in range(valid_len):
                for j in range(valid_len):
                    if i != j and abs(i - j) <= self.temporal_window:
                        pos_mask[i, j] = True
            
            # Mask out self-similarity
            sim_matrix.fill_diagonal_(-1e9)
            
            # Compute NT-Xent for this batch
            batch_loss = self._nt_xent_loss(sim_matrix, pos_mask)
            
            if batch_loss is not None:
                loss += batch_loss
                count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss / count
    
    def _nt_xent_loss(
        self,
        sim_matrix: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
  
        seq_len = sim_matrix.shape[0]
        device = sim_matrix.device
        
        if pos_mask.sum() == 0:
            return None
        
        # For each frame, compute cross-entropy loss
        loss = 0.0
        valid_count = 0
        
        for i in range(seq_len):
            pos_indices = torch.where(pos_mask[i])[0]
            
            if len(pos_indices) == 0:
                continue
            
            # Get logits for this anchor
            logits = sim_matrix[i]  # [T]
            pos_logits = logits[pos_indices]  # [num_pos]
            
            # Compute NT-Xent loss properly
            # Numerator: mean of exponentials of positive similarities
            pos_exp = torch.exp(pos_logits)  # [num_pos]
            pos_exp_sum = pos_exp.sum()
            
            # Denominator: sum of exponentials of all similarities
            all_exp_sum = torch.exp(logits).sum()
            
            # NT-Xent loss = -log(pos_mean / (pos_mean + neg_mean))
            # Which is: -log(pos_sum / all_sum)
            frame_loss = -torch.log(pos_exp_sum / (all_exp_sum + 1e-8))
            loss += frame_loss
            valid_count += 1
        
        if valid_count == 0:
            return None
        
        return loss / valid_count

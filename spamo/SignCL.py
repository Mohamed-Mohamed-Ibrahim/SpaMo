import torch
import torch.nn as nn
import torch.nn.functional as F

class SignCL(nn.Module):
    def __init__(self, temperature=0.07, margin=0.2):
        """
        Args:
            temperature: Scaling factor for logits (if using Softmax/InfoNCE)
            margin: The margin for the hinge loss (pos_sim > neg_sim + margin)
        """
        super(SignCL, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, inputs_embeds, mask=None, temporal_margin=1):
        """
        Args:
            inputs_embeds: (B, T, D) tensor of features.
            mask: (B, T) boolean or binary tensor (1 for valid, 0 for pad).
            temporal_margin: Time steps required between anchor and negative.
        """
        B, T, D = inputs_embeds.size()
        
        # 1. Align Geometry: Normalize inputs for Cosine Similarity (matches CLIP)
        embeds = F.normalize(inputs_embeds, p=2, dim=-1)
        
        total_loss = torch.tensor(0.0, device=inputs_embeds.device)
        n_valid_steps = torch.tensor(0.0, device=inputs_embeds.device)
        
        # Iterate through the sequence (skipping first and last to have neighbors)
        for t in range(1, T - 1):
            anchor = embeds[:, t, :]       # (B, D)
            positive_prev = embeds[:, t-1, :] # (B, D)
            positive_next = embeds[:, t+1, :] # (B, D)
            
            # 2. Similarity with Positives (Temporal Smoothness)
            sim_prev = torch.sum(anchor * positive_prev, dim=-1) 
            sim_next = torch.sum(anchor * positive_next, dim=-1)
            pos_sim = (sim_prev + sim_next) / 2.0 
            
            # 3. Similarity with Negatives (Distinctiveness)
            # Find valid negative indices (outside the local window)
            valid_neg_indices = [x for x in range(T) if abs(x - t) > temporal_margin]
            
            if not valid_neg_indices:
                continue 
                
            # Randomly select one negative time step for the whole batch 
            # (Vectorization trick: picking the same relative negative time step is faster)
            neg_t = valid_neg_indices[torch.randint(0, len(valid_neg_indices), (1,)).item()]
            negative = embeds[:, neg_t, :]
            neg_sim = torch.sum(anchor * negative, dim=-1)
            
            # 4. Hinge Loss: We want pos_sim > neg_sim + margin
            # Loss = max(0, neg_sim - pos_sim + margin)
            loss_step = F.relu(neg_sim - pos_sim + self.margin)
            
            # 5. Apply Masking (CRITICAL FIX)
            if mask is not None:
                # A step is valid only if anchor, pos_prev, pos_next, AND negative are all valid
                # mask[:, t] is the anchor validity
                # mask[:, neg_t] is the negative validity
                valid_mask = (
                    mask[:, t] & 
                    mask[:, t-1] & 
                    mask[:, t+1] & 
                    mask[:, neg_t]
                ).float()
                
                loss_step = loss_step * valid_mask
                n_valid_steps += valid_mask.sum()
            else:
                n_valid_steps += B
                
            total_loss += loss_step.sum()

        if n_valid_steps > 0:
            return total_loss / n_valid_steps
        else:
            return torch.tensor(0.0, device=inputs_embeds.device, requires_grad=True)

if __name__ == "__main__":
    # --- Test Case Configuration ---
    batch_size = 4
    seq_len = 10
    embed_dim = 16
    
    # 1. Generate Random Embeddings
    inputs_embeds = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

    # 2. Create a Mask to simulate padding
    # Sample 0: Full length (10 frames)
    # Sample 1: Short (3 frames) -> Note: SignCL needs at least 3 frames (prev, curr, next)
    # Sample 2: Medium (6 frames)
    # Sample 3: Full length
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[0, :10] = True
    mask[1, :3]  = True 
    mask[2, :6]  = True
    mask[3, :10] = True
    
    print(f"Input Shape: {inputs_embeds.shape}")
    print(f"Mask Shape: {mask.shape}")
    print(f"Mask (Sample 1 - Short): {mask[1].int().tolist()}")

    # 3. Initialize Model
    # margin=0.2 is standard for cosine similarity (which ranges -1 to 1)
    sign_cl = SignCL(margin=0.2)

    # 4. Forward Pass
    # temporal_margin=2 means negatives must be at least 2 frames away
    loss = sign_cl(inputs_embeds, mask=mask, temporal_margin=2)
    
    print("\n--- Results ---")
    print(f"Calculated Loss: {loss.item():.6f}")

    # 5. Backward Pass Check (Verify gradients exist)
    loss.backward()
    print(f"Gradients computed? {inputs_embeds.grad is not None}")
    
    # Check if padding caused gradients (Should be 0 for padded regions)
    # Checking gradients for Sample 1 (index 1) at frame 8 (which is padded/False)
    grad_at_padding = inputs_embeds.grad[1, 8, :].sum().item()
    print(f"Gradient at padded frame (should be 0.0): {grad_at_padding}")
    
    if grad_at_padding == 0.0:
        print("\u2705 SUCCESS: Padding was correctly ignored.")
    else:
        print("\u274C FAILURE: Gradients leaked into padded regions.")
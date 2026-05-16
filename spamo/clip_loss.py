import torch
import torch.nn.functional as F
 
def siglip_loss(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    n = logits.shape[0]
    labels = 2.0 * torch.eye(n, device=logits.device, dtype=logits.dtype) - 1.0
    return -torch.mean(F.logsigmoid(labels * (logits + bias)))
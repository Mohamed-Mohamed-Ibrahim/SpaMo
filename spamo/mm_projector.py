import torch
import torch.nn as nn
import re
import torch.nn.functional as F


# Credit by https://github1s.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py
class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class AdaptiveFusion(nn.Module):
    """
    Adaptive Fusion Mechanism inspired by AVRET.
    
    Instead of simple addition or concatenation, this module learns to adaptively
    weight two input representations at each timestep, allowing the model to
    dynamically emphasize one stream over the other.
    
    Formula: fused = input_1 + input_2 + λ₁ × input_1 + λ₂ × input_2
    where λ₁ and λ₂ are learnable per-timestep weights.
    """
    def __init__(self, input_size_1=512, input_size_2=512, output_size=2, bias=False):
        """
        Args:
            input_size_1: dimensionality of first input
            input_size_2: dimensionality of second input (should match input_size_1)
            output_size: number of adaptive weight channels (default 2 for λ₁, λ₂)
            bias: whether to use bias in linear layers
        """
        super(AdaptiveFusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.weight_input_1 = nn.Linear(input_size_1, output_size, bias=bias)
        self.weight_input_2 = nn.Linear(input_size_2, output_size, bias=bias)
        self.layer_norm = nn.LayerNorm(input_size_1, eps=1e-5)
        
    def forward(self, input_1, input_2):
        """
        Fuse two input representations adaptively.
        
        Args:
            input_1: First input tensor [B, T, D]
            input_2: Second input tensor [B, T, D] (same shape as input_1)
            
        Returns:
            Fused representation [B, T, D]
        """
        # Compute adaptive weights: [B, T, 2]
        fm_sigmoid = self.sigmoid(self.weight_input_1(input_1) + self.weight_input_2(input_2))
        
        # Extract lambda weights (using detach to prevent gradient flow through weights)
        # This allows the fusion to be adaptive but doesn't backprop through lambda computation
        lambda1 = fm_sigmoid.clone().detach()[:, :, 0].unsqueeze(-1)  # [B, T, 1]
        lambda2 = fm_sigmoid.clone().detach()[:, :, 1].unsqueeze(-1)  # [B, T, 1]
        
        # Adaptive fusion formula
        fused_output = input_1 + input_2 + torch.mul(lambda1, input_1) + torch.mul(lambda2, input_2)
        fused_output = self.layer_norm(fused_output)
        return fused_output


class AdaptiveFusionWithProjection(nn.Module):
    """
    Adaptive Fusion with projection for inputs of different dimensions.
    Projects both inputs to a common dimension before fusion.
    """
    def __init__(self, input_size_1, input_size_2, hidden_size, output_size=2, bias=False):
        super(AdaptiveFusionWithProjection, self).__init__()
        self.proj_1 = nn.Linear(input_size_1, hidden_size)
        self.proj_2 = nn.Linear(input_size_2, hidden_size)
        self.adaptive_fusion = AdaptiveFusion(hidden_size, hidden_size, output_size, bias)
        
    def forward(self, input_1, input_2):
        proj_1 = self.proj_1(input_1)
        proj_2 = self.proj_2(input_2)
        return self.adaptive_fusion(proj_1, proj_2)

def build_vision_projector(mm_projector_type='linear', mm_hidden_size=512, hidden_size=768, mlp_depth=1):
    if mm_projector_type == 'linear':
        return nn.Linear(mm_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', mm_projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1)) if mlp_gelu_match.group(1).isdigit() else mlp_depth
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    if mm_projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {mm_projector_type}')


# https://github1s.com/facebookresearch/jepa/blob/main/src/models/utils/modules.py
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        use_sdpa=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim*2), bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = (xattn @ v)

        q = q.transpose(1, 2).reshape(B, n, C)
        q = self.proj(q)
    
        return q


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q

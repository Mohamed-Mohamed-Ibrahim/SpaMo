"""
Spatial–Spatiotemporal Contrastive Loss
========================================
Computes a CLIP-style contrastive loss between spatial features (high frame-rate,
~100-300 frames) and spatiotemporal features (low frame-rate, ~20-50 clips) **before**
they are merged.

Both streams share the same feature dimensionality (``inter_hidden``) after their
respective linear projections in the main model.

Because the two streams have different temporal lengths we offer six alignment
strategies, controlled by ``alignment_mode``:

  padding         – zero-pad the shorter sequence to match the longer one, then mean-pool
  interp_nearest  – 1-D nearest-neighbour interpolation along the time axis, then mean-pool
  interp_linear   – 1-D linear interpolation along the time axis, then mean-pool
  reduction       – independently mean-pool both streams to (B, D) — no length alignment
  attention       – cross-attention: spatiotemporal queries attend to spatial keys/values,
                    then mean-pool the attended output
  upsample_st     – dynamic 1D interpolation upsamples the spatiotemporal stream to the spatial
                    length, refined by a learnable depthwise Conv1d, then element-wise averages
                    both aligned streams and mean-pools; the spatial resolution is preserved
                    (ST is never downsampled) and the upsample kernel is fully trainable

An optional ``proj_dim > 0`` projects both streams into a smaller shared space before
computing cosine similarity (useful to decouple the similarity head from the feature dim).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spamo.clip_loss import clip_loss


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mean_pool_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``x`` (B, T, D) over valid time steps indicated by ``mask`` (B, T) bool."""
    mask_f = mask.float().unsqueeze(-1)       # (B, T, 1)
    lengths = mask_f.sum(dim=1).clamp(min=1)  # (B, 1)
    return (x * mask_f).sum(dim=1) / lengths  # (B, D)


def _pad_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Zero-pad ``x`` (B, T, D) along the time axis to ``target_len``."""
    pad = target_len - x.size(1)
    if pad <= 0:
        return x[:, :target_len, :]
    return F.pad(x, (0, 0, 0, pad))


def _interp_time(x: torch.Tensor, target_len: int, mode: str = "linear") -> torch.Tensor:
    """Interpolate ``x`` (B, T, D) to ``target_len`` along the time axis."""
    x_t = x.permute(0, 2, 1)  # (B, D, T)  – required by F.interpolate
    kwargs: dict = dict(size=target_len, mode=mode)
    if mode == "linear":
        kwargs["align_corners"] = False
    x_t = F.interpolate(x_t, **kwargs)
    return x_t.permute(0, 2, 1)  # (B, target_len, D)


def _extend_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    """Extend a (B, T) bool mask to (B, target_len), filling added positions with False."""
    B, T = mask.shape
    if T >= target_len:
        return mask[:, :target_len]
    extra = torch.zeros(B, target_len - T, dtype=torch.bool, device=mask.device)
    return torch.cat([mask, extra], dim=1)


# ---------------------------------------------------------------------------
# Dynamic upsampler
# ---------------------------------------------------------------------------

class DynamicUpsampler(nn.Module):
    """Learnable temporal upsampler using dynamic interpolation + depthwise Conv1d.

    First, it uses 1-D linear interpolation to dynamically match the exact target length.
    Then, it applies a depthwise ``Conv1d`` to refine the features and keep it learnable.

    Parameters
    ----------
    dim : int
        Number of feature channels (= ``inter_hidden`` or ``proj_dim``).
    kernel_size : int
        Kernel size for the refinement conv. Defaults to 3.
    """

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        # Depthwise 1D conv: one kernel per channel, keeps it lightweight
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor, target: int) -> torch.Tensor:
        """Upsample ``x`` (B, T, D) to (B, target, D)."""
        x_t = x.permute(0, 2, 1)          # (B, D, T)
        
        # 1. Dynamic interpolation to match target exactly
        x_t = F.interpolate(x_t, size=target, mode="linear", align_corners=False)
        
        # 2. Refine with learnable depthwise conv
        x_t = self.conv(x_t)              # (B, D, target)
        
        return x_t.permute(0, 2, 1)       # (B, target, D)


# ---------------------------------------------------------------------------
# Cross-attention aligner
# ---------------------------------------------------------------------------

class CrossAttentionAligner(nn.Module):
    """Spatiotemporal queries attend over spatial keys/values.

    Since both streams share the same dim ``D``, no key/value projection is needed;
    standard ``nn.MultiheadAttention`` handles it directly.
    The output retains the temporal shape of the spatiotemporal stream; a subsequent
    mean-pool reduces it to (B, D) for CLIP similarity.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,                          # (B, Tt, D) – spatiotemporal
        key:   torch.Tensor,                          # (B, Ts, D) – spatial
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, Ts) True = ignore
    ) -> torch.Tensor:
        out, _ = self.attn(query, key, key, key_padding_mask=key_padding_mask)
        return out  # (B, Tt, D)


# ---------------------------------------------------------------------------
# Main loss module
# ---------------------------------------------------------------------------

class SpatialTemporalCLLoss(nn.Module):
    """CLIP-style contrastive loss between spatial and spatiotemporal streams.

    Parameters
    ----------
    dim : int
        Shared feature dimensionality of both streams (= ``inter_hidden``).
    alignment_mode : str
        One of ``padding | interp_nearest | interp_linear | reduction | attention |
        upsample_st``.
    proj_dim : int
        If > 0, both streams are projected to this dim before similarity is computed.
        If <= 0, the raw ``dim``-dimensional features are used directly.
    temperature_init : float
        Initial value of the learnable log-temperature (CLIP default ≈ 2.6592 = ln 14).
    num_heads : int
        Attention heads (only used when ``alignment_mode == "attention"``).
    attn_dropout : float
        Dropout inside the cross-attention module.
    """

    VALID_MODES = {"padding", "interp_nearest", "interp_linear", "reduction", "attention", "upsample_st"}

    def __init__(
        self,
        dim: int,
        alignment_mode: str = "reduction",
        proj_dim: int = 0,
        temperature_init: float = 2.6592,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        if alignment_mode not in self.VALID_MODES:
            raise ValueError(
                f"alignment_mode must be one of {self.VALID_MODES}, got '{alignment_mode}'"
            )
        self.alignment_mode = alignment_mode

        # Single shared projection (same weights applied to both streams)
        self.feat_proj = nn.Linear(dim, proj_dim, bias=False) if proj_dim > 0 else nn.Identity()
        self._out_dim  = proj_dim if proj_dim > 0 else dim

        # Cross-attention aligner (only built when needed)
        if alignment_mode == "attention":
            self.cross_attn = CrossAttentionAligner(
                dim=self._out_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
            )
        else:
            self.cross_attn = None

        # Learnable dynamic upsampler (only built when needed)
        if alignment_mode == "upsample_st":
            self.dynamic_up = DynamicUpsampler(
                dim=self._out_dim,
            )
        else:
            self.dynamic_up = None

        self.logit_scale = nn.Parameter(torch.tensor(temperature_init))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def forward(
        self,
        spatial: torch.Tensor,               # (B, Ts, D)
        spatial_mask: torch.Tensor,          # (B, Ts) bool – True = valid
        spatiotemporal: torch.Tensor,        # (B, Tt, D)
        spatiotemporal_mask: torch.Tensor,   # (B, Tt) bool
    ) -> torch.Tensor:
        """Return the scalar symmetric CLIP loss."""
        # 1. (Optional) shared projection
        s  = self.feat_proj(spatial)        # (B, Ts, D')
        st = self.feat_proj(spatiotemporal) # (B, Tt, D')

        # 2. Align temporal lengths → get (B, D') embeddings per stream
        s_emb, st_emb = self._align_and_embed(s, spatial_mask, st, spatiotemporal_mask)

        # 3. L2 normalise
        s_emb  = F.normalize(s_emb,  dim=-1)
        st_emb = F.normalize(st_emb, dim=-1)

        # 4. Scaled cosine-similarity matrix + symmetric CLIP loss
        logits = torch.matmul(s_emb, st_emb.t()) * self.logit_scale.exp()  # (B, B)
        return clip_loss(logits)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _align_and_embed(
        self,
        s:       torch.Tensor, s_mask:  torch.Tensor,
        st:      torch.Tensor, st_mask: torch.Tensor,
    ):
        mode = self.alignment_mode

        if mode == "reduction":
            # Independently mean-pool each stream — simplest, no temporal alignment
            s_emb  = _mean_pool_masked(s,  s_mask)
            st_emb = _mean_pool_masked(st, st_mask)

        elif mode == "padding":
            target = max(s.size(1), st.size(1))
            s_emb  = _mean_pool_masked(_pad_to_length(s,  target), _extend_mask(s_mask,  target))
            st_emb = _mean_pool_masked(_pad_to_length(st, target), _extend_mask(st_mask, target))

        elif mode in ("interp_nearest", "interp_linear"):
            interp_mode = "nearest" if mode == "interp_nearest" else "linear"
            target = max(s.size(1), st.size(1))
            # After interpolation the mask is dense; plain mean-pool suffices
            s_emb  = _interp_time(s,  target, interp_mode).mean(dim=1)
            st_emb = _interp_time(st, target, interp_mode).mean(dim=1)

        elif mode == "attention":
            # Spatiotemporal queries attend to spatial keys/values;
            # key_padding_mask: True = ignore (inverse of our valid mask)
            spatial_key_mask = ~s_mask                                   # (B, Ts)
            st_attended      = self.cross_attn(st, s, spatial_key_mask) # (B, Tt, D')
            st_emb = _mean_pool_masked(st_attended, st_mask)
            s_emb  = _mean_pool_masked(s, s_mask)

        elif mode == "upsample_st":
            # Upsample the short spatiotemporal stream to the spatial length using
            # dynamic interpolation followed by a learnable depthwise Conv1d.
            # Only ST is upsampled so the spatial resolution is fully preserved.
            # After upsampling the two aligned streams are element-wise averaged and
            # then mean-pooled over the spatial mask.
            target = s.size(1)                                           # Ts
            st_up  = self.dynamic_up(st, target)                         # (B, Ts, D')
            fused  = (s + st_up) * 0.5                                   # (B, Ts, D')
            s_emb  = _mean_pool_masked(fused, s_mask)                    # (B, D')
            # For the other side of the CLIP matrix keep a clean ST embedding
            # (independently pooled, unpolluted by the spatial signal).
            st_emb = _mean_pool_masked(st, st_mask)                      # (B, D')

        else:
            raise ValueError(f"Unknown alignment_mode: {mode}")

        return s_emb, st_emb

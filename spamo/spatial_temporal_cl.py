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
  upsample_st     – learnable deconvolution (depthwise ConvTranspose1d) upsamples the
                    spatiotemporal stream to the spatial length, then element-wise averages
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
# Deconvolution upsampler
# ---------------------------------------------------------------------------

class DeconvUpsampler(nn.Module):
    """Learnable temporal upsampler using a depthwise ``ConvTranspose1d``.

    Each feature channel gets its own learnable 1-D transposed-conv kernel
    (``groups=dim``), keeping the total parameter count to ``dim * kernel_size``.

    The transposed convolution produces an approximate target length; the output is
    then trimmed (or zero-padded) to the exact ``target`` length requested at
    forward time, so the module works for any ratio of input to target length.

    Parameters
    ----------
    dim : int
        Number of feature channels (= ``inter_hidden`` or ``proj_dim``).
    upsample_factor : int
        Nominal stride of the transposed convolution (≈ ``Ts / Tt``).
        A value of ``kernel_size = upsample_factor`` gives a non-overlapping
        deconv; overlap can be added by increasing ``kernel_size``.
    kernel_size : int | None
        Kernel size for the transposed conv.  Defaults to ``upsample_factor``
        (non-overlapping).  Use ``upsample_factor * 2`` for smoother output.
    """

    def __init__(self, dim: int, upsample_factor: int = 6, kernel_size: Optional[int] = None):
        super().__init__()
        if kernel_size is None:
            kernel_size = upsample_factor
        self.stride = upsample_factor
        # Depthwise transposed conv: one kernel per channel
        self.deconv = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=upsample_factor,
            groups=dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor, target: int) -> torch.Tensor:
        """Upsample ``x`` (B, T, D) to (B, target, D)."""
        x_t = x.permute(0, 2, 1)          # (B, D, T)
        x_t = self.deconv(x_t)            # (B, D, T_out)  T_out ≈ T * stride
        T_out = x_t.size(-1)
        if T_out >= target:
            x_t = x_t[:, :, :target]      # trim to exact length
        else:
            pad = target - T_out
            x_t = F.pad(x_t, (0, pad))    # zero-pad if deconv undershoots
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
    upsample_factor : int
        Nominal stride of the depthwise ``ConvTranspose1d`` used in ``upsample_st``
        mode (≈ ``Ts / Tt``; default 6).  Ignored for all other modes.
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
        upsample_factor: int = 6,
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

        # Learnable deconvolution upsampler (only built when needed)
        if alignment_mode == "upsample_st":
            self.deconv_up = DeconvUpsampler(
                dim=self._out_dim,
                upsample_factor=upsample_factor,
            )
        else:
            self.deconv_up = None

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
            # a learnable depthwise ConvTranspose1d (deconvolution).
            # Only ST is upsampled so the spatial resolution is fully preserved.
            # After deconv the two aligned streams are element-wise averaged and
            # then mean-pooled over the spatial mask.
            target = s.size(1)                                           # Ts
            st_up  = self.deconv_up(st, target)                          # (B, Ts, D')
            fused  = (s + st_up) * 0.5                                   # (B, Ts, D')
            s_emb  = _mean_pool_masked(fused, s_mask)                    # (B, D')
            # For the other side of the CLIP matrix keep a clean ST embedding
            # (independently pooled, unpolluted by the spatial signal).
            st_emb = _mean_pool_masked(st, st_mask)                      # (B, D')

        else:
            raise ValueError(f"Unknown alignment_mode: {mode}")

        return s_emb, st_emb

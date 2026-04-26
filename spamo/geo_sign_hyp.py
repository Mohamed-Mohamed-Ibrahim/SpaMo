"""
spamo/geo_sign_hyp.py

Self-contained Geo-Sign hyperbolic regularisation module for SPAMO.

Reference:
    Fish & Bowden, "Geo-Sign: Hyperbolic Contrastive Regularisation for
    Geometrically Aware Sign Language Translation", NeurIPS 2025.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import geoopt
    from geoopt.manifolds import PoincareBall
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False
    PoincareBall = None


# ============================================================================
# Internal utilities  (ported verbatim from Geo-Sign models.py)
# ============================================================================

def _trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    with torch.no_grad():
        lo = norm_cdf((a - mean) / std)
        hi = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lo - 1, 2 * hi - 1).erfinv_()
        tensor.mul_(std * math.sqrt(2.0)).add_(mean).clamp_(min=a, max=b)
    return tensor


def weighted_frechet_mean(
    points: torch.Tensor,       # (K, B, D)  – K parts, B batch, D hyp_dim
    weights: torch.Tensor,      # (K,)       – softmax weights per part
    manifold: "PoincareBall",
    max_iter: int = 50,
    tol: float = 1e-5,
) -> torch.Tensor:              # (B, D)
    """
    Iterative weighted Fréchet mean on the Poincaré ball.
    Ported from Geo-Sign weighted_frechet_mean_origin().
    """
    w_norm = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)  # (K,)
    mu = points[0]                                                  # (B, D)
    for _ in range(max_iter):
        log_pts = manifold.logmap(mu.unsqueeze(0), points)          # (K, B, D)
        bar_tan = (w_norm.unsqueeze(-1) * log_pts).sum(dim=0)       # (B, D)
        mu_next = manifold.expmap(mu, bar_tan, project=True)
        if (manifold.dist(mu_next, mu) < tol).all():
            break
        mu = mu_next
    return mu


# ============================================================================
# Core modules (ported / adapted from Geo-Sign HyperbolicProjection &
# HyperbolicContrastiveLoss)
# ============================================================================

class HyperbolicProjection(nn.Module):
    """
    Linear -> scale -> expmap0 onto the Poincaré ball.

    Autocast-safe: linear runs in weight dtype; geoopt ops run in fp32.
    Initialised with trunc_normal (std=0.02) to match Geo-Sign.
    """
    def __init__(self, dim_in: int, dim_out: int, manifold: "PoincareBall"):
        super().__init__()
        if not GEOOPT_AVAILABLE:
            raise ImportError("geoopt is required. Install with: pip install geoopt")
        self.manifold = manifold
        self.proj = nn.Linear(dim_in, dim_out, bias=True)
        # Learnable log-scale (= 0 → scale = 1 at init)
        self.log_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        _trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., dim_in) → (..., dim_out) on Poincaré ball"""
        w_dtype = self.proj.weight.dtype
        y_tan = self.proj(x.to(w_dtype)) * self.log_scale.to(w_dtype).exp()
        # geoopt expmap0 must run in fp32
        return self.manifold.expmap0(y_tan.float(), project=True).to(x.dtype)


class HyperbolicContrastiveLoss(nn.Module):
    """
    Geodesic-distance contrastive loss on the Poincaré ball.

    Learnable temperature (sigmoid-bounded to (0.01, 2.0)) and margin.
    Ported from Geo-Sign HyperbolicContrastiveLoss.pair_loss().
    """
    def __init__(self, manifold: "PoincareBall", label_smoothing: float = 0.1):
        super().__init__()
        if not GEOOPT_AVAILABLE:
            raise ImportError("geoopt is required. Install with: pip install geoopt")
        self.manifold = manifold
        self.temp = nn.Parameter(torch.ones(1))
        self.margin = nn.Parameter(torch.full((1,), 0.3))
        self.loss_fct = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, ignore_index=-100
        )

    def forward(self, pose_hyp: torch.Tensor, text_hyp: torch.Tensor) -> dict:
        """
        pose_hyp, text_hyp: (B, hyp_dim) – both already on the Poincaré ball.
        Returns dict with 'loss' and diagnostic scalars.
        """
        B = pose_hyp.shape[0]
        if B == 0:
            zero = torch.tensor(0.0, device=pose_hyp.device, requires_grad=True)
            return {"loss": zero, "sim_mean": zero.detach(),
                    "margin": self.margin.detach(), "temp": self.temp.detach()}

        # (B, B) pairwise geodesic distances → similarities
        dist = self.manifold.dist(pose_hyp.unsqueeze(1), text_hyp.unsqueeze(0))
        sims = -dist

        tau = torch.sigmoid(self.temp) * 1.99 + 0.01          # ∈ (0.01, 2.0)
        logits = sims / tau

        # Apply off-diagonal margin (push negatives apart)
        eye = torch.eye(B, device=logits.device, dtype=torch.bool)
        logits = logits + self.margin.to(logits.dtype) * (~eye)

        targets = torch.arange(B, device=pose_hyp.device)
        loss = self.loss_fct(logits, targets)

        return {
            "loss": loss,
            "sim_mean": sims.diagonal().mean().detach(),
            "margin": self.margin.detach(),
            "temp": tau.detach(),
        }


# ============================================================================
# HyperbolicRegulariser – the single object SPAMO's t5_slt.py imports
# ============================================================================

class HyperbolicRegulariser(nn.Module):
    """
    Drop-in Geo-Sign hyperbolic regularisation module for SPAMO.

    Wraps:
      - A shared Poincaré manifold (learnable curvature)
      - HyperbolicProjection  for pose features
      - HyperbolicProjection  for text features
      - Weighted Fréchet mean aggregation
      - HyperbolicContrastiveLoss

    Usage in t5_slt.py::

        # __init__
        if self.use_hyperbolic:
            from spamo.geo_sign_hyp import HyperbolicRegulariser
            self.hyp_reg = HyperbolicRegulariser(
                pose_dim   = self.inter_hidden,   # dim after pose_proj
                text_dim   = self.t5_model.config.hidden_size,
                hyp_dim    = hyp_dim,             # e.g. 256
                init_c     = hyp_init_c,          # e.g. 1.0
                label_smoothing = 0.1,
            )

        # shared_step  (before fusion_proj, after prepare_visual_inputs)
        hyp_loss = self.hyp_reg(pose_feats, pose_mask, text_embeds, text_mask)
        loss = t5_loss + self.hyp_alpha * hyp_loss
    """

    def __init__(
        self,
        pose_dim: int,
        text_dim: int,
        hyp_dim: int = 256,
        init_c: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        if not GEOOPT_AVAILABLE:
            raise ImportError(
                "geoopt is required for HyperbolicRegulariser. "
                "Install with: pip install geoopt"
            )

        self.hyp_dim = hyp_dim
        # Shared manifold with learnable curvature – exactly as in Geo-Sign
        self.manifold = PoincareBall(c=init_c, learnable=True)

        # Pose → hyperbolic  (one projector; SPAMO has one unified pose stream)
        self.hyp_proj_pose = HyperbolicProjection(pose_dim, hyp_dim, self.manifold)
        # Text → hyperbolic
        self.hyp_proj_text = HyperbolicProjection(text_dim, hyp_dim, self.manifold)

        self.geom_loss = HyperbolicContrastiveLoss(self.manifold, label_smoothing)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        pose_feats: torch.Tensor,    # (B, T_pose, pose_dim)  – output of pose_proj
        pose_mask: torch.Tensor,     # (B, T_pose) bool, True = valid
        text_embeds: torch.Tensor,   # (B, T_text, text_dim)  – T5 token embeddings
        text_mask: torch.Tensor,     # (B, T_text) bool, True = valid
    ) -> torch.Tensor:
        """
        Returns scalar hyperbolic contrastive loss.
        All geoopt ops run in fp32 regardless of AMP context.
        """
        # ── Pose side ────────────────────────────────────────────────
        # Project each token to the Poincaré ball, then aggregate via
        # distance-weighted Fréchet mean (mirroring Geo-Sign's per-part pooling)
        with torch.cuda.amp.autocast(enabled=False):
            pose_f32 = pose_feats.float()                           # (B, T, D_pose)
            hyp_pose_seq = self.hyp_proj_pose(pose_f32)             # (B, T, hyp_dim)

            # Compute per-token weights: softmax over dist0 (distance from origin)
            # Tokens farther from origin carry more structure → higher weight
            d0 = self.manifold.dist0(hyp_pose_seq)                  # (B, T)
            d0 = d0.masked_fill(~pose_mask, -1e9)
            w = torch.softmax(d0, dim=1)                            # (B, T)

            # Fréchet mean: (T, B, hyp_dim) format expected by helper
            hyp_pose_t = hyp_pose_seq.permute(1, 0, 2)             # (T, B, hyp_dim)
            w_t = w.permute(1, 0)                                   # (T, B)

            mu_pose = _frechet_mean_sequence(
                hyp_pose_t, w_t, self.manifold
            )                                                        # (B, hyp_dim)

            # ── Text side ────────────────────────────────────────────────
            text_f32 = text_embeds.float()                          # (B, T_txt, D_text)
            # Masked mean pool in Euclidean space, then project once
            valid_len = text_mask.float().sum(dim=1, keepdim=True).clamp_min(1)  # (B,1)
            txt_mean = (text_f32 * text_mask.unsqueeze(-1).float()).sum(1) / valid_len
            hyp_text = self.hyp_proj_text(txt_mean)                 # (B, hyp_dim)

            # ── Contrastive loss ─────────────────────────────────────────
            out = self.geom_loss(mu_pose, hyp_text)

        return out["loss"]

    def log_dict(self) -> dict:
        """Call after forward() to get diagnostic metrics for wandb/lightning."""
        return {
            "hyp/curvature": self.manifold.c.abs().item(),
            "hyp/temperature": self.geom_loss.temp.item(),
            "hyp/margin": self.geom_loss.margin.item(),
        }


# ============================================================================
# Private helper (avoids modifying weighted_frechet_mean signature)
# ============================================================================

def _frechet_mean_sequence(
    points: torch.Tensor,   # (T, B, D)
    weights: torch.Tensor,  # (T, B)
    manifold: "PoincareBall",
    max_iter: int = 50,
    tol: float = 1e-5,
) -> torch.Tensor:          # (B, D)
    """
    Weighted Fréchet mean over the sequence dimension (T).
    Each batch element gets its own mean.
    Ported from Geo-Sign weighted_frechet_mean_origin, adapted for (T,B,D).
    """
    # Normalise weights along T for each batch element
    w_norm = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)   # (T, B)
    mu = points[0]                                                   # (B, D)
    for _ in range(max_iter):
        log_pts = manifold.logmap(mu.unsqueeze(0), points)          # (T, B, D)
        bar_tan = (w_norm.unsqueeze(-1) * log_pts).sum(dim=0)       # (B, D)
        mu_next = manifold.expmap(mu, bar_tan, project=True)
        if (manifold.dist(mu_next, mu) < tol).all():
            break
        mu = mu_next
    return mu
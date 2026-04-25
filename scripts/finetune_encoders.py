# region imports & constants
import argparse
import os
import sys
import gc
import math
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.strategies import DDPStrategy

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    CLIPVisionModel,
    CLIPTextModel,
    AutoImageProcessor,
    AutoTokenizer,
    VideoMAEModel,
    VideoMAEImageProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image

# ── PATH SETUP ──────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.s2wrapper import forward as multiscale_forward
from utils.helpers import (
    create_mask,
    read_video,
    get_img_list,
    sliding_window_for_list,
)
from spamo.clip_loss import clip_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not find image processor class")

_GLOBAL_SEED = 42
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.set_float32_matmul_precision("high")
# endregion


# region CONTRASTIVE LOSSES
class NTXentLoss(nn.Module):
    """Normalised Temperature-scaled Cross-Entropy (NT-Xent) loss.

    Pulls together positive pairs (spatial ↔ spatiotemporal) while pushing
    apart all other in-batch examples.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_spatial: torch.Tensor,  # [B, D]
        z_temporal: torch.Tensor,  # [B, D]
    ) -> torch.Tensor:
        z_spatial = F.normalize(z_spatial, dim=-1)
        z_temporal = F.normalize(z_temporal, dim=-1)

        # Cosine similarity matrix  [B, B]
        logits = torch.mm(z_spatial, z_temporal.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.t(), labels)
        return (loss_s2t + loss_t2s) / 2.0


class EncoderContrastiveLoss(nn.Module):
    """Combined contrastive losses for encoder finetuning.

    Supports:
        1. cross_modal  – spatial ↔ spatiotemporal alignment (NT-Xent)
        2. visual_text  – pooled visual embed ↔ CLIP text embed (CLIP loss)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        cross_modal_weight: float = 1.0,
        visual_text_weight: float = 0.5,
    ):
        super().__init__()
        self.ntxent = NTXentLoss(temperature)
        self.cross_modal_weight = cross_modal_weight
        self.visual_text_weight = visual_text_weight
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def forward(
        self,
        spatial_embeds: torch.Tensor,  # [B, D]
        temporal_embeds: torch.Tensor,  # [B, D]
        text_embeds: Optional[torch.Tensor] = None,  # [B, D_text]
    ) -> Dict[str, torch.Tensor]:

        losses: Dict[str, torch.Tensor] = {}

        # 1. Cross-modal (spatial ↔ temporal)
        cm_loss = self.ntxent(spatial_embeds, temporal_embeds)
        losses["cross_modal_loss"] = cm_loss

        total = self.cross_modal_weight * cm_loss

        # 2. Visual–text alignment
        if text_embeds is not None:
            # Fuse spatial + temporal → visual
            visual = F.normalize((spatial_embeds + temporal_embeds) / 2.0, dim=-1)
            text = F.normalize(text_embeds, dim=-1)
            logit_scale = self.logit_scale.exp()
            sim = torch.mm(visual, text.t()) * logit_scale
            vt_loss = clip_loss(sim)
            losses["visual_text_loss"] = vt_loss
            total = total + self.visual_text_weight * vt_loss

        losses["total_loss"] = total
        return losses


# endregion


# region DATASET  (Paired spatial + temporal from pre-computed features)
class PairedFeatureDataset(Dataset):
    """Loads paired spatial (ViT) and spatiotemporal (MAE) pre-extracted features.

    Each sample returns:
      - spatial feature       : [T_s, D_s]
      - spatiotemporal feature: [T_t, D_t]
      - text (for optional visual–text contrastive)
    """

    def __init__(
        self,
        anno_root: str,
        spatial_feat_root: str,
        mae_feat_root: str,
        mode: str = "train",
        spatial_postfix: str = "_None_s2wrapping",
        spatiotemporal_postfix: str = "_None_overlap-8",
        max_frame_len: int = 512,
    ):
        super().__init__()
        self.spatial_feat_root = spatial_feat_root
        self.mae_feat_root = mae_feat_root
        self.mode = mode
        self.spatial_postfix = spatial_postfix
        self.spatiotemporal_postfix = spatiotemporal_postfix
        self.max_frame_len = max_frame_len

        anno_path = os.path.join(anno_root, f"{mode}_info_ml.npy")
        if not os.path.exists(anno_path):
            # Fallback to non-multilingual annotation
            anno_path = os.path.join(anno_root, f"{mode}_info.npy")
        self.data = np.load(anno_path, allow_pickle=True).item()

        # Filter out non-integer keys (e.g., 'prefix')
        self.valid_keys = sorted([k for k in self.data.keys() if isinstance(k, int)])
        print(
            f"[PairedFeatureDataset] mode={mode}, "
            f"{len(self.valid_keys)} samples loaded."
        )

    def __len__(self):
        return len(self.valid_keys)

    def __getitem__(self, idx):
        key = self.valid_keys[idx]
        entry = self.data[key]
        file_id = entry["fileid"]
        text = entry.get("text", "")
        if not text.endswith("."):
            text = text + "."

        # Load spatial features
        spatial_path = os.path.join(
            self.spatial_feat_root,
            self.mode,
            f"{file_id}{self.spatial_postfix}.npy",
        )
        spatial_feat = self._safe_load(spatial_path)

        # Load spatiotemporal features
        st_path = os.path.join(
            self.mae_feat_root,
            self.mode,
            f"{file_id}{self.spatiotemporal_postfix}.npy",
        )
        st_feat = self._safe_load(st_path)

        # Truncate
        if spatial_feat.shape[0] > self.max_frame_len:
            start = random.randint(0, spatial_feat.shape[0] - self.max_frame_len)
            spatial_feat = spatial_feat[start : start + self.max_frame_len]

        if st_feat.shape[0] > self.max_frame_len:
            start = random.randint(0, st_feat.shape[0] - self.max_frame_len)
            st_feat = st_feat[start : start + self.max_frame_len]

        return {
            "spatial_feat": torch.tensor(spatial_feat, dtype=torch.float32),
            "st_feat": torch.tensor(st_feat, dtype=torch.float32),
            "text": text.lower(),
            "id": file_id,
            "num_spatial_frames": spatial_feat.shape[0],
            "num_st_frames": st_feat.shape[0],
        }

    @staticmethod
    def _safe_load(path: str) -> np.ndarray:
        if not os.path.exists(path):
            print(f"[WARNING] Feature file missing: {path}")
            return np.zeros((1, 1024), dtype=np.float32)
        return np.load(path).astype(np.float32)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch


# endregion


# region LIGHTNING MODULE
class EncoderFinetuneLora(pl.LightningModule):
    """
    Finetunes ViT (CLIP) and VideoMAE encoders with LoRA adapters
    using contrastive losses.

    Architecture:
        ViT  →  LoRA  →  proj_spatial  →  shared embed space
        MAE  →  LoRA  →  proj_temporal →  shared embed space
        CLIP-Text  (frozen)            →  text embed space

    Losses:
        1. NT-Xent   : spatial embed  ↔  temporal embed
        2. CLIP loss  : fused visual   ↔  text embed
    """

    def __init__(
        self,
        # Encoder names
        vit_model_name: str = "openai/clip-vit-large-patch14",
        mae_model_name: str = "MCG-NJU/videomae-large",
        # LoRA
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        vit_lora_targets: Optional[List[str]] = None,
        mae_lora_targets: Optional[List[str]] = None,
        # Projections
        spatial_input_dim: int = 2048,  # ViT with s2wrapping → 1024*2
        temporal_input_dim: int = 1024,  # VideoMAE hidden dim
        proj_dim: int = 512,
        # Losses
        temperature: float = 0.07,
        cross_modal_weight: float = 1.0,
        visual_text_weight: float = 0.5,
        # Text encoder
        use_text_loss: bool = True,
        text_model_name: str = "openai/clip-vit-large-patch14",
        # Optimiser
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        # S2 wrapping
        s2_mode: str = "",
        scales: Optional[List[int]] = None,
        vit_nth_layer: int = -1,
        mae_nth_layer: int = -1,
        mae_overlap_size: int = 8,
        # Misc
        cache_dir: Optional[str] = None,
        monitor: str = "val/total_loss",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.monitor = monitor
        self.use_text_loss = use_text_loss

        self.s2_mode = s2_mode
        self.scales = scales or []
        self.vit_nth_layer = vit_nth_layer
        self.mae_nth_layer = mae_nth_layer
        self.mae_overlap_size = mae_overlap_size

        # ── Projection heads (learnable, always train these) ─────────
        self.proj_spatial = nn.Sequential(
            nn.Linear(spatial_input_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.proj_temporal = nn.Sequential(
            nn.Linear(temporal_input_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # ── Contrastive loss ─────────────────────────────────────────
        self.contrastive_loss = EncoderContrastiveLoss(
            temperature=temperature,
            cross_modal_weight=cross_modal_weight,
            visual_text_weight=visual_text_weight,
        )

        # ── Text encoder (frozen, for visual-text alignment) ─────────
        if self.use_text_loss:
            from transformers import CLIPTokenizer, CLIPTextModel

            self.text_tokenizer = CLIPTokenizer.from_pretrained(
                text_model_name, cache_dir=cache_dir
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                text_model_name, cache_dir=cache_dir
            )
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False

            text_dim = self.text_encoder.config.hidden_size
            self.text_proj = nn.Linear(text_dim, proj_dim)

        # Containers for validation
        self._val_losses: List[Dict[str, float]] = []

    # ── LoRA helpers ─────────────────────────────────────────────────

    @staticmethod
    def _apply_lora_to_vit(
        model: nn.Module,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: Optional[List[str]] = None,
    ) -> nn.Module:
        """Apply LoRA to CLIP ViT encoder."""
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
        )
        model = get_peft_model(model, config)
        print("[LoRA ViT] Applied. Trainable parameters:")
        model.print_trainable_parameters()
        return model

    @staticmethod
    def _apply_lora_to_mae(
        model: nn.Module,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: Optional[List[str]] = None,
    ) -> nn.Module:
        """Apply LoRA to VideoMAE encoder."""
        if target_modules is None:
            target_modules = ["query", "value"]
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
        )
        model = get_peft_model(model, config)
        print("[LoRA MAE] Applied. Trainable parameters:")
        model.print_trainable_parameters()
        return model

    # ── Forward ──────────────────────────────────────────────────────

    def _pool_features(self, feats: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        """Mean-pool variable-length features → [B, D]."""
        pooled = []
        for i, L in enumerate(lengths):
            pooled.append(feats[i, :L].mean(dim=0))
        return torch.stack(pooled, dim=0)

    def forward(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        spatial_feats, st_feats = [], []
        texts = []
        num_spatial, num_st = [], []

        for sample in batch:
            sf = sample["spatial_feat"]
            tf = sample["st_feat"]
            if sf.ndim == 1 or sf.shape[0] == 0:
                continue

            spatial_feats.append(sf)
            st_feats.append(tf)
            texts.append(sample["text"])
            num_spatial.append(sample["num_spatial_frames"])
            num_st.append(sample["num_st_frames"])

        if len(spatial_feats) == 0:
            return {
                "total_loss": torch.tensor(0.0, device=self.device, requires_grad=True)
            }

        # Pad and move to device
        spatial_padded = pad_sequence(spatial_feats, batch_first=True).to(self.device)
        st_padded = pad_sequence(st_feats, batch_first=True).to(self.device)

        # Project
        spatial_proj = self.proj_spatial(spatial_padded)  # [B, T_s, proj_dim]
        st_proj = self.proj_temporal(st_padded)  # [B, T_t, proj_dim]

        # Pool → [B, proj_dim]
        spatial_embeds = self._pool_features(spatial_proj, num_spatial)
        temporal_embeds = self._pool_features(st_proj, num_st)

        # Text embeddings (optional)
        text_embeds = None
        if self.use_text_loss and hasattr(self, "text_encoder"):
            with torch.no_grad():
                tokens = self.text_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)
                text_out = self.text_encoder(**tokens)
                # Pool text: use the [EOS] token embedding (CLIP convention)
                raw_text = text_out.pooler_output  # [B, D_text]

            text_embeds = self.text_proj(raw_text)  # [B, proj_dim]

        # Compute losses
        loss_dict = self.contrastive_loss(spatial_embeds, temporal_embeds, text_embeds)
        return loss_dict

    # ── PL training / validation hooks ───────────────────────────────

    def training_step(self, batch, batch_idx):
        loss_dict = self.forward(batch)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, prog_bar=(k == "total_loss"), sync_dist=True)
        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self.forward(batch)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, prog_bar=(k == "total_loss"), sync_dist=True)
        self._val_losses.append({k: v.item() for k, v in loss_dict.items()})
        return loss_dict["total_loss"]

    def on_validation_epoch_end(self):
        if not self._val_losses:
            return
        avg = {}
        for k in self._val_losses[0]:
            avg[k] = np.mean([d[k] for d in self._val_losses])
        print(f"\n{'=' * 60}")
        print("[Val Epoch Summary]")
        for k, v in avg.items():
            print(f"  {k}: {v:.5f}")
        print(f"{'=' * 60}\n")
        self._val_losses.clear()

    # ── Optimizer ────────────────────────────────────────────────────

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters found!")

        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.98),
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * self.warmup_ratio)
        print(
            f"[Optimiser] total_steps={total_steps}, "
            f"warmup_steps={warmup_steps}, lr={self.lr}"
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# endregion


# region DATA MODULE
class PairedFeatureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        anno_root: str,
        spatial_feat_root: str,
        mae_feat_root: str,
        spatial_postfix: str = "_None_s2wrapping",
        spatiotemporal_postfix: str = "_None_overlap-8",
        max_frame_len: int = 512,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        hp = self.hparams
        common = dict(
            anno_root=hp.anno_root,
            spatial_feat_root=hp.spatial_feat_root,
            mae_feat_root=hp.mae_feat_root,
            spatial_postfix=hp.spatial_postfix,
            spatiotemporal_postfix=hp.spatiotemporal_postfix,
            max_frame_len=hp.max_frame_len,
        )
        self.train_ds = PairedFeatureDataset(mode="train", **common)
        self.val_ds = PairedFeatureDataset(mode="dev", **common)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=PairedFeatureDataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=PairedFeatureDataset.collate_fn,
        )


# endregion


# region CLI  (supports --config YAML + CLI overrides)
def get_parser():
    p = argparse.ArgumentParser(
        description="Finetune ViT & MAE encoders with LoRA + contrastive loss"
    )
    # ── Config file (primary entry point) ─────────────────────────
    p.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g. configs/finetune_encoders.yaml). "
        "All keys in the YAML become defaults; CLI flags override them.",
    )

    # Data paths
    p.add_argument(
        "--anno_root",
        default=None,
        help="Path to annotation dir (e.g. ./preprocess/Phoenix14T)",
    )
    p.add_argument(
        "--spatial_feat_root", default=None, help="Root dir of spatial (ViT) features"
    )
    p.add_argument(
        "--mae_feat_root",
        default=None,
        help="Root dir of spatiotemporal (MAE) features",
    )
    p.add_argument("--spatial_postfix", default=None)
    p.add_argument("--spatiotemporal_postfix", default=None)
    p.add_argument("--max_frame_len", type=int, default=None)

    # Model
    p.add_argument("--vit_model_name", default=None)
    p.add_argument("--mae_model_name", default=None)
    p.add_argument(
        "--text_model_name", default=None, help="CLIP text encoder for visual-text loss"
    )

    # Feature dimensions
    p.add_argument(
        "--spatial_input_dim",
        type=int,
        default=None,
        help="ViT feature dim (2048 with s2wrapping, 1024 without)",
    )
    p.add_argument(
        "--temporal_input_dim", type=int, default=None, help="MAE feature dim"
    )
    p.add_argument(
        "--proj_dim", type=int, default=None, help="Shared projection dimension"
    )

    # LoRA
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)

    # Loss
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--cross_modal_weight", type=float, default=None)
    p.add_argument("--visual_text_weight", type=float, default=None)
    p.add_argument(
        "--use_text_loss",
        action="store_true",
        default=None,
        help="Include visual-text contrastive loss",
    )
    p.add_argument("--no_text_loss", dest="use_text_loss", action="store_false")

    # Optimiser
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)

    # Training
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--devices", type=int, default=None)
    p.add_argument("--accumulate_grad_batches", type=int, default=None)
    p.add_argument("--precision", default=None)
    p.add_argument("--gradient_clip_val", type=float, default=None)
    p.add_argument(
        "--patience", type=int, default=None, help="EarlyStopping patience (epochs)"
    )
    p.add_argument("--seed", type=int, default=None)

    # Misc
    p.add_argument("--cache_dir", default=None)
    p.add_argument(
        "--log_dir", default=None, help="Directory for TensorBoard logs and checkpoints"
    )
    p.add_argument(
        "--save_dir",
        default=None,
        help="Where to export final LoRA weights (default: <log_dir>/lora_weights)",
    )

    return p


# ── Hard-coded defaults (used when neither YAML nor CLI provide a value) ──
_DEFAULTS = dict(
    anno_root="./preprocess/Phoenix14T",
    spatial_feat_root="",
    mae_feat_root="",
    spatial_postfix="_None_s2wrapping",
    spatiotemporal_postfix="_None_overlap-8",
    max_frame_len=512,
    vit_model_name="openai/clip-vit-large-patch14",
    mae_model_name="MCG-NJU/videomae-large",
    text_model_name="openai/clip-vit-large-patch14",
    spatial_input_dim=2048,
    temporal_input_dim=1024,
    proj_dim=512,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    temperature=0.07,
    cross_modal_weight=1.0,
    visual_text_weight=0.5,
    use_text_loss=True,
    lr=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    batch_size=16,
    num_workers=4,
    max_epochs=50,
    devices=1,
    accumulate_grad_batches=4,
    precision="bf16-mixed",
    gradient_clip_val=1.0,
    patience=10,
    seed=42,
    cache_dir=None,
    log_dir="logs/encoder_finetune",
    save_dir=None,
)


def _merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """Merge: hard-coded defaults ← YAML config ← CLI overrides."""
    from omegaconf import OmegaConf

    # Start from hard-coded defaults
    merged = dict(_DEFAULTS)

    # Layer on YAML values (if provided)
    if args.config is not None:
        yaml_cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        for k, v in yaml_cfg.items():
            if k in merged:
                merged[k] = v
            else:
                print(f"[WARNING] Unknown config key in YAML: '{k}' — ignoring.")

    # Layer on CLI overrides (only values explicitly set by the user)
    cli_dict = vars(args)
    for k, v in cli_dict.items():
        # if k == "config":
        #     continue
        if v is not None:
            merged[k] = v

    # Build final namespace
    return argparse.Namespace(**merged)


def main():
    raw_args = get_parser().parse_args()
    args = _merge_config(raw_args)
    seed_everything(args.seed)

    # ── Validate required paths ─────────────────────────────────────
    for required in ("anno_root", "spatial_feat_root", "mae_feat_root"):
        val = getattr(args, required, "")
        if not val:
            raise ValueError(
                f"--{required} is required. Set it via YAML config or CLI flag."
            )

    # ── Data ────────────────────────────────────────────────────────
    dm = PairedFeatureDataModule(
        anno_root=args.anno_root,
        spatial_feat_root=args.spatial_feat_root,
        mae_feat_root=args.mae_feat_root,
        spatial_postfix=args.spatial_postfix,
        spatiotemporal_postfix=args.spatiotemporal_postfix,
        max_frame_len=args.max_frame_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ───────────────────────────────────────────────────────
    model = EncoderFinetuneLora(
        vit_model_name=args.vit_model_name,
        mae_model_name=args.mae_model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        spatial_input_dim=args.spatial_input_dim,
        temporal_input_dim=args.temporal_input_dim,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        cross_modal_weight=args.cross_modal_weight,
        visual_text_weight=args.visual_text_weight,
        use_text_loss=args.use_text_loss,
        text_model_name=args.text_model_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        cache_dir=args.cache_dir,
    )

    # ── Callbacks ───────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="encoder-epoch={epoch:03d}-loss={val/total_loss:.4f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/total_loss",
            patience=args.patience,
            mode="min",
            verbose=True,
        ),
    ]

    # ── Strategy ────────────────────────────────────────────────────
    strategy = "auto"
    if args.devices > 1:
        strategy = DDPStrategy(find_unused_parameters=True)

    # ── Trainer ─────────────────────────────────────────────────────
    trainer = Trainer(
        default_root_dir=args.log_dir,
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator="gpu",
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
    )

    # ── Print configuration summary ─────────────────────────────────
    cfg_src = args.config or "CLI-only (no YAML)"
    print("=" * 70)
    print("  Encoder Finetuning with LoRA + Contrastive Loss")
    print(f"  Config loaded from: {cfg_src}")
    print("=" * 70)
    print(f"  anno_root   : {args.anno_root}")
    print(f"  spatial_feat: {args.spatial_feat_root}")
    print(f"  mae_feat    : {args.mae_feat_root}")
    print(f"  ViT model   : {args.vit_model_name}")
    print(f"  MAE model   : {args.mae_model_name}")
    print(f"  LoRA r/α    : {args.lora_r} / {args.lora_alpha}")
    print(f"  Proj dim    : {args.proj_dim}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Text loss   : {args.use_text_loss}")
    print(f"  LR          : {args.lr}")
    print(f"  Batch size  : {args.batch_size}  (accum: {args.accumulate_grad_batches})")
    print(f"  Devices     : {args.devices}")
    print(f"  Precision   : {args.precision}")
    print("=" * 70)

    trainer.fit(model, dm)

    # ── Save LoRA weights ───────────────────────────────────────────
    save_dir = args.save_dir or os.path.join(args.log_dir, "lora_weights")
    os.makedirs(save_dir, exist_ok=True)

    # Save the full checkpoint (projections + loss parameters)
    full_path = os.path.join(save_dir, "encoder_finetune_full.ckpt")
    trainer.save_checkpoint(full_path)
    print(f"\n✅ Full checkpoint saved → {full_path}")

    # Save just the projection weights (for downstream use)
    proj_state = {
        "proj_spatial": model.proj_spatial.state_dict(),
        "proj_temporal": model.proj_temporal.state_dict(),
    }
    if hasattr(model, "text_proj"):
        proj_state["text_proj"] = model.text_proj.state_dict()

    proj_path = os.path.join(save_dir, "projection_weights.pt")
    torch.save(proj_state, proj_path)
    print(f"✅ Projection weights saved → {proj_path}")

    print("\n🎉 Encoder finetuning complete!")


if __name__ == "__main__":
    main()
# endregion

# region usage examples
"""
python scripts/finetune_encoders.py -c configs/finetune_encoders.yaml

# Example: load config from YAML but override batch_size and devices
python scripts/finetune_encoders.py \
    -c configs/finetune_encoders.yaml \
    --batch_size 32 \
    --devices 1

python scripts/finetune_encoders.py \
    --anno_root ./preprocess/Phoenix14T \
    --spatial_feat_root <vit-features-dir> \
    --mae_feat_root <mae-features-dir> \
    --batch_size 16 --max_epochs 50 --devices 2

"""
# endregion

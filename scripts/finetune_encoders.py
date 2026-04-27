"""
Finetune ViT (CLIP) and VideoMAE encoders using LoRA + Contrastive Loss.

This script loads raw video frames, passes them through the ViT and MAE
encoders (with LoRA adapters applied), projects the outputs into a shared
embedding space, and optimises using contrastive losses:
  - NT-Xent  : spatial (ViT) ↔ spatiotemporal (MAE)  alignment
  - CLIP loss: fused visual   ↔ text                  alignment

After training, LoRA adapters are exported as PEFT directories that can
be loaded directly by vit_extract_feature.py and mae_extract_feature.py
via ``PeftModel.from_pretrained()``.

Usage:
    python scripts/finetune_encoders.py -c configs/finetune_encoders.yaml
"""

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
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from torch.utils.data import Dataset, DataLoader

from transformers import (
    CLIPVisionModel,
    CLIPTextModel,
    AutoImageProcessor,
    AutoTokenizer,
    VideoMAEModel,
    VideoMAEImageProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
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


# region LOSSES
# ═══════════════════════════════════════════════════════════════════════════
#  CONTRASTIVE LOSSES
# ═══════════════════════════════════════════════════════════════════════════


class NTXentLoss(nn.Module):
    """Normalised Temperature-scaled Cross-Entropy (NT-Xent) loss."""

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
        logits = torch.mm(z_spatial, z_temporal.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.t(), labels)
        return (loss_s2t + loss_t2s) / 2.0


class EncoderContrastiveLoss(nn.Module):
    """Combined contrastive losses for encoder finetuning."""

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
        spatial_embeds: torch.Tensor,
        temporal_embeds: torch.Tensor,
        text_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        cm_loss = self.ntxent(spatial_embeds, temporal_embeds)
        losses["cross_modal_loss"] = cm_loss
        total = self.cross_modal_weight * cm_loss

        if text_embeds is not None:
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


# region DATASET
# ═══════════════════════════════════════════════════════════════════════════
#  DATASET  (Raw video frames — same loading logic as extraction scripts)
# ═══════════════════════════════════════════════════════════════════════════


class VideoFrameDataset(Dataset):
    """Loads raw video frames for paired ViT + MAE finetuning.

    Each sample returns:
        - pil_frames  : List[PIL.Image]  (all frames of the video)
        - text        : str
        - file_id     : str
    """

    RESIZE_SIZE = 256
    CROP_SIZE = 224

    def __init__(
        self,
        anno_root: str,
        video_root: str,
        mode: str = "train",
        max_frames: int = 64,
    ):
        super().__init__()
        self.video_root = video_root
        self.mode = mode
        self.max_frames = max_frames

        anno_path = os.path.join(anno_root, f"{mode}_info_ml.npy")
        if not os.path.exists(anno_path):
            anno_path = os.path.join(anno_root, f"{mode}_info.npy")
        self.data = np.load(anno_path, allow_pickle=True).item()
        self.ds_name = os.path.split(anno_root)[-1]

        # Filter non-integer keys
        self.valid_keys = sorted([k for k in self.data.keys() if isinstance(k, int)])
        print(
            f"[VideoFrameDataset] mode={mode}, {len(self.valid_keys)} samples loaded."
        )

    def __len__(self):
        return len(self.valid_keys)

    def _center_crop(self, img: Image.Image) -> Image.Image:
        """Resize + center crop to 224x224 (for MAE)."""
        img = img.resize((self.RESIZE_SIZE, self.RESIZE_SIZE), resample=Image.BILINEAR)
        left = (self.RESIZE_SIZE - self.CROP_SIZE) // 2
        top = (self.RESIZE_SIZE - self.CROP_SIZE) // 2
        return img.crop((left, top, left + self.CROP_SIZE, top + self.CROP_SIZE))

    def __getitem__(self, idx):
        key = self.valid_keys[idx]
        entry = self.data[key]
        fname = entry["folder"]
        file_id = entry["fileid"]
        text = entry.get("text", "")
        if not text.endswith("."):
            text = text + "."

        # ── Load frames (same logic as extraction scripts) ───────
        pil_frames = []
        if self.ds_name in ["Phoenix14T", "CSL-Daily"]:
            image_paths = get_img_list(self.ds_name, self.video_root, fname)
            for p in image_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    pil_frames.append(img)
                except Exception as e:
                    print(f"[WARNING] Failed to load {p}: {e}")

        elif self.ds_name in ["How2Sign", "Phoenix14TCompressed"]:
            s = entry.get("original_info", {}).get("START_REALIGNED", None)
            e = entry.get("original_info", {}).get("END_REALIGNED", None)
            try:
                s = float(s) if s is not None else None
            except (ValueError, TypeError):
                s = None
            try:
                e = float(e) if e is not None else None
            except (ValueError, TypeError):
                e = None
            raw_frames = read_video(fname, start_time=s, end_time=e)
            for f in raw_frames:
                if isinstance(f, np.ndarray):
                    pil_frames.append(Image.fromarray(f).convert("RGB"))
                elif isinstance(f, Image.Image):
                    pil_frames.append(f.convert("RGB"))
        else:
            raise NotImplementedError(f"Unknown dataset: {self.ds_name}")

        if len(pil_frames) == 0:
            return None  # Will be filtered in collate_fn

        # ── Sub-sample if too many frames ────────────────────────
        if len(pil_frames) > self.max_frames:
            # Uniform sub-sample to max_frames
            indices = np.linspace(0, len(pil_frames) - 1, self.max_frames, dtype=int)
            pil_frames = [pil_frames[i] for i in indices]

        # ── Create MAE-cropped copies ────────────────────────────
        mae_frames = [self._center_crop(f) for f in pil_frames]

        return {
            "pil_frames": pil_frames,  # Original size for ViT
            "mae_frames": mae_frames,  # 224x224 cropped for MAE
            "text": text.lower(),
            "file_id": file_id,
            "n_frames": len(pil_frames),
        }

    @staticmethod
    def collate_fn(batch):
        """Filter out None samples and return as a list of dicts."""
        return [s for s in batch if s is not None]


# endregion


# region LIGHTNING MODULE
# ═══════════════════════════════════════════════════════════════════════════
#  LIGHTNING MODULE
# ═══════════════════════════════════════════════════════════════════════════


class EncoderFinetuneLora(pl.LightningModule):
    """
    Finetunes ViT (CLIP) and VideoMAE encoders with LoRA adapters
    using contrastive losses on raw video frames.

    Architecture:
        Raw frames → ViT (+ LoRA) → [CLS] → proj_spatial  → spatial_embed
        Raw frames → MAE (+ LoRA) → [CLS] → proj_temporal → temporal_embed
        Text       → CLIP-Text (frozen)   → text_proj     → text_embed

    Forward pass extracts features from the actual encoders so LoRA
    gradients flow back through the attention layers.
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
        # ViT feature extraction settings
        s2_mode: str = "",
        scales: Optional[List[int]] = None,
        vit_nth_layer: int = -1,
        # MAE feature extraction settings
        mae_nth_layer: int = -1,
        mae_overlap_size: int = 8,
        # Optimiser
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        # Misc
        cache_dir: Optional[str] = None,
        monitor: str = "val/total_loss",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.monitor = monitor
        self.use_text_loss = use_text_loss
        self.verbose = verbose

        self.s2_mode = s2_mode
        self.scales = scales or []
        self.vit_nth_layer = vit_nth_layer
        self.mae_nth_layer = mae_nth_layer
        self.mae_overlap_size = mae_overlap_size

        # ── Load ViT encoder + apply LoRA ────────────────────────────
        print("[EncoderFinetuneLora] Loading ViT encoder...")
        self.vit_encoder = CLIPVisionModel.from_pretrained(
            vit_model_name,
            output_hidden_states=True,
            cache_dir=cache_dir,
            use_safetensors=False,
        )
        for p in self.vit_encoder.parameters():
            p.requires_grad = False
        self.vit_encoder = self._apply_lora(
            self.vit_encoder,
            lora_r,
            lora_alpha,
            lora_dropout,
            vit_lora_targets or ["q_proj", "v_proj"],
            name="ViT",
        )

        # ── Load MAE encoder + apply LoRA ────────────────────────────
        print("[EncoderFinetuneLora] Loading MAE encoder...")
        self.mae_encoder = VideoMAEModel.from_pretrained(
            mae_model_name, cache_dir=cache_dir, use_safetensors=False
        )
        for p in self.mae_encoder.parameters():
            p.requires_grad = False
        self.mae_encoder = self._apply_lora(
            self.mae_encoder,
            lora_r,
            lora_alpha,
            lora_dropout,
            mae_lora_targets or ["query", "value"],
            name="MAE",
        )
        
        # ── Enable Gradient Checkpointing ────────────────────────────
        self.vit_encoder.gradient_checkpointing_enable()
        self.mae_encoder.gradient_checkpointing_enable()

        # ── Image processors (needed to preprocess raw frames) ───────
        self.vit_image_processor = AutoImageProcessor.from_pretrained(
            vit_model_name, cache_dir=cache_dir
        )
        self.mae_image_processor = VideoMAEImageProcessor.from_pretrained(
            mae_model_name, cache_dir=cache_dir
        )

        # ── Projection heads ─────────────────────────────────────────
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

        # ── Text encoder (frozen) ────────────────────────────────────
        if self.use_text_loss:
            from transformers import CLIPTokenizer

            self.text_tokenizer = CLIPTokenizer.from_pretrained(
                text_model_name, cache_dir=cache_dir
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                text_model_name, cache_dir=cache_dir, use_safetensors=False
            )
            self.text_encoder.to("cpu")  # Keep text encoder on CPU to save VRAM
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False

            text_dim = self.text_encoder.config.hidden_size
            self.text_proj = nn.Linear(text_dim, proj_dim)

        self._val_losses: List[Dict[str, float]] = []

    # ── LoRA helper ──────────────────────────────────────────────────

    @staticmethod
    def _apply_lora(
        model: nn.Module,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: List[str],
        name: str = "",
    ) -> nn.Module:
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
        )
        model = get_peft_model(model, config)
        print(f"[LoRA {name}] Applied. Trainable parameters:")
        model.print_trainable_parameters()
        return model

    def save_lora_adapters(self, save_dir: str) -> None:
        """Save LoRA adapters as PEFT adapter directories.

        Creates:
            <save_dir>/vit_lora/  and  <save_dir>/mae_lora/
        """
        vit_dir = os.path.join(save_dir, "vit_lora")
        mae_dir = os.path.join(save_dir, "mae_lora")
        os.makedirs(vit_dir, exist_ok=True)
        os.makedirs(mae_dir, exist_ok=True)

        self.vit_encoder.save_pretrained(vit_dir)
        print(f"✅ ViT LoRA adapter saved → {vit_dir}")

        self.mae_encoder.save_pretrained(mae_dir)
        print(f"✅ MAE LoRA adapter saved → {mae_dir}")

    # ── Encoder feature extraction ───────────────────────────────────

    def _extract_vit_features(
        self, pil_frames: List[Image.Image], chunk_size: int = 1
    ) -> torch.Tensor:
        """Run frames through ViT encoder in chunks to save memory."""
        all_cls = []
        
        for i in range(0, len(pil_frames), chunk_size):
            chunk = pil_frames[i : i + chunk_size]
            pixel_values = self.vit_image_processor(
                chunk, return_tensors="pt"
            ).pixel_values.to(self.device)

            if self.s2_mode == "s2wrapping":
                def _vit_forward(inputs):
                    return self.vit_encoder(inputs).hidden_states[self.vit_nth_layer]

                outputs = multiscale_forward(
                    _vit_forward,
                    pixel_values,
                    scales=self.scales,
                    num_prefix_token=1,
                )
            else:
                outputs = self.vit_encoder(pixel_values).hidden_states[self.vit_nth_layer]
            
            all_cls.append(outputs[:, 0])  # [chunk_size, D_vit]

        return torch.cat(all_cls, dim=0)  # [N_frames, D_vit]

    def _extract_mae_features(
        self, mae_frames: List[Image.Image], chunk_size: int = 1
    ) -> torch.Tensor:
        """Run frames through MAE encoder in chunks of clips."""
        if len(mae_frames) < 16:
            mae_frames = mae_frames + [mae_frames[-1]] * (16 - len(mae_frames))

        clips = sliding_window_for_list(mae_frames, 16, self.mae_overlap_size)
        all_feats = []
        
        for i in range(0, len(clips), chunk_size):
            chunk = clips[i : i + chunk_size]
            inputs = self.mae_image_processor(images=chunk, return_tensors="pt").to(
                self.device
            )

            outputs = self.mae_encoder(
                **inputs, output_hidden_states=True
            ).hidden_states[self.mae_nth_layer]

            all_feats.append(outputs[:, 0])  # [chunk_size, D_mae]

        return torch.cat(all_feats, dim=0)  # [N_clips, D_mae]

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process a batch of raw video samples through both encoders."""
        spatial_embeds_list = []
        temporal_embeds_list = []
        texts = []

        for sample in batch:
            pil_frames = sample["pil_frames"]
            mae_frames = sample["mae_frames"]

            # ── ViT: extract per-frame [CLS] → mean pool → project ─
            vit_feats = self._extract_vit_features(pil_frames)  # [N, D_vit]
            vit_pooled = vit_feats.mean(dim=0, keepdim=True)  # [1, D_vit]
            spatial_embed = self.proj_spatial(vit_pooled)  # [1, proj_dim]
            spatial_embeds_list.append(spatial_embed.squeeze(0))

            # Free cache before temporal pass
            torch.cuda.empty_cache()

            # ── MAE: extract per-clip [CLS] → mean pool → project ──
            mae_feats = self._extract_mae_features(mae_frames)  # [N_clips, D_mae]
            mae_pooled = mae_feats.mean(dim=0, keepdim=True)  # [1, D_mae]
            temporal_embed = self.proj_temporal(mae_pooled)  # [1, proj_dim]
            temporal_embeds_list.append(temporal_embed.squeeze(0))

            # Free cache before next sample or loss
            torch.cuda.empty_cache()

            texts.append(sample["text"])

            if self.verbose:
                print(
                    f"[Verbose] sample={sample['file_id']} | "
                    f"vit_feats={tuple(vit_feats.shape)} -> "
                    f"vit_pooled={tuple(vit_pooled.shape)} -> "
                    f"spatial_embed={tuple(spatial_embed.shape)}"
                )
                print(
                    f"[Verbose] sample={sample['file_id']} | "
                    f"mae_feats={tuple(mae_feats.shape)} -> "
                    f"mae_pooled={tuple(mae_pooled.shape)} -> "
                    f"temporal_embed={tuple(temporal_embed.shape)}"
                )

        if len(spatial_embeds_list) == 0:
            return {
                "total_loss": torch.tensor(0.0, device=self.device, requires_grad=True)
            }

        spatial_embeds = torch.stack(spatial_embeds_list)  # [B, proj_dim]
        temporal_embeds = torch.stack(temporal_embeds_list)  # [B, proj_dim]

        if self.verbose:
            print(
                f"[Verbose] batch spatial_embeds={tuple(spatial_embeds.shape)} "
                f"temporal_embeds={tuple(temporal_embeds.shape)}"
            )

        # ── Text embeddings (optional) ───────────────────────────
        text_embeds = None
        if self.use_text_loss and hasattr(self, "text_encoder"):
            # Force text encoder to CPU if Lightning moved it to GPU
            if next(self.text_encoder.parameters()).device.type != "cpu":
                self.text_encoder.to("cpu")

            with torch.no_grad():
                tokens = self.text_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to("cpu")
                raw_text = self.text_encoder(**tokens).pooler_output

            # Move raw_text to GPU for projection (proj head is small)
            text_embeds = self.text_proj(raw_text.to(self.device))  # [B, proj_dim]
            if self.verbose:
                print(f"[Verbose] text_embeds={tuple(text_embeds.shape)}")

        return self.contrastive_loss(spatial_embeds, temporal_embeds, text_embeds)

    # ── PL hooks ─────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        loss_dict = self.forward(batch)
        batch_size = len(batch)
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                prog_bar=(k == "total_loss"),
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self.forward(batch)
        batch_size = len(batch)
        for k, v in loss_dict.items():
            self.log(
                f"val/{k}",
                v,
                prog_bar=(k == "total_loss"),
                sync_dist=True,
                batch_size=batch_size,
            )
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

        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
        # This is the optimized version for CPU Offloading
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(
                trainable,
                lr=self.lr,
                weight_decay=self.weight_decay,
                eps=1e-8,
                betas=(0.9, 0.98),
            )
        else:
            # Fallback for standard DDP or single GPU
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
# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODULE
# ═══════════════════════════════════════════════════════════════════════════


class VideoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        anno_root: str,
        video_root: str,
        max_frames: int = 64,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        hp = self.hparams
        common = dict(
            anno_root=hp.anno_root,
            video_root=hp.video_root,
            max_frames=hp.max_frames,
        )
        self.train_ds = VideoFrameDataset(mode="train", **common)
        self.val_ds = VideoFrameDataset(mode="dev", **common)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=VideoFrameDataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=VideoFrameDataset.collate_fn,
        )


# endregion


# region CLI & CONFIG
# ═══════════════════════════════════════════════════════════════════════════
#  CLI  (supports --config YAML + CLI overrides)
# ═══════════════════════════════════════════════════════════════════════════


def get_parser():
    p = argparse.ArgumentParser(
        description="Finetune ViT & MAE encoders with LoRA + contrastive loss"
    )
    p.add_argument(
        "-c", "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Data paths
    p.add_argument("--anno_root", default=None)
    p.add_argument("--video_root", default=None, help="Root dir of raw video frames")
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames per video (sub-sampled uniformly)",
    )

    # Model
    p.add_argument("--vit_model_name", default=None)
    p.add_argument("--mae_model_name", default=None)
    p.add_argument("--text_model_name", default=None)

    # Feature dimensions
    p.add_argument("--spatial_input_dim", type=int, default=None)
    p.add_argument("--temporal_input_dim", type=int, default=None)
    p.add_argument("--proj_dim", type=int, default=None)

    # ViT extraction settings
    p.add_argument("--s2_mode", default=None)
    p.add_argument("--scales", nargs="+", type=int, default=None)
    p.add_argument("--vit_nth_layer", type=int, default=None)

    # MAE extraction settings
    p.add_argument("--mae_nth_layer", type=int, default=None)
    p.add_argument("--mae_overlap_size", type=int, default=None)

    # LoRA
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)

    # Loss
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--cross_modal_weight", type=float, default=None)
    p.add_argument("--visual_text_weight", type=float, default=None)
    p.add_argument("--use_text_loss", action="store_true", default=None)
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
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    # Misc
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--log_dir", default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Print tensor dimensions for each forward step.",
    )

    return p


_DEFAULTS = dict(
    anno_root="./preprocess/Phoenix14T",
    video_root="",
    max_frames=64,
    vit_model_name="openai/clip-vit-large-patch14",
    mae_model_name="MCG-NJU/videomae-large",
    text_model_name="openai/clip-vit-large-patch14",
    spatial_input_dim=2048,
    temporal_input_dim=1024,
    proj_dim=512,
    s2_mode="s2wrapping",
    scales=[1, 2],
    vit_nth_layer=-1,
    mae_nth_layer=-1,
    mae_overlap_size=8,
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
    batch_size=4,
    num_workers=4,
    max_epochs=50,
    devices=1,
    accumulate_grad_batches=8,
    precision="16",
    gradient_clip_val=1.0,
    patience=10,
    seed=42,
    cache_dir=None,
    log_dir="logs/encoder_finetune",
    save_dir=None,
    verbose=False,
)


def _merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """Merge: hard-coded defaults ← YAML config ← CLI overrides."""
    from omegaconf import OmegaConf

    merged = dict(_DEFAULTS)

    if args.config is not None:
        yaml_cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        for k, v in yaml_cfg.items():
            if k in merged:
                merged[k] = v
            else:
                print(f"[WARNING] Unknown config key in YAML: '{k}' — ignoring.")

    cli_dict = vars(args)
    for k, v in cli_dict.items():
        if v is not None:
            merged[k] = v

    return argparse.Namespace(**merged)


# endregion


# region MAIN
# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    raw_args = get_parser().parse_args()
    args = _merge_config(raw_args)
    seed_everything(args.seed)

    # ── Validate required paths ─────────────────────────────────────
    for required in ("anno_root", "video_root"):
        val = getattr(args, required, "")
        if not val:
            raise ValueError(
                f"--{required} is required. Set it via YAML config or CLI flag."
            )

    # ── Data ────────────────────────────────────────────────────────
    dm = VideoDataModule(
        anno_root=args.anno_root,
        video_root=args.video_root,
        max_frames=args.max_frames,
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
        s2_mode=args.s2_mode,
        scales=args.scales,
        vit_nth_layer=args.vit_nth_layer,
        mae_nth_layer=args.mae_nth_layer,
        mae_overlap_size=args.mae_overlap_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
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
    # if args.devices > 1:
        # strategy = DDPStrategy(find_unused_parameters=True)

    strategy = DeepSpeedStrategy(
        stage=3,                       # Shards parameters, gradients, and optimizer states
        offload_optimizer=True,        # Moves optimizer states to CPU (huge VRAM saver)
        offload_parameters=True,       # Moves model weights to CPU (Sequential Offloading)
        remote_device="cpu",
        pin_memory=True,
        logging_batch_size_per_gpu=args.batch_size
    )

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

    # ── Print config summary ────────────────────────────────────────
    cfg_src = args.config or "CLI-only (no YAML)"
    print("=" * 70)
    print("  Encoder Finetuning with LoRA + Contrastive Loss")
    print(f"  Config: {cfg_src}")
    print("=" * 70)
    print(f"  anno_root   : {args.anno_root}")
    print(f"  video_root  : {args.video_root}")
    print(f"  max_frames  : {args.max_frames}")
    print(f"  ViT model   : {args.vit_model_name}")
    print(f"  MAE model   : {args.mae_model_name}")
    print(f"  s2_mode     : {args.s2_mode}")
    print(f"  LoRA r/α    : {args.lora_r} / {args.lora_alpha}")
    print(f"  Proj dim    : {args.proj_dim}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Text loss   : {args.use_text_loss}")
    print(f"  Verbose     : {args.verbose}")
    print(f"  LR          : {args.lr}")
    print(f"  Batch size  : {args.batch_size}  (accum: {args.accumulate_grad_batches})")
    print(f"  Devices     : {args.devices}")
    print(f"  Precision   : {args.precision}")
    print("=" * 70)

    trainer.fit(model, dm)

    # ── Save outputs ────────────────────────────────────────────────
    save_dir = args.save_dir or os.path.join(args.log_dir, "lora_weights")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Full PL checkpoint (for resuming training)
    full_path = os.path.join(save_dir, "encoder_finetune_full.ckpt")
    trainer.save_checkpoint(full_path)
    print(f"\n✅ Full checkpoint saved → {full_path}")

    # 2. LoRA adapters as PEFT directories (for extraction scripts)
    model.save_lora_adapters(save_dir)

    # 3. Projection weights (for downstream SpaMo pipeline)
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

# Example override:
python scripts/finetune_encoders.py \
    -c configs/finetune_encoders.yaml \
    --batch_size 2 \
    --accumulate_grad_batches 16 \
    --devices 2
"""

"""
python scripts/finetune_vit.py -c configs/finetune_vit.yaml

python scripts/finetune_mae.py -c configs/finetune_mae.yaml

"""
# endregion

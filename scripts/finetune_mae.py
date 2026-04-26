"""
Finetune VideoMAE encoder using LoRA + Contrastive Loss.

Loads raw video frames, passes them through the VideoMAE encoder (with LoRA),
projects into a shared embedding space, and optimises using CLIP-style
contrastive loss against CLIP text embeddings.

After training, the LoRA adapter is exported as a PEFT directory that can
be loaded by mae_extract_feature.py via ``PeftModel.from_pretrained()``.

Usage:
    python scripts/finetune_mae.py -c configs/finetune_mae.yaml
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
from pytorch_lightning.strategies import DDPStrategy

from torch.utils.data import Dataset, DataLoader

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
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


# region DATASET
# ═══════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════


class VideoFrameDataset(Dataset):
    """Loads raw video frames for MAE finetuning.

    Each sample returns:
        - mae_frames  : List[PIL.Image]  (224x224 cropped for MAE)
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

        self.valid_keys = sorted([k for k in self.data.keys() if isinstance(k, int)])
        print(
            f"[VideoFrameDataset-MAE] mode={mode}, {len(self.valid_keys)} samples loaded."
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

        # ── Load frames ──────────────────────────────────────────
        raw_frames = []
        if self.ds_name in ["Phoenix14T", "CSL-Daily"]:
            image_paths = get_img_list(self.ds_name, self.video_root, fname)
            for p in image_paths:
                try:
                    with Image.open(p) as img:
                        raw_frames.append(img.convert("RGB"))
                except Exception as e:
                    print(f"[WARNING] Failed to load {p}: {e}")

        elif self.ds_name in ["How2Sign", "Phoenix14TCompressed"]:
            s = entry.get("original_info", {}).get("START_REALIGNED", None)
            e = entry.get("original_info", {}).get("END_REALIGNED", None)
            try: s = float(s) if s is not None else None
            except: s = None
            try: e = float(e) if e is not None else None
            except: e = None
            
            raw_frames = read_video(fname, start_time=s, end_time=e)
            # read_video returns PIL images
        else:
            raise NotImplementedError(f"Unknown dataset: {self.ds_name}")

        if not raw_frames:
            return None

        # ── Sub-sample BEFORE processing all frames ────────────────
        if len(raw_frames) > self.max_frames:
            indices = np.linspace(0, len(raw_frames) - 1, self.max_frames, dtype=int)
            raw_frames = [raw_frames[i] for i in indices]

        # ── Process & Convert to Tensor ──────────────────────────
        # Resizing and cropping immediately saves RAM
        processed_frames = []
        for f in raw_frames:
            # Resize + Crop
            p_img = self._center_crop(f)
            # Convert to numpy then to torch
            arr = np.array(p_img, dtype=np.uint8)
            processed_frames.append(arr)
            
            # Explicitly close original frame to free memory
            if hasattr(f, 'close'):
                f.close()
            p_img.close()

        # Stack into [T, H, W, C]
        video_tensor = torch.from_numpy(np.stack(processed_frames)) # uint8 saves RAM over float32
        
        return {
            "mae_frames": video_tensor, # Now a [T, H, W, C] uint8 tensor
            "text": text.lower(),
            "file_id": file_id,
        }

    @staticmethod
    def collate_fn(batch):
        return [s for s in batch if s is not None]


# endregion


# region LIGHTNING MODULE
# ═══════════════════════════════════════════════════════════════════════════
#  LIGHTNING MODULE
# ═══════════════════════════════════════════════════════════════════════════


class MAEFinetuneLora(pl.LightningModule):
    """
    Finetunes VideoMAE encoder with LoRA adapters using
    CLIP-style contrastive loss (visual ↔ text).

    Architecture:
        Raw frames → MAE (+ LoRA) → [CLS] → proj_temporal → temporal_embed
        Text       → CLIP-Text (CPU, frozen) → text_proj   → text_embed
    """

    def __init__(
        self,
        mae_model_name: str = "MCG-NJU/videomae-large",
        # LoRA
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_targets: Optional[List[str]] = None,
        # Projections
        temporal_input_dim: int = 1024,
        proj_dim: int = 512,
        # Loss
        temperature: float = 0.07,
        # Text encoder
        text_model_name: str = "openai/clip-vit-large-patch14",
        # MAE settings
        mae_nth_layer: int = -1,
        mae_overlap_size: int = 8,
        # Optimiser
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        # Misc
        cache_dir: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.verbose = verbose
        self.temperature = temperature

        self.mae_nth_layer = mae_nth_layer
        self.mae_overlap_size = mae_overlap_size

        # ── Load MAE encoder + apply LoRA ────────────────────────────
        print("[MAEFinetuneLora] Loading MAE encoder...")
        self.mae_encoder = VideoMAEModel.from_pretrained(
            mae_model_name, cache_dir=cache_dir
        )
        for p in self.mae_encoder.parameters():
            p.requires_grad = False
        self.mae_encoder = self._apply_lora(
            self.mae_encoder,
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_targets or ["query", "value"],
            name="MAE",
        )

        # ── Enable Gradient Checkpointing ────────────────────────────
        self.mae_encoder.gradient_checkpointing_enable()

        # ── Image processor ──────────────────────────────────────────
        self.mae_image_processor = VideoMAEImageProcessor.from_pretrained(
            mae_model_name, cache_dir=cache_dir
        )

        # ── Projection head ──────────────────────────────────────────
        self.proj_temporal = nn.Sequential(
            nn.Linear(temporal_input_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # ── Logit scale (learnable) ──────────────────────────────────
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        # ── Text encoder (frozen) ───────────────────────────────────
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

        self._val_losses: List[float] = []

    # ── LoRA helper ──────────────────────────────────────────────────

    @staticmethod
    def _apply_lora(model, r, alpha, dropout, target_modules, name=""):
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

    def save_lora_adapter(self, save_dir: str) -> None:
        """Save LoRA adapter as a PEFT directory."""
        mae_dir = os.path.join(save_dir, "mae_lora")
        os.makedirs(mae_dir, exist_ok=True)
        self.mae_encoder.save_pretrained(mae_dir)
        print(f"✅ MAE LoRA adapter saved → {mae_dir}")

    # ── Feature extraction ───────────────────────────────────────────

    def _extract_mae_features(
        self, mae_frames: torch.Tensor, chunk_size: int = 1
    ) -> torch.Tensor:
        """Run frames through MAE encoder one clip at a time to save memory."""
        # mae_frames: [T, H, W, C] uint8
        T = mae_frames.shape[0]
        if T < 16:
            padding = mae_frames[-1:].expand(16 - T, -1, -1, -1)
            mae_frames = torch.cat([mae_frames, padding], dim=0)
            T = 16

        # Create windows manually (since sliding_window_for_list is for lists)
        step_size = 16 - self.mae_overlap_size
        clips = []
        for i in range(0, T, step_size):
            if i + 16 <= T:
                clips.append(mae_frames[i : i + 16])
        
        if not clips: # Fallback
            clips = [mae_frames[:16]]

        all_feats = []
        for i in range(0, len(clips), chunk_size):
            batch_clips = clips[i : i + chunk_size]
            # Convert back to numpy or list of images for the processor if needed, 
            # but processor usually accepts a list of [T, C, H, W] or a list of lists of PIL.
            # Actually, the processor for VideoMAE usually expects List[List[PIL]] or List[np.ndarray].
            
            # Let's convert to list of numpy arrays [T, H, W, C]
            clip_list = [c.cpu().numpy() for c in batch_clips]
            
            inputs = self.mae_image_processor(images=clip_list, return_tensors="pt").to(
                self.device
            )

            outputs = self.mae_encoder(
                **inputs, output_hidden_states=True
            ).hidden_states[self.mae_nth_layer]

            all_feats.append(outputs[:, 0])

            # Free intermediate memory
            del inputs, outputs
            torch.cuda.empty_cache()

        return torch.cat(all_feats, dim=0)  # [N_clips, D_mae]

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        temporal_embeds_list = []
        texts = []

        for sample in batch:
            mae_frames = sample["mae_frames"]

            mae_feats = self._extract_mae_features(mae_frames)
            mae_pooled = mae_feats.mean(dim=0, keepdim=True)
            temporal_embed = self.proj_temporal(mae_pooled)
            temporal_embeds_list.append(temporal_embed.squeeze(0))
            texts.append(sample["text"])

            if self.verbose:
                print(
                    f"[Verbose] sample={sample['file_id']} | "
                    f"mae_feats={tuple(mae_feats.shape)} -> "
                    f"temporal_embed={tuple(temporal_embed.shape)}"
                )

            # Free intermediate memory between samples
            del mae_feats, mae_pooled
            torch.cuda.empty_cache()

        if len(temporal_embeds_list) == 0:
            return {
                "total_loss": torch.tensor(0.0, device=self.device, requires_grad=True)
            }

        temporal_embeds = torch.stack(temporal_embeds_list)  # [B, proj_dim]

        # ── Text embeddings ─────────────────────────────────────
        with torch.no_grad():
            tokens = self.text_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            raw_text = self.text_encoder(**tokens).pooler_output

        text_embeds = self.text_proj(raw_text)  # [B, proj_dim]

        if self.verbose:
            print(
                f"[Verbose] temporal_embeds={tuple(temporal_embeds.shape)} "
                f"text_embeds={tuple(text_embeds.shape)}"
            )

        # ── CLIP-style contrastive loss ──────────────────────────
        visual = F.normalize(temporal_embeds, dim=-1)
        text = F.normalize(text_embeds, dim=-1)
        logit_scale = self.logit_scale.exp()
        sim = torch.mm(visual, text.t()) * logit_scale
        loss = clip_loss(sim)

        return {"total_loss": loss}

    # ── PL hooks ─────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
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
        torch.cuda.empty_cache()
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
        self._val_losses.append(loss_dict["total_loss"].item())
        return loss_dict["total_loss"]

    def on_validation_epoch_end(self):
        if not self._val_losses:
            return
        avg = np.mean(self._val_losses)
        print(f"\n{'=' * 60}")
        print(f"[Val Epoch Summary] total_loss: {avg:.5f}")
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
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def get_parser():
    p = argparse.ArgumentParser(
        description="Finetune VideoMAE encoder with LoRA + contrastive loss"
    )
    p.add_argument(
        "-c", "--config", type=str, default=None, help="Path to YAML config file"
    )

    p.add_argument("--anno_root", default=None)
    p.add_argument("--video_root", default=None)
    p.add_argument("--max_frames", type=int, default=None)

    p.add_argument("--mae_model_name", default=None)
    p.add_argument("--text_model_name", default=None)

    p.add_argument("--temporal_input_dim", type=int, default=None)
    p.add_argument("--proj_dim", type=int, default=None)

    p.add_argument("--mae_nth_layer", type=int, default=None)
    p.add_argument("--mae_overlap_size", type=int, default=None)

    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)

    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--devices", type=int, default=None)
    p.add_argument("--accumulate_grad_batches", type=int, default=None)
    p.add_argument("--precision", default=None)
    p.add_argument("--gradient_clip_val", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--cache_dir", default=None)
    p.add_argument("--log_dir", default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--verbose", action="store_true", default=None)

    return p


_DEFAULTS = dict(
    anno_root="./preprocess/Phoenix14T",
    video_root="",
    max_frames=64,
    mae_model_name="MCG-NJU/videomae-large",
    text_model_name="openai/clip-vit-large-patch14",
    temporal_input_dim=1024,
    proj_dim=512,
    mae_nth_layer=-1,
    mae_overlap_size=8,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    temperature=0.07,
    lr=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    batch_size=4,
    num_workers=4,
    max_epochs=50,
    devices=1,
    accumulate_grad_batches=8,
    precision="bf16-mixed",
    gradient_clip_val=1.0,
    patience=10,
    seed=42,
    cache_dir=None,
    log_dir="logs/finetune_mae",
    save_dir=None,
    verbose=False,
)


def _merge_config(args: argparse.Namespace) -> argparse.Namespace:
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

    for required in ("anno_root", "video_root"):
        val = getattr(args, required, "")
        if not val:
            raise ValueError(f"--{required} is required.")

    # ── Data ────────────────────────────────────────────────────────
    dm = VideoDataModule(
        anno_root=args.anno_root,
        video_root=args.video_root,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ───────────────────────────────────────────────────────
    model = MAEFinetuneLora(
        mae_model_name=args.mae_model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        temporal_input_dim=args.temporal_input_dim,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        text_model_name=args.text_model_name,
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
            filename="mae-epoch={epoch:03d}-loss={val/total_loss:.4f}",
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

    # ── Print config ────────────────────────────────────────────────
    cfg_src = args.config or "CLI-only (no YAML)"
    print("=" * 70)
    print("  VideoMAE Finetuning with LoRA + Contrastive Loss")
    print(f"  Config: {cfg_src}")
    print("=" * 70)
    print(f"  anno_root   : {args.anno_root}")
    print(f"  video_root  : {args.video_root}")
    print(f"  max_frames  : {args.max_frames}")
    print(f"  MAE model   : {args.mae_model_name}")
    print(f"  LoRA r/α    : {args.lora_r} / {args.lora_alpha}")
    print(f"  Proj dim    : {args.proj_dim}")
    print(f"  Temperature : {args.temperature}")
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

    full_path = os.path.join(save_dir, "mae_finetune_full.ckpt")
    trainer.save_checkpoint(full_path)
    print(f"\n✅ Full checkpoint saved → {full_path}")

    model.save_lora_adapter(save_dir)

    proj_path = os.path.join(save_dir, "mae_projection_weights.pt")
    torch.save(
        {
            "proj_temporal": model.proj_temporal.state_dict(),
            "text_proj": model.text_proj.state_dict(),
        },
        proj_path,
    )
    print(f"✅ Projection weights saved → {proj_path}")
    print("\n🎉 MAE finetuning complete!")


if __name__ == "__main__":
    main()
# endregion


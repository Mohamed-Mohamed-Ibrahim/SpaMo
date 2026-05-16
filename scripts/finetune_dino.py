"""
Finetune DINOv2 encoder with LoRA and CLIP Text Anchor.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from transformers import (
    Dinov2Model,
    AutoImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from typing import List, Dict

# Reuse dataset from vit script
from scripts.finetune_vit import VideoFrameDataset, VideoDataModule
from spamo.clip_loss import clip_loss

class DINOFinetuneLora(pl.LightningModule):
    def __init__(self, dino_model_name="facebook/dinov2-base", text_model_name="openai/clip-vit-base-patch16", proj_dim=512, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.dino = Dinov2Model.from_pretrained(dino_model_name, use_safetensors=False)
        # Apply LoRA
        config = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"], lora_dropout=0.1)
        self.dino = get_peft_model(self.dino, config)
        self.processor = AutoImageProcessor.from_pretrained(dino_model_name)
        
        self.proj = nn.Linear(self.dino.config.hidden_size, proj_dim)
        
        # Text anchor
        self.text_tokenizer = CLIPTokenizer.from_pretrained(text_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(
            text_model_name, use_safetensors=False
        ).to("cpu")
        self.text_encoder.eval()
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, proj_dim)

    def forward(self, batch):
        visual_embeds = []
        texts = []
        for s in batch:
            imgs = s["pil_frames"]
            pixel_values = self.processor(imgs, return_tensors="pt").pixel_values.to(self.device)
            # DINOv2 is fast
            feats = self.dino(pixel_values).pooler_output # [N, D]
            pooled = feats.mean(dim=0, keepdim=True)
            visual_embeds.append(self.proj(pooled).squeeze(0))
            texts.append(s["text"])
        
        visual_embeds = torch.stack(visual_embeds)
        
        # Text on CPU
        tokens = self.text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cpu")
        with torch.no_grad():
            text_feats = self.text_encoder(**tokens).pooler_output
        text_embeds = self.text_proj(text_feats.to(self.device))
        
        # Loss
        sim = torch.mm(F.normalize(visual_embeds, dim=-1), F.normalize(text_embeds, dim=-1).t()) * 30.0
        return clip_loss(sim)

    def training_step(self, batch, idx):
        loss = self.forward(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        loss = self.forward(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

def main():
    # Simple runner
    dm = VideoDataModule(anno_root="./preprocess/Phoenix14TCompressed", 
                        video_root="/kaggle/input/datasets/mariusschmidtmengin/phoenixweather2014t-3rd-attempt/videos_phoenix/videos",
                        max_frames=32, batch_size=4)
    model = DINOFinetuneLora()
    trainer = Trainer(max_epochs=10, accelerator="gpu", precision="16")
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()

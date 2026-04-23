import os
import torch
import torch.nn as nn
import random
import math
from typing import Dict, List, Optional, Tuple, Any

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, T5ForConditionalGeneration, MT5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

from spamo.tconv import TemporalConv
from utils.helpers import create_mask, derangement
from spamo.mm_projector import build_vision_projector
from utils.evaluate import evaluate_results
from spamo.clip_loss import clip_loss
from spamo.sign_cl import TemporalSignCLLoss
from spamo.asb import AbstractSLT
from spamo.data_augmentation import FeatureAugmenter
from torch.optim.lr_scheduler import LambdaLR
from spamo.lr_scheduler import LambdaWarmUpCosineScheduler

# NEW IMPORTS FOR UNI-SIGN PGF & ST-GCN HYBRID
from deformable_attention_2d import DeformableAttention2D
from gcn_utils import Graph
from stgcn_block import get_stgcn_chain

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

# MMPose 133 to Uni-Sign 69 Keypoint Mapping
# 11 Upper Body, 16 Face (Lips/Eyes), 21 Left Hand, 21 Right Hand
UNI_SIGN_69_INDICES = (
    list(range(0, 11)) +       # Body 
    list(range(23, 39)) +      # Face (Partial)
    list(range(91, 112)) +     # Left Hand
    list(range(112, 133))      # Right Hand
)

class FlanT5SLT(AbstractSLT):
    """
    Hybrid SpaMo + Uni-Sign Sign Language Translation model.
    Features: Spatial (CLIP ViT) + Motion (VideoMAE) + Pose (ST-GCN -> PGF).
    """
    def __init__(
        self, 
        tuning_type: str = 'lora', 
        model_name: str = 'google/mt5-base',  # Enforcing multilingual LLM
        weight_decay=0.01,
        frame_sample_rate: int = 1, 
        prompt: str = '',
        lr: float = 3e-4,
        min_lr: float = 5e-5,
        input_size: int = 1024,
        pose_hidden_dim: int = 64,
        inter_hidden: int = 512,
        max_frame_len: int = 512,
        max_txt_len: int = 64,
        cross_modal_align: bool = False,
        warm_up_steps: Optional[int] = 4000,
        lr_warmup_steps: Optional[int] = 10000,
        combined_loss: bool = False,
        alpha: float = 0.1,
        sign_cl_loss: bool = False,
        sign_cl_alpha: float = 0.5,
        sign_cl_temperature: float = 0.07,
        sign_cl_temporal_window: int = 5,
        sign_cl_every_n_steps: int = 1,
        cache_dir: str = "/data3/models",
        use_in_context: bool = False,
        num_in_context: int = 0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_data_augmentation: bool = True,
        use_gradient_checkpointing: bool = False,
        augmentation_prob: float = 0.5,
        aug_frame_prob: float = 0.1,
        aug_span_prob: float = 0.1,
        aug_channel_prob: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_size = input_size
        self.prompt = prompt
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.model_name = model_name
        self.frame_sample_rate = frame_sample_rate
        self.inter_hidden = inter_hidden
        self.max_frame_len = max_frame_len
        self.max_txt_len = max_txt_len
        self.tuning_type = tuning_type
        self.cross_modal_align = cross_modal_align
        self.warm_up_steps = warm_up_steps
        self.lr_warmup_steps = lr_warmup_steps
        self.combined_loss = combined_loss
        self.alpha = alpha
        self.sign_cl_loss = sign_cl_loss
        self.sign_cl_alpha = sign_cl_alpha
        self.sign_cl_temperature = sign_cl_temperature
        self.sign_cl_temporal_window = sign_cl_temporal_window
        self.sign_cl_every_n_steps = sign_cl_every_n_steps
        self.cache_dir = cache_dir
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        
        if self.num_in_context == 0:
            self.use_in_context = False

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_data_augmentation = use_data_augmentation
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.save_hyperparameters()
        self.prepare_models(model_name, pose_hidden_dim)

        if tuning_type == 'freeze':
            self._freeze_model()
        elif tuning_type == 'lora':
            self._apply_lora()

        if self.use_gradient_checkpointing:
            self.t5_model.gradient_checkpointing_enable()
            if hasattr(self.t5_model, "enable_input_require_grads"):
                self.t5_model.enable_input_require_grads()

        if self.use_data_augmentation:
            self.augmenter = FeatureAugmenter(
                aug_prob=augmentation_prob,
                frame_dropout_prob=aug_frame_prob,
                span_mask_prob=aug_span_prob,
                channel_drop_prob=aug_channel_prob
            )

        self.set_container()
        
    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])

    def _apply_lora(self) -> None:
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)

    def _freeze_model(self) -> None:
        self.t5_model.eval()
        for params in self.t5_model.parameters():
            params.requires_grad = False

    def set_container(self) -> None:
        self.generated = []
        self.references = []

    def prepare_models(self, t5_model: str, pose_hidden_dim: int) -> None:
        # Load the textual model (mT5 for multilingual support)
        if 'mt5' in t5_model.lower():
            self.t5_model = MT5ForConditionalGeneration.from_pretrained(
                t5_model, cache_dir=self.cache_dir, torch_dtype=torch.bfloat16, use_safetensors=True
            )
        else:
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                t5_model, cache_dir=self.cache_dir, torch_dtype=torch.bfloat16, use_safetensors=True
            )

        self.t5_model.tie_weights()

        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model, cache_dir=self.cache_dir, max_length=self.max_txt_len,
        )

        # ---------------------------------------------------------
        # PHASE 2: ONLINE PIPELINE MODULES
        # ---------------------------------------------------------
        
        # 1. Pose Encoding: Single ST-GCN block for all 69 points
        self.pose_graph = Graph(layout='mediapipe_69', strategy='distance', max_hop=1) # Adapting to 69 nodes
        A = torch.tensor(self.pose_graph.A, dtype=torch.float32, requires_grad=False)
        self.pose_proj_in = nn.Linear(2, pose_hidden_dim) # Project (x,y) to hidden dim
        self.pose_gcn, final_gcn_dim = get_stgcn_chain(pose_hidden_dim, 'spatial', (1, A.size(0)), A.clone(), True)
        
        # Project ST-GCN output to match CLIP Spatial dimension (1024)
        self.pose_to_pgf = nn.Linear(final_gcn_dim, 1024)

        # 2. Prior-Guided Fusion: Deformable Attention
        self.pgf_attention = DeformableAttention2D(
            dim = 1024,
            dim_head = 32,
            heads = 8,
            dropout = 0.1,
            downsample_factor = 1,
            offset_kernel_size = 1,
        )

        # 3. Modality Merging: Flattened Refined Spatial (69*1024) + VideoMAE Motion (1024)
        fused_input_size = (69 * 1024) + 1024
        self.fusion_proj = build_vision_projector('mlp2x_gelu', fused_input_size, self.t5_model.config.hidden_size)
        
        # 4. Temporal Compression: 1D Convolution
        self.temporal_encoder = TemporalConv(self.t5_model.config.hidden_size, self.t5_model.config.hidden_size)

        if self.sign_cl_loss:
            self.sign_cl = TemporalSignCLLoss(
                temperature=self.sign_cl_temperature,
                temporal_window=self.sign_cl_temporal_window,
            )
        else:
            self.sign_cl = None

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def prepare_inputs(
        self, visual_outputs: torch.Tensor, visual_mask: torch.Tensor, samples: Dict, split: str, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        # FIXED
        bs = visual_outputs.shape[0]
        
        prompts = [f'{self.prompt}'] * bs
        prompts = [p.format(l) for p, l in zip(prompts, samples['lang'])]
        
        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples['ex_lang_trans'])]
        
        input_tokens = self.t5_tokenizer(
            prompts, padding="longest", truncation=True, return_tensors="pt",
        ).to(self.device)
        
        visual_lengths = visual_mask.sum(1)
        prompt_lengths = input_tokens.attention_mask.sum(1)
        new_lengths = visual_lengths + prompt_lengths
        
        input_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        
        joint_outputs = []
        for i in range(bs):
            vis_out = visual_outputs[i, :visual_lengths[i], :]
            prompt_embeds = input_embeds[i, :prompt_lengths[i], :]
            concat_sample = torch.cat((vis_out, prompt_embeds), dim=0)
            joint_outputs.append(concat_sample)
        
        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        joint_mask = create_mask(seq_lengths=new_lengths.tolist(), device=self.device)
        
        output_tokens = self.t5_tokenizer(
            samples['text'], padding="longest", return_tensors="pt",
        ).to(self.device)
        
        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )
        
        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the exact Uni-Sign integration pipeline step-by-step.
        """
        # Load offline modalities
        spatial_outputs = pad_sequence(samples['pixel_values'], batch_first=True) # (B, T, 1024, D)
        motion_outputs = pad_sequence(samples['glor_values'], batch_first=True)   # (B, T, D)
        
        # FIXED — offline extraction already gives (T, 133, 2), just cast to float
        pose_values_local = [pv.float() for pv in samples['pose_values']]
        pose_outputs = pad_sequence(pose_values_local, batch_first=True).float()  # (B, T, 133, 2)

        B, T = spatial_outputs.shape[:2]

        # STEP 1: Pose Encoding & Filtering
        pose_69 = pose_outputs[:, :, UNI_SIGN_69_INDICES, :] # Filter down to (B, T, 69, 2)
        pose_init = pose_69.clone() # Save raw coordinates [1] for PGF vgrid calculation

        # Project coordinates and format for ST-GCN: (B*T, C, V, 1)
        pose_proj = self.pose_proj_in(pose_69) 
        pose_proj = pose_proj.view(B*T, 69, -1).permute(0, 2, 1).unsqueeze(-1) 

        # Single Graph Pass (No Left/Right splitting)
        pose_feat = self.pose_gcn(pose_proj) # (B*T, D_gcn, 69, 1)
        pose_feat = pose_feat.squeeze(-1).permute(0, 2, 1) # (B*T, 69, D_gcn)
        
        # Align dimension with CLIP (1024) and format for Deformable Attention: (B*T, 1024, 69)
        pose_feat = self.pose_to_pgf(pose_feat).permute(0, 2, 1) 

        # STEP 2: Prior-Guided Fusion (PGF)
        # Format Spatial grid as Key/Value: (B*T, D, H, W) -> e.g., (B*T, 1024, 32, 32)
        grid_size = int(math.sqrt(spatial_outputs.shape[2]))


        # FIXED — add .contiguous() and use .reshape() defensively throughout
        spatial_reshaped = spatial_outputs.reshape(B*T, grid_size * grid_size, 1024)
        spatial_flat = spatial_reshaped.permute(0, 2, 1).contiguous()
        rgb_feat = spatial_flat.view(B*T, 1024, grid_size, grid_size)
        
        # FIXED — transpose to (B*T, 2, 69) as the module requires
        pose_init_flat = pose_init.view(B*T, 69, 2).permute(0, 2, 1).contiguous()

        # Execute Deformable Attention
        refined_spatial_feat = self.pgf_attention(pose_feat, rgb_feat, pose_init_flat) # (B*T, 1024, 69)

        # STEP 3: Modality Merging
        # Flatten the refined spatial features to (B*T, 69 * 1024)
        refined_spatial_flat = refined_spatial_feat.permute(0, 2, 1).reshape(B*T, 69 * 1024)

        # Also apply the same fix to motion_outputs:
        motion_flat = motion_outputs.reshape(B*T, -1)  # reshape is safer than view on padded seqs

        # Concatenate and pass through MLP Gate
        fused_features = torch.cat([refined_spatial_flat, motion_flat], dim=-1)
        projected_features = self.fusion_proj(fused_features) # (B*T, LLM_Hidden)
        projected_features = projected_features.view(B, T, -1)

        # STEP 4: Temporal Compression
        lengths = torch.tensor(samples['num_frames'], device=self.device)
        conv_outputs = self.temporal_encoder(projected_features.permute(0, 2, 1), lengths)

        visual_outputs = conv_outputs['visual_feat'].permute(1, 0, 2)
        visual_masks = create_mask(
            seq_lengths=conv_outputs['feat_len'].to(torch.int).tolist(), 
            device=self.device
        )

        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        pixel_values, glor_values, masks, ids = [], [], [], []
        pose_values = []
        texts, glosses = [], []
        num_frames, glor_lengths, langs = [], [], []
        ex_lang_translations = []

        max_frame_len = self.max_frame_len

        for sample in batch:
            if sample.get('pixel_value') is None or sample['pixel_value'].shape == 0:
                continue

            nframe = math.ceil(sample['num_frames'] / self.frame_sample_rate)

            if nframe > max_frame_len:
                continue
            
            pval = sample['pixel_value'][::self.frame_sample_rate]

            ids.append(sample['id'])
            texts.append(sample['text'].lower())
            glosses.append(sample['gloss'])
            langs.append(sample['lang'])

            _ex_lang_trans = []
            if self.num_in_context > 0:
                if 'ctx_en_text' in sample and 'ctx_text' in sample:
                    _ex_lang_trans = [f"{sample.get('ctx_en_text','')}={sample['ctx_text']}"]
                trimmed = _ex_lang_trans[:self.num_in_context]
                ex_lang_translations.append(' '.join(trimmed))
            else:
                ex_lang_translations.append("")

            num_frames.append(nframe)
            pixel_values.append(pval)
            
            if sample.get('pose_value') is not None:
                pose_values.append(sample['pose_value'])

            if sample.get('glor_value') is not None:
                if isinstance(sample['glor_value'], list):
                    glor_values.append(torch.cat(sample['glor_value'], dim=0))
                    glor_lengths.append(sum(len(g) for g in sample['glor_value']))
                else:
                    glor_values.append(sample['glor_value'])
                    glor_lengths.append(len(sample['glor_value']))

        if self.use_in_context and len(ex_lang_translations) > 1:
            ex_lang_translations = derangement(ex_lang_translations)

        return {
            'pixel_values': pixel_values,
            'glor_values': glor_values,
            'pose_values': pose_values,
            'bool_mask_pos': masks,
            'ids': ids,
            'text': texts,
            'ex_lang_trans': ex_lang_translations,
            'gloss': glosses,
            'lang': langs,
            'num_frames': num_frames,
            'glor_lengths': glor_lengths,
        }

    def visual_textual_align(self, visual_outputs: torch.Tensor, visual_masks: torch.Tensor, samples: Dict) -> torch.Tensor:
        output_tokens = self.t5_tokenizer(
            samples['text'], padding="longest", return_tensors="pt",
        ).to(self.device)
        
        text_embeds = self.t5_model.encoder.embed_tokens(output_tokens.input_ids)
        
        image_embeds = visual_outputs.mean(1) 
        text_embeds = text_embeds.mean(1)
        
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

        loss = clip_loss(logits_per_text)
        return loss

    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        # STEP 5: LLM Translation execution (Uses updated visual outputs)
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        log_dict = {}
        
        if self.cross_modal_align:
            if self.warm_up_steps is None and not self.combined_loss:
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss
                
            elif self.warm_up_steps is not None and self.global_step <= self.warm_up_steps:
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss
                
                if (self.sign_cl_loss and self.sign_cl is not None 
                    and (self.sign_cl_every_n_steps <= 1 or (self.global_step % self.sign_cl_every_n_steps) == 0)):
                    sign_cl_loss_val = self.sign_cl(visual_outputs, visual_masks)
                    if sign_cl_loss_val is not None and sign_cl_loss_val > 0:
                        loss = loss + self.sign_cl_alpha * sign_cl_loss_val
                        log_dict[f"{split}/sign_cl_loss"] = sign_cl_loss_val
                        log_dict[f"{split}/warmup_total_loss"] = loss
                
            else:
                input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                    visual_outputs, visual_masks, inputs, split, batch_idx
                )
                
                outputs = self.t5_model(
                    inputs_embeds=input_embeds,
                    attention_mask=input_masks,
                    decoder_attention_mask=output_tokens.attention_mask,
                    labels=targets,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                t5_loss = outputs.loss
                log_dict[f"{split}/loss"] = t5_loss
                
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss = t5_loss + self.alpha * cont_loss
                log_dict[f"{split}/contra_loss"] = cont_loss
                
                if (self.sign_cl_loss and self.sign_cl is not None 
                    and (self.sign_cl_every_n_steps <= 1 or (self.global_step % self.sign_cl_every_n_steps) == 0)):
                    sign_cl_loss_val = self.sign_cl(visual_outputs, visual_masks)
                    if sign_cl_loss_val is not None and sign_cl_loss_val > 0:
                        loss = loss + self.sign_cl_alpha * sign_cl_loss_val
                        log_dict[f"{split}/sign_cl_loss"] = sign_cl_loss_val
                
                log_dict[f"{split}/combined_loss"] = loss
        else:
            input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                visual_outputs, visual_masks, inputs, split, batch_idx
            )
            
            outputs = self.t5_model(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                decoder_attention_mask=output_tokens.attention_mask,
                labels=targets,
                output_hidden_states=True,
                return_dict=True
            )
            
            loss = outputs.loss
            log_dict[f"{split}/loss"] = loss

        if split != "train":
            input_embeds, input_masks, _, _ = self.prepare_inputs(
                visual_outputs, visual_masks, inputs, split, batch_idx
            )
            
            generated = self.t5_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                num_beams=5,
                max_length=self.max_txt_len,
                top_p=0.9,
                do_sample=True,
            )
            
            generated_strings = self.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_strings = [gen.lower() for gen in generated_strings]
            
            reference_strings = self.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            reference_strings = [ref.lower() for ref in reference_strings]

            self.generated.extend(generated_strings)
            self.references.extend(reference_strings)

        return loss, log_dict

    def on_validation_epoch_end(self) -> None:
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='val',
            device=self.device
        )
        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def on_test_epoch_end(self) -> None:
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='test',
            device=self.device
        )
        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=1.0,  
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.98)
        )
        
        if hasattr(self.trainer, 'estimated_stepping_batches'):
            total_steps = int(self.trainer.estimated_stepping_batches)
        else:
            max_epochs = self.trainer.max_epochs
            train_loader = self.trainer.train_dataloader
            if hasattr(train_loader, 'dataloader'): 
                train_loader = train_loader.dataloader
            batches_per_epoch = len(train_loader)
            acc_batches = self.trainer.accumulate_grad_batches if hasattr(self.trainer, 'accumulate_grad_batches') else 1
            total_steps = (batches_per_epoch // acc_batches) * max_epochs
        
        if self.lr_warmup_steps is not None:
            warmup_steps = self.lr_warmup_steps
        else:
            warmup_steps = int(total_steps * 0.1)

        custom_scheduler_fn = LambdaWarmUpCosineScheduler(
            warm_up_steps=warmup_steps,
            lr_min=self.hparams.min_lr,
            lr_max=self.hparams.lr,
            lr_start=1e-6,
            max_decay_steps=total_steps
        )

        scheduler = LambdaLR(optimizer, lr_lambda=custom_scheduler_fn)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

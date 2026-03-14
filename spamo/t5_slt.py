import os
import torch
import torch.nn as nn
import random
import math

from typing import Dict, List, Optional, Tuple, Any

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, get_cosine_schedule_with_warmup
from transformers import BertConfig, BertModel
from peft import LoraConfig, get_peft_model, TaskType

from spamo.tconv import TemporalConv
from spamo.encoders import (
    DeformableTransformerEncoder,
    TransformerEncoder,
    RecurrentEncoder,
)
from utils.helpers import create_mask, derangement
from spamo.mm_projector import build_vision_projector, AdaptiveFusion
from utils.evaluate import evaluate_results
from spamo.clip_loss import clip_loss
from spamo.sign_cl import TemporalSignCLLoss
from spamo.asb import AbstractSLT
from spamo.data_augmentation import FeatureAugmenter



os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')


def build_encoder(cfg: Dict) -> 'RecurrentEncoder | TransformerEncoder | DeformableTransformerEncoder':
    """
    Build a second-stage encoder from a config dict, mirroring the pattern in
    GASLT/signjoey/model.py (build_model, L506-L536).

    Supported ``type`` values:
      - ``'deformable_transformer'``  – DeformableTransformerEncoder
      - ``'transformer'``             – TransformerEncoder
      - ``'recurrent'``               – RecurrentEncoder (default)

    All remaining keys are forwarded verbatim to the chosen encoder class
    (same as the ``**cfg["encoder"]`` style used in model.py).
    """
    enc_dropout = cfg.get("dropout", 0.0)
    enc_emb_dropout = cfg.get("emb_dropout", enc_dropout)
    encoder_type = cfg.get("type", "recurrent")

    # Strip meta-keys that must not be forwarded to encoder constructors
    fwd_cfg = {k: v for k, v in cfg.items() if k not in ("type", "emb_dropout")}

    if encoder_type == "deformable_transformer":
        encoder = DeformableTransformerEncoder(
            **fwd_cfg,
            emb_size=fwd_cfg["hidden_size"],
            emb_dropout=enc_emb_dropout,
        )
    elif encoder_type == "transformer":
        encoder = TransformerEncoder(
            **fwd_cfg,
            emb_size=fwd_cfg["hidden_size"],
            emb_dropout=enc_emb_dropout,
        )
    else:  # 'recurrent' (default)
        encoder = RecurrentEncoder(
            **fwd_cfg,
            emb_size=fwd_cfg["hidden_size"],
            emb_dropout=enc_emb_dropout,
        )
    return encoder


class FlanT5SLT(AbstractSLT):
    """
    FlanT5-based Sign Language Translation model.
    Features: Spatial (ViT/ResNet) + Spatiotemporal (VideoMAE/C3D).
    Removed: Pose and I3D features.
    """
    def __init__(
        self, 
        tuning_type: str = 'lora', 
        model_name: Optional[str] = None,
        weight_decay=0.01,
        frame_sample_rate: int = 1, 
        prompt: str = '',
        lr: float = 3e-4,
        input_size: int = 1024,
        pose_input_size: int = 33*3,
        fusion_mode: str = 'joint',
        inter_hidden: int = 512,
        max_frame_len: int = 512,
        max_txt_len: int = 64,
        cross_modal_align: bool = False,
        warm_up_steps: Optional[int] = None,
        combined_loss: bool = False,
        alpha: float = 0.1,
        sign_cl_loss: bool = False,
        sign_cl_alpha: float = 0.5,
        sign_cl_temperature: float = 0.07,
        sign_cl_temporal_window: int = 5,
        sign_cl_every_n_steps: int = 1,
        use_resampler: bool = False,
        sampling_length: int = 64,
        cache_dir: str = "/data3/models",
        use_in_context: bool = False,
        num_in_context: int = 0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_data_augmentation: bool = True,

        # NEW: 3 Augmentation Parameters
        augmentation_prob: float = 0.5,
        aug_frame_prob: float = 0.1,
        aug_span_prob: float = 0.1,
        aug_channel_prob: float = 0.05,

        # Second-stage encoder applied after TemporalConv.
        # Mirrors GASLT model.py build_model() pattern (L506-L536).
        # Set use_second_encoder=True and provide encoder_cfg dict to activate.
        # encoder_cfg example (deformable_transformer):
        #   {type: deformable_transformer, hidden_size: 768, num_layers: 2,
        #    num_heads: 8, ff_size: 2048, dropout: 0.1,
        #    query_type: mean, query_nb: 3,
        #    num_keys: [5, 5], attentions_type: weighted_local_global}
        use_second_encoder: bool = False,
        encoder_cfg: Optional[Dict] = None,

        # Primary TemporalConv encoder toggle
        use_temporal_conv: bool = True,

        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Configuration parameters
        self.input_size = input_size
        self.pose_input_size = pose_input_size
        self.prompt = prompt
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_name = model_name
        self.frame_sample_rate = frame_sample_rate
        self.fusion_mode = fusion_mode
        self.inter_hidden = inter_hidden
        self.max_frame_len = max_frame_len
        self.max_txt_len = max_txt_len
        self.tuning_type = tuning_type
        self.cross_modal_align = cross_modal_align
        self.warm_up_steps = warm_up_steps
        self.combined_loss = combined_loss
        self.alpha = alpha
        self.sign_cl_loss = sign_cl_loss
        self.sign_cl_alpha = sign_cl_alpha
        self.sign_cl_temperature = sign_cl_temperature
        self.sign_cl_temporal_window = sign_cl_temporal_window
        self.sign_cl_every_n_steps = sign_cl_every_n_steps
        self.use_resampler = use_resampler
        self.sampling_length = sampling_length
        self.cache_dir = cache_dir
        
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        
        # <--- FIX: Force disable context if count is 0
        if self.num_in_context == 0:
            self.use_in_context = False

        
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_data_augmentation = use_data_augmentation
        self.use_temporal_conv = use_temporal_conv

        self.use_second_encoder = use_second_encoder
        if use_second_encoder:
            self._encoder_cfg = dict(encoder_cfg) if encoder_cfg else {}
            self._encoder_cfg.setdefault("hidden_size", self.inter_hidden)
        else:
            self._encoder_cfg = {}
        print("==="*40)
        print(f"use_data_augmentation: {use_data_augmentation}")
        print(f"sign_cl_loss{sign_cl_loss}")
        print("==="*40)
        
        # Save hyperparameters (ensures self.hparams.lr exists)
        self.save_hyperparameters()
        
        self.prepare_models(model_name)

        # Apply the selected tuning strategy
        if tuning_type == 'freeze':
            self._freeze_model()
        elif tuning_type == 'lora':
            self._apply_lora()

        # Data augmenter
        if self.use_data_augmentation:
            self.augmenter = FeatureAugmenter(
                aug_prob=augmentation_prob,
                frame_dropout_prob=aug_frame_prob,
                span_mask_prob=aug_span_prob,
                channel_drop_prob=aug_channel_prob
            )
            print(
                f"[AUG] Enabled | frame={aug_frame_prob}, span={aug_span_prob}, channel={aug_channel_prob}"
            )


        self.set_container()
        
    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        print(f'Checkpoint is loaded from {checkpoint_path}.')

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
        print("LoRA adapter applied to T5 model.")

    def _freeze_model(self) -> None:
        self.t5_model.eval()
        for params in self.t5_model.parameters():
            params.requires_grad = False
        print("T5 model frozen.")

    def set_container(self) -> None:
        self.generated = []
        self.references = []

    def prepare_models(self, t5_model: str) -> None:
        # Load the textual model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,
            use_safetensors=True 
        )
        
        # Load the tokenizer
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,
        )

        # Load the vision projectors (Spatial + Spatiotemporal ONLY)
        self.spatio_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', 1024, self.inter_hidden)
        # Pose projector: default pose size is 33 keypoints * 3 coords = 99
        self.pose_proj = build_vision_projector('linear', self.pose_input_size, self.inter_hidden)
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)
        
        if self.use_temporal_conv:
            self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

        if self.use_second_encoder:
            print(f"[SecondEncoder] Building encoder type='{self._encoder_cfg.get('type', 'recurrent')}' "
                  f"hidden_size={self._encoder_cfg.get('hidden_size', self.inter_hidden)}")
            self.second_encoder = build_encoder(self._encoder_cfg)
        
        # Initialize adaptive fusion if fusion_mode is 'adaptive'
        if self.fusion_mode == 'adaptive':
            self.adaptive_fusion = AdaptiveFusion(
                input_size_1=self.inter_hidden, 
                input_size_2=self.inter_hidden, 
                input_size_3=self.inter_hidden, 
                output_size=3
            )
            

        # Initialize SignCL loss if enabled
        if self.sign_cl_loss:
            self.sign_cl = TemporalSignCLLoss(
                temperature=self.sign_cl_temperature,
                temporal_window=self.sign_cl_temporal_window,
            )
        else:
            self.sign_cl = None

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def prepare_inputs(
        self, 
        visual_outputs: torch.Tensor, 
        visual_mask: torch.Tensor, 
        samples: Dict, 
        split: str, 
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        bs = visual_outputs.shape[0]
        
        # Prepare the prompt
        prompts = [f'{self.prompt}'] * bs
        prompts = [p.format(l) for p, l in zip(prompts, samples['lang'])]
        
        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples['ex_lang_trans'])]
        
        input_tokens = self.t5_tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
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
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        
        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )
        
        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare visual inputs based on the fusion mode.
        
        Args:
            samples: Input samples containing visual features
            
        Returns:
            Tuple of (visual_outputs, visual_masks)
        """
        # Determine which visual features to use based on fusion mode
        if self.fusion_mode in ['joint', 'adaptive']:
            spatial = spatiotemporal = pose = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'
            pose = self.fusion_mode == 'pose'

        # Process spatial features
        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)

            if self.training and hasattr(self, 'use_data_augmentation') and self.use_data_augmentation:
                pixel_values = self.augmenter(pixel_values, samples['num_frames'])

            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
        
        # Process spatiotemporal features
        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            
            if self.training and hasattr(self, 'use_data_augmentation') and self.use_data_augmentation:
                spatiotemporal_outputs = self.augmenter(spatiotemporal_outputs, samples['glor_lengths'])
            
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
        
        # Process pose features if needed
        if pose:
            # pose_values should be a list of [T, pose_input_size] tensors (cpu tensors from get_inputs)
            raw_pose_values = samples.get('pose_values', [])
            pose_values_local = [pv if pv.dim() == 2 else pv.view(pv.shape[0], -1) for pv in raw_pose_values]
            if len(pose_values_local) > 0:
                # pad and move to device; ensure float dtype
                pose_padded = pad_sequence(pose_values_local, batch_first=True).to(self.device).float()
                pose_lengths = [int(p.size(0)) for p in pose_values_local]
            else:
                B = len(samples['pixel_values'])
                pose_padded = torch.zeros((B, 1, self.pose_input_size), device=self.device, dtype=torch.float32)
                pose_lengths = [0] * B
            pose_outputs = self.pose_proj(pose_padded)
            pose_mask = create_mask(seq_lengths=pose_lengths, device=self.device)
        
        # Combine features for joint mode
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            pose_length = pose_mask.sum(1) if pose else torch.zeros_like(spatial_length)
            new_length = spatial_length + spatiotemporal_length + pose_length

            # Concatenate spatial, spatiotemporal and pose features for each sample
            joint_outputs = []
            for i in range(bs):
                parts = []
                if spatial:
                    parts.append(spatial_outputs[i, :spatial_length[i], :])
                if spatiotemporal:
                    parts.append(spatiotemporal_outputs[i, :spatiotemporal_length[i], :])
                if pose:
                    parts.append(pose_outputs[i, :pose_length[i], :])
                concat_sample = torch.cat(parts, dim=0)
                joint_outputs.append(concat_sample)
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)

            if self.use_temporal_conv:
                visual_conv_outputs = self.temporal_encoder(
                    joint_outputs.permute(0,2,1), torch.tensor(new_length.tolist(), device=self.device)
                )
                visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(
                    seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(),
                    device=self.device
                )
            else:
                visual_outputs = joint_outputs
                visual_masks = create_mask(seq_lengths=new_length.tolist(), device=self.device)

            if self.use_second_encoder:
                enc_mask = visual_masks.unsqueeze(1)
                visual_outputs, _ = self.second_encoder(
                    embed_src=visual_outputs,
                    src_length=visual_masks.sum(1).long(),
                    mask=enc_mask,
                )
        
        elif self.fusion_mode == 'adaptive':
            
            # 1. Align Dimensions: Interpolate spatiotemporal (T_st) to match spatial (T_s) length
            # Input shapes are (B, T, C). Interpolate expects (B, C, T)
            if spatial_outputs.shape[1] != spatiotemporal_outputs.shape[1]:
                spatiotemporal_outputs = F.interpolate(
                    spatiotemporal_outputs.permute(0, 2, 1), 
                    size=spatial_outputs.shape[1], 
                    mode='linear', 
                    align_corners=False
                ).permute(0, 2, 1)

            # 2. Apply Adaptive Fusion
            # Returns (B, T, C)
            fused_outputs = self.adaptive_fusion(spatial_outputs, spatiotemporal_outputs, pose_outputs)

            if self.use_temporal_conv:
                visual_conv_outputs = self.temporal_encoder(
                    fused_outputs.permute(0, 2, 1),
                    torch.tensor(samples['num_frames'], device=self.device)
                )
                visual_outputs = visual_conv_outputs['visual_feat'].permute(1, 0, 2)
                visual_masks = create_mask(
                    seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(),
                    device=self.device
                )
            else:
                visual_outputs = fused_outputs
                visual_masks = create_mask(seq_lengths=samples['num_frames'], device=self.device)

            if self.use_second_encoder:
                enc_mask = visual_masks.unsqueeze(1)
                visual_outputs, _ = self.second_encoder(
                    embed_src=visual_outputs,
                    src_length=visual_masks.sum(1).long(),
                    mask=enc_mask,
                )

        else:
            # Single feature mode
            if spatial:
                active_outputs, active_lens = spatial_outputs, samples['num_frames']
            elif spatiotemporal:
                active_outputs, active_lens = spatiotemporal_outputs, samples['glor_lengths']
                visual_outputs = spatiotemporal_outputs
                visual_masks = spatiotemporal_mask
            elif pose:
                # For pose-only mode, run temporal encoder on pose features
                if self.use_temporal_conv:
                    pose_conv_outputs = self.temporal_encoder(
                        pose_outputs.permute(0,2,1), torch.tensor(pose_lengths, device=self.device)
                    )
                    visual_outputs = pose_conv_outputs['visual_feat'].permute(1,0,2)
                    visual_masks = create_mask(
                        seq_lengths=pose_conv_outputs['feat_len'].to(torch.int).tolist(),
                        device=self.device
                    )
                else:
                    visual_outputs = pose_outputs
                    visual_masks = create_mask(seq_lengths=pose_lengths, device=self.device)

                if self.use_second_encoder:
                    enc_mask = visual_masks.unsqueeze(1)
                    visual_outputs, _ = self.second_encoder(
                        embed_src=visual_outputs,
                        src_length=visual_masks.sum(1).long(),
                        mask=enc_mask,
                    )
            else:
                raise NotImplementedError("Invalid fusion mode")
            
            if self.fusion_mode == 'spatiotemporal':
                visual_outputs = active_outputs
                visual_masks = create_mask(seq_lengths=active_lens, device=self.device)
            else:
                if self.use_temporal_conv:
                    conv_outputs = self.temporal_encoder(
                        active_outputs.permute(0,2,1), torch.tensor(active_lens, device=self.device)
                    )
                    visual_outputs = conv_outputs['visual_feat'].permute(1,0,2)
                    visual_masks = create_mask(
                        seq_lengths=conv_outputs['feat_len'].to(torch.int).tolist(),
                        device=self.device
                    )
                else:
                    visual_outputs = active_outputs
                    visual_masks = create_mask(seq_lengths=active_lens, device=self.device)

                if self.use_second_encoder:
                    enc_mask = visual_masks.unsqueeze(1)
                    visual_outputs, _ = self.second_encoder(
                        embed_src=visual_outputs,
                        src_length=visual_masks.sum(1).long(),
                        mask=enc_mask,
                    )

        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        """
        Process batch inputs into a structured dictionary.
        
        Args:
            batch: Raw batch from dataloader

        Returns:
            Processed inputs dictionary
        """
        pixel_values, glor_values, masks, ids = [], [], [], []
        pose_values = []
        texts, glosses = [], []
        num_frames, glor_lengths, langs = [], [], []
        ex_lang_translations = []

        max_frame_len = self.max_frame_len

        for sample in batch:
            if sample.get('pixel_value') is None or sample['pixel_value'].shape[0] == 0:
                continue

            nframe = math.ceil(sample['num_frames'] / self.frame_sample_rate)
            pval = sample['pixel_value'][::self.frame_sample_rate]

            ids.append(sample['id'])
            texts.append(sample['text'].lower())
            glosses.append(sample['gloss'])
            langs.append(sample['lang'])

            # <--- FIX: Clean context handling
            _ex_lang_trans = []
            if self.num_in_context > 0:
                if 'en_text' in sample and 'text' in sample:
                    _ex_lang_trans = [
                        f"{sample.get('en_text','')}={sample['text']}",
                        f"{sample.get('fr_text','')}={sample['text']}",
                        f"{sample.get('es_text','')}={sample['text']}"
                    ]
            
                # Keep only the number requested
                trimmed = _ex_lang_trans[:self.num_in_context]
            
                # Join them into one string
                ex_lang_translations.append(' '.join(trimmed))
            else:
                ex_lang_translations.append("")


            if nframe > max_frame_len:
                nframe = max_frame_len
                start_index = random.randint(0, pval.size(0) - max_frame_len)
                pval = pval[start_index:start_index + max_frame_len]

            num_frames.append(nframe)
            pixel_values.append(pval)

            # Removed Pose and I3D processing blocks

            if sample.get('glor_value') is not None:
                if isinstance(sample['glor_value'], list):
                    glor_values.append(torch.cat(sample['glor_value'], dim=0))
                    glor_lengths.append(sum(len(g) for g in sample['glor_value']))
                else:
                    glor_values.append(sample['glor_value'])
                    glor_lengths.append(len(sample['glor_value']))

        # Only shuffle if we are actually USING context
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
            samples['text'],
            padding="longest",
            return_tensors="pt",
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
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        visual_outputs = self.fusion_proj(visual_outputs)
        
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
                
                # Visual-textual contrastive loss (CLIP-style)
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss
                
                # Sign Contrastive Learning loss (SignCL) - temporal neighborhoods
                if (
                    self.sign_cl_loss
                    and self.sign_cl is not None
                    and (self.sign_cl_every_n_steps <= 1 or (self.global_step % self.sign_cl_every_n_steps) == 0)
                ):
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
                
                # Add visual-textual contrastive component
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss = t5_loss + self.alpha * cont_loss
                log_dict[f"{split}/contra_loss"] = cont_loss
                
                # Add SignCL loss if enabled (reduces representation density)
                if (
                    self.sign_cl_loss
                    and self.sign_cl is not None
                    and (self.sign_cl_every_n_steps <= 1 or (self.global_step % self.sign_cl_every_n_steps) == 0)
                ):
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
        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m") 
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m") 
            print("-" * 50)
            
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='val',
            device=self.device
        )
        
        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def on_test_epoch_end(self) -> None:
        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m") 
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m") 
            print("-" * 50)
            
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='test',
            device=self.device
        )

        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def configure_optimizers(self):
        # 1. Filter parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found.")

        # 2. Setup AdamW 
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.98)
        )
        
        # 3. Dynamic Step Calculation
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
        
        # 4. Warmup Logic (Priority: YAML value)
        if self.warm_up_steps is not None:
            warmup_steps = self.warm_up_steps
        else:
            warmup_steps = int(total_steps * 0.1)

        print(f"--> Optimizer Setup: Total Steps={total_steps}, Warmup Steps={warmup_steps}")

        # 5. Cosine Scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # log parameter counts for debugging DDP issues
        try:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in trainable_params)
            self.log('model/total_params', float(total), prog_bar=False)
            self.log('model/trainable_params', float(trainable), prog_bar=False)
        except Exception:
            # logging may not be available at construction time in some contexts
            pass

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

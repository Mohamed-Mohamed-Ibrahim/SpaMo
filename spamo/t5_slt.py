import os
import torch
import torch.nn as nn
import random
import math
from typing import Dict, List, Optional, Tuple, Any

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from spamo.tconv import TemporalConv
from utils.helpers import create_mask, derangement
from spamo.mm_projector import build_vision_projector, AdaptiveFusion, EmotionEnhancer, EmotionModulator
from utils.evaluate import evaluate_results
from spamo.clip_loss import clip_loss
from spamo.sign_cl import TemporalSignCLLoss
from spamo.asb import AbstractSLT
from spamo.data_augmentation import FeatureAugmenter
from spamo.spatial_temporal_cl import SpatialTemporalCLLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')


class FlanT5SLT(AbstractSLT):
    def __init__(
        self,
        tuning_type: str = 'lora',
        model_name: Optional[str] = None,
        weight_decay: float = 0.01,
        frame_sample_rate: int = 1,
        prompt: str = '',
        lr: float = 6e-4,
        input_size: int = 2048,
        pose_input_size: int = 33 * 3,
        emotion_input_size: int = 768,
        fusion_mode: str = 'joint',
        inter_hidden: int = 1024,
        max_frame_len: int = 512,
        max_txt_len: int = 128,
        cross_modal_align: bool = True,
        warm_up_steps: Optional[int] = None,
        combined_loss: bool = True,
        alpha: float = 1.0,
        sign_cl_loss: bool = False,
        sign_cl_alpha: float = 0.1,
        sign_cl_temperature: float = 0.3,
        sign_cl_temporal_window: int = 12,
        sign_cl_every_n_steps: int = 1,
        use_resampler: bool = False,
        sampling_length: int = 64,
        cache_dir: str = "/data/models",
        use_in_context: bool = True,
        num_in_context: int = 3,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_data_augmentation: bool = False,
        augmentation_prob: float = 0.5,
        aug_frame_prob: float = 0.1,
        aug_span_prob: float = 0.1,
        aug_channel_prob: float = 0.05,
        use_spatial: bool = True,
        use_spatiotemporal: bool = True,
        use_pose: bool = False,
        use_emotion: bool = True,
        # --- Spatial <-> Spatiotemporal (+ Emotion) contrastive loss ---
        st_cl_loss: bool = False,
        st_cl_alpha: float = 1,
        st_cl_alignment_mode: str = "padding",
        st_cl_proj_dim: int = 0,
        st_cl_num_heads: int = 4,
        st_cl_attn_dropout: float = 0.0,
        st_cl_temperature_init: float = 2.6592,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.pose_input_size = pose_input_size
        self.emotion_input_size = emotion_input_size
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
        self.use_spatial = use_spatial
        self.use_spatiotemporal = use_spatiotemporal
        self.use_pose = use_pose
        self.use_emotion = use_emotion

        self.st_cl_loss = st_cl_loss
        self.st_cl_alpha = st_cl_alpha
        self.st_cl_alignment_mode = st_cl_alignment_mode
        self.st_cl_proj_dim = st_cl_proj_dim
        self.st_cl_num_heads = st_cl_num_heads
        self.st_cl_attn_dropout = st_cl_attn_dropout
        self.st_cl_temperature_init = st_cl_temperature_init
        
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_data_augmentation = use_data_augmentation

        if self.num_in_context == 0:
            self.use_in_context = False

        self.save_hyperparameters()
        self.prepare_models(model_name)

        if tuning_type == 'freeze':
            self._freeze_model()
        elif tuning_type == 'lora':
            self._apply_lora()

        if self.use_data_augmentation:
            self.augmenter = FeatureAugmenter(
                aug_prob=augmentation_prob,
                frame_dropout_prob=aug_frame_prob,
                span_mask_prob=aug_span_prob,
                channel_drop_prob=aug_channel_prob
            )

        self.set_container()

    def load_pretrained_weights(self, checkpoint_path: str) -> None:
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
        for param in self.t5_model.parameters():
            param.requires_grad = False

    def set_container(self) -> None:
        self.generated = []
        self.references = []

    def prepare_models(self, t5_model: str) -> None:
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float32,
            use_safetensors=True
        )

        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model,
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,
        )

        self.spatio_proj = build_vision_projector('linear', 2048, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', 1024, self.inter_hidden)
        self.pose_proj = build_vision_projector('linear', self.pose_input_size, self.inter_hidden)
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)

        if self.use_emotion:
            self.emotion_proj = build_vision_projector('linear', self.emotion_input_size, self.inter_hidden)
            self.emotion_enhancer = EmotionEnhancer(self.inter_hidden)
            self.emotion_modulator_s = EmotionModulator(self.inter_hidden)
            self.emotion_modulator_m = EmotionModulator(self.inter_hidden)
            self.emotion_modulator_e = EmotionModulator(self.inter_hidden)

        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

        if self.fusion_mode == 'adaptive':
            self.adaptive_fusion = AdaptiveFusion(
                input_size_1=self.inter_hidden,
                input_size_2=self.inter_hidden,
                input_size_3=self.inter_hidden,
                output_size=3
            )

        if self.sign_cl_loss:
            self.sign_cl = TemporalSignCLLoss(
                temperature=self.sign_cl_temperature,
                temporal_window=self.sign_cl_temporal_window,
            )
        else:
            self.sign_cl = None

        # Spatial <-> Spatiotemporal (+ Emotion) contrastive loss (before merge)
        if self.st_cl_loss:
            self.st_cl = SpatialTemporalCLLoss(
                dim=self.inter_hidden,
                alignment_mode=self.st_cl_alignment_mode,
                proj_dim=self.st_cl_proj_dim,
                temperature_init=self.st_cl_temperature_init,
                num_heads=self.st_cl_num_heads,
                attn_dropout=self.st_cl_attn_dropout,
            )
            print(
                f"[ST-CL] Enabled | mode={self.st_cl_alignment_mode}, "
                f"proj_dim={self.st_cl_proj_dim}, alpha={self.st_cl_alpha}"
            )
        else:
            self.st_cl = None

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
            joint_outputs.append(torch.cat((vis_out, prompt_embeds), dim=0))

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

    def prepare_visual_inputs(
        self, samples: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.fusion_mode == 'joint':
            spatial = self.use_spatial
            spatiotemporal = self.use_spatiotemporal
            pose = self.use_pose
        elif self.fusion_mode == 'adaptive':
            spatial = spatiotemporal = pose = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'
            pose = self.fusion_mode == 'pose'

        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            if self.training and hasattr(self, 'use_data_augmentation') and self.use_data_augmentation:
                pixel_values = self.augmenter(pixel_values, samples['num_frames'])
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)

        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            if self.training and hasattr(self, 'use_data_augmentation') and self.use_data_augmentation:
                spatiotemporal_outputs = self.augmenter(spatiotemporal_outputs, samples['glor_lengths'])
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)

        if pose:
            raw_pose_values = samples.get('pose_values', [])
            pose_values_local = [pv if pv.dim() == 2 else pv.view(pv.shape[0], -1) for pv in raw_pose_values]
            if len(pose_values_local) > 0:
                pose_padded = pad_sequence(pose_values_local, batch_first=True).to(self.dtype)
                pose_lengths = [int(p.size(0)) for p in pose_values_local]
            else:
                B = len(samples['pixel_values'])
                pose_padded = torch.zeros((B, 1, self.pose_input_size), device=self.device, dtype=self.dtype)
                pose_lengths = [0] * B
            pose_outputs = self.pose_proj(pose_padded)
            pose_mask = create_mask(seq_lengths=pose_lengths, device=self.device)

        # ---- Emotion modulation (applied BEFORE ST-CL so the loss sees emotion-aware features) ----
        emotion_mask = None
        Ze_proj = None
        Ze_mod = None
        if self.use_emotion and len(samples.get('emotion_values', [])) > 0:
            Ze_padded = pad_sequence(samples['emotion_values'], batch_first=True).to(self.device).float()
            emotion_mask = create_mask(seq_lengths=samples['emotion_lengths'], device=self.device)
            Ze_proj = self.emotion_proj(Ze_padded)
            Ze_g = self.emotion_enhancer(Ze_proj, emotion_mask)
            if spatial:
                spatial_outputs = self.emotion_modulator_s(spatial_outputs, Ze_g)
            if spatiotemporal:
                spatiotemporal_outputs = self.emotion_modulator_m(spatiotemporal_outputs, Ze_g)
            Ze_mod = self.emotion_modulator_e(Ze_proj, Ze_g)

        # ---- Spatial <-> Spatiotemporal (<-> Emotion) contrastive loss (before merge) ----
        # NOTE: computed AFTER emotion modulation so all three streams are emotion-aware.
        st_cl_loss_val: Optional[torch.Tensor] = None
        if (
            self.st_cl_loss
            and self.st_cl is not None
            and spatial
            and spatiotemporal
        ):
            st_cl_loss_val = self.st_cl(
                spatial_outputs,
                spatial_mask,
                spatiotemporal_outputs,
                spatiotemporal_mask,
                emotion=Ze_proj,
                emotion_mask=emotion_mask,
            )

        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            pose_length = pose_mask.sum(1) if pose else torch.zeros_like(spatial_length)
            emotion_length = emotion_mask.sum(1) if (self.use_emotion and emotion_mask is not None) else torch.zeros_like(spatial_length)
            new_length = spatial_length + spatiotemporal_length + pose_length + emotion_length

            joint_outputs = []
            for i in range(bs):
                parts = []
                if spatial:
                    parts.append(spatial_outputs[i, :spatial_length[i], :])
                if spatiotemporal:
                    parts.append(spatiotemporal_outputs[i, :spatiotemporal_length[i], :])
                if pose:
                    parts.append(pose_outputs[i, :pose_length[i], :])
                if self.use_emotion and Ze_mod is not None:
                    parts.append(Ze_mod[i, :emotion_length[i], :])
                joint_outputs.append(torch.cat(parts, dim=0))

            joint_outputs = pad_sequence(joint_outputs, batch_first=True)
            visual_conv_outputs = self.temporal_encoder(
                joint_outputs.permute(0, 2, 1),
                torch.tensor(new_length.tolist(), device=self.device)
            )
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1, 0, 2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(),
                device=self.device
            )

        elif self.fusion_mode == 'adaptive':
            if spatial_outputs.shape[1] != spatiotemporal_outputs.shape[1]:
                spatiotemporal_outputs = F.interpolate(
                    spatiotemporal_outputs.permute(0, 2, 1),
                    size=spatial_outputs.shape[1],
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)
            fused_outputs = self.adaptive_fusion(spatial_outputs, spatiotemporal_outputs, pose_outputs)
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
            if spatial:
                active_outputs, active_lens = spatial_outputs, samples['num_frames']
            elif spatiotemporal:
                active_outputs, active_lens = spatiotemporal_outputs, samples['glor_lengths']
            elif pose:
                pose_conv_outputs = self.temporal_encoder(
                    pose_outputs.permute(0, 2, 1),
                    torch.tensor(pose_lengths, device=self.device)
                )
                visual_outputs = pose_conv_outputs['visual_feat'].permute(1, 0, 2)
                visual_masks = create_mask(
                    seq_lengths=pose_conv_outputs['feat_len'].to(torch.int).tolist(),
                    device=self.device
                )
                return visual_outputs, visual_masks
            else:
                raise NotImplementedError("Invalid fusion mode")

            if self.fusion_mode == 'spatiotemporal':
                visual_outputs = active_outputs
                visual_masks = create_mask(seq_lengths=active_lens, device=self.device)
            else:
                conv_outputs = self.temporal_encoder(
                    active_outputs.permute(0, 2, 1),
                    torch.tensor(active_lens, device=self.device)
                )
                visual_outputs = conv_outputs['visual_feat'].permute(1, 0, 2)
                visual_masks = create_mask(
                    seq_lengths=conv_outputs['feat_len'].to(torch.int).tolist(),
                    device=self.device
                )

        return visual_outputs, visual_masks, st_cl_loss_val

    def get_inputs(self, batch: List) -> Dict:
        pixel_values, glor_values, masks, ids = [], [], [], []
        pose_values = []
        emotion_values, emotion_lengths = [], []
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

            _ex_lang_trans = []
            if self.num_in_context > 0:
                if 'en_text' in sample and 'text' in sample:
                    _ex_lang_trans = [
                        f"{sample.get('en_text', '')}={sample['text']}",
                        f"{sample.get('fr_text', '')}={sample['text']}",
                        f"{sample.get('es_text', '')}={sample['text']}"
                    ]
                trimmed = _ex_lang_trans[:self.num_in_context]
                ex_lang_translations.append(' '.join(trimmed))
            else:
                ex_lang_translations.append("")

            if nframe > max_frame_len:
                nframe = max_frame_len
                start_index = random.randint(0, pval.size(0) - max_frame_len)
                pval = pval[start_index:start_index + max_frame_len]

            num_frames.append(nframe)
            pixel_values.append(pval.float())

            pv = sample.get('pose_value')
            if pv is not None and isinstance(pv, torch.Tensor) and pv.numel() > 0:
                if pv.dim() == 1:
                    pv = pv.unsqueeze(0)
                pose_values.append(pv.float())
            else:
                pose_values.append(torch.zeros(1, self.pose_input_size, dtype=torch.float32))

            ev = sample.get('emotion_value')
            if ev is not None and isinstance(ev, torch.Tensor) and ev.numel() > 0:
                emotion_values.append(ev.float())
                emotion_lengths.append(len(ev))
            else:
                emotion_values.append(torch.zeros(1, self.emotion_input_size, dtype=torch.float32))
                emotion_lengths.append(0)

            if sample.get('glor_value') is not None:
                if isinstance(sample['glor_value'], list):
                    glor_values.append(torch.cat(sample['glor_value'], dim=0).float())
                    glor_lengths.append(sum(len(g) for g in sample['glor_value']))
                else:
                    glor_values.append(sample['glor_value'].float())
                    glor_lengths.append(len(sample['glor_value']))

        if self.use_in_context and len(ex_lang_translations) > 1:
            if self.training:
                ex_lang_translations = derangement(ex_lang_translations)

        return {
            'pixel_values': pixel_values,
            'glor_values': glor_values,
            'pose_values': pose_values,
            'emotion_values': emotion_values,
            'emotion_lengths': emotion_lengths,
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

        text_embeds_raw = self.t5_model.encoder.embed_tokens(output_tokens.input_ids)

        mask_v = visual_masks.unsqueeze(-1).float()
        image_embeds = (visual_outputs.float() * mask_v).sum(1) / mask_v.sum(1).clamp(min=1e-6)

        mask_t = output_tokens.attention_mask.unsqueeze(-1).float()
        text_embeds = (text_embeds_raw.float() * mask_t).sum(1) / mask_t.sum(1).clamp(min=1e-6)

        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

        return clip_loss(logits_per_text)

    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        visual_outputs, visual_masks, st_cl_loss_val = self.prepare_visual_inputs(inputs)
        visual_outputs = self.fusion_proj(visual_outputs)

        log_dict = {}

        # ---- Spatial <-> Spatiotemporal contrastive loss (before merge) ----
        if st_cl_loss_val is not None:
            log_dict[f"{split}/st_cl_loss"] = st_cl_loss_val

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
                    return_dict=True
                )
                t5_loss = F.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                    label_smoothing=0.1
                )
                log_dict[f"{split}/loss"] = t5_loss

                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss = t5_loss + self.alpha * cont_loss
                log_dict[f"{split}/contra_loss"] = cont_loss

                if st_cl_loss_val is not None:
                    loss = loss + self.st_cl_alpha * st_cl_loss_val

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
                return_dict=True
            )
            loss = F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                label_smoothing=0.1
            )
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
                do_sample=False,
                early_stopping=True,
            )
            generated_strings = self.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_strings = [g.lower() for g in generated_strings]

            reference_strings = self.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            reference_strings = [r.lower() for r in reference_strings]

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
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found.")

        lora_params = [p for n, p in self.named_parameters()
                    if p.requires_grad and ('lora_' in n or 'logit_scale' in n)]

        bridge_params = [p for n, p in self.named_parameters()
                        if p.requires_grad and 'fusion_proj' in n]

        other_params = [p for n, p in self.named_parameters()
                        if p.requires_grad
                        and 'lora_' not in n
                        and 'logit_scale' not in n
                        and 'fusion_proj' not in n]

        optimizer = torch.optim.AdamW([
            {'params': lora_params,   'lr': self.hparams.lr},
            {'params': bridge_params, 'lr': self.hparams.lr * 1.3},
            {'params': other_params,  'lr': self.hparams.lr * 1.7},
        ], weight_decay=self.hparams.weight_decay, eps=1e-8, betas=(0.9, 0.98))

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

        if self.warm_up_steps is not None:
            warmup_steps = self.warm_up_steps
        else:
            warmup_steps = int(total_steps * 0.1)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        try:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in trainable_params)
            self.log('model/total_params', float(total), prog_bar=False)
            self.log('model/trainable_params', float(trainable), prog_bar=False)
        except Exception:
            pass

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
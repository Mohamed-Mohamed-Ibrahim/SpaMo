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
from utils.helpers import create_mask, derangement
from spamo.mm_projector import build_vision_projector
from utils.evaluate import evaluate_results
from spamo.clip_loss import clip_loss
from spamo.asb import AbstractSLT

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PoseGatingUnit(nn.Module):
    """
    A gating mechanism that filters noisy pose features.
    It learns a sigmoid gate: Output = Input * Sigmoid(Linear(Input))
    """
    def __init__(self, hidden_size, dropout_rate=0.3):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        x = x * gate
        return self.dropout(self.norm(x))


class FlanT5SLT(AbstractSLT):
    """
    FlanT5-based Sign Language Translation model with multimodal capabilities.
    """
    def __init__(
        self, 
        tuning_type: str = 'lora', 
        model_name: Optional[str] = None, 
        frame_sample_rate: int = 1, 
        prompt: str = '',
        lr: float = 1e-4,             # <--- FIXED: Added lr here
        input_size: int = 1024,
        pose_input_size: int = 33*3,
        i3d_input_size: int = 1024,
        fusion_mode: str = 'joint',
        inter_hidden: int = 768,
        max_frame_len: int = 1024,
        max_txt_len: int = 64,
        cross_modal_align: bool = False,
        warm_up_steps: Optional[int] = None,
        combined_loss: bool = False,
        alpha: float = 0.1,
        use_resampler: bool = False,
        sampling_length: int = 64,
        cache_dir: str = "/data3/models",
        use_in_context: bool = False,
        num_in_context: int = 0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Configuration parameters
        self.input_size = input_size
        self.pose_input_size = pose_input_size
        self.i3d_input_size = i3d_input_size
        self.prompt = prompt
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
        self.use_resampler = use_resampler
        self.sampling_length = sampling_length
        self.cache_dir = cache_dir
        
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        
        # <--- FIXED: Force disable context if set to 0
        if self.num_in_context == 0:
            self.use_in_context = False
        
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Save hyperparameters explicitly ensures 'lr' is available
        self.save_hyperparameters() 
        
        self.prepare_models(model_name)

        # Apply the selected tuning strategy
        if tuning_type == 'freeze':
            self._freeze_model()
        elif tuning_type == 'lora':
            self._apply_lora()

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

        # Load the vision projectors
        self.spatio_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', 1024, self.inter_hidden)
        
        # Pose projector with Gating
        self.pose_proj = build_vision_projector('mlp2x_gelu', self.pose_input_size, self.inter_hidden)
        self.pose_gating = PoseGatingUnit(self.inter_hidden, dropout_rate=0.3)
        
        # I3D Projector
        self.i3d_proj = build_vision_projector('mlp2x_gelu', self.i3d_input_size, self.inter_hidden)

        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)
        
        # Load the temporal encoder
        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

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
        # Determine fusion mode
        if self.fusion_mode in ['joint']:
            spatial = spatiotemporal = pose = i3d = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'
            pose = self.fusion_mode == 'pose'
            i3d = self.fusion_mode == 'i3d'

        # Process spatial features
        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
        
        # Process spatiotemporal features
        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
        
        # Process pose features
        if pose:
            raw_pose_values = samples.get('pose_values', [])
            pose_values_local = [pv if pv.dim() == 2 else pv.view(pv.shape[0], -1) for pv in raw_pose_values]
            target_dtype = self.t5_model.dtype
            
            if len(pose_values_local) > 0:
                pose_padded = pad_sequence(pose_values_local, batch_first=True).to(self.device).to(target_dtype)
                pose_lengths = [int(p.size(0)) for p in pose_values_local]
            else:
                B = len(samples['pixel_values'])
                pose_padded = torch.zeros((B, 1, self.pose_input_size), device=self.device, dtype=target_dtype)
                pose_lengths = [0] * B
                
            pose_outputs = self.pose_proj(pose_padded)
            pose_outputs = self.pose_gating(pose_outputs)
            pose_mask = create_mask(seq_lengths=pose_lengths, device=self.device)

        # Process I3D features
        if i3d:
            raw_i3d_values = samples.get('i3d_values', [])
            i3d_values_local = [iv.view(iv.shape[0], -1) for iv in raw_i3d_values]
            target_dtype = self.t5_model.dtype
            
            if len(i3d_values_local) > 0:
                i3d_padded = pad_sequence(i3d_values_local, batch_first=True).to(self.device).to(target_dtype)
                i3d_lengths = [int(p.size(0)) for p in i3d_values_local]
            else:
                B = len(samples['pixel_values'])
                i3d_padded = torch.zeros((B, 1, self.i3d_input_size), device=self.device, dtype=target_dtype)
                i3d_lengths = [0] * B
            
            i3d_outputs = self.i3d_proj(i3d_padded)
            i3d_mask = create_mask(seq_lengths=i3d_lengths, device=self.device)

        # Combine features
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            pose_length = pose_mask.sum(1) if pose else torch.zeros_like(spatial_length)
            i3d_length = i3d_mask.sum(1) if i3d else torch.zeros_like(spatial_length)

            new_length = spatial_length + spatiotemporal_length + pose_length + i3d_length

            # Concatenate features
            joint_outputs = []
            for i in range(bs):
                parts = []
                if spatial: parts.append(spatial_outputs[i, :spatial_length[i], :])
                if spatiotemporal: parts.append(spatiotemporal_outputs[i, :spatiotemporal_length[i], :])
                if pose: parts.append(pose_outputs[i, :pose_length[i], :])
                if i3d: parts.append(i3d_outputs[i, :i3d_length[i], :])
                
                concat_sample = torch.cat(parts, dim=0)
                joint_outputs.append(concat_sample)
            
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)

            visual_conv_outputs = self.temporal_encoder(
                joint_outputs.permute(0,2,1), torch.tensor(new_length.tolist(), device=self.device)
            )

            visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(), 
                device=self.device
            ) 
        else:
            # Single feature mode
            if spatial:
                active_outputs, active_lens = spatial_outputs, samples['num_frames']
            elif spatiotemporal:
                active_outputs, active_lens = spatiotemporal_outputs, samples['glor_lengths']
            elif pose:
                active_outputs, active_lens = pose_outputs, pose_lengths
            elif i3d:
                active_outputs, active_lens = i3d_outputs, i3d_lengths
            else:
                raise NotImplementedError("Invalid fusion mode")
            
            if self.fusion_mode == 'spatiotemporal':
                 visual_outputs = active_outputs
                 visual_masks = create_mask(seq_lengths=active_lens, device=self.device)
            else:
                conv_outputs = self.temporal_encoder(
                    active_outputs.permute(0,2,1), torch.tensor(active_lens, device=self.device)
                )
                visual_outputs = conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(
                    seq_lengths=conv_outputs['feat_len'].to(torch.int).tolist(), 
                    device=self.device
                )

        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        pixel_values, glor_values, masks, ids = [], [], [], []
        pose_values = []
        i3d_values = []
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

            # <--- FIXED: Check for empty context
            _ex_lang_trans = []
            if self.num_in_context > 0:
                if 'en_text' in sample and 'text' in sample:
                    _ex_lang_trans = [
                        f"{sample.get('en_text','')}={sample['text']}",
                        f"{sample.get('fr_text','')}={sample['text']}",
                        f"{sample.get('es_text','')}={sample['text']}"
                    ]
                ex_lang_trans = _ex_lang_trans[:self.num_in_context]
                ex_lang_translations.append(' '.join(_ex_lang_trans))
            else:
                ex_lang_translations.append("")

            if nframe > max_frame_len:
                nframe = max_frame_len
                start_index = random.randint(0, pval.size(0) - max_frame_len)
                pval = pval[start_index:start_index + max_frame_len]

            num_frames.append(nframe)
            pixel_values.append(pval)

            # --- POSE PROCESSING ---
            if 'pose_value' in sample and sample['pose_value'] is not None and sample['pose_value'].numel() != 0:
                pose_arr = sample['pose_value'][::self.frame_sample_rate]
                if pose_arr.dim() == 3:
                    pose_arr = pose_arr.view(pose_arr.shape[0], -1)
                
                # Instance Normalization for Pose
                mask = (pose_arr != 0).float()
                mean = (pose_arr * mask).sum(dim=0, keepdim=True) / (mask.sum(dim=0, keepdim=True) + 1e-6)
                pose_arr = pose_arr - mean
                std = (pose_arr * mask).std(dim=0, keepdim=True) + 1e-6
                pose_arr = pose_arr / std
                pose_arr = pose_arr * mask

                if pose_arr.size(0) > max_frame_len:
                    start_index = random.randint(0, pose_arr.size(0) - max_frame_len)
                    pose_arr = pose_arr[start_index:start_index + max_frame_len]
                
                pose_values.append(pose_arr)

            # --- I3D PROCESSING ---
            if 'i3d_feat' in sample and sample['i3d_feat'] is not None and sample['i3d_feat'].numel() != 0:
                i3d_arr = sample['i3d_feat'] 
                
                if i3d_arr.size(0) > max_frame_len:
                    start_index = random.randint(0, i3d_arr.size(0) - max_frame_len)
                    i3d_arr = i3d_arr[start_index:start_index + max_frame_len]
                
                i3d_values.append(i3d_arr)

            if sample.get('glor_value') is not None:
                if isinstance(sample['glor_value'], list):
                    glor_values.append(torch.cat(sample['glor_value'], dim=0))
                    glor_lengths.append(sum(len(g) for g in sample['glor_value']))
                else:
                    glor_values.append(sample['glor_value'])
                    glor_lengths.append(len(sample['glor_value']))

        if len(ex_lang_translations) > 1:
            ex_lang_translations = derangement(ex_lang_translations)

        return {
            'pixel_values': pixel_values,
            'glor_values': glor_values,
            'pose_values': pose_values,
            'i3d_values': i3d_values,
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
                
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss
                
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
            raise RuntimeError("No trainable parameters found. Check freezing/LoRA setup.")

        # 2. Setup AdamW 
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.lr,  
            eps=1e-8,
            weight_decay=0.01,
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
        
        # 4. Warmup Logic
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

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
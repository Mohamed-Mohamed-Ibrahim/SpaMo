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
from spamo.mm_projector import build_vision_projector, AdaptiveFusion
from utils.evaluate import evaluate_results
from spamo.clip_loss import clip_loss
from spamo.sign_cl import TemporalSignCLLoss
from spamo.asb import AbstractSLT
from spamo.data_augmentation import FeatureAugmenter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

class FlanT5SLT(AbstractSLT):
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
        gfslt_dim: int = 1024,           
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
        augmentation_prob: float = 0.5,
        aug_frame_prob: float = 0.1,
        aug_span_prob: float = 0.1,
        aug_channel_prob: float = 0.05,
        use_spatial: bool = True,        # Restored to allow switching via YAML
        use_spatiotemporal: bool = True, # Restored to allow switching via YAML
        use_pose: bool = False,
        use_gfslt: bool = True,          # New GFSLT flag
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_size = input_size
        self.pose_input_size = pose_input_size
        self.gfslt_dim = gfslt_dim       
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
        self.use_gfslt = use_gfslt       
        
        if self.num_in_context == 0:
            self.use_in_context = False
        
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_data_augmentation = use_data_augmentation
        print("==="*40)
        print(f"Features Enabled -> Spatial: {self.use_spatial}, SpatioTemporal: {self.use_spatiotemporal}, Pose: {self.use_pose}, GFSLT: {self.use_gfslt}")
        print("==="*40)
        
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

    def _freeze_model(self) -> None:
        self.t5_model.eval()
        for params in self.t5_model.parameters():
            params.requires_grad = False

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

        # Initialize projectors
        self.spatio_proj = build_vision_projector('linear', 2048, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', 1024, self.inter_hidden)
        self.pose_proj = build_vision_projector('linear', self.pose_input_size, self.inter_hidden)
        self.gfslt_proj = build_vision_projector('linear', self.gfslt_dim, self.inter_hidden)
        
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)
        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)
        
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
            vis_out = visual_outputs[i, :int(visual_lengths[i].item()), :]
            prompt_embeds = input_embeds[i, :int(prompt_lengths[i].item()), :]
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
        spatial_outputs, spatial_mask = None, None
        spatiotemporal_outputs, spatiotemporal_mask = None, None
        pose_outputs, pose_mask = None, None
        gfslt_outputs, gfslt_mask = None, None

        if self.use_spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            if self.training and hasattr(self, 'use_data_augmentation') and self.use_data_augmentation:
                pixel_values = self.augmenter(pixel_values, samples['num_frames'])
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
        
        if self.use_spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            if self.training and hasattr(self, 'use_data_augmentation') and self.use_data_augmentation:
                spatiotemporal_outputs = self.augmenter(spatiotemporal_outputs, samples['glor_lengths'])
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
        
        if self.use_pose:
            pose_padded = pad_sequence(samples['pose_values'], batch_first=True).to(self.device).float()
            pose_outputs = self.pose_proj(pose_padded)
            pose_mask = create_mask(seq_lengths=samples['pose_lengths'], device=self.device)

        if self.use_gfslt:
            gfslt_padded = pad_sequence(samples['gfslt_values'], batch_first=True).to(self.device).float()
            if self.training and hasattr(self, 'use_data_augmentation') and self.use_data_augmentation:
                gfslt_padded = self.augmenter(gfslt_padded, samples['gfslt_lengths'])
            gfslt_outputs = self.gfslt_proj(gfslt_padded)
            gfslt_mask = create_mask(seq_lengths=samples['gfslt_lengths'], device=self.device)

        # Dynamic Fusion: ONLY concatenates features that are turned ON. No zeros.
        if self.fusion_mode == 'joint':
            bs = len(samples['ids'])
            joint_outputs = []
            new_lengths = []
            
            for i in range(bs):
                parts = []
                cur_len = 0
                
                if self.use_spatial:
                    length = int(spatial_mask.sum(1)[i].item())
                    parts.append(spatial_outputs[i, :length, :])
                    cur_len += length
                    
                if self.use_spatiotemporal:
                    length = int(spatiotemporal_mask.sum(1)[i].item())
                    parts.append(spatiotemporal_outputs[i, :length, :])
                    cur_len += length
                    
                if self.use_pose:
                    length = int(pose_mask.sum(1)[i].item())
                    parts.append(pose_outputs[i, :length, :])
                    cur_len += length
                    
                if self.use_gfslt:
                    length = int(gfslt_mask.sum(1)[i].item())
                    parts.append(gfslt_outputs[i, :length, :])
                    cur_len += length
                    
                concat_sample = torch.cat(parts, dim=0)
                joint_outputs.append(concat_sample)
                new_lengths.append(cur_len)
                
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)

            visual_conv_outputs = self.temporal_encoder(
                joint_outputs.permute(0,2,1), torch.tensor(new_lengths, device=self.device)
            )

            visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(), 
                device=self.device
            )
            
        else:
            # Independent pass if not using joint fusion
            if self.use_gfslt:
                active_outputs, active_lens = gfslt_outputs, samples['gfslt_lengths']
            elif self.use_spatial:
                active_outputs, active_lens = spatial_outputs, samples['num_frames']
            elif self.use_spatiotemporal:
                active_outputs, active_lens = spatiotemporal_outputs, samples['glor_lengths']
            elif self.use_pose:
                active_outputs, active_lens = pose_outputs, samples['pose_lengths']
            else:
                raise NotImplementedError("No valid visual features selected for projection.")
                
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
        result = {
            'ids': [], 'text': [], 'gloss': [], 'lang': [], 'ex_lang_trans': [],
            'pixel_values': [], 'num_frames': [],
            'glor_values': [], 'glor_lengths': [],
            'pose_values': [], 'pose_lengths': [],
            'gfslt_values': [], 'gfslt_lengths': [],
        }

        max_frame_len = self.max_frame_len

        for sample in batch:
            if sample.get('num_frames', 0) == 0:
                continue

            result['ids'].append(sample['id'])
            result['text'].append(sample['text'].lower())
            result['gloss'].append(sample['gloss'])
            result['lang'].append(sample['lang'])

            _ex_lang_trans = []
            if self.num_in_context > 0:
                if 'en_text' in sample and 'text' in sample:
                    _ex_lang_trans = [
                        f"{sample.get('en_text','')}=>{sample['text']}",
                        f"{sample.get('fr_text','')}=>{sample['text']}",
                        f"{sample.get('es_text','')}=>{sample['text']}"
                    ]
                trimmed = _ex_lang_trans[:self.num_in_context]
                result['ex_lang_trans'].append(' '.join(trimmed))
            else:
                result['ex_lang_trans'].append("")

            # Only extract features explicitly turned ON. No zero-padding for OFF features.
            if self.use_spatial:
                pval = sample.get('pixel_value')
                if pval is not None and len(pval) > 0:
                    nframe = math.ceil(len(pval) / self.frame_sample_rate)
                    pval = pval[::self.frame_sample_rate]
                    if nframe > max_frame_len:
                        start_index = random.randint(0, pval.size(0) - max_frame_len)
                        pval = pval[start_index:start_index + max_frame_len]
                        nframe = max_frame_len
                    result['pixel_values'].append(pval)
                    result['num_frames'].append(nframe)

            if self.use_gfslt:
                gval = sample.get('gfslt_value')
                if gval is not None and len(gval) > 0:
                    nframe = math.ceil(len(gval) / self.frame_sample_rate)
                    gval = gval[::self.frame_sample_rate]
                    if nframe > max_frame_len:
                        start_index = random.randint(0, gval.size(0) - max_frame_len)
                        gval = gval[start_index:start_index + max_frame_len]
                        nframe = max_frame_len
                    result['gfslt_values'].append(gval)
                    result['gfslt_lengths'].append(nframe)

            if self.use_pose:
                pv = sample.get('pose_value')
                if pv is not None and isinstance(pv, torch.Tensor) and pv.numel() > 0:
                    if pv.dim() == 1: pv = pv.unsqueeze(0)
                    result['pose_values'].append(pv.float())
                    result['pose_lengths'].append(int(pv.size(0)))

            if self.use_spatiotemporal:
                glor_val = sample.get('glor_value')
                if glor_val is not None and len(glor_val) > 0:
                    if isinstance(glor_val, list):
                        result['glor_values'].append(torch.cat(glor_val, dim=0))
                        result['glor_lengths'].append(sum(len(g) for g in glor_val))
                    else:
                        result['glor_values'].append(glor_val)
                        result['glor_lengths'].append(len(glor_val))

        if self.use_in_context and len(result['ex_lang_trans']) > 1:
            result['ex_lang_trans'] = derangement(result['ex_lang_trans'])

        return result

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
                
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss = t5_loss + self.alpha * cont_loss
                log_dict[f"{split}/contra_loss"] = cont_loss
                
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

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.lr,
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
        
        warmup_steps = self.warm_up_steps if self.warm_up_steps is not None else int(total_steps * 0.1)

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
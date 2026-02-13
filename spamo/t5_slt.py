%%writefile /kaggle/working/SpaMo/spamo/t5_slt.py
import os
import torch
import torch.nn as nn
import random
import math
from typing import Dict, List, Optional, Tuple, Any

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

from spamo.tconv import TemporalConv
from utils.helpers import create_mask, derangement
from spamo.mm_projector import build_vision_projector
from utils.evaluate import evaluate_results
from spamo.clip_loss import clip_loss
from spamo.asb import AbstractSLT
from transformers import get_cosine_schedule_with_warmup

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

class FlanT5SLT(AbstractSLT):
    def __init__(
        self, 
        tuning_type: str = 'lora', 
        model_name: Optional[str] = None, 
        frame_sample_rate: int = 1, 
        prompt: str = '',
        input_size: int = 1024,
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
        
        self.input_size = input_size
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
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self.prepare_models(model_name)

        if tuning_type == 'freeze':
            self._freeze_model()
        elif tuning_type == 'lora':
            self._apply_lora()

        self.set_container()
        
    def load_pretrained_weights(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model_dict = self.state_dict()
        
        # --- FIXED LOGIC: DIRECT FILTERING ---
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_dict:
                # ONLY keep if shapes match exactly
                if v.shape == model_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"[WARN] Skipping layer {k}: Checkpoint {v.shape} != Model {model_dict[k].shape}")
        
        # Load ONLY the filtered matching weights
        self.load_state_dict(filtered_state_dict, strict=False)
        print("Successfully loaded matching weights.")

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
            torch_dtype=torch.bfloat16, 
        )
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,
        )

        self.spatio_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', 1024, self.inter_hidden)
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)
        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def prepare_inputs(self, visual_outputs, visual_mask, samples, split, batch_idx):
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
        
        input_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        
        joint_outputs = []
        for i in range(bs):
            vis_out = visual_outputs[i, :visual_lengths[i], :]
            prompt_embeds = input_embeds[i, :prompt_lengths[i], :]
            concat_sample = torch.cat((vis_out, prompt_embeds), dim=0)
            joint_outputs.append(concat_sample)
        
        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        # Use simple attention mask based on lengths
        new_lengths = visual_lengths + prompt_lengths
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
        if self.fusion_mode in ['joint']:
            spatial = spatiotemporal = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'

        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
        
        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
        
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            new_length = spatial_length + spatiotemporal_length
            
            joint_outputs = []
            for i in range(bs):
                valid_spatial_output = spatial_outputs[i, :spatial_length[i], :]
                valid_spatiotemporal_output = spatiotemporal_outputs[i, :spatiotemporal_length[i], :]
                concat_sample = torch.cat((valid_spatial_output, valid_spatiotemporal_output), dim=0)
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
            if spatial:
                spatial_conv_outputs = self.temporal_encoder(
                    spatial_outputs.permute(0,2,1), torch.tensor(samples['num_frames'], device=self.device)
                )
                visual_outputs = spatial_conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(
                    seq_lengths=spatial_conv_outputs['feat_len'].to(torch.int).tolist(), 
                    device=self.device
                )
            elif spatiotemporal:
                visual_outputs = spatiotemporal_outputs
                visual_masks = spatiotemporal_mask
            
        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        pixel_values, glor_values, masks, ids = [], [], [], []
        texts, glosses = [], []
        num_frames, glor_lengths, langs = [], [], []
        ex_lang_translations = []
        max_frame_len = self.max_frame_len

        for sample in batch:
            if sample['pixel_value'].shape[0] != 0:
                nframe = math.ceil(sample['num_frames'] / self.frame_sample_rate)
                pval = sample['pixel_value'][::self.frame_sample_rate]
                ids.append(sample['id'])
                texts.append(sample['text'].lower())
                glosses.append(sample.get('gloss', ''))
                langs.append(sample.get('lang', 'en'))
                
                if self.use_in_context and self.num_in_context > 0:
                    try:
                        _ex_lang_trans = [
                            f"{sample.get('en_text', '')}={sample['text']}",
                            f"{sample.get('fr_text', '')}={sample['text']}",
                            f"{sample.get('es_text', '')}={sample['text']}"
                        ]
                        _ex_lang_trans = _ex_lang_trans[:self.num_in_context]
                        ex_lang_translations.append(' '.join(_ex_lang_trans))
                    except: ex_lang_translations.append("")
                else:
                    ex_lang_translations.append("")
                
                if nframe > max_frame_len:
                    nframe = max_frame_len
                    start_index = random.randint(0, pval.size(0) - max_frame_len)
                    pval = pval[start_index:start_index + max_frame_len]
                
                num_frames.append(nframe)
                pixel_values.append(pval)
                
                if sample['glor_value'] is not None:
                    if isinstance(sample['glor_value'], list):
                        glor_values.append(torch.cat(sample['glor_value'], dim=0))
                        glor_lengths.append(sum(len(g) for g in sample['glor_value']))
                    else:
                        glor_values.append(sample['glor_value'])
                        glor_lengths.append(len(sample['glor_value']))
        
        if self.use_in_context:
            ex_lang_translations = derangement(ex_lang_translations)
        
        return {
            'pixel_values': pixel_values,
            'glor_values': glor_values,
            'bool_mask_pos': masks,
            'ids': ids,
            'text': texts,
            'ex_lang_trans': ex_lang_translations,
            'gloss': glosses,
            'lang': langs,
            'num_frames': num_frames,
            'glor_lengths': glor_lengths,
        }

    def visual_textual_align(self, visual_outputs, visual_masks, samples):
        output_tokens = self.t5_tokenizer(
            samples['text'], padding="longest", return_tensors="pt"
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

    def shared_step(self, inputs, split, batch_idx):
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        visual_outputs = self.fusion_proj(visual_outputs)
        log_dict = {}
        loss = torch.tensor(0.0, device=self.device)

        if self.cross_modal_align:
             # Simplified for inference context: just get loss if needed
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
             loss = outputs.loss
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
            reference_strings = self.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            self.generated.extend([g.lower() for g in generated_strings])
            self.references.extend([r.lower() for r in reference_strings])

        return loss, log_dict

    def on_validation_epoch_end(self) -> None:
        eval_res = evaluate_results(self.generated, self.references, split='val', device=self.device)
        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def on_test_epoch_end(self) -> None:
        eval_res = evaluate_results(self.generated, self.references, split='test', device=self.device)
        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
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
from spamo.mm_projector import build_vision_projector, AdaptiveFusion
from utils.evaluate import evaluate_results
from spamo.clip_loss import clip_loss
from spamo.sign_cl import TemporalSignCLLoss
from spamo.asb import AbstractSLT
from spamo.data_augmentation import FeatureAugmenter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

class SpatialGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.A = nn.Parameter(torch.eye(num_nodes) + torch.randn(num_nodes, num_nodes) * 0.01)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.conv(x)
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        return self.relu(self.bn(x) + res)

class TemporalGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        return self.relu(self.bn(self.conv(x)) + res)

class SubPoseStream(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, out_dim=256, num_nodes=21):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            SpatialGCNLayer(in_channels, hidden_dim, num_nodes),
            SpatialGCNLayer(hidden_dim, hidden_dim * 2, num_nodes),
            SpatialGCNLayer(hidden_dim * 2, out_dim, num_nodes)
        )
        self.temporal_encoder = nn.Sequential(
            TemporalGCNLayer(out_dim, out_dim),
            TemporalGCNLayer(out_dim, out_dim),
            TemporalGCNLayer(out_dim, out_dim)
        )

    def forward(self, x):
        x = self.spatial_encoder(x)
        x = self.temporal_encoder(x)
        x = x.mean(dim=-1)
        return x

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
        use_spatial: bool = True,
        use_spatiotemporal: bool = True,
        use_pose: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
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
        self.use_spatial = use_spatial
        self.use_spatiotemporal = use_spatiotemporal
        self.use_pose = use_pose
        
        if self.num_in_context == 0:
            self.use_in_context = False
        
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_data_augmentation = use_data_augmentation
        
        self.save_hyperparameters()
        self.prepare_models(model_name)

        if tuning_type == 'freeze':
            self._freeze_model()
        elif tuning_type == 'lora':
            self._apply_lora()

        if self.use_data_augmentation:
            self.augmenter = FeatureAugmenter(
                aug_prob=augmentation_prob, frame_dropout_prob=aug_frame_prob,
                span_mask_prob=aug_span_prob, channel_drop_prob=aug_channel_prob
            )

        self.set_container()
        
    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])

    def _apply_lora(self) -> None:
        lora_config = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_alpha,
            target_modules=["q", "v"], lora_dropout=self.lora_dropout,
            bias="none", task_type=TaskType.SEQ_2_SEQ_LM
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
            t5_model, cache_dir=self.cache_dir, torch_dtype=torch.bfloat16, use_safetensors=True 
        )
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model, cache_dir=self.cache_dir, max_length=self.max_txt_len,
        )

        self.spatio_proj = build_vision_projector('linear', 2048, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', 1024, self.inter_hidden)
        
        out_dim_per_stream = self.inter_hidden // 4
        self.pose_stream_face = SubPoseStream(in_channels=3, out_dim=out_dim_per_stream, num_nodes=11)
        self.pose_stream_lhand = SubPoseStream(in_channels=3, out_dim=out_dim_per_stream, num_nodes=6)
        self.pose_stream_rhand = SubPoseStream(in_channels=3, out_dim=out_dim_per_stream, num_nodes=6)
        self.pose_stream_body = SubPoseStream(in_channels=3, out_dim=out_dim_per_stream, num_nodes=10)
        
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)
        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)
        
        if self.fusion_mode == 'adaptive':
            self.adaptive_fusion = AdaptiveFusion(self.inter_hidden, self.inter_hidden, self.inter_hidden, 3)
            
        if self.sign_cl_loss:
            self.sign_cl = TemporalSignCLLoss(self.sign_cl_temperature, self.sign_cl_temporal_window)
        else:
            self.sign_cl = None

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def prepare_inputs(self, visual_outputs, visual_mask, samples, split, batch_idx):
        bs = visual_outputs.shape[0]
        prompts = [f'{self.prompt}'] * bs
        prompts = [p.format(l) for p, l in zip(prompts, samples['lang'])]
        
        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples['ex_lang_trans'])]
        
        input_tokens = self.t5_tokenizer(prompts, padding="longest", truncation=True, return_tensors="pt").to(self.device)
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
        
        output_tokens = self.t5_tokenizer(samples['text'], padding="longest", return_tensors="pt").to(self.device)
        targets = output_tokens.input_ids.masked_fill(output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100)
        
        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial = self.use_spatial if self.fusion_mode == 'joint' else (self.fusion_mode in ['spatial', 'adaptive'])
        spatiotemporal = self.use_spatiotemporal if self.fusion_mode == 'joint' else (self.fusion_mode in ['spatiotemporal', 'adaptive'])
        pose = self.use_pose if self.fusion_mode == 'joint' else (self.fusion_mode in ['pose', 'adaptive'])

        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            if self.training and getattr(self, 'use_data_augmentation', False):
                pixel_values = self.augmenter(pixel_values, samples['num_frames'])
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
        
        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            if self.training and getattr(self, 'use_data_augmentation', False):
                spatiotemporal_outputs = self.augmenter(spatiotemporal_outputs, samples['glor_lengths'])
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
        
        if pose:
            raw_pose_values = samples.get('pose_values', [])
            pose_values_local = []
            for pv in raw_pose_values:
                if pv.dim() == 1: pv = pv.view(-1, 33, 3)
                elif pv.dim() == 2: pv = pv.view(pv.shape[0], 33, 3)
                pose_values_local.append(pv)
                
            if len(pose_values_local) > 0:
                pose_padded = pad_sequence(pose_values_local, batch_first=True).to(self.device).float()
                pose_lengths = [int(p.size(0)) for p in pose_values_local]
            else:
                B = len(samples['pixel_values'])
                pose_padded = torch.zeros((B, 1, 33, 3), device=self.device, dtype=torch.float32)
                pose_lengths = [0] * B
                
            face_kps = pose_padded[:, :, 0:11, :] 
            lhand_kps = pose_padded[:, :, [11, 13, 15, 17, 19, 21], :]
            rhand_kps = pose_padded[:, :, [12, 14, 16, 18, 20, 22], :]
            body_kps = pose_padded[:, :, 23:33, :]

            face_root = face_kps[:, :, 0:1, :].clone()
            lhand_root = lhand_kps[:, :, 0:1, :].clone()
            rhand_root = rhand_kps[:, :, 0:1, :].clone()
            
            face_kps = face_kps - face_root
            lhand_kps = lhand_kps - lhand_root
            rhand_kps = rhand_kps - rhand_root

            face_kps = face_kps.permute(0, 3, 1, 2)
            lhand_kps = lhand_kps.permute(0, 3, 1, 2)
            rhand_kps = rhand_kps.permute(0, 3, 1, 2)
            body_kps = body_kps.permute(0, 3, 1, 2)

            feat_face = self.pose_stream_face(face_kps)
            feat_lhand = self.pose_stream_lhand(lhand_kps)
            feat_rhand = self.pose_stream_rhand(rhand_kps)
            feat_body = self.pose_stream_body(body_kps)
            
            pose_outputs = torch.cat([feat_face, feat_lhand, feat_rhand, feat_body], dim=1)
            pose_outputs = pose_outputs.permute(0, 2, 1)
            pose_mask = create_mask(seq_lengths=pose_lengths, device=self.device)
        
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0] if spatial else (spatiotemporal_outputs.shape[0] if spatiotemporal else pose_outputs.shape[0])
            spatial_length = spatial_mask.sum(1) if spatial else torch.zeros(bs, device=self.device, dtype=torch.int)
            spatiotemporal_length = spatiotemporal_mask.sum(1) if spatiotemporal else torch.zeros(bs, device=self.device, dtype=torch.int)
            pose_length = pose_mask.sum(1) if pose else torch.zeros(bs, device=self.device, dtype=torch.int)
            new_length = spatial_length + spatiotemporal_length + pose_length

            joint_outputs = []
            for i in range(bs):
                parts = []
                if spatial: parts.append(spatial_outputs[i, :spatial_length[i], :])
                if spatiotemporal: parts.append(spatiotemporal_outputs[i, :spatiotemporal_length[i], :])
                if pose: parts.append(pose_outputs[i, :pose_length[i], :])
                joint_outputs.append(torch.cat(parts, dim=0))
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)

            visual_conv_outputs = self.temporal_encoder(joint_outputs.permute(0,2,1), torch.tensor(new_length.tolist(), device=self.device))
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
            visual_masks = create_mask(seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(), device=self.device)
        
        elif self.fusion_mode == 'adaptive':
            if spatial_outputs.shape[1] != spatiotemporal_outputs.shape[1]:
                spatiotemporal_outputs = F.interpolate(spatiotemporal_outputs.permute(0, 2, 1), size=spatial_outputs.shape[1], mode='linear', align_corners=False).permute(0, 2, 1)
            fused_outputs = self.adaptive_fusion(spatial_outputs, spatiotemporal_outputs, pose_outputs)
            visual_conv_outputs = self.temporal_encoder(fused_outputs.permute(0, 2, 1), torch.tensor(samples['num_frames'], device=self.device))
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1, 0, 2)
            visual_masks = create_mask(seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(), device=self.device)

        else:
            if spatial:
                active_outputs, active_lens = spatial_outputs, samples['num_frames']
            elif spatiotemporal:
                active_outputs, active_lens = spatiotemporal_outputs, samples['glor_lengths']
                visual_outputs = spatiotemporal_outputs
                visual_masks = spatiotemporal_mask
            elif pose:
                pose_conv_outputs = self.temporal_encoder(pose_outputs.permute(0,2,1), torch.tensor(pose_lengths, device=self.device))
                visual_outputs = pose_conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(seq_lengths=pose_conv_outputs['feat_len'].to(torch.int).tolist(), device=self.device)
            
            if self.fusion_mode != 'spatiotemporal' and self.fusion_mode != 'pose':
                conv_outputs = self.temporal_encoder(active_outputs.permute(0,2,1), torch.tensor(active_lens, device=self.device))
                visual_outputs = conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(seq_lengths=conv_outputs['feat_len'].to(torch.int).tolist(), device=self.device)

        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        pixel_values, glor_values, masks, ids = [], [], [], []
        pose_values, texts, glosses, langs = [], [], [], []
        num_frames, glor_lengths, ex_lang_translations = [], [], []

        for sample in batch:
            if sample.get('pixel_value') is None or sample['pixel_value'].shape[0] == 0: continue
            nframe = math.ceil(sample['num_frames'] / self.frame_sample_rate)
            pval = sample['pixel_value'][::self.frame_sample_rate]

            ids.append(sample['id'])
            texts.append(sample['text'].lower())
            glosses.append(sample['gloss'])
            langs.append(sample['lang'])

            _ex_lang_trans = []
            if self.num_in_context > 0 and 'en_text' in sample and 'text' in sample:
                _ex_lang_trans = [f"{sample.get('en_text','')}={sample['text']}", f"{sample.get('fr_text','')}={sample['text']}", f"{sample.get('es_text','')}={sample['text']}"]
                ex_lang_translations.append(' '.join(_ex_lang_trans[:self.num_in_context]))
            else:
                ex_lang_translations.append("")

            if nframe > self.max_frame_len:
                nframe = self.max_frame_len
                start_index = random.randint(0, pval.size(0) - self.max_frame_len)
                pval = pval[start_index:start_index + self.max_frame_len]

            num_frames.append(nframe)
            pixel_values.append(pval)

            pv = sample.get('pose_value')
            if pv is not None and isinstance(pv, torch.Tensor) and pv.numel() > 0:
                pose_values.append(pv.unsqueeze(0).float() if pv.dim() == 1 else pv.float())
            else:
                pose_values.append(torch.zeros(1, self.pose_input_size, dtype=torch.float32))

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
            'pixel_values': pixel_values, 'glor_values': glor_values, 'pose_values': pose_values,
            'bool_mask_pos': masks, 'ids': ids, 'text': texts, 'ex_lang_trans': ex_lang_translations,
            'gloss': glosses, 'lang': langs, 'num_frames': num_frames, 'glor_lengths': glor_lengths,
        }

    def visual_textual_align(self, visual_outputs, visual_masks, samples):
        output_tokens = self.t5_tokenizer(samples['text'], padding="longest", return_tensors="pt").to(self.device)
        text_embeds = self.t5_model.encoder.embed_tokens(output_tokens.input_ids)
        
        image_embeds = F.normalize(visual_outputs.mean(1), dim=-1)
        text_embeds = F.normalize(text_embeds.mean(1), dim=-1)

        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale.exp()
        return clip_loss(logits_per_text)

    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        visual_outputs = self.fusion_proj(visual_outputs)
        log_dict = {}
        
        if self.cross_modal_align:
            if (self.warm_up_steps is None and not self.combined_loss) or (self.warm_up_steps is not None and self.global_step <= self.warm_up_steps):
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(visual_outputs, visual_masks, inputs, split, batch_idx)
                loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = loss
                
                if self.sign_cl_loss and self.sign_cl and (self.sign_cl_every_n_steps <= 1 or (self.global_step % self.sign_cl_every_n_steps) == 0):
                    sign_cl_loss_val = self.sign_cl(visual_outputs, visual_masks)
                    if sign_cl_loss_val > 0:
                        loss += self.sign_cl_alpha * sign_cl_loss_val
                        log_dict[f"{split}/sign_cl_loss"] = sign_cl_loss_val
            else:
                input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(visual_outputs, visual_masks, inputs, split, batch_idx)
                outputs = self.t5_model(inputs_embeds=input_embeds, attention_mask=input_masks, decoder_attention_mask=output_tokens.attention_mask, labels=targets, output_hidden_states=True, return_dict=True)
                t5_loss = outputs.loss
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss = t5_loss + self.alpha * cont_loss
                log_dict[f"{split}/loss"] = t5_loss
                log_dict[f"{split}/contra_loss"] = cont_loss
                
                if self.sign_cl_loss and self.sign_cl and (self.sign_cl_every_n_steps <= 1 or (self.global_step % self.sign_cl_every_n_steps) == 0):
                    sign_cl_loss_val = self.sign_cl(visual_outputs, visual_masks)
                    if sign_cl_loss_val > 0:
                        loss += self.sign_cl_alpha * sign_cl_loss_val
                        log_dict[f"{split}/sign_cl_loss"] = sign_cl_loss_val
                log_dict[f"{split}/combined_loss"] = loss
        else:
            input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(visual_outputs, visual_masks, inputs, split, batch_idx)
            outputs = self.t5_model(inputs_embeds=input_embeds, attention_mask=input_masks, decoder_attention_mask=output_tokens.attention_mask, labels=targets, output_hidden_states=True, return_dict=True)
            loss = outputs.loss
            log_dict[f"{split}/loss"] = loss

        if split != "train":
            input_embeds, input_masks, _, _ = self.prepare_inputs(visual_outputs, visual_masks, inputs, split, batch_idx)
            generated = self.t5_model.generate(inputs_embeds=input_embeds, attention_mask=input_masks, num_beams=5, max_length=self.max_txt_len, top_p=0.9, do_sample=True)
            self.generated.extend([gen.lower() for gen in self.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)])
            self.references.extend([ref.lower() for ref in self.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)])

        return loss, log_dict

    def on_validation_epoch_end(self) -> None:
        eval_res = evaluate_results(predictions=self.generated, references=self.references, split='val', device=self.device)
        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def on_test_epoch_end(self) -> None:
        eval_res = evaluate_results(predictions=self.generated, references=self.references, split='test', device=self.device)
        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, eps=1e-8, betas=(0.9, 0.98))
        
        total_steps = int(self.trainer.estimated_stepping_batches) if hasattr(self.trainer, 'estimated_stepping_batches') else (len(getattr(self.trainer.train_dataloader, 'dataloader', self.trainer.train_dataloader)) // getattr(self.trainer, 'accumulate_grad_batches', 1)) * self.trainer.max_epochs
        warmup_steps = self.warm_up_steps if self.warm_up_steps is not None else int(total_steps * 0.1)

        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        try:
            self.log('model/total_params', float(sum(p.numel() for p in self.parameters())), prog_bar=False)
            self.log('model/trainable_params', float(sum(p.numel() for p in trainable_params)), prog_bar=False)
        except Exception: pass

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
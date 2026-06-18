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
from spamo.ctc_mixin import CTCMixin
from spamo.geo_sign_hyp import HyperbolicRegulariser
from sentence_transformers import SentenceTransformer
from spamo.dynamic_segmentation import DynamicSegmenter, AdaptiveMasker


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
#torch.set_float32_matmul_precision('high')


class FlanT5SLT(CTCMixin, AbstractSLT):
    """
    FlanT5-based Sign Language Translation model.
    Features: Spatial (ViT/ResNet) + Spatiotemporal (VideoMAE/C3D).
    Removed: Pose and I3D features.

    Merged from both branches:
      - Branch 1 (flan-t5-base-easlt-different-lr):
          emotion modality in joint fusion, label-smoothed cross-entropy loss.
      - Branch 2 (flan-t5-base-easlt-different-lr-makeing-kt-ctc-hyperpolic):
          dynamic segmentation / adaptive masking, InfoLOOB loss, KT loss,
          CTC loss, hyperbolic regularisation, SignCL projection head,
          updated contrastive loss tuple API.
    """

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
        combined_loss: bool = False,
        alpha: float = 0.1,
        use_kt_loss: bool = True,
        kt_lambda: float = 0.1,
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
        conv_type: int = 2,
        use_ctc: bool = False,
        ctc_weight: float = 0.3,
        ctc_blank_id: int = -1,
        use_spatial: bool = True,
        use_spatiotemporal: bool = True,
        use_pose: bool = False,
        use_emotion: bool = True,
        use_hyperbolic: bool = False,
        hyp_dim: int = 256,
        hyp_init_c: float = 1.0,
        hyp_alpha: float = 0.1,
        # InfoLOOB Loss
        use_infoloob_loss: bool = False,
        infoloob_temperature: float = 0.07,
        infoloob_weight: float = 1.0,
        # Dynamic Temporal Segmentation and Masking
        use_dynamic_segmentation: bool = False,
        motion_threshold: float = 0.5,
        use_adaptive_masking: bool = False,
        mask_prob: float = 0.15,
        min_mask_len: int = 5,
        max_mask_len: int = 20,
        num_layers: int = 3,
        mask_type: str = 'noise',
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
        self.use_kt_loss = use_kt_loss
        self.kt_lambda = kt_lambda
        self.sign_cl_loss = sign_cl_loss
        self.sign_cl_alpha = sign_cl_alpha
        self.sign_cl_temperature = sign_cl_temperature
        self.sign_cl_temporal_window = sign_cl_temporal_window
        self.sign_cl_every_n_steps = sign_cl_every_n_steps
        self.use_resampler = use_resampler
        self.sampling_length = sampling_length
        self.cache_dir = cache_dir

        # InfoLOOB
        self.use_infoloob_loss = use_infoloob_loss
        self.infoloob_temperature = infoloob_temperature
        self.infoloob_weight = infoloob_weight

        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        self.use_spatial = use_spatial
        self.use_spatiotemporal = use_spatiotemporal
        self.use_pose = use_pose
        self.use_emotion = use_emotion

        # Geo-Sign hyperbolic branch
        self.use_hyperbolic = use_hyperbolic
        self.hyp_dim = hyp_dim
        self.hyp_init_c = hyp_init_c
        self.hyp_alpha = hyp_alpha

        if self.num_in_context == 0:
            self.use_in_context = False

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_data_augmentation = use_data_augmentation

        self.conv_type = conv_type
        self._ctc_use = use_ctc
        self._ctc_weight = ctc_weight
        self._ctc_blank_id = ctc_blank_id

        print("===" * 40)
        print(f"use_data_augmentation: {use_data_augmentation}")
        print(f"sign_cl_loss: {sign_cl_loss}")
        print(f"use_ctc: {use_ctc}  |  ctc_weight: {ctc_weight}")
        print(f"conv_type: {conv_type}")
        print(f"use_emotion: {use_emotion}")
        print("===" * 40)

        # Dynamic Temporal Segmentation and Masking
        self.use_dynamic_segmentation = use_dynamic_segmentation
        self.motion_threshold = motion_threshold
        self.use_adaptive_masking = use_adaptive_masking
        self.mask_prob = mask_prob
        self.min_mask_len = min_mask_len
        self.max_mask_len = max_mask_len
        self.num_layers = num_layers
        self.mask_type = mask_type

        # Disable segmentation for first N epochs (warmup strategy)
        self.segmentation_warmup_epochs = 5

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

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

        # Emotion modules (from branch 1)
        if self.use_emotion:
            self.emotion_proj = build_vision_projector('linear', self.emotion_input_size, self.inter_hidden)
            self.emotion_enhancer = EmotionEnhancer(self.inter_hidden)
            self.emotion_modulator_s = EmotionModulator(self.inter_hidden)
            self.emotion_modulator_m = EmotionModulator(self.inter_hidden)
            self.emotion_modulator_e = EmotionModulator(self.inter_hidden)

        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden, conv_type=self.conv_type)

        # Geo-Sign hyperbolic regulariser (from branch 2)
        if self.use_hyperbolic:
            self.hyp_reg = HyperbolicRegulariser(
                pose_dim=self.inter_hidden,
                text_dim=self.t5_model.config.hidden_size,
                hyp_dim=self.hyp_dim,
                init_c=self.hyp_init_c,
                label_smoothing=0.1,
            )
        print(
            f"use_hyperbolic: {self.use_hyperbolic}"
            + (f" | hyp_dim={self.hyp_dim}, init_c={self.hyp_init_c}, hyp_alpha={self.hyp_alpha}"
               if self.use_hyperbolic else "")
        )

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

        # SignCL projection head (from branch 2)
        proj_dim = self.inter_hidden
        self.sign_cl_proj = nn.Sequential(
            nn.Linear(self.t5_model.config.hidden_size, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        # CTC head via mixin
        self.init_ctc(
            hidden_size=self.t5_model.config.hidden_size,
            vocab_size=self.t5_model.config.vocab_size,
            blank_id_cfg=self._ctc_blank_id,
            tokenizer=self.t5_tokenizer,
            ctc_weight=self._ctc_weight,
            use_ctc=self._ctc_use,
        )

        # Knowledge Transfer components (from branch 2)
        if self.use_kt_loss:
            self.sbert = SentenceTransformer('all-mpnet-base-v2')
            for param in self.sbert.parameters():
                param.requires_grad = False
            self.kt_proj = nn.Linear(self.t5_model.config.hidden_size, 768)

        # Dynamic Segmentation and Adaptive Masking (from branch 2)
        if self.use_dynamic_segmentation or self.use_adaptive_masking:
            self.segmenter = DynamicSegmenter(
                hidden_dim=self.inter_hidden,
                motion_threshold=self.motion_threshold,
                num_layers=self.num_layers
            )
        if self.use_adaptive_masking:
            self.masker = AdaptiveMasker(
                mask_prob=self.mask_prob,
                min_mask_len=self.min_mask_len,
                max_mask_len=self.max_mask_len,
                mask_type=self.mask_type
            )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        visual_outputs: torch.Tensor,
        visual_mask: torch.Tensor,
        samples: Dict,
        split: str,
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        bs = visual_outputs.shape[0]

        # 1. Create the base prompts
        prompts = [f'{self.prompt}'] * bs
        prompts = [p.format(l) for p, l in zip(prompts, samples['lang'])]

        # 2. Append context if enabled
        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples['ex_lang_trans'])]

        # 3. Tokenize
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

    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # ── Process spatial features ──────────────────────────────────────────
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
            pose_values_local = [
                pv if pv.dim() == 2 else pv.view(pv.shape[0], -1)
                for pv in raw_pose_values
            ]
            if len(pose_values_local) > 0:
                pose_padded = pad_sequence(pose_values_local, batch_first=True).to(self.device).float()
                pose_lengths = [int(p.size(0)) for p in pose_values_local]
            else:
                B = len(samples['pixel_values'])
                pose_padded = torch.zeros((B, 1, self.pose_input_size), device=self.device, dtype=torch.float32)
                pose_lengths = [0] * B

            pose_outputs = self.pose_proj(pose_padded)
            pose_mask = create_mask(seq_lengths=pose_lengths, device=self.device)

        # ── Emotion modulation (from branch 1) ───────────────────────────────
        # Applied before fusion so that spatial / spatiotemporal features are
        # modulated before they are concatenated or fused.
        emotion_mask = None
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

        # ── Per-modality segmentation / masking for single-modality modes ────
        # (from branch 2) Applied BEFORE joint/adaptive concat so importance
        # scores are computed on the right (pre-concat) tensor.
        if self.fusion_mode == 'spatial' and spatial and (self.use_dynamic_segmentation or self.use_adaptive_masking):
            importance_scores = None
            if self.use_dynamic_segmentation:
                importance_scores = self.segmenter(spatial_outputs, samples['num_frames'])
            if self.use_adaptive_masking:
                if importance_scores is None:
                    importance_scores = torch.full_like(spatial_outputs[:, :, 0], 0.5)
                spatial_outputs = self.masker(spatial_outputs, importance_scores, samples['num_frames'], self.training)

        elif self.fusion_mode == 'spatiotemporal' and spatiotemporal and (self.use_dynamic_segmentation or self.use_adaptive_masking):
            importance_scores = None
            if self.use_dynamic_segmentation:
                importance_scores = self.segmenter(spatiotemporal_outputs, samples['glor_lengths'])
            if self.use_adaptive_masking:
                if importance_scores is None:
                    importance_scores = torch.full_like(spatiotemporal_outputs[:, :, 0], 0.5)
                spatiotemporal_outputs = self.masker(spatiotemporal_outputs, importance_scores, samples['glor_lengths'], self.training)

        elif self.fusion_mode == 'pose' and pose and (self.use_dynamic_segmentation or self.use_adaptive_masking):
            importance_scores = None
            if self.use_dynamic_segmentation:
                importance_scores = self.segmenter(pose_outputs, pose_lengths)
            if self.use_adaptive_masking:
                if importance_scores is None:
                    importance_scores = torch.full_like(pose_outputs[:, :, 0], 0.5)
                pose_outputs = self.masker(pose_outputs, importance_scores, pose_lengths, self.training)

        # =========================================================
        # JOINT FUSION
        # =========================================================
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]

            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            pose_length = (
                pose_mask.sum(1) if pose else torch.zeros_like(spatial_length)
            )
            # Emotion length (from branch 1)
            emotion_length = (
                emotion_mask.sum(1)
                if (self.use_emotion and emotion_mask is not None)
                else torch.zeros_like(spatial_length)
            )
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
                # Append emotion tokens (from branch 1)
                if self.use_emotion and Ze_mod is not None:
                    parts.append(Ze_mod[i, :emotion_length[i], :])
                joint_outputs.append(torch.cat(parts, dim=0))

            joint_outputs = pad_sequence(joint_outputs, batch_first=True)

            # Segmentation / masking on the concatenated joint tensor (from branch 2)
            importance_scores = None
            if self.use_dynamic_segmentation:
                importance_scores = self.segmenter(joint_outputs, new_length)
                joint_outputs, new_length = self.segmenter.segment(
                    joint_outputs, importance_scores, new_length
                )
            if self.use_adaptive_masking:
                if importance_scores is None:
                    importance_scores = self.segmenter(joint_outputs, new_length)
                joint_outputs = self.masker(
                    joint_outputs, importance_scores, new_length, self.training
                )

            visual_conv_outputs = self.temporal_encoder(
                joint_outputs.permute(0, 2, 1),
                torch.tensor(new_length.tolist() if hasattr(new_length, 'tolist') else new_length,
                              device=self.device)
            )
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1, 0, 2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(),
                device=self.device
            )

        # =========================================================
        # ADAPTIVE FUSION
        # =========================================================
        elif self.fusion_mode == 'adaptive':
            if spatial_outputs.shape[1] != spatiotemporal_outputs.shape[1]:
                spatiotemporal_outputs = F.interpolate(
                    spatiotemporal_outputs.permute(0, 2, 1),
                    size=spatial_outputs.shape[1],
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)
            fused_outputs = self.adaptive_fusion(spatial_outputs, spatiotemporal_outputs, pose_outputs)
            fused_lengths = samples['num_frames']

            # Segmentation / masking on fused tensor (from branch 2)
            importance_scores = None
            if self.use_dynamic_segmentation:
                importance_scores = self.segmenter(fused_outputs, fused_lengths)
                fused_outputs, fused_lengths = self.segmenter.segment(
                    fused_outputs, importance_scores, fused_lengths
                )
            if self.use_adaptive_masking:
                if importance_scores is None:
                    importance_scores = self.segmenter(fused_outputs, fused_lengths)
                fused_outputs = self.masker(
                    fused_outputs, importance_scores, fused_lengths, self.training
                )

            visual_conv_outputs = self.temporal_encoder(
                fused_outputs.permute(0, 2, 1),
                torch.tensor(fused_lengths, device=self.device)
            )
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1, 0, 2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(),
                device=self.device
            )

        # =========================================================
        # SINGLE MODALITY — temporal encoder + mask creation
        # =========================================================
        else:
            if spatial:
                active_outputs, active_lens = spatial_outputs, samples['num_frames']
            elif spatiotemporal:
                active_outputs, active_lens = spatiotemporal_outputs, samples['glor_lengths']
            elif pose:
                active_outputs, active_lens = pose_outputs, pose_lengths
            else:
                raise NotImplementedError("Invalid fusion mode")

            # spatiotemporal skips the temporal encoder (mirrors original design)
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

        return visual_outputs, visual_masks

    # ------------------------------------------------------------------
    # Batch collation
    # ------------------------------------------------------------------

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
                if self.training:
                    # TRAINING: Pull the video's own text (which will be shuffled away by derangement later)
                    if 'en_text' in sample and 'text' in sample:
                        _ex_lang_trans = [
                            f"{sample.get('en_text', '')}={sample['text']}",
                            f"{sample.get('fr_text', '')}={sample['text']}",
                            f"{sample.get('es_text', '')}={sample['text']}"
                        ]
                    trimmed = _ex_lang_trans[:self.num_in_context]
                    ex_lang_translations.append(' '.join(trimmed))
                else:
                    # INFERENCE: Hard-code the fixed examples from the EASLT paper (Table 8)
                    hard_coded_ctx = "It can occasionally thunderstorms.=vereinzelt kann es gewittern."
                    ex_lang_translations.append(hard_coded_ctx)
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

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def infoloob_loss(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """
        InfoLOOB Loss: "Improving Contrastive Learning by Leaving Out the Positive"

        Args:
            sim_matrix: Similarity matrix of shape [batch_size, batch_size]

        Returns:
            Scalar loss value
        """
        sim_matrix_stable = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim_matrix_stable)
        pos = torch.diag(exp_sim)
        neg = exp_sim.sum(dim=1) - pos
        loss = -torch.log(pos / (neg + 1e-8))
        return loss.mean()

    def visual_textual_align(
        self,
        visual_outputs: torch.Tensor,
        visual_masks: torch.Tensor,
        samples: Dict
    ) -> Tuple[torch.Tensor, str]:
        """
        Visual-textual alignment loss.

        Supports both CLIP-style loss and InfoLOOB loss based on configuration.
        Always returns a (loss, loss_type) tuple where loss_type is
        'infoloob' or 'clip'.
        """
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

        similarity = torch.matmul(text_embeds, image_embeds.t())

        if self.use_infoloob_loss:
            similarity_scaled = similarity / self.infoloob_temperature
            loss = self.infoloob_loss(similarity_scaled)
            return loss, 'infoloob'
        else:
            logit_scale = self.logit_scale.exp()
            logits_per_text = similarity * logit_scale
            loss = clip_loss(logits_per_text)
            return loss, 'clip'

    def _compute_hyp_loss(self, samples: Dict) -> torch.Tensor:
        """
        Computes the Geo-Sign hyperbolic contrastive loss between
        projected pose features and T5 text embeddings.

        Returns a scalar tensor (0.0 if use_hyperbolic is False or
        pose_values are empty).
        """
        zero = torch.tensor(0.0, device=self.device)

        if not self.use_hyperbolic or not self.training:
            return zero

        raw_pose = samples.get('pose_values', [])
        if not raw_pose:
            return zero

        pose_local = [
            pv if pv.dim() == 2 else pv.view(pv.shape[0], -1)
            for pv in raw_pose
        ]
        pose_padded = pad_sequence(pose_local, batch_first=True).to(self.device).float()
        pose_lengths = [int(p.size(0)) for p in pose_local]

        pose_feats = self.pose_proj(pose_padded)            # (B, T, inter_hidden)
        pose_mask = create_mask(seq_lengths=pose_lengths, device=self.device)

        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_embeds = self.t5_model.encoder.embed_tokens(output_tokens.input_ids)

        text_mask = output_tokens.attention_mask.bool()

        return self.hyp_reg(pose_feats, pose_mask, text_embeds, text_mask)

    def _apply_sign_cl(
        self,
        visual_outputs: torch.Tensor,
        visual_masks: torch.Tensor,
        split: str,
        log_dict: Dict,
        current_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Helper that applies SignCL loss and updates loss + log_dict in place."""
        if not (
            self.sign_cl_loss
            and self.sign_cl is not None
            and (
                self.sign_cl_every_n_steps <= 1
                or (self.global_step % self.sign_cl_every_n_steps) == 0
            )
        ):
            return current_loss, log_dict

        proj_vis = self.sign_cl_proj(visual_outputs)
        proj_vis = F.normalize(proj_vis, dim=2)
        sign_cl_loss_val = self.sign_cl(proj_vis, visual_masks)

        if sign_cl_loss_val is not None and sign_cl_loss_val > 0:
            current_loss = current_loss + self.sign_cl_alpha * sign_cl_loss_val
            log_dict[f"{split}/sign_cl_loss"] = sign_cl_loss_val

        return current_loss, log_dict

    def _apply_kt_loss(
        self,
        visual_outputs: torch.Tensor,
        inputs: Dict,
        split: str,
        log_dict: Dict,
        current_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Helper that applies Knowledge Transfer loss."""
        if not self.use_kt_loss:
            return current_loss, log_dict

        video_emb = self.kt_proj(visual_outputs.mean(dim=1))
        text_emb = torch.tensor(self.sbert.encode(inputs["text"])).to(self.device)
        loss_kt = F.mse_loss(video_emb, text_emb)
        current_loss = current_loss + self.kt_lambda * loss_kt
        log_dict[f"{split}/kt_loss"] = loss_kt
        return current_loss, log_dict

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def shared_step(
        self,
        inputs: Dict,
        split: str,
        batch_idx: int
    ) -> Tuple[torch.Tensor, Dict]:

        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)

        # Geo-Sign Hyperbolic Regularisation
        hyp_loss = self._compute_hyp_loss(inputs)

        # Project to T5 hidden size
        visual_outputs = self.fusion_proj(visual_outputs)

        log_dict = {}

        # ==================================================
        # Cross-modal alignment branch
        # ==================================================
        if self.cross_modal_align:

            # --------------------------------------------------
            # Stage 1: Contrastive-only (no warm-up, no decoder)
            # --------------------------------------------------
            if self.warm_up_steps is None and not self.combined_loss:

                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )

                cont_loss, loss_type = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss_key = (
                    f"{split}/infoloob_loss" if loss_type == "infoloob"
                    else f"{split}/contra_loss"
                )
                log_dict[loss_key] = cont_loss

                loss = cont_loss + self.hyp_alpha * hyp_loss
                log_dict[f"{split}/hyp_loss"] = hyp_loss

            # --------------------------------------------------
            # Stage 2: Warm-up (contrastive + hyperbolic + SignCL)
            # --------------------------------------------------
            elif (
                self.warm_up_steps is not None
                and self.global_step <= self.warm_up_steps
            ):

                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )

                cont_loss, loss_type = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss_key = (
                    f"{split}/infoloob_loss" if loss_type == "infoloob"
                    else f"{split}/contra_loss"
                )
                log_dict[loss_key] = cont_loss

                loss = cont_loss + self.hyp_alpha * hyp_loss
                log_dict[f"{split}/hyp_loss"] = hyp_loss

                loss, log_dict = self._apply_sign_cl(
                    visual_outputs, visual_masks, split, log_dict, loss
                )
                log_dict[f"{split}/warmup_total_loss"] = loss

            # --------------------------------------------------
            # Stage 3: Full decoder training
            # --------------------------------------------------
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

                # T5 loss — use label-smoothed CE from branch 1 explicitly
                t5_loss = F.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                    label_smoothing=0.1
                )
                log_dict[f"{split}/loss"] = t5_loss

                cont_loss, loss_type = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss_key = (
                    f"{split}/infoloob_loss" if loss_type == "infoloob"
                    else f"{split}/contra_loss"
                )
                log_dict[loss_key] = cont_loss

                loss = t5_loss + self.alpha * cont_loss

                # SignCL
                loss, log_dict = self._apply_sign_cl(
                    visual_outputs, visual_masks, split, log_dict, loss
                )

                # Knowledge Transfer
                loss, log_dict = self._apply_kt_loss(
                    visual_outputs, inputs, split, log_dict, loss
                )

                # CTC
                ctc_loss = self.compute_ctc_loss(visual_outputs, visual_masks, inputs)
                loss = loss + ctc_loss
                log_dict[f"{split}/ctc_loss"] = ctc_loss

                # Hyperbolic
                loss = loss + self.hyp_alpha * hyp_loss
                log_dict[f"{split}/hyp_loss"] = hyp_loss

                log_dict[f"{split}/combined_loss"] = loss

        # ==================================================
        # No cross-modal alignment branch
        # ==================================================
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

            # Use label-smoothed CE from branch 1
            base_loss = F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                label_smoothing=0.1
            )
            log_dict[f"{split}/loss"] = base_loss

            loss = base_loss + self.hyp_alpha * hyp_loss
            log_dict[f"{split}/hyp_loss"] = hyp_loss

            # CTC (train only)
            if split == "train":
                ctc_loss = self.compute_ctc_loss(visual_outputs, visual_masks, inputs)
                loss = loss + ctc_loss
                log_dict[f"{split}/ctc_loss"] = ctc_loss

            # Knowledge Transfer
            loss, log_dict = self._apply_kt_loss(
                visual_outputs, inputs, split, log_dict, loss
            )

        # ==================================================
        # Validation / Test generation
        # ==================================================
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

            reference_strings = self.t5_tokenizer.batch_decode(
                output_tokens.input_ids, skip_special_tokens=True
            )
            reference_strings = [r.lower() for r in reference_strings]

            self.generated.extend(generated_strings)
            self.references.extend(reference_strings)

        return loss, log_dict

    # ------------------------------------------------------------------
    # Epoch-end hooks
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found.")

        lora_params = [
            p for n, p in self.named_parameters()
            if p.requires_grad and ('lora_' in n or 'logit_scale' in n)
        ]
        bridge_params = [
            p for n, p in self.named_parameters()
            if p.requires_grad and 'fusion_proj' in n
        ]
        other_params = [
            p for n, p in self.named_parameters()
            if p.requires_grad
            and 'lora_' not in n
            and 'logit_scale' not in n
            and 'fusion_proj' not in n
        ]

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
            acc_batches = (
                self.trainer.accumulate_grad_batches
                if hasattr(self.trainer, 'accumulate_grad_batches')
                else 1
            )
            total_steps = (batches_per_epoch // acc_batches) * max_epochs

        warmup_steps = (
            self.warm_up_steps
            if self.warm_up_steps is not None
            else int(total_steps * 0.1)
        )

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

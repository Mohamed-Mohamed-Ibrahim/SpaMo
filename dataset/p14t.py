import torch
import os
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from spamo.constants import *
import random


class Phoenix14T(torch.utils.data.Dataset):
    def __init__(
        self,
        anno_root: str,
        vid_root: str,
        feat_root: str,
        mae_feat_root: str,
        pose_root: str,
        mode: str = 'dev',
        spatial: bool = False,
        spatiotemporal: bool = False,
        spatial_postfix: str = '',
        spatiotemporal_postfix: Union[str, List[str]] = '',
        pose_postfix: str = '',
        pose: bool = False,
        emotion: bool = False,
        emotion_postfix: str = '_Ze',
        emo_feat_root: str = ''
    ):
        super().__init__()
        
        self.anno_root = Path(anno_root)
        self.vid_root = Path(vid_root)
        self.feat_root = Path(feat_root)
        self.mae_feat_root = Path(mae_feat_root)
        self.mode = mode
        self.spatial = spatial
        self.spatiotemporal = spatiotemporal
        self.spatial_postfix = spatial_postfix
        self.spatiotemporal_postfix = spatiotemporal_postfix
        self.pose_root = Path(pose_root) if pose_root else None
        self.pose_postfix = pose_postfix
        self.pose = pose
        
        self.emotion = emotion
        self.emotion_postfix = emotion_postfix
        self.emo_feat_root = Path(emo_feat_root) if emo_feat_root else None
        
        if not (spatial or spatiotemporal):
            raise ValueError("At least one of 'spatial' or 'spatiotemporal' must be True")
        
        if not (pose):
            print("No Pose features will be loaded.")

        if not emotion:
            print("No emotion features will be loaded.")
 
        anno_path = self.anno_root / f'{mode}_info_ml.npy'
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        
        self.data = np.load(anno_path, allow_pickle=True).item()
        
        if isinstance(self.data, dict):
            self.valid_keys = [k for k, v in self.data.items() if isinstance(v, dict) and 'fileid' in v]
            self.valid_keys.sort()
        else:
            self.valid_keys = list(range(len(self.data)))
        
        self.spatial_dir = self.feat_root / self.mode
        self.spatiotemporal_dir = self.mae_feat_root / self.mode
        self.pose_dir = (self.pose_root / self.mode) if (self.pose_root is not None) else None
        self.emotion_dir = (self.emo_feat_root / self.mode) if self.emo_feat_root else None
        
        self._validate_directories()

    def _validate_directories(self) -> None:
        if self.spatial and not self.spatial_dir.exists():
            raise FileNotFoundError(f"Spatial feature directory not found: {self.spatial_dir}")
        
        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            raise FileNotFoundError(f"Spatiotemporal feature directory not found: {self.spatiotemporal_dir}")
        
        if self.pose:
            if self.pose_dir is None or not self.pose_dir.exists():
                raise FileNotFoundError(f"Pose feature directory not found: {self.pose_dir}")

        if self.emotion:
            if self.emotion_dir is None or not self.emotion_dir.exists():
                raise FileNotFoundError(f"Emotion feature directory not found: {self.emotion_dir}")
        
    def _load_spatial_features(self, file_id: str) -> torch.Tensor:
        feat_path = self.spatial_dir / f"{file_id}{self.spatial_postfix}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Spatial feature file not found: {feat_path}")
        
        return torch.tensor(np.load(feat_path))

    def _load_spatiotemporal_features(self, file_id: str) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(self.spatiotemporal_postfix, str):
            glor_path = self.spatiotemporal_dir / f"{file_id}{self.spatiotemporal_postfix}.npy"
            if not glor_path.exists():
                raise FileNotFoundError(f"Spatiotemporal feature file not found: {glor_path}")
            return torch.tensor(np.load(glor_path))
        else:
            features = []
            for postfix in self.spatiotemporal_postfix:
                path = self.spatiotemporal_dir / f"{file_id}{postfix}.npy"
                if not path.exists():
                    raise FileNotFoundError(f"Spatiotemporal feature file not found: {path}")
                features.append(torch.tensor(np.load(path)))
            return features
        
    def _load_pose_features(self, file_id: str) -> torch.Tensor:
        if self.pose_dir is None:
            return torch.tensor([])

        pose_path = self.pose_dir / f"{file_id}{self.pose_postfix}.npy"
        if not pose_path.exists():
            print(f"Warning: Pose feature file not found: {pose_path}")
            return torch.tensor([])

        return torch.tensor(np.load(pose_path), dtype=torch.float32)

    def _heal_emotion_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        valid_mask = ~torch.isnan(tensor).any(dim=1) & (tensor.abs().sum(dim=1) > 0)
        
        if valid_mask.sum() == 0:
            return torch.zeros_like(tensor)
            
        if valid_mask.all():
            return tensor

        tensor_np = tensor.numpy()
        valid_indices = torch.where(valid_mask)[0].numpy()
        all_indices = np.arange(tensor.shape[0])

        for i in range(tensor.shape[1]):
            tensor_np[:, i] = np.interp(all_indices, valid_indices, tensor_np[valid_indices, i])

        return torch.tensor(tensor_np, dtype=torch.float32)

    def _load_emotion_features(self, file_id: str) -> torch.Tensor:
        path = self.emotion_dir / f"{file_id}{self.emotion_postfix}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Emotion feature file not found: {path}")
        
        raw_tensor = torch.tensor(np.load(path), dtype=torch.float32)
        return self._heal_emotion_tensor(raw_tensor)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        actual_key = self.valid_keys[index]
        data = self.data[actual_key]
        
        file_id = data['fileid']
        pixel_value = None
        glor_value = None
        pose_value = None
        emotion_value = None
        
        if self.spatial:
            try:
                pixel_value = self._load_spatial_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                pixel_value = torch.tensor([])
        
        if self.spatiotemporal:
            try:
                glor_value = self._load_spatiotemporal_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                if isinstance(self.spatiotemporal_postfix, str):
                    glor_value = torch.tensor([])
                else:
                    glor_value = [torch.tensor([])]

        if self.pose:
            try:
                pose_value = self._load_pose_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                pose_value = torch.tensor([])

        if self.emotion:
            try:
                emotion_value = self._load_emotion_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning zero tensor.")
                emotion_value = torch.zeros(1, 768, dtype=torch.float32)
        
        result = {
            'pixel_value': pixel_value,
            'glor_value': glor_value,
            'pose_value': pose_value,
            'emotion_value': emotion_value,
            'bool_mask_pos': None,
            'text': self._normalize_text(data['text']),
            'gloss': data['gloss'],
            'id': file_id,
            'num_frames': len(pixel_value) if pixel_value is not None else 0,
            'vid_path': str(self.vid_root / 'features' / 'fullFrame-256x256px' / data['folder']),
            'lang': 'German'
        }
        
        # Add language texts if available
        for lang in ['en', 'es', 'fr']:
            if f'{lang}_text' in data:
                result[f'{lang}_text'] = data[f'{lang}_text']
        
        # --- NEW: Reproducible Random Context ---
        # Seed the random number generator using the current index
        # This guarantees the exact same "random" context is chosen every single epoch
        rng = random.Random(index)
        rand_idx = rng.randint(0, len(self.data) - 2)
        
        # Make sure we don't accidentally pick the same video
        while rand_idx == index:
            rand_idx = rng.randint(0, len(self.data) - 2)
            
        rand_data = self.data[rand_idx]
        result['ctx_text'] = self._normalize_text(rand_data['text'])
        
        for lang in ['en', 'es', 'fr']:
            if f'{lang}_text' in rand_data:
                result[f'ctx_{lang}_text'] = rand_data[f'{lang}_text']
        # ----------------------------------------

        # Store original data for reference
        result['original_info'] = data
        
        return result

    def _normalize_text(self, text: str) -> str:
        text = text.strip()
        if not text.endswith('.'):
            text = f"{text}."
        return text

    def __len__(self) -> int:
        return len(self.valid_keys)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch

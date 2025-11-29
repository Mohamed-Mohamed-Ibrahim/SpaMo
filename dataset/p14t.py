import torch
import os
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from spamo.constants import *
import random


class Phoenix14T(torch.utils.data.Dataset):
    """
    Dataset class for the Phoenix14T sign language dataset.
    
    This class handles loading video features and annotations for sign language translation,
    supporting spatial, spatiotemporal, pose, and I3D feature types.
    """
    def __init__(
        self,
        anno_root: str,
        vid_root: str,
        feat_root: str,
        mae_feat_root: str,
        pose_root: str = None,
        i3d_root: str = None,      # <--- NEW: I3D Path
        mode: str = 'dev',
        spatial: bool = False,
        spatiotemporal: bool = False,
        spatial_postfix: str = '',
        spatiotemporal_postfix: Union[str, List[str]] = '',
        pose_postfix: str = '',
        pose: bool = False,
        i3d: bool = False          # <--- NEW: I3D Flag
    ):
        """
        Initialize the Phoenix14T dataset.
        """
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
        
        # Pose configuration
        self.pose_root = Path(pose_root) if pose_root else None
        self.pose_postfix = pose_postfix
        self.pose = pose

        # I3D configuration <--- NEW
        self.i3d_root = Path(i3d_root) if i3d_root else None
        self.i3d = i3d
        
        # Validate inputs
        if not (spatial or spatiotemporal):
            raise ValueError("At least one of 'spatial' or 'spatiotemporal' must be True")
        
        if not pose:
            print("No Pose features will be loaded.")
            
        if not i3d:
            print("No I3D features will be loaded.")

        # Load annotations
        anno_path = self.anno_root / f'{mode}_info_ml.npy'
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        
        self.data = np.load(anno_path, allow_pickle=True).item()
        
        # Set up directory paths
        # Assumption: Your folders have subfolders named 'train', 'test', 'dev'
        self.spatial_dir = self.feat_root / self.mode
        self.spatiotemporal_dir = self.mae_feat_root / self.mode
        self.pose_dir = (self.pose_root / self.mode) if (self.pose_root is not None) else None
        self.i3d_dir = (self.i3d_root / self.mode) if (self.i3d_root is not None) else None # <--- NEW
        
        # Validate that key directories exist
        self._validate_directories()

    def _validate_directories(self) -> None:
        """Validate that all necessary directories exist."""
        if self.spatial and not self.spatial_dir.exists():
            raise FileNotFoundError(f"Spatial feature directory not found: {self.spatial_dir}")
        
        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            raise FileNotFoundError(f"Spatiotemporal feature directory not found: {self.spatiotemporal_dir}")
        
        if self.pose:
            if self.pose_dir is None or not self.pose_dir.exists():
                raise FileNotFoundError(f"Pose feature directory not found: {self.pose_dir}")
                
        # <--- NEW: I3D Validation
        if self.i3d:
            if self.i3d_dir is None or not self.i3d_dir.exists():
                # Fallback check: maybe the root points directly to files?
                if self.i3d_root and self.i3d_root.exists():
                     print(f"Warning: Mode subfolder {self.mode} not found in I3D root. Trying root directly.")
                     self.i3d_dir = self.i3d_root
                else:
                    raise FileNotFoundError(f"I3D feature directory not found: {self.i3d_dir}")

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

    # <--- NEW: I3D Loading Method
    def _load_i3d_features(self, file_id: str) -> torch.Tensor:
        if self.i3d_dir is None:
            return torch.tensor([])

        # Try loading directly
        i3d_path = self.i3d_dir / f"{file_id}.npy"
        
        if not i3d_path.exists():
             print(f"Warning: I3D feature file not found: {i3d_path}")
             return torch.tensor([])
             
        return torch.tensor(np.load(i3d_path), dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.data[index]
        file_id = data['fileid']
        pixel_value = None
        glor_value = None
        pose_value = None
        i3d_value = None # <--- NEW
        
        # Load spatial features
        if self.spatial:
            try:
                pixel_value = self._load_spatial_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                pixel_value = torch.tensor([])
        
        # Load spatiotemporal features
        if self.spatiotemporal:
            try:
                glor_value = self._load_spatiotemporal_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                if isinstance(self.spatiotemporal_postfix, str):
                    glor_value = torch.tensor([])
                else:
                    glor_value = [torch.tensor([])]

        # Load pose features
        if self.pose:
            try:
                pose_value = self._load_pose_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                pose_value = torch.tensor([])

        # <--- NEW: Load I3D features
        if self.i3d:
            try:
                i3d_value = self._load_i3d_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                i3d_value = torch.tensor([])
        
        # Create result dictionary
        result = {
            'pixel_value': pixel_value,
            'glor_value': glor_value,
            'pose_value': pose_value,
            'i3d_feat': i3d_value,   # <--- NEW: Using 'i3d_feat' to match typical Projector keys
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
        
        # Store original data for reference
        result['original_info'] = data
        
        return result

    def _normalize_text(self, text: str) -> str:
        text = text.strip()
        if not text.endswith('.'):
            text = f"{text}."
        return text

    def __len__(self) -> int:
        return len(self.data) - 1

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch
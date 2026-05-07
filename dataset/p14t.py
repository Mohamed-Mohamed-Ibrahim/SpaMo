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
    supporting spatial, spatiotemporal, pose, and gfslt feature types.
    """
    def __init__(
        self,
        anno_root: str,
        vid_root: str,
        feat_root: str,
        mae_feat_root: str,
       pose_root: str,    
        gfslt_root: str = "",               # <--- NEW
        mode: str = 'dev',
        spatial: bool = False,
        spatiotemporal: bool = False,
        spatial_postfix: str = '',
        spatiotemporal_postfix: Union[str, List[str]] = '',
        pose_postfix: str = '',            
        pose: bool = False,                
        gfslt: bool = False,               # <--- NEW
        gfslt_postfix: str = ''            # <--- NEW
    ):
        """
        Initialize the Phoenix14T dataset.
        
        Args:
            anno_root: Root directory for annotation files
            vid_root: Root directory for video files
            feat_root: Root directory for spatial features
            mae_feat_root: Root directory for spatiotemporal features
            pose_root: Root directory for pose features
            gfslt_root: Root directory for gfslt features
            mode: Dataset split ('train', 'dev', or 'test')
            spatial: Whether to load spatial features
            spatiotemporal: Whether to load spatiotemporal features
            spatial_postfix: Filename postfix for spatial features
            spatiotemporal_postfix: Filename postfix for spatiotemporal features
            pose_postfix: Filename postfix for pose features
            pose: Whether to load pose features
            gfslt: Whether to load gfslt features
            gfslt_postfix: Filename postfix for gfslt features
        """
        super().__init__()
        
        self.anno_root = Path(anno_root)
        self.vid_root = Path(vid_root)
        self.feat_root = Path(feat_root)
        self.mae_feat_root = Path(mae_feat_root)
        self.mode = mode
        
        # Feature flags
        self.spatial = spatial
        self.spatiotemporal = spatiotemporal
        self.pose = pose
        self.gfslt = gfslt
        
        # Postfixes
        self.spatial_postfix = spatial_postfix
        self.spatiotemporal_postfix = spatiotemporal_postfix
        self.pose_postfix = pose_postfix
        self.gfslt_postfix = gfslt_postfix
        
        # Roots
        self.pose_root = Path(pose_root) if pose_root else None
        self.gfslt_root = Path(gfslt_root) if gfslt_root else None
        
        # Validate inputs
        if not (spatial or spatiotemporal or pose or gfslt):
            raise ValueError("At least one of 'spatial', 'spatiotemporal', 'pose', or 'gfslt' must be True")
        
        if not pose:
            print("No Pose features will be loaded.")
            
        if not gfslt:
            print("No GFSLT features will be loaded.")
 
        # Load annotations
        anno_path = self.anno_root / f'{mode}_info_ml.npy'
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        
        self.data = np.load(anno_path, allow_pickle=True).item()
        
        # Set up directory paths
        self.spatial_dir = self.feat_root / self.mode
        self.spatiotemporal_dir = self.mae_feat_root / self.mode
        self.pose_dir = (self.pose_root / self.mode) if (self.pose_root is not None) else None
        self.gfslt_dir = (self.gfslt_root / self.mode) if (self.gfslt_root is not None) else None
        
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
                
        if self.gfslt:
            if self.gfslt_dir is None or not self.gfslt_dir.exists():
                raise FileNotFoundError(f"GFSLT feature directory not found: {self.gfslt_dir}")
        

    def _load_spatial_features(self, file_id: str) -> torch.Tensor:
        """Load spatial features for a given file ID."""
        feat_path = self.spatial_dir / f"{file_id}{self.spatial_postfix}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Spatial feature file not found: {feat_path}")
        
        return torch.tensor(np.load(feat_path))

    def _load_spatiotemporal_features(self, file_id: str) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Load spatiotemporal features for a given file ID."""
        if isinstance(self.spatiotemporal_postfix, str):
            glor_path = self.spatiotemporal_dir / f"{file_id}{self.spatiotemporal_postfix}.npy"
            if not glor_path.exists():
                raise FileNotFoundError(f"Spatiotemporal feature file not found: {glor_path}")
            return torch.tensor(np.load(glor_path))
        else:
            # Handle multiple spatiotemporal features
            features = []
            for postfix in self.spatiotemporal_postfix:
                path = self.spatiotemporal_dir / f"{file_id}{postfix}.npy"
                if not path.exists():
                    raise FileNotFoundError(f"Spatiotemporal feature file not found: {path}")
                features.append(torch.tensor(np.load(path)))
            return features
        
    def _load_pose_features(self, file_id: str) -> torch.Tensor:
        """Load pose (skeletal) features for a given file ID."""
        if self.pose_dir is None:
            return torch.tensor([])

        pose_path = self.pose_dir / f"{file_id}{self.pose_postfix}.npy"
        if not pose_path.exists():
            print(f"Warning: Pose feature file not found: {pose_path}")
            return torch.tensor([])

        return torch.tensor(np.load(pose_path), dtype=torch.float32)

    def _load_gfslt_features(self, file_id: str) -> torch.Tensor:
        """Load GFSLT features for a given file ID."""
        if self.gfslt_dir is None:
            return torch.tensor([])

        gfslt_path = self.gfslt_dir / f"{file_id}{self.gfslt_postfix}.npy"
        if not gfslt_path.exists():
            raise FileNotFoundError(f"GFSLT feature file not found: {gfslt_path}")
            
        return torch.tensor(np.load(gfslt_path), dtype=torch.float32)


    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a dataset item by index."""
        data = self.data[index]
        file_id = data['fileid']
        pixel_value = None
        glor_value = None
        pose_value = None
        gfslt_value = None
        
        # Load spatial features if enabled
        if self.spatial:
            try:
                pixel_value = self._load_spatial_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                pixel_value = torch.tensor([])
        
        # Load spatiotemporal features if enabled
        if self.spatiotemporal:
            try:
                glor_value = self._load_spatiotemporal_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                if isinstance(self.spatiotemporal_postfix, str):
                    glor_value = torch.tensor([])
                else:
                    glor_value = [torch.tensor([])]

        # Load pose features if enabled
        if self.pose:
            try:
                pose_value = self._load_pose_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                pose_value = torch.tensor([])
                
        # Load gfslt features if enabled
        if self.gfslt:
            try:
                gfslt_value = self._load_gfslt_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                gfslt_value = torch.tensor([])
        
        # Calculate dynamic frame length based on whatever feature is available
        num_frames = 0
        if gfslt_value is not None and len(gfslt_value) > 0:
            num_frames = len(gfslt_value)
        elif pixel_value is not None and len(pixel_value) > 0:
            num_frames = len(pixel_value)
        elif glor_value is not None and isinstance(glor_value, torch.Tensor) and len(glor_value) > 0:
            num_frames = len(glor_value)
        
        # Create result dictionary with normalized text
        result = {
            'pixel_value': pixel_value,
            'glor_value': glor_value,
            'pose_value': pose_value,
            'gfslt_value': gfslt_value,     # <--- NEW
            'bool_mask_pos': None,
            'text': self._normalize_text(data['text']),
            'gloss': data['gloss'],
            'id': file_id,
            'num_frames': num_frames,
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
        """Normalize text by ensuring it ends with a period."""
        text = text.strip()
        if not text.endswith('.'):
            text = f"{text}."
        return text

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.data) - 1

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch
import torch
import os
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from spamo.constants import *
import random


class How2Sign(torch.utils.data.Dataset):
    """
    Dataset class for the How2Sign sign language dataset.
    
    This class handles loading video features and annotations for How2Sign,
    specifically adapted for the .npy metadata format where clips are pre-trimmed.
    """
    def __init__(
        self,
        anno_root: str,
        vid_root: str,
        feat_root: str,
        mae_feat_root: str,
        mode: str = 'test',
        spatial: bool = False,
        spatiotemporal: bool = False,
        spatial_postfix: str = '',
        spatiotemporal_postfix: Union[str, List[str]] = ''
    ):
        """
        Initialize the How2Sign dataset.
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
        
        # Validate inputs
        if not (spatial or spatiotemporal):
            raise ValueError("At least one of 'spatial' or 'spatiotemporal' must be True")
        
        # Load annotations (Looking for test_info.npy)
        anno_path = self.anno_root / f'{mode}_info.npy'

        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        
        full_data = np.load(anno_path, allow_pickle=True).item()
        
        # Filter out 'prefix' key and ensure we only have integer-indexed data
        self.data = {k: v for k, v in full_data.items() if isinstance(k, (int, np.integer))}
        
        # Set up directory paths
        # Note: If your feature reader saves directly into mode folders:
        self.spatial_dir = self.feat_root / self.mode
        self.spatiotemporal_dir = self.mae_feat_root / self.mode
        
        # Validate that key directories exist
        self._validate_directories()

    def _validate_directories(self) -> None:
        """Validate that all necessary directories exist."""
        if self.spatial and not self.spatial_dir.exists():
            raise FileNotFoundError(f"Spatial feature directory not found: {self.spatial_dir}")
        
        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            raise FileNotFoundError(f"Spatiotemporal feature directory not found: {self.spatiotemporal_dir}")

    def _get_feature_filename(self, file_id: str, start_time: Any, postfix: str) -> str:
        """
        Reconstruct the filename logic. 
        If START_REALIGNED is null (None), it omits the timestamp to match standard saving.
        """
        if start_time is None or str(start_time).lower() == 'none':
            return f"{file_id}{postfix}.npy"
        return f"{file_id}_{start_time}{postfix}.npy"

    def _load_spatial_features(self, file_id: str, start_time: Any) -> torch.Tensor:
        """Load spatial features for a given file ID."""
        fname = self._get_feature_filename(file_id, start_time, self.spatial_postfix)
        feat_path = self.spatial_dir / fname
        
        if not feat_path.exists():
            raise FileNotFoundError(f"Spatial feature file not found: {feat_path}")
        
        return torch.tensor(np.load(feat_path))

    def _load_spatiotemporal_features(self, file_id: str, start_time: Any) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Load spatiotemporal features for a given file ID."""
        if isinstance(self.spatiotemporal_postfix, str):
            fname = self._get_feature_filename(file_id, start_time, self.spatiotemporal_postfix)
            glor_path = self.spatiotemporal_dir / fname
            if not glor_path.exists():
                raise FileNotFoundError(f"Spatiotemporal feature file not found: {glor_path}")
            return torch.tensor(np.load(glor_path))
        else:
            features = []
            for postfix in self.spatiotemporal_postfix:
                fname = self._get_feature_filename(file_id, start_time, postfix)
                path = self.spatiotemporal_dir / fname
                if not path.exists():
                    raise FileNotFoundError(f"Spatiotemporal feature file not found: {path}")
                features.append(torch.tensor(np.load(path)))
            return features

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a dataset item by index."""
        data = self.data[index]
        file_id = data['fileid']
        start_time = data['original_info'].get('START_REALIGNED', None)
        
        pixel_value = None
        glor_value = None
        
        # Load spatial features if enabled
        if self.spatial:
            try:
                pixel_value = self._load_spatial_features(file_id, start_time)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                pixel_value = torch.tensor([])
        
        # Load spatiotemporal features if enabled
        if self.spatiotemporal:
            try:
                glor_value = self._load_spatiotemporal_features(file_id, start_time)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                if isinstance(self.spatiotemporal_postfix, str):
                    glor_value = torch.tensor([])
                else:
                    glor_value = [torch.tensor([])]
        
        # Create result dictionary with How2Sign normalization
        result = {
            'pixel_value': pixel_value,
            'glor_value': glor_value,
            'bool_mask_pos': None,
            'text': self._normalize_text(data['text']),
            'gloss': data['gloss'],
            'id': file_id,
            'num_frames': len(pixel_value) if pixel_value is not None and pixel_value.numel() > 0 else 0,
            'vid_path': data['folder'],
            'lang': 'English'
        }
        
        # Store original data for reference
        result['original_info'] = data
        
        return result

    def _normalize_text(self, text: str) -> str:
        """Normalize text for SLT (ensure period at end)."""
        text = text.strip()
        if not text.endswith('.'):
            text = f"{text}."
        return text

    def __len__(self) -> int:
        """Get the number of items in the dataset (prefix excluded)."""
        return len(self.data)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch
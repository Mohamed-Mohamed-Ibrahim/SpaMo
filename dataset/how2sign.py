import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Dict, Any

class How2Sign(torch.utils.data.Dataset):
    """
    How2Sign Dataset class for test/eval splits.
    Handles data validation and feature loading (spatial/spatiotemporal).
    """

    def __init__(
        self,
        anno_root: str,
        vid_root: str,
        feat_root: str,
        mae_feat_root: str,
        mode: str = "test",
        spatial: bool = False,
        spatiotemporal: bool = False,
        spatial_postfix: str = "",
        spatiotemporal_postfix: Union[str, List[str]] = "",
    ):
        super().__init__()
        self.anno_root = Path(anno_root)
        self.vid_root = Path(vid_root)
        self.spatial_dir = Path(feat_root)
        self.spatiotemporal_dir = Path(mae_feat_root)

        self.mode = mode
        self.spatial = spatial
        self.spatiotemporal = spatiotemporal
        self.spatial_postfix = spatial_postfix
        self.spatiotemporal_postfix = spatiotemporal_postfix

        # Load and validate annotations
        if not self.anno_root.exists():
            raise FileNotFoundError(f"Annotation file missing: {self.anno_root}")

        raw_data = np.load(self.anno_root, allow_pickle=True).item()
        
        # Filter for valid dict entries containing 'fileid' to prevent key errors
        self.data = raw_data
        self.valid_keys = [
            k for k, v in raw_data.items() 
            if isinstance(v, dict) and "fileid" in v
        ]
        print(f"Initialized {len(self.valid_keys)}/{len(raw_data)} valid samples.")
        
        self._validate_dirs()

    def _validate_dirs(self):
        """Ensure feature directories exist if flags are enabled."""
        if self.spatial and not self.spatial_dir.exists():
            raise FileNotFoundError(f"Spatial dir missing: {self.spatial_dir}")
        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            raise FileNotFoundError(f"Spatiotemporal dir missing: {self.spatiotemporal_dir}")

    def _load_feature(self, path: Path) -> torch.Tensor:
        """Safe loader: returns empty tensor if file missing."""
        if not path.exists():
            print(f"[WARN] Missing feature: {path}")
            return torch.tensor([])
        return torch.tensor(np.load(path))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.valid_keys[idx]
        d = self.data[key]
        file_id = d["fileid"]

        # 1. Load Spatial Features
        pixel_val = torch.tensor([])
        if self.spatial:
            pixel_val = self._load_feature(self.spatial_dir / f"{file_id}{self.spatial_postfix}.npy")

        # 2. Load Spatiotemporal Features (Single or List)
        glor_val = torch.tensor([])
        if self.spatiotemporal:
            post = self.spatiotemporal_postfix
            if isinstance(post, list):
                # Load multiple and handle as list; logic may require stacking depending on model
                glor_val = [self._load_feature(self.spatiotemporal_dir / f"{file_id}{p}.npy") for p in post]
            else:
                glor_val = self._load_feature(self.spatiotemporal_dir / f"{file_id}{post}.npy")

        # 3. Construct Output
        return {
            "pixel_value": pixel_val,
            "glor_value": glor_val,
            "bool_mask_pos": None,
            "text": d.get("text", ""),
            "gloss": d.get("gloss", ""),
            "id": file_id,
            "num_frames": len(pixel_val) if isinstance(pixel_val, torch.Tensor) and len(pixel_val) > 0 else d.get("num_frames", 0),
            "vid_path": str(self.vid_root),
            "lang": "English",
            "original_info": d,
        }

    def __len__(self) -> int:
        return len(self.valid_keys)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch
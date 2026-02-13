import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union


class How2Sign(torch.utils.data.Dataset):
    """
    Dataset class for How2Sign test/eval splits.
    Works similarly to Phoenix14T but adapted to How2Sign metadata structure.
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
        self.feat_root = Path(feat_root)
        self.mae_feat_root = Path(mae_feat_root)

        self.mode = mode
        self.spatial = spatial
        self.spatiotemporal = spatiotemporal
        self.spatial_postfix = spatial_postfix
        self.spatiotemporal_postfix = spatiotemporal_postfix

        # Load annotation dictionary created during preprocessing
        if not self.anno_root.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.anno_root}")

        self.data = np.load(self.anno_root, allow_pickle=True).item()

        self.spatial_dir = Path(self.feat_root)
        self.spatiotemporal_dir = Path(self.mae_feat_root)

        self._validate_directories()

    def _validate_directories(self):
        if self.spatial and not self.spatial_dir.exists():
            raise FileNotFoundError(f"Spatial feature dir missing: {self.spatial_dir}")

        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            raise FileNotFoundError(
                f"Spatiotemporal feature dir missing: {self.spatiotemporal_dir}"
            )

    # ------------ FEATURE LOADERS ------------------

    def _load_spatial(self, file_id: str):
        path = self.spatial_dir / f"{file_id}{self.spatial_postfix}.npy"
        if not path.exists():
            print(f"[WARN] Missing spatial feature: {path}")
            return torch.tensor([])
        return torch.tensor(np.load(path))

    def _load_spatiotemporal(self, file_id: str):
        if isinstance(self.spatiotemporal_postfix, str):
            path = self.spatiotemporal_dir / f"{file_id}{self.spatiotemporal_postfix}.npy"
            if not path.exists():
                print(f"[WARN] Missing motion feature: {path}")
                return torch.tensor([])
            return torch.tensor(np.load(path))

        # multiple features
        tensors = []
        for p in self.spatiotemporal_postfix:
           fp = self.spatiotemporal_dir / f"{file_id}{p}.npy"
           if not fp.exists():
            print(f"[WARN] Missing motion feature: {fp}")
            tensors.append(torch.tensor([]))
           else:
            tensors.append(torch.tensor(np.load(fp)))
        return tensors

    # --------------- MAIN ENTRY ---------------------

    def __getitem__(self, idx):
        d = self.data[idx]
        file_id = d["fileid"]

        pixel_value = self._load_spatial(file_id) if self.spatial else torch.tensor([])
        glor_value = (
            self._load_spatiotemporal(file_id)
            if self.spatiotemporal
            else torch.tensor([])
        )

        return {
            "pixel_value": pixel_value,
            "glor_value": glor_value,
            "bool_mask_pos": None,
            "text": d.get("text", ""),
            "gloss": d.get("gloss", ""),
            "id": file_id,
            "num_frames": len(pixel_value)
            if isinstance(pixel_value, torch.Tensor)
            else 0,
            "vid_path": str(self.vid_root),
            "lang": "English",
            "original_info": d,
        }

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return batch
